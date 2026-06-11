import argparse

import torch
from torch.export import Dim
from torch_mlir.fx import export_and_import, OutputType
from transformers import AutoModelForCausalLM
import transformers.models.qwen2.modeling_qwen2 as _qwen2_mod


def _simple_causal_mask(config, inputs_embeds, attention_mask, past_key_values=None, position_ids=None, **kwargs):
    """替换 create_causal_mask：避免其内部 mask_interface/vmap 路径对 torch.export 动态 shape 不友好。"""
    batch, seq, _ = inputs_embeds.shape
    dtype = inputs_embeds.dtype
    device = inputs_embeds.device
    positions = torch.arange(seq, device=device)
    causal = torch.zeros((batch, 1, seq, seq), dtype=dtype, device=device)
    causal = causal.masked_fill(positions.view(1, 1, 1, seq) > positions.view(1, 1, seq, 1), torch.finfo(dtype).min)
    return causal


_qwen2_mod.create_causal_mask = _simple_causal_mask


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """repeat_interleave 替代 expand+reshape，避免非连续 stride 产生 min(64S,448S) guard。"""
    if n_rep == 1:
        return hidden_states
    return hidden_states.repeat_interleave(n_rep, dim=1)


_qwen2_mod.repeat_kv = _repeat_kv


class _DynamicRoPE(torch.nn.Module):
    """RoPE without precomputed cos/sin cache (cache 变成大 literal，torch_mlir 无法 lower)."""

    def __init__(self, orig):
        super().__init__()
        self.inv_freq = orig.inv_freq  # shape [head_dim/2]

    def forward(self, x, position_ids):
        # position_ids: [batch, seq]
        # 保持 D (inv_freq 维度) 在最后，避免中间出现 [B, D, S] 让 solver 误判 D 为动态
        pos = position_ids.float().unsqueeze(-1)  # [B, S, 1]
        inv = self.inv_freq.float()               # [D]
        freqs = pos * inv                         # [B, S, D]  D 是静态维
        emb = torch.cat([freqs, freqs], dim=-1)   # [B, S, 2D]
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class Qwen2Forward(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        # eager: 避免 sdpa/flash_attn 的 trace 问题
        # float32: 避免 bfloat16 在 torch_mlir 的兼容问题
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        model.eval()
        model.config.use_cache = False
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.use_cache = False

        # 替换 model 级别的 RoPE，避免预计算 cache 变成大 literal
        model.model.rotary_emb = _DynamicRoPE(model.model.rotary_emb)

        self.model = model

    def forward(self, input_ids):
        batch, seq = input_ids.shape
        torch._check(seq > 0)  # 显式告知 solver S>0，帮助化简 min(64*S, 448*S)
        positions = torch.arange(seq, device=input_ids.device)
        position_ids = positions.unsqueeze(0).expand(batch, -1)
        attention_mask = torch.ones((batch, seq), dtype=torch.long, device=input_ids.device)
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=False,
            use_cache=False,
        )[0]


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5 -> linalg MLIR")
    parser.add_argument("-o", "--output", type=str, default="qwen2_5.mlir")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=8)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--keep-weights", action="store_true")
    args = parser.parse_args()

    model = Qwen2Forward(args.model)
    trace_batch = max(args.batch, 2) if args.dynamic else args.batch
    trace_seq = max(args.seq, 2) if args.dynamic else args.seq
    input_ids = torch.zeros((trace_batch, trace_seq), dtype=torch.int64)

    dynamic_shapes = None
    if args.dynamic:
        max_positions = model.model.config.max_position_embeddings
        b_dim = Dim("B", min=1, max=128)
        s_dim = Dim("S", min=1, max=max_positions)
        dynamic_shapes = {
            "input_ids": {0: b_dim, 1: s_dim},
        }

    module = export_and_import(
        model,
        input_ids,
        func_name="kernel",
        output_type=OutputType.LINALG_ON_TENSORS,
        dynamic_shapes=dynamic_shapes,
    )

    if args.keep_weights:
        mlir_text = module.operation.get_asm()
    else:
        mlir_text = module.operation.get_asm(
            large_elements_limit=16,
            large_resource_limit=16,
        )

    with open(args.output, "w") as f:
        f.write(mlir_text)
    lines = mlir_text.count("\n")
    print(f"写入 {args.output} ({lines} 行, {len(mlir_text)} 字节)")


if __name__ == "__main__":
    main()