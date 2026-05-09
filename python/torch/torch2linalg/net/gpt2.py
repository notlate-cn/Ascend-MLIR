import argparse

import torch
from torch.export import Dim
from torch_mlir.fx import export_and_import, OutputType
from transformers import AutoModelForCausalLM


class GPT2Forward(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        model.config.use_cache = False
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.use_cache = False
        self.model = model

    def forward(self, input_ids):
        batch, seq = input_ids.shape
        positions = torch.arange(seq, device=input_ids.device)
        q_pos = positions.view(1, 1, seq, 1)
        k_pos = positions.view(1, 1, 1, seq)
        allowed = k_pos <= q_pos
        attention_mask = torch.zeros(
            (batch, 1, seq, seq),
            dtype=self.model.dtype,
            device=input_ids.device,
        )
        attention_mask = attention_mask.masked_fill(
            ~allowed,
            torch.finfo(self.model.dtype).min,
        )
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
            use_cache=False,
        )[0]


def main():
    parser = argparse.ArgumentParser(description="openai-community/gpt2 -> linalg MLIR")
    parser.add_argument("-o", "--output", type=str, default="gpt2.mlir")
    parser.add_argument("--model", type=str, default="openai-community/gpt2")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=8)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--keep-weights", action="store_true")
    args = parser.parse_args()

    model = GPT2Forward(args.model)
    trace_batch = max(args.batch, 2) if args.dynamic else args.batch
    trace_seq = max(args.seq, 2) if args.dynamic else args.seq
    input_ids = torch.zeros((trace_batch, trace_seq), dtype=torch.int64)

    dynamic_shapes = None
    if args.dynamic:
        max_positions = model.model.config.max_position_embeddings
        b_dim = Dim("B", min=1, max=128)
        s_dim = Dim("S", min=1, max=max_positions)
        dynamic_shapes = {"input_ids": {0: b_dim, 1: s_dim}}

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