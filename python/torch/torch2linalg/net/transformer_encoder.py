"""
Transformer Encoder Layer → linalg MLIR 转换脚本

使用 torch.nn.TransformerEncoderLayer（PyTorch 内置，无额外依赖）。
输入 (B, S, d_model)，B 和 S 都可标记为动态维度。

用法:
    python transformer_encoder.py                  # 静态 shape
    python transformer_encoder.py --dynamic        # B 和 S 动态
"""

import argparse
import torch
import torch.nn as nn
from torch.export import Dim
from torch_mlir.fx import export_and_import, OutputType


def build_model(d_model: int, nhead: int, dim_ff: int) -> nn.Module:
    """构造单层 TransformerEncoder。

    用 nn.TransformerEncoder 包一层是为了通过 enable_nested_tensor=False
    明确禁掉 nested-tensor fast path，避免 torch.export 路径切换问题。
    """
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_ff,
        batch_first=True,
        norm_first=False,
    )
    return nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)


def main():
    parser = argparse.ArgumentParser(
        description="TransformerEncoderLayer → linalg MLIR")
    parser.add_argument("-o", "--output", type=str,
                        default="transformer_encoder.mlir")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32",
                        help="fp32 推荐：避免 layernorm/softmax 数值问题")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim-ff", type=int, default=512)
    parser.add_argument("--dynamic", action="store_true",
                        help="将 batch 和 seq_len 标记为动态")
    parser.add_argument("--keep-weights", action="store_true")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    model = build_model(args.d_model, args.nhead, args.dim_ff).eval().to(dtype)

    # 动态 trace 必须用 size >= 2 避免特化为常量
    trace_b = max(args.batch, 2) if args.dynamic else args.batch
    trace_s = max(args.seq, 2) if args.dynamic else args.seq
    x = torch.randn(trace_b, trace_s, args.d_model, dtype=dtype)

    dynamic_shapes = None
    if args.dynamic:
        b_dim = Dim("B", min=1, max=128)
        s_dim = Dim("S", min=1, max=2048)
        # nn.TransformerEncoder.forward(self, src, mask=None, ...) 第一个参数名 "src"
        dynamic_shapes = {"src": {0: b_dim, 1: s_dim}}

    module = export_and_import(
        model, x,
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