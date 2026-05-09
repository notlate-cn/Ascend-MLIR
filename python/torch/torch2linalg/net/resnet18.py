"""
ResNet-18 → linalg MLIR 转换脚本

用法:
    python resnet18.py                  # 输出到 stdout
    python resnet18.py -o resnet18.mlir # 输出到文件
"""

import argparse
import torch
import torchvision.models as models
from torch.export import Dim
from torch_mlir.fx import export_and_import, OutputType


def main():
    parser = argparse.ArgumentParser(description="ResNet-18 → linalg MLIR")
    parser.add_argument("-o", "--output", type=str, default="resnet18.mlir",
                        help="输出文件路径")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dynamic", action="store_true",
                        help="将 batch 维度标记为动态 "
                             "(C 受 conv 权重约束、H/W 受 AdaptiveAvgPool 约束都必须静态)")
    parser.add_argument("--keep-weights", action="store_true",
                        help="保留权重数据；默认省略以减小文件体积")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    model = models.resnet18(weights=None).eval().to(dtype)

    # 动态 trace 必须用 size >= 2 的样本，否则 torch.export 会把 dim 特化为常量
    trace_batch = max(args.batch, 2) if args.dynamic else args.batch
    x = torch.randn(trace_batch, 3, 224, 224, dtype=dtype)

    dynamic_shapes = None
    if args.dynamic:
        # Channel C 由 conv1 权重 (64x3x7x7) 决定，必须是 3
        # H/W 须为 32 的倍数（5 次 stride=2 下采样）+ 不小于 224
        n_dim = Dim("N", min=1, max=64)
        h_dim = 32 * Dim("H_div32", min=7, max=64)
        w_dim = 32 * Dim("W_div32", min=7, max=64)
        dynamic_shapes = {"x": {0: n_dim, 2: h_dim, 3: w_dim}}

    module = export_and_import(
        model, x,
        func_name="kernel",
        output_type=OutputType.LINALG_ON_TENSORS,
        dynamic_shapes=dynamic_shapes,
    )
    if args.keep_weights:
        mlir_text = module.operation.get_asm()
    else:
        # 省略 >16 个元素的权重数据，IR 结构可读但保留 shape/dtype
        mlir_text = module.operation.get_asm(
            large_elements_limit=16,
            large_resource_limit=16,
        )

    if args.output:
        with open(args.output, "w") as f:
            f.write(mlir_text)
        lines = mlir_text.count("\n")
        print(f"写入 {args.output} ({lines} 行, {len(mlir_text)} 字节)")
    else:
        print(mlir_text)


if __name__ == "__main__":
    main()