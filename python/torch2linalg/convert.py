"""
torch2linalg 核心转换模块

将 PyTorch nn.Module 通过 torch-mlir 转换为 linalg MLIR IR。
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch_mlir.fx import export_and_import, OutputType


def torch_to_linalg(
    model: nn.Module,
    sample_inputs: list[torch.Tensor],
    output_path: Optional[str | Path] = None,
) -> str:
    """
    将 PyTorch 模型转换为 linalg MLIR IR 文本。

    Args:
        model: PyTorch 模型（调用方负责 dtype）
        sample_inputs: 示例输入张量（调用方负责 dtype）
        output_path: 可选，输出 .mlir 文件路径

    Returns:
        linalg MLIR IR 文本
    """
    model = model.eval()
    inputs = tuple(sample_inputs)

    # torch-mlir: export + import → linalg
    module = export_and_import(model, *inputs, output_type=OutputType.LINALG_ON_TENSORS)
    mlir_text = module.operation.get_asm()

    # 可选：写入文件
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(mlir_text)

    return mlir_text