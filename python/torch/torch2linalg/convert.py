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
    dynamic_shapes: dict | None = None,
) -> str:
    """
    将 PyTorch 模型转换为 linalg MLIR IR 文本。

    Args:
        model: PyTorch 模型（调用方负责 dtype）
        sample_inputs: 示例输入张量（调用方负责 dtype）
        dynamic_shapes: 可选，torch.export 的动态维度映射

    Returns:
        linalg MLIR IR 文本
    """
    model = model.eval()
    inputs = tuple(sample_inputs)

    # torch-mlir: export + import → linalg
    kwargs = dict(output_type=OutputType.LINALG_ON_TENSORS)
    if dynamic_shapes is not None:
        kwargs["dynamic_shapes"] = dynamic_shapes
    module = export_and_import(model, *inputs, func_name="kernel", **kwargs)
    mlir_text = module.operation.get_asm()

    return mlir_text