"""
torch2linalg - 将 PyTorch 模型转换为 linalg MLIR IR

依赖：
  pip install --pre torch-mlir -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
  pip install torch --index-url https://download.pytorch.org/whl/cpu

用法：
  from torch2linalg import torch_to_linalg
  mlir_text = torch_to_linalg(model, sample_inputs)
"""

from .convert import torch_to_linalg
from .pytest_plugin import torch_e2e_test

__all__ = ["torch_to_linalg", "torch_e2e_test"]