"""TensorSpec: 张量形状描述符，支持动态/静态维度声明。"""

from dataclasses import dataclass
from typing import Any

import torch
from torch.export import Dim


@dataclass
class TensorSpec:
    """描述张量的形状模式和数据类型。

    shape 中每个维度可以是：
      - int: 静态维度（IR 里是固定值）
      - str: 命名动态维度（同名跨 spec 共享同一个 Dim，IR 里是 ?）
      - None: 匿名动态维度（每个独立，IR 里是 ?）

    例：("M", "N") 两个命名动态维度，(32, "N") 第一维静态。
    """
    shape: tuple
    dtype: torch.dtype = torch.float16

    def make_sample(self, dim_sizes: dict[str, int] | None = None) -> torch.Tensor:
        """生成具体 tensor。

        Args:
            dim_sizes: 动态维度名 → 具体大小的映射，如 {"M": 128}。
                       未指定的动态维度默认为 64。
        """
        sizes = dim_sizes or {}
        concrete = tuple(
            sizes.get(d, 64) if isinstance(d, str) else
            64 if d is None else
            d
            for d in self.shape
        )
        return torch.randn(concrete, dtype=self.dtype)

    def dynamic_dims(self, dim_pool: dict[str, Any] | None = None) -> dict[int, Any]:
        """返回 torch.export 需要的 dynamic_shapes 映射。

        Args:
            dim_pool: 共享 Dim 池，同名 str 维度复用同一个 Dim 对象。
                      调用方传入一个 dict，本方法会自动填充。
        """
        if dim_pool is None:
            dim_pool = {}
        result = {}
        for i, d in enumerate(self.shape):
            if isinstance(d, int):
                continue
            if isinstance(d, str):
                if d not in dim_pool:
                    dim_pool[d] = Dim(d)
                result[i] = dim_pool[d]
            else:  # None
                result[i] = Dim(f"_anon_{id(self)}_{i}")
        return result