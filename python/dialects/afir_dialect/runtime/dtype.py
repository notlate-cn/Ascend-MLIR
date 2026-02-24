# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

from enum import Enum
import re
from typing import Optional

import mlir.ir as ir

class DataType:
    _re_matcher = re.compile("^(int|float|uint)([0-9]+)$")

    class Kind(Enum):
        Any = ""
        Int = "int"
        Float = "float"
        UInt = "uint"

    def __init__(self, name: str):
        self.name = name
        self.kind = self.Kind.Any
        self.bitwidth: Optional[int] = None
        match = self._re_matcher.match(name)
        if match:
            self.kind = self.Kind(match[1])
            self.bitwidth = int(match[2])

    def __eq__(self, other: DataType) -> bool:
        if not isinstance(other, DataType):
            return False
        return self.name == other.name

    def __neq__(self, other: DataType) -> bool:
        return not (self == other)

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_ir(cls, ir_type: ir.Type):
        name = ir_type.__str__()
        if name:
            return cls(name)
        raise ValueError(f"IR type doesn't have a Pythonic name: {ir_type}")

    def is_int(self) -> bool:
        return self.is_signed() or self.is_unsigned()

    def is_float(self) -> bool:
        return self.kind == self.Kind.Float

    def is_numeric(self) -> bool:
        return self.is_int() or self.is_float()

    def is_signed(self) -> bool:
        return self.kind == self.Kind.Int

    def is_unsigned(self) -> bool:
        return self.kind == self.Kind.UInt

    def is_void(self) -> bool:
        return self == void

    def sizeof(self) -> int:
        if self.bitwidth is None:
            raise ValueError(f"DataType does not have a bitwidth: {self.name}")
        div, mod = divmod(self.bitwidth, 8)
        if mod != 0:
            raise ValueError(f"DataType bitwidth does not fit an integer number of bytes: {self.bitwidth}")
        return div

    def to_ir(self) -> ir.Type:
        factories = {
            "void": ir.none,
            "int1": ir.bool,
            "int8": ir.si8,
            "int16": ir.si16,
            "int32": ir.si32,
            "int64": ir.si64,
            "uint8": ir.ui8,
            "uint16": ir.ui16,
            "uint32": ir.ui32,
            "uint64": ir.ui64,
            "float16": ir.f16,
            "float32": ir.f32,
            "float64": ir.f64,
        }
        factory = factories.get(self.name)
        if factory:
            return factory()
        raise ValueError(f"Unsupported DataType name: {self.name}")


# Predefined data types
void = DataType("void")
int1 = DataType("int1")
int8 = DataType("int8")
int16 = DataType("int16")
int32 = DataType("int32")
int64 = DataType("int64")
float16 = DataType("float16")
float32 = DataType("float32")
float64 = DataType("float64")
uint8 = DataType("uint8")
uint16 = DataType("uint16")
uint32 = DataType("uint32")
uint64 = DataType("uint64")
bit = int1
bool_ = int8
int_ = int32
half = float16
float_ = float32
double = float64


class KnownTypes:
    void = void
    int1 = int1
    int8 = int8
    int16 = int16
    int32 = int32
    int64 = int64
    float16 = float16
    float32 = float32
    float64 = float64
    uint8 = uint8
    uint16 = uint16
    uint32 = uint32
    uint64 = uint64
    bit = bit
    bool_ = int8
    int_ = int32
    half = half
    float_ = float_
    double = double
