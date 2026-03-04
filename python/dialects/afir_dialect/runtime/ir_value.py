# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import abc
from typing import Any, NoReturn, Optional, Union, Self, TypeAlias

from .dtype import DataType, KnownTypes as KT

import mlir.ir as ir

import mlir.dialects.arith as arith
import mlir.dialects.afir as afir
import mlir.dialects.math as mlirMath

IRHandle: TypeAlias = ir.Value


class IRValue(abc.ABC):

    # @classmethod
    # @abc.abstractmethod
    # def from_ir(cls, handle: IRHandle) -> Self:
    #     raise NotImplementedError

    @abc.abstractmethod
    def to_ir(self) -> IRHandle:
        raise NotImplementedError


# class GlobalAddress(IRValue):
# 
#     def __init__(self, handle: IRHandle, dtype: Optional[DataType] = None):
#         """This contructor should not be called by user"""
#         self.handle = handle
#         self.dtype = dtype
# 
#     def __repr__(self) -> str:
#         return f"GlobalAddress(dtype={self.dtype}, handle=...)"
# 
#     
#     def __add__(self, offset: "RuntimeInt") -> GlobalAddress:
#         offset = materialize_ir_value(offset, KT.int_)
#         builder = global_builder.get_ir_builder()
#         offset_index = builder.create_arith_IndexCastOp(offset.to_ir(), builder.get_index_type())
#         handle = builder.create_emitasc_PtrOffsetOp(self.to_ir(), offset_index)
#         return GlobalAddress(handle, self.dtype)
# 
#     @classmethod
#     def from_ir(cls, handle: IRHandle) -> Self:
#         return GlobalAddress(handle, DataType.from_ir(ir.get_element_type(handle.get_type())))
# 
#     def to_ir(self) -> IRHandle:
#         return self.handle

BinaryOperationIndex = {
    "Add": ("AddI", "AddF"),
    "Sub": ("SubI", "SubF"),
    "Mul": ("MulI", "MulF"),
    "TrueDiv": ("DivF"),
    "FloorDiv": ("DivSI"),
    "Pow": ("Pow"),
    "BitwiseAnd": ("AndI")
}

# CompareOperationIndex = {
#     
# }

class PlainValue(IRValue):

    def __init__(self, handle: IRHandle, dtype: Optional[ir.Type] = None, loc : Optional[ir.Location] = None, context : Optional[ir.Context] = None):
        """This contructor should not be called by user"""
        self.handle = handle
        self.dtype = dtype
        self.loc = loc
        self.context = context

    
    def __rxor__(self, other) -> PlainValue:
        return self.apply_binary_op(other, self, "XOrI", None)

    # Binary operations

    
    def __add__(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "Add")

    
    def __sub__(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "Sub")

    
    def __mul__(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "Mul")

    
    def __truediv__(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "TrueDiv")

    
    def __floordiv__(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "FloorDiv")

    
    def __mod__(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "RemSI", None)

    
    def __pow__(self, other) -> NoReturn:
        # raise NotImplementedError("Power operator is not implemented for PlainValue")
        return self.apply_binary_op(self, other, "Pow")

    
    def __lshift__(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "ShLI", None)

    
    def __rshift__(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "ShRSI", None)

    
    def __and__(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "AndI", None)

    
    def __or__(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "OrI", None)

    
    def __xor__(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "XOrI", None)

    def __repr__(self) -> str:
        return f"PlainValue(dtype={self.dtype}, handle=...)"

    # Binary operations (reversed)

    
    def __radd__(self, other) -> PlainValue:
        return self.apply_binary_op(other, self, "Add")

    
    def __rsub__(self, other) -> PlainValue:
        return self.apply_binary_op(other, self, "Sub")

    
    def __rmul__(self, other) -> PlainValue:
        return self.apply_binary_op(other, self, "Mul")

    
    def __rtruediv__(self, other) -> PlainValue:
        return self.apply_binary_op(other, self, "TrueDiv")

    
    def __rfloordiv__(self, other) -> PlainValue:
        return self.apply_binary_op(other, self, "FloorDiv")

    
    def __rmod__(self, other) -> PlainValue:
        return self.apply_binary_op(other, self, "RemSI", None)

    
    def __rpow__(self, other) -> NoReturn:
        # raise NotImplementedError("Power operator is not implemented for PlainValue")
        return self.apply_binary_op(other, self, "Pow")

    
    def __rlshift__(self, other) -> PlainValue:
        return self.apply_binary_op(other, self, "ShLI", None)

    
    def __rrshift__(self, other) -> PlainValue:
        return self.apply_binary_op(other, self, "ShRSI", None)

    
    def __rand__(self, other) -> PlainValue:
        return self.apply_binary_op(other, self, "AndI", None)

    
    def __ror__(self, other) -> PlainValue:
        return self.apply_binary_op(other, self, "OrI", None)

    # Comparison operations

    
    def __eq__(self, other) -> PlainValue:
        return self.apply_compare_op(self, other, arith.CmpIPredicate.eq, arith.CmpFPredicate.OEQ)

    
    def __ne__(self, other) -> PlainValue:
        return self.apply_compare_op(self, other, arith.CmpIPredicate.ne, arith.CmpFPredicate.ONE)

    
    def __ge__(self, other) -> PlainValue:
        return self.apply_compare_op(self, other, arith.CmpIPredicate.sge, arith.CmpFPredicate.OGE)

    
    def __gt__(self, other) -> PlainValue:
        return self.apply_compare_op(self, other, arith.CmpIPredicate.sgt, arith.CmpFPredicate.OGT)

    
    def __le__(self, other) -> PlainValue:
        return self.apply_compare_op(self, other, arith.CmpIPredicate.sle, arith.CmpFPredicate.OLE)

    
    def __lt__(self, other) -> PlainValue:
        return self.apply_compare_op(self, other, arith.CmpIPredicate.slt, arith.CmpFPredicate.OLT)

    # Comparison operations (reversed)

    
    def __req__(self, other) -> PlainValue:
        return self.apply_compare_op(other, self, arith.CmpIPredicate.eq, arith.CmpFPredicate.OEQ)

    
    def __rne__(self, other) -> PlainValue:
        return self.apply_compare_op(other, self, arith.CmpIPredicate.ne, arith.CmpFPredicate.ONE)

    
    def __rge__(self, other) -> PlainValue:
        return self.apply_compare_op(other, self, arith.CmpIPredicate.sge, arith.CmpFPredicate.OGE)

    
    def __rgt__(self, other) -> PlainValue:
        return self.apply_compare_op(other, self, arith.CmpIPredicate.sgt, arith.CmpFPredicate.OGT)

    
    def __rle__(self, other) -> PlainValue:
        return self.apply_compare_op(other, self, arith.CmpIPredicate.sle, arith.CmpFPredicate.OLE)

    
    def __rlt__(self, other) -> PlainValue:
        return self.apply_compare_op(other, self, arith.CmpIPredicate.slt, arith.CmpFPredicate.OLT)

    # Unary operations

    
    def __neg__(self) -> PlainValue:
        if self.dtype.is_float():
            return arith.NegFOp(self.to_ir())
        return self.__mul__(-1)

    
    def __pos__(self) -> PlainValue:
        return self

    
    def __not__(self) -> PlainValue:
        return self.__eq__(0)

    
    def __invert__(self) -> PlainValue:
        raise NotImplementedError("Inversion operator is not implemented for PlainValue")

    @staticmethod
    def infer_common_type(lhs: PlainValue, rhs: PlainValue) -> ir.Type:
        result_type = lhs.dtype
        if rhs.dtype != lhs.dtype:
            if not isinstance(lhs.dtype, ir.RankedTensorType):
                return lhs.dtype if isinstance(rhs.dtype, ir.IntegerType) else rhs.dtype
            else:
                # return lhs.dtype if isinstance(rhs.dtype.element_type, ir.IntegerType) else rhs.dtype
                elementType = lhs.dtype.element_type if isinstance(rhs.dtype.element_type, ir.IntegerType) else rhs.dtype.element_type
                finalRank = max(lhs.dtype.rank, rhs.dtype.rank)
                rankOffset = finalRank - min(lhs.dtype.rank, rhs.dtype.rank)
                minShape, maxShape = [rhs.dtype.shape, lhs.dtype.shape] if finalRank == lhs.dtype.rank else [lhs.dtype.shape, rhs.dtype.shape]
                finalShape = maxShape[:rankOffset]
                for i, dim in enumerate(minShape):
                    if dim != 1 and dim != maxShape[i+rankOffset] and maxShape[i+rankOffset] != 1:
                        raise TypeError("input shape must be broadcastable to result shape")
                    finalShape.append(max(dim, maxShape[i+rankOffset]))
                return ir.RankedTensorType.get(finalShape, elementType)
                    

        
        return result_type

    @classmethod
    def apply_binary_op(cls, lhs: Any, rhs: Any, oper: str) -> PlainValue:
        if isinstance(lhs.dtype, ir.RankedTensorType) and isinstance(rhs.dtype, ir.RankedTensorType):
            handle = getattr(afir, f"{oper}Op")(lhs.handle, rhs.handle)
        else:
            result_type = cls.infer_common_type(lhs, rhs)
            if result_type != lhs.dtype:
                lhs = cls.cast(lhs, result_type)
            if result_type != rhs.dtype:
                rhs = cls.cast(rhs, result_type)
            if len(BinaryOperationIndex[oper]) == 1:
                builder_attr = BinaryOperationIndex[oper][0]
                if hasattr(arith, f"{builder_attr}Op"):
                    handle = getattr(arith, f"{builder_attr}Op")(lhs.handle, rhs.handle)
                elif hasattr(mlirMath, f"{builder_attr}Op"):
                    handle = getattr(mlirMath, f"{builder_attr}Op")(lhs.handle, rhs.handle)
                else:
                    raise ValueError(f"Binary operation {oper} is not supported between {lhs} and {rhs}")
            else:
                build_int, build_float = BinaryOperationIndex[oper]
                if isinstance(result_type, ir.VectorType):
                    builder_attr = build_int if arith._is_integer_like_type(result_type.element_type) else build_float
                elif isinstance(result_type, ir.Type):
                    builder_attr = build_int if arith._is_integer_like_type(result_type) else build_float
                if builder_attr is None:
                    raise ValueError(f"Binary operation is not supported between {lhs} and {rhs}")
                handle = getattr(arith, f"{builder_attr}Op")(lhs.handle, rhs.handle)
        return handle

    @classmethod
    def apply_bool_op(cls, lhs: Any, rhs: Any, builder_attr: str) -> PlainValue:
        lhs = materialize_ir_value(lhs, KT.bit)
        rhs = materialize_ir_value(rhs, KT.bit)
        handle = getattr(arith, f"{builder_attr}Op")(lhs.to_ir(), rhs.to_ir())
        return PlainValue(handle=handle, dtype=KT.bit)

    @classmethod
    def apply_compare_op(cls, lhs: Any, rhs: Any, pred_int: int, pred_float: int) -> PlainValue:
        common_type = cls.infer_common_type(lhs, rhs)
        if common_type != lhs.dtype:
            lhs = cls.cast(lhs, common_type)
        if common_type != rhs.dtype:
            rhs = cls.cast(rhs, common_type)
        if isinstance(common_type, ir.RankedTensorType):
            method, pred = [arith.CmpIOp, pred_int] if arith._is_integer_like_type(common_type.element_type) else [arith.CmpFOp, pred_float]
        elif isinstance(common_type, ir.Type):
            method, pred = [arith.CmpIOp, pred_int] if arith._is_integer_like_type(common_type) else [arith.CmpFOp, pred_float]
        handle = method(pred, lhs.handle, rhs.handle)
        return handle

    # @classmethod
    # def from_ir(cls, handle: IRHandle) -> Self:
    #     return PlainValue(handle, DataType.from_ir(handle.type))

        

    def cast(self, dtype: ir.Type) -> PlainValue:
        if self.dtype == dtype:
            return self
        baseFromType = self.dtype if not isinstance(self.dtype, ir.ShapedType) else self.dtype.element_type
        baseToType = dtype if not isinstance(dtype, ir.ShapedType) else dtype.element_type
        # if isinstance(self.dtype, ir.RankedTensorType) and isinstance(dtype, ir.RankedTensorType):
        #     return self.hiddenBroadcast(dtype)
        if baseFromType == baseToType:
            return self
        from_i = arith._is_integer_like_type(baseFromType)
        from_f = arith._is_float_type(baseFromType)
        to_i = arith._is_integer_like_type(baseToType)
        to_f = arith._is_float_type(baseToType)
        method = None
        if not (from_i or from_f) or not (to_i or to_f):
            pass
        elif baseFromType.width == baseToType.width:
            if from_f and to_i:
                method = arith.FPToSIOp
            elif from_i and to_f:
                method = arith.SIToFPOp
        elif (from_i and to_i) or (from_f and to_f):
            ext = baseFromType.width < baseToType.width
            if from_i:
                method = arith.ExtSIOp if ext else arith.TruncIOp
            else:
                method = arith.ExtFOp if ext else arith.TruncFOp
        if method is None:
            raise NotImplementedError(f"Arithmetic cast from {self.dtype} to {dtype} is not supported")
        return PlainValue(handle=method(dtype, self.handle).result, dtype=dtype)

    
    def ceildiv(self, other) -> PlainValue:
        return self.apply_binary_op(self, other, "CeilDivSI", None)

    # Logical (bool) operations

    
    def logical_and(self, other) -> PlainValue:
        return self.apply_bool_op(self, other, "AndI")

    
    def logical_or(self, other) -> PlainValue:
        return self.apply_bool_op(self, other, "OrI")

    def to_ir(self) -> IRHandle:
        return self.handle


RuntimeBool: TypeAlias = Union[PlainValue, bool]
RuntimeInt: TypeAlias = Union[PlainValue, int]
RuntimeFloat: TypeAlias = Union[PlainValue, float]
RuntimeNumeric: TypeAlias = Union[RuntimeInt, RuntimeFloat]


def materialize_ir_value(value: RuntimeNumeric, required_type: Optional[DataType] = None) -> PlainValue:
    if isinstance(value, PlainValue):
        return value if required_type is None else value.cast(required_type)
    if isinstance(value, IRValue):
        if required_type is not None:
            raise ValueError("Required type cannot be specified for IRValue which is not PlainValue")
        return value
    if not isinstance(value, (int, float)):
        raise TypeError(f"Unsupported value type for materialization: {value.__class__.__name__}")
    if required_type is not None:
        if required_type == KT.bit:
            value = bool(value)
        if required_type.is_int():
            value = int(value)
        elif required_type.is_float():
            value = float(value)

    return convert_value(value, required_type)


def convert_value(value: Any, required_type: Optional[DataType] = None) -> PlainValue:
    type_to_builder = {
        bool: {
            "bit": lambda x: arith.ConstantOp(ir.BoolAttr(x))
        },
        int: {
            "int1": lambda x: arith.ConstantOp(ir.BoolAttr(x)),
            "int8": lambda x: arith.ConstantOp(ir.SI8Attr(x)),
            "int16": lambda x: arith.ConstantOp(ir.SI16Attr(x)),
            "int32": lambda x: arith.ConstantOp(ir.IntegerType.get_signless(32), x),
            "int64": lambda x: arith.ConstantOp(ir.SI64Attr(x)),
            "uint8": lambda x: arith.ConstantOp(ir.UI8Attr(x)),
            "uint16": lambda x: arith.ConstantOp(ir.UI16Attr(x)),
            "uint32": lambda x: arith.ConstantOp(ir.UI32Attr(x)),
            "uint64": lambda x: arith.ConstantOp(ir.UI64Attr(x)),
        },
        float: {
            "float16": lambda x: arith.ConstantOp(ir.F16Type.get(), x),
            "float32": lambda x: arith.ConstantOp(ir.F32Attr(x)),
            "float64": lambda x: arith.ConstantOp(ir.F64Attr(x)),
        }
    }

    if isinstance(value, bool):
        if required_type is not None and required_type != KT.bit:
            raise ValueError("Required type must be None or KT.bit")
        return PlainValue(arith.ConstantOp(ir.BoolAttr(value)))

    if isinstance(value, int):
        if required_type is None:
            required_type = KT.int_
        if str(required_type) not in type_to_builder[int]:
            raise ValueError(f"Unsupported DataType for materialization: {required_type}")
        factory = type_to_builder[int][str(required_type)]

    if isinstance(value, float):
        if required_type is None:
            required_type = KT.float_
        if str(required_type) not in type_to_builder[float]:
            raise ValueError(f"Unsupported DataType for materialization: {required_type}")
        factory = type_to_builder[float][str(required_type)]

    return PlainValue(factory(value), required_type)


def cast_to_index(value: Union[RuntimeNumeric, IRHandle]) -> IRHandle:
    if isinstance(value, int):
        return arith.ConstantOp(ir.IndexAttr(value))
    if isinstance(value, PlainValue):
        return cast_to_index(value.to_ir())
    if isinstance(value, IRHandle):
        return arith.IndexCastOp(value, ir.IndexType.get())
    raise TypeError(f"Unsupported type for index materialization: {value.__class__.__name__}")
