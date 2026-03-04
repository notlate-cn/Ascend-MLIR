from ._afir_ops_gen import *
from ._afir_enum_gen import *
from .._mlir_libs._afir import *
from ._arith_ops_gen import _Dialect
from .afir_dialect.runtime.jit import jit
from typing import overload
import builtins
try:
    from ..ir import *
    from ._ods_common import (
        get_default_loc_context as _get_default_loc_context,
        _cext as _ods_cext,
        get_op_result_or_op_results as _get_op_result_or_op_results,
    )

    from typing import Any, List, Union
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

def hiddenBroadCast(*operands):
    if builtins.all(isinstance(oper.type, RankedTensorType) for oper in operands):
        maxRank = builtins.max([oper.type.rank for oper in operands])
        newOperands = list()
        for oper in operands:
            if oper.type.rank < maxRank:
                newOperands.append(broadcast(RankedTensorType.get([1] *(maxRank - oper.type.rank) +oper.type.shape, oper.type.element_type), oper))
            else :
                newOperands.append(oper)
        return newOperands
    return operands

@_ods_cext.register_operation(_Dialect, replace=True)
class AddOp(AddOp):

    def __init__(self, lhs, rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None):
        lhs, rhs = hiddenBroadCast(lhs, rhs)
        super().__init__(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip)


def add(lhs , rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None) -> Value:
    return _get_op_result_or_op_results(AddOp(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip))

@_ods_cext.register_operation(_Dialect, replace=True)
class MulOp(MulOp):

    def __init__(self, lhs, rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None):
        lhs, rhs = hiddenBroadCast(lhs, rhs)
        super().__init__(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip)


def mul(lhs , rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None) -> Value:
    return _get_op_result_or_op_results(MulOp(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip))
@_ods_cext.register_operation(_Dialect, replace=True)
class SubOp(SubOp):

    def __init__(self, lhs, rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None):
        lhs, rhs = hiddenBroadCast(lhs, rhs)
        super().__init__(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip)


def sub(lhs , rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None) -> Value:
    return _get_op_result_or_op_results(SubOp(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip))

@_ods_cext.register_operation(_Dialect, replace=True)
class DivOp(DivOp):

    def __init__(self, lhs, rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None):
        lhs, rhs = hiddenBroadCast(lhs, rhs)
        super().__init__(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip)


def div(lhs , rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None) -> Value:
    return _get_op_result_or_op_results(DivOp(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip))

@_ods_cext.register_operation(_Dialect, replace=True)
class MinimumOp(MinimumOp):

    def __init__(self, lhs, rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None):
        lhs, rhs = hiddenBroadCast(lhs, rhs)
        super().__init__(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip)


def minimum(lhs , rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None) -> Value:
    return _get_op_result_or_op_results(MinimumOp(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip))

@_ods_cext.register_operation(_Dialect, replace=True)
class MaximumOp(MaximumOp):

    def __init__(self, lhs, rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None):
        lhs, rhs = hiddenBroadCast(lhs, rhs)
        super().__init__(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip)


def maximum(lhs , rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None) -> Value:
    return _get_op_result_or_op_results(MaximumOp(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip))

@_ods_cext.register_operation(_Dialect, replace=True)
class TrueDivOp(TrueDivOp):

    def __init__(self, lhs, rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None):
        lhs, rhs = hiddenBroadCast(lhs, rhs)
        super().__init__(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip)


def truediv(lhs , rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None) -> Value:
    return _get_op_result_or_op_results(TrueDivOp(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip))

@_ods_cext.register_operation(_Dialect, replace=True)
class PowOp(PowOp):

    def __init__(self, lhs, rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None):
        lhs, rhs = hiddenBroadCast(lhs, rhs)
        super().__init__(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip)


def pow(lhs , rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None) -> Value:
    return _get_op_result_or_op_results(PowOp(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip))


@_ods_cext.register_operation(_Dialect, replace=True)
class BitwiseAndOp(BitwiseAndOp):

    def __init__(self, lhs, rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None):
        lhs, rhs = hiddenBroadCast(lhs, rhs)
        super().__init__(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip)


def bitwise_and(lhs , rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None) -> Value:
    return _get_op_result_or_op_results(BitwiseAndOp(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip))

@_ods_cext.register_operation(_Dialect, replace=True)
class FloorDivOp(FloorDivOp):

    def __init__(self, lhs, rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None):
        lhs, rhs = hiddenBroadCast(lhs, rhs)
        super().__init__(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip)


def floor_div(lhs , rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None) -> Value:
    return _get_op_result_or_op_results(FloorDivOp(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip))

@_ods_cext.register_operation(_Dialect, replace=True)
class GeluOp(GeluOp):

    def __init__(self, lhs, rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None):
        lhs, rhs = hiddenBroadCast(lhs, rhs)
        super().__init__(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip)


def gelu(lhs , rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None) -> Value:
    return _get_op_result_or_op_results(GeluOp(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip))

@_ods_cext.register_operation(_Dialect, replace=True)
class SignOp(SignOp):

    def __init__(self, lhs, rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None):
        lhs, rhs = hiddenBroadCast(lhs, rhs)
        super().__init__(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip)


def sign(lhs , rhs, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None) -> Value:
    return _get_op_result_or_op_results(SignOp(lhs, rhs, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip))

@_ods_cext.register_operation(_Dialect, replace=True)
class ClipByValue(ClipByValue):

    def __init__(self, input1, input2, input3, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None):
        input1, input2, input3 = hiddenBroadCast(input1, input2, input3)
        super().__init__(input1, input2, input3, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip)


def clip_by_value(input1 , input2, input3, *, indexing_maps=None, loop_axis=None, ir_attr_def=None, tmp_buffers=None, outputs=None, loc=None, ip=None) -> Value:
    return _get_op_result_or_op_results(ClipByValue(input1, input2, input3, indexing_maps=indexing_maps, loop_axis=loop_axis, ir_attr_def=ir_attr_def, tmp_buffers=tmp_buffers, outputs=outputs, loc=loc, ip=ip))

