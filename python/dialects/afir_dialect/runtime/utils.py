# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from typing import Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union, get_args, get_origin, ParamSpec, TypeAlias


import mlir.ir as ir

FnT = TypeVar("FnT", bound=Callable)
T = TypeVar("T")
P = ParamSpec("P")

JIT_INTERNAL = "__jit__"


def ConvertPythonTypeToMLIRType(param, ctx, loc):
    import numpy as np
    if isinstance(param, np.ndarray):
        elementType = None
        if param.dtype == np.float32:
            elementType = ir.F32Type.get(ctx)
        if param.dtype == np.int32:
            elementType = ir.IntegerType.get_signless(32, ctx)
        return ir.RankedTensorType.get(param.shape, elementType, loc=loc)
    if isinstance(param, int):
        return ir.IntegerType.get_signless(32, ctx)
    if isinstance(param, float):
        return ir.F32Type.get(ctx)
    if isinstance(param, ir.Type):
        return param