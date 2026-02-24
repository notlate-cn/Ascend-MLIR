# RUN: %PYTHON %s | FileCheck %s
import mlir.dialects.afir as afir
import mlir.dialects.func as func
import mlir.dialects.arith as arith
import mlir.dialects.scf as scf
from mlir.ir import *
from mlir.extras import types as T

ctx = Context()
afir.register_dialect(ctx)
afir.register_afir_passes()

# CHECK-LABEL: test_add_f16
with Location.unknown(ctx) as loc:
    testModule = Module.create(loc=loc)
    with InsertionPoint(testModule.body):
        @func.func(
            T.tensor(1,4,T.f16()), T.tensor(1,4,T.f16())
        )
        def test_add_f16(A, B):
            C = afir.add(A, B)
            # CHECK: afir.add %arg0, %arg1 : (tensor<1x4xf16>, tensor<1x4xf16>) -> tensor<1x4xf16>
            return C
    print(testModule)


#  CHECK-LABEL:   func.func @HashCopyAscGraph(
#  CHECK-SAME:      %[[ARG0:.*]]: tensor<4x1x31xf32>,
#  CHECK-SAME:      %[[ARG1:.*]]: tensor<25x1xf32>) -> tensor<4x25x31xf32> {
#  CHECK:           %[[VAL_0:.*]] = afir.load %[[ARG0]] : tensor<4x1x31xf32> -> tensor<4x1x31xf32>
#  CHECK:           %[[VAL_1:.*]] = afir.load %[[ARG1]] : tensor<25x1xf32> -> tensor<25x1xf32>
#  CHECK:           %[[VAL_2:.*]] = afir.broadcast %[[VAL_1]] : tensor<25x1xf32> -> tensor<1x25x1xf32>
#  CHECK:           %[[VAL_3:.*]] = afir.add %[[VAL_0]], %[[VAL_2]] : (tensor<4x1x31xf32>, tensor<1x25x1xf32>) -> tensor<4x25x31xf32>
#  CHECK:           %[[VAL_4:.*]] = afir.broadcast %[[VAL_1]] : tensor<25x1xf32> -> tensor<1x25x1xf32>
#  CHECK:           %[[VAL_5:.*]] = afir.mul %[[VAL_3]], %[[VAL_4]] : (tensor<4x25x31xf32>, tensor<1x25x1xf32>) -> tensor<4x25x31xf32>
#  CHECK:           %[[VAL_6:.*]] = afir.broadcast %[[VAL_1]] : tensor<25x1xf32> -> tensor<1x25x1xf32>
#  CHECK:           %[[VAL_7:.*]] = afir.sub %[[VAL_6]], %[[VAL_5]] : (tensor<1x25x1xf32>, tensor<4x25x31xf32>) -> tensor<4x25x31xf32>
#  CHECK:           %[[VAL_8:.*]] = afir.store %[[VAL_7]] : tensor<4x25x31xf32> -> tensor<4x25x31xf32>
#  CHECK:           return %[[VAL_8]] : tensor<4x25x31xf32>
#  CHECK:         }

@afir.jit
def HashCopyAscGraph(A, B):
    C = afir.load(A.type, A)
    D = afir.load(B.type, B)
    F = C + D
    G = F * D
    H = D - G
    I = afir.store(H.type, H)
    return I

import numpy as np

A = np.zeros((4, 1, 31), np.float32)
B = np.zeros((25, 1), np.float32)
mod = HashCopyAscGraph[(1,)](A, B)
print(mod)