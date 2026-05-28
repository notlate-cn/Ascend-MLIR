/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it
 * under terms and conditions of the CANN Open Software License Agreement
 * Version 2.0 (the "License"). Please refer to LICENSE in the root of the
 * software repository for the full text of the License.
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the
 * License.
 */

#include "ComputeLoweringInternal.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "Conversion/Ascend/Translate/KernelIR/Capabilities/ElementwiseBodyOpRegistry.h"
#include "Conversion/Ascend/Translate/KernelIR/Capabilities/LinalgBodyClassifier.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#include <algorithm>
#include <limits>
#include <optional>

#define DEBUG_TYPE "ascend-compute-lower-compute"

using namespace mlir;
using namespace mlir::ascendc;

namespace mlir {
namespace ascend {

LogicalResult lowerTransposeComputes(ComputeLoweringContext &lowering) {
  func::FuncOp funcOp = lowering.funcOp;
  OpBuilder &builder = lowering.builder;
  // --- linalg.transpose ---
  SmallVector<linalg::TransposeOp> transposeOps;
  funcOp.walk([&](linalg::TransposeOp op) { transposeOps.push_back(op); });
  for (linalg::TransposeOp transposeOp : transposeOps) {
    FailureOr<TransposeLoweringSpec> spec =
        buildTransposeLoweringSpec(transposeOp);
    if (failed(spec))
      continue;
    TransposeLoweringPlan plan = planTransposeLowering(*spec);
    if (plan.kind == TransposeLoweringKind::Unsupported)
      continue;

    Value inMemref = transposeOp.getDpsInputOperand(0)->get();
    Value outMemref = transposeOp.getDpsInitOperand(0)->get();
    if (plan.kind == TransposeLoweringKind::ScalarMemRefLoop) {
      Location loc = transposeOp.getLoc();
      builder.setInsertionPoint(transposeOp);
      if (succeeded(lowerRank2GmTransposeToLocalDataCopy(
              builder, loc, inMemref, outMemref, spec->permutation,
              lowering.ctx.pipe))) {
        transposeOp.erase();
        continue;
      }
      if (failed(lowerTransposeToLoops(builder, loc, inMemref, outMemref,
                                       spec->permutation))) {
        transposeOp.emitError("failed to lower transpose scalar fallback");
        return failure();
      }
      transposeOp.erase();
      continue;
    }

    if (getMemorySpace(outMemref.getType()) <= 0)
      continue;

    Location loc = transposeOp.getLoc();
    builder.setInsertionPoint(transposeOp);
    Value srcLt = lowering.readTensor(builder, loc, inMemref);
    Value dstLt = lowering.writeTensor(builder, loc, outMemref);
    auto lowered = builder.create<TransposeOp>(loc, dstLt, srcLt);
    lowering.copyAscendCUnitAttr(transposeOp.getOperation(), lowered.getOperation());
    if (Value queue = lowering.ctx.getQueue(outMemref))
      builder.create<TQueBindEnqueTensorOp>(loc, queue, dstLt);
    transposeOp.erase();
  }

  return success();
}

} // namespace ascend
} // namespace mlir
