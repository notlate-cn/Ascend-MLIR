//===- Passes.h - Ascend conversion passes declarations --------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_PASSES_H
#define ASCEND_MLIR_CONVERSION_ASCEND_PASSES_H

#include "Conversion/Ascend/Backend/AscendCParallelizePass.h"
#include "Conversion/Ascend/Backend/AscendCPrepareForEmitPass.h"
#include "Conversion/Ascend/Backend/BackendWrapperPasses.h"
#include "Conversion/Ascend/Backend/CanonicalizeCannSignaturePass.h"
#include "Conversion/Ascend/Backend/ComputeLoweringPass.h"
#include "Conversion/Ascend/Backend/LinalgToAscendCPass.h"
#include "Conversion/Ascend/Kernelize/FuseGatherElementwisePass.h"
#include "Conversion/Ascend/Kernelize/KernelizePass.h"
#include "Conversion/Ascend/Kernelize/MarkStructuredOpsPass.h"
#include "Conversion/Ascend/Normalize/NormalizePass.h"
#include "Conversion/Ascend/Realize/AscendCBufferPlacementPass.h"
#include "Conversion/Ascend/Realize/AscendCFoldConcatAllocPass.h"
#include "Conversion/Ascend/Realize/RealizePass.h"
#include "Conversion/Ascend/Schedule/SchedulePass.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;
}

namespace mlir::afir {

std::unique_ptr<Pass> createAscendPrintTargetProfilePass();
std::unique_ptr<Pass> createAscendKernelSplitPass();

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Conversion/Ascend/Passes.h.inc"

} // namespace mlir::afir

#endif // ASCEND_MLIR_CONVERSION_ASCEND_PASSES_H
