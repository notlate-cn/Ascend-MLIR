//===- Passes.h - Conversion passes declarations ----------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_CONVERSION_PASSES
#define AFIR_CONVERSION_PASSES

#include "Conversion/AFIRToASCIR/AFIRToASCIR.h"
#include "Conversion/AFIRToASCIRText/AFIRToASCIRText.h"
#include "Conversion/AscendCBufferPlacement/AscendCBufferPlacementPass.h"
#include "Conversion/AscendCFoldConcatAlloc/AscendCFoldConcatAllocPass.h"
#include "Conversion/LinalgToAscendC/LinalgToAscendCPass.h"
#include "Conversion/AscendCParallelize/AscendCParallelizePass.h"
#include "Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace afir {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"

std::unique_ptr<Pass> createConvertAFIRToASCIRTextPass();
std::unique_ptr<Pass> createAscendCBufferPlacementPass();

}  // namespace afir
}  // namespace mlir

#endif  // AFIR_CONVERSION_PASSES
