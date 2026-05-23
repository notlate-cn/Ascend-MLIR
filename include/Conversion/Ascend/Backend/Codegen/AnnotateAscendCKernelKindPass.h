//===- AnnotateAscendCKernelKindPass.h - AscendC kernel kind ----*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_CODEGEN_ANNOTATE_ASCENDC_KERNEL_KIND_PASS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_CODEGEN_ANNOTATE_ASCENDC_KERNEL_KIND_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::afir {

std::unique_ptr<Pass> createAnnotateAscendCKernelKindPass();

} // namespace mlir::afir

#endif // ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_CODEGEN_ANNOTATE_ASCENDC_KERNEL_KIND_PASS_H
