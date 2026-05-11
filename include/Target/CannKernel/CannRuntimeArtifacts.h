//===- CannRuntimeArtifacts.h - CANN runtime artifact emission --*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_TARGET_CANNKERNEL_RUNTIME_ARTIFACTS_H
#define AFIR_TARGET_CANNKERNEL_RUNTIME_ARTIFACTS_H

#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir::afir::cann {

struct CannRuntimeArtifactOptions {
  StringRef kernelFile;
  StringRef soc = "Ascend910B1";
};

LogicalResult emitTilingSpaceJson(ModuleOp module, StringRef outPath,
                                  const CannRuntimeArtifactOptions &options);
LogicalResult emitRuntimeManifestJson(ModuleOp module, StringRef outPath,
                                      const CannRuntimeArtifactOptions &options);
LogicalResult emitHostTilingCpp(ModuleOp module, StringRef outPath,
                                const CannRuntimeArtifactOptions &options);

} // namespace mlir::afir::cann

#endif // AFIR_TARGET_CANNKERNEL_RUNTIME_ARTIFACTS_H
