//===- CannTranslation.h - CANN kernel C++ translation ----------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_TARGET_CANNKERNEL_CANNTRANSLATION_H
#define ASCEND_TARGET_CANNKERNEL_CANNTRANSLATION_H

#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
struct CannTranslationOptions {
  StringRef tilingSpaceOutPath;
  StringRef artifactManifestOutPath;
  StringRef hostTilingOutPath;
  StringRef kernelFile;
  StringRef soc = "Ascend910B1";
};

/// Translates a module containing CANN-signature aicore functions to C++.
/// Expects func.func args in order: inputs, outputs, workspace:memref<ui8>,
/// tiling:!emitasc.py_struct<...>, with cann.num_inputs attr.
/// If artifact paths are non-empty, writes the requested runtime artifacts.
LogicalResult translateToCannKernel(Operation *op, raw_ostream &os,
                                    const CannTranslationOptions &options);

/// Compatibility overload.
/// tilingSpaceOutPath: if non-empty, write tiling_space.json to this path.
/// kernelFile: value for "kernel_file" field in the JSON (may be empty).
LogicalResult translateToCannKernel(Operation *op, raw_ostream &os,
                                    StringRef tilingSpaceOutPath = "",
                                    StringRef kernelFile = "");
} // namespace mlir

#endif // ASCEND_TARGET_CANNKERNEL_CANNTRANSLATION_H
