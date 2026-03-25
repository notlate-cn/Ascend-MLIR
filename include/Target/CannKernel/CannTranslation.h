//===- CannTranslation.h - CANN kernel C++ translation ----------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_TARGET_CANNKERNEL_CANNTRANSLATION_H
#define AFIR_TARGET_CANNKERNEL_CANNTRANSLATION_H

#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
/// Translates a module containing CANN-signature aicore functions to C++.
/// Expects func.func args in order: inputs, outputs, workspace:memref<ui8>,
/// tiling:!emitasc.py_struct<...>, with cann.num_inputs attr.
/// tilingSpaceOutPath: if non-empty, write tiling_space.json skeleton to this path.
/// kernelFile: value for "kernel_file" field in the JSON (may be empty).
LogicalResult translateToCannKernel(Operation *op, raw_ostream &os,
                                    StringRef tilingSpaceOutPath = "",
                                    StringRef kernelFile = "");
} // namespace mlir

#endif // AFIR_TARGET_CANNKERNEL_CANNTRANSLATION_H
