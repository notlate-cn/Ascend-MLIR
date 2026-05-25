//===- KernelizeOpRegistry.h - Ascend kernelize op registry -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_KERNELIZEOPREGISTRY_H
#define ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_KERNELIZEOPREGISTRY_H

#include "Conversion/Ascend/Kernelize/KernelizeOpInterface.h"

namespace mlir::afir::ascend::kernelize {

void registerDefaultKernelizeOpModels(KernelizeOpModelRegistry &registry);

} // namespace mlir::afir::ascend::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_KERNELIZEOPREGISTRY_H
