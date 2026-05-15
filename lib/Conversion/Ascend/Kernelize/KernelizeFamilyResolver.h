//===- KernelizeFamilyResolver.h - Kernelize family resolver -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_FAMILY_RESOLVER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_FAMILY_RESOLVER_H

#include "KernelizeTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::afir::ascend::kernelize {

struct KernelizeFamilyResolution {
  SmallVector<std::string, 2> templateFamilies;
  std::string resolverName = "kernelize_trait_resolver";
};

KernelizeFamilyResolution resolveKernelizeTemplateFamilies(
    ArrayRef<std::string> lhsFamilies, ArrayRef<std::string> rhsFamilies,
    ArrayRef<KernelizePrimitiveKind> primitiveCombo);

} // namespace mlir::afir::ascend::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_FAMILY_RESOLVER_H
