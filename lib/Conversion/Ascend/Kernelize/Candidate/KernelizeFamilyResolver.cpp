//===- KernelizeFamilyResolver.cpp - Kernelize family resolver --------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/Candidate/KernelizeFamilyResolver.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir::afir::ascend::kernelize {
namespace {

bool containsFamily(ArrayRef<std::string> families, StringRef family) {
  return llvm::is_contained(families, family);
}

bool containsFamilyPair(ArrayRef<std::string> lhsFamilies,
                        ArrayRef<std::string> rhsFamilies,
                        StringRef lhsFamily, StringRef rhsFamily) {
  return (containsFamily(lhsFamilies, lhsFamily) &&
          containsFamily(rhsFamilies, rhsFamily)) ||
         (containsFamily(lhsFamilies, rhsFamily) &&
          containsFamily(rhsFamilies, lhsFamily));
}

void appendUniqueFamily(SmallVectorImpl<std::string> &families,
                        StringRef family) {
  if (!containsFamily(families, family))
    families.push_back(family.str());
}

} // namespace

KernelizeFamilyResolution resolveKernelizeTemplateFamilies(
    ArrayRef<std::string> lhsFamilies, ArrayRef<std::string> rhsFamilies,
    ArrayRef<KernelizePrimitiveKind>) {
  KernelizeFamilyResolution result;

  if (containsFamilyPair(lhsFamilies, rhsFamilies, kOpRoleVector,
                         kOpRoleReduction)) {
    appendUniqueFamily(result.templateFamilies, kOpRoleReduction);
    return result;
  }

  if (containsFamilyPair(lhsFamilies, rhsFamilies, kOpRoleCube,
                         kOpRoleVector)) {
    appendUniqueFamily(result.templateFamilies, kOpRoleCube);
    return result;
  }

  for (const std::string &lhsFamily : lhsFamilies)
    if (containsFamily(rhsFamilies, lhsFamily))
      appendUniqueFamily(result.templateFamilies, lhsFamily);

  return result;
}

} // namespace mlir::afir::ascend::kernelize
