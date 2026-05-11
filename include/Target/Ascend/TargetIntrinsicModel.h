//===- TargetIntrinsicModel.h - Ascend target intrinsic model ---*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_TARGET_ASCEND_TARGET_INTRINSIC_MODEL_H
#define ASCEND_MLIR_TARGET_ASCEND_TARGET_INTRINSIC_MODEL_H

#include "Target/Ascend/TargetMemoryModel.h"
#include "Target/Ascend/TargetProfile.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Support/LLVM.h"
#include <string>

namespace mlir::ascend {

enum class ComputeKind {
  Matmul,
  VectorAdd,
  VectorExp,
  VectorTranspose,
  VectorGather,
  VectorReduce,
};

struct IntrinsicCapability {
  std::string name;
  SmallVector<std::string> dtypes;
};

class TargetIntrinsicModel {
public:
  bool supportsIntrinsic(StringRef name) const;
  FailureOr<IntrinsicCapability> getIntrinsic(StringRef name) const;
  bool supportsDTypePattern(StringRef name, StringRef dtypePattern) const;
  SmallVector<std::string> getIntrinsicsForUnit(ExecutionUnit unit) const;
  SmallVector<std::string> getIntrinsicsForPathKind(PathKind kind) const;
  SmallVector<std::string> getIntrinsicsForComputeKind(ComputeKind kind) const;

private:
  friend class TargetIntrinsicModelBuilder;
  llvm::StringMap<IntrinsicCapability> intrinsicTable;
  SmallVector<std::string> dmaIntrinsics;
  SmallVector<std::string> cubeIntrinsics;
  SmallVector<std::string> vectorIntrinsics;
  SmallVector<std::string> directCopyIntrinsics;
  SmallVector<std::string> load2DIntrinsics;
  SmallVector<std::string> load2DTransposeIntrinsics;
  SmallVector<std::string> fixPipeIntrinsics;
  SmallVector<std::string> matmulIntrinsics;
  SmallVector<std::string> vectorAddIntrinsics;
  SmallVector<std::string> vectorExpIntrinsics;
  SmallVector<std::string> vectorTransposeIntrinsics;
  SmallVector<std::string> vectorGatherIntrinsics;
  SmallVector<std::string> vectorReduceIntrinsics;
};

class TargetIntrinsicModelBuilder {
public:
  FailureOr<TargetIntrinsicModel> build(const TargetProfile &profile) const;
};

} // namespace mlir::ascend

#endif // ASCEND_MLIR_TARGET_ASCEND_TARGET_INTRINSIC_MODEL_H
