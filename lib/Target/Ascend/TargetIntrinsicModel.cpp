//===- TargetIntrinsicModel.cpp - Ascend target intrinsic model -----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/Ascend/TargetIntrinsicModel.h"

#include "llvm/ADT/STLExtras.h"
#include <algorithm>

using namespace mlir;

namespace mlir::ascend {
namespace {

void sortAndUnique(SmallVector<std::string> &values) {
  llvm::sort(values);
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

void appendUnique(SmallVector<std::string> &values, StringRef value) {
  if (!llvm::is_contained(values, value))
    values.push_back(value.str());
}

void appendUnit(SmallVector<ExecutionUnit> &units, ExecutionUnit unit) {
  if (!llvm::is_contained(units, unit))
    units.push_back(unit);
}

bool isLoad2DIntrinsic(StringRef name) {
  return name == "Intrinsic_data_move_out2l1" ||
         name == "Intrinsic_data_move_out2l0a" ||
         name == "Intrinsic_data_move_out2l0b";
}

bool isFixPipePathIntrinsic(StringRef name) {
  return name.starts_with("Intrinsic_fix_pipe_l");
}

SmallVector<ExecutionUnit> inferUnitsFromName(StringRef name) {
  SmallVector<ExecutionUnit> units;
  if (name == "Intrinsic_mmad")
    units.push_back(ExecutionUnit::Cube);
  else if (name.starts_with("Intrinsic_v"))
    units.push_back(ExecutionUnit::Vector);
  else if (name.starts_with("Intrinsic_data_move_") ||
           isFixPipePathIntrinsic(name))
    units.push_back(ExecutionUnit::DMA);
  return units;
}

} // namespace

bool TargetIntrinsicModel::supportsIntrinsic(StringRef name) const {
  return intrinsicTable.contains(name);
}

FailureOr<IntrinsicCapability>
TargetIntrinsicModel::getIntrinsic(StringRef name) const {
  auto it = intrinsicTable.find(name);
  if (it == intrinsicTable.end())
    return failure();
  return it->second;
}

bool TargetIntrinsicModel::supportsDTypePattern(StringRef name,
                                                StringRef dtypePattern) const {
  auto intrinsic = getIntrinsic(name);
  if (failed(intrinsic))
    return false;
  return llvm::is_contained(intrinsic->dtypes, dtypePattern);
}

SmallVector<std::string>
TargetIntrinsicModel::getIntrinsicsForUnit(ExecutionUnit unit) const {
  switch (unit) {
  case ExecutionUnit::DMA:
    return dmaIntrinsics;
  case ExecutionUnit::Cube:
    return cubeIntrinsics;
  case ExecutionUnit::Vector:
    return vectorIntrinsics;
  }
  llvm_unreachable("unknown execution unit");
}

SmallVector<std::string>
TargetIntrinsicModel::getIntrinsicsForPathKind(PathKind kind) const {
  switch (kind) {
  case PathKind::DirectCopy:
    return directCopyIntrinsics;
  case PathKind::Load2D:
    return load2DIntrinsics;
  case PathKind::Load2DTranspose:
    return load2DTransposeIntrinsics;
  case PathKind::FixPipe:
    return fixPipeIntrinsics;
  case PathKind::QueueTransfer:
    return {};
  }
  llvm_unreachable("unknown path kind");
}

SmallVector<std::string>
TargetIntrinsicModel::getIntrinsicsForComputeKind(ComputeKind kind) const {
  switch (kind) {
  case ComputeKind::Matmul:
    return matmulIntrinsics;
  case ComputeKind::VectorAdd:
    return vectorAddIntrinsics;
  case ComputeKind::VectorExp:
    return vectorExpIntrinsics;
  case ComputeKind::VectorTranspose:
    return vectorTransposeIntrinsics;
  case ComputeKind::VectorGather:
    return vectorGatherIntrinsics;
  case ComputeKind::VectorReduce:
    return vectorReduceIntrinsics;
  }
  llvm_unreachable("unknown compute kind");
}

FailureOr<TargetIntrinsicModel>
TargetIntrinsicModelBuilder::build(const TargetProfile &profile) const {
  TargetIntrinsicModel model;
  llvm::StringMap<SmallVector<ExecutionUnit>> unitsByIntrinsic;

  auto addCapability = [&](const TargetIntrinsicInfo &intrinsic) {
    IntrinsicCapability &capability = model.intrinsicTable[intrinsic.name];
    capability.name = intrinsic.name;
    for (StringRef dtype : intrinsic.dtypes)
      appendUnique(capability.dtypes, dtype);

    SmallVector<ExecutionUnit> &units = unitsByIntrinsic[intrinsic.name];
    for (ExecutionUnit unit : intrinsic.units)
      appendUnit(units, unit);
  };

  for (const TargetIntrinsicInfo &intrinsic : profile.intrinsics)
    addCapability(intrinsic);

  for (auto &entry : model.intrinsicTable) {
    IntrinsicCapability &capability = entry.second;
    sortAndUnique(capability.dtypes);
    StringRef name = capability.name;

    SmallVector<ExecutionUnit> units = unitsByIntrinsic[name];
    if (units.empty())
      units = inferUnitsFromName(name);
    for (ExecutionUnit unit : units) {
      switch (unit) {
      case ExecutionUnit::DMA:
        model.dmaIntrinsics.push_back(name.str());
        break;
      case ExecutionUnit::Cube:
        model.cubeIntrinsics.push_back(name.str());
        break;
      case ExecutionUnit::Vector:
        model.vectorIntrinsics.push_back(name.str());
        break;
      }
    }

    if (name.contains("_transpose_"))
      model.load2DTransposeIntrinsics.push_back(name.str());
    else if (isFixPipePathIntrinsic(name))
      model.fixPipeIntrinsics.push_back(name.str());
    else if (isLoad2DIntrinsic(name))
      model.load2DIntrinsics.push_back(name.str());
    else if (name.starts_with("Intrinsic_data_move_"))
      model.directCopyIntrinsics.push_back(name.str());

    if (name == "Intrinsic_mmad")
      model.matmulIntrinsics.push_back(name.str());
    else if (name == "Intrinsic_vadd")
      model.vectorAddIntrinsics.push_back(name.str());
    else if (name == "Intrinsic_vexp")
      model.vectorExpIntrinsics.push_back(name.str());
    else if (name == "Intrinsic_vtranspose")
      model.vectorTransposeIntrinsics.push_back(name.str());
    else if (name == "Intrinsic_vgather")
      model.vectorGatherIntrinsics.push_back(name.str());
    else if (name == "Intrinsic_vreduce")
      model.vectorReduceIntrinsics.push_back(name.str());
  }

  for (SmallVector<std::string> *values :
       {&model.dmaIntrinsics, &model.cubeIntrinsics, &model.vectorIntrinsics,
        &model.directCopyIntrinsics, &model.load2DIntrinsics,
        &model.load2DTransposeIntrinsics, &model.fixPipeIntrinsics,
        &model.matmulIntrinsics, &model.vectorAddIntrinsics,
        &model.vectorExpIntrinsics, &model.vectorTransposeIntrinsics,
        &model.vectorGatherIntrinsics, &model.vectorReduceIntrinsics})
    sortAndUnique(*values);

  return model;
}

} // namespace mlir::ascend
