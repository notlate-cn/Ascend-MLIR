//===- CannTranslation.cpp - CANN kernel C++ translation --------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/CannKernel/CannTranslation.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Target/Asc/CodeEmitter.h"
#include "ascir/Target/Asc/Common.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

using namespace mlir;

namespace {

enum class AscendCKernelKind {
  Unknown,
  Vec,
  Cube,
  Mix,
};

static AscendCKernelKind getKernelKind(func::FuncOp funcOp) {
  auto kindAttr = funcOp->getAttrOfType<StringAttr>("ascendc.kernel_kind");
  if (!kindAttr)
    return AscendCKernelKind::Unknown;
  StringRef kind = kindAttr.getValue();
  if (kind == "vec")
    return AscendCKernelKind::Vec;
  if (kind == "cube")
    return AscendCKernelKind::Cube;
  if (kind == "mix")
    return AscendCKernelKind::Mix;
  return AscendCKernelKind::Unknown;
}

static func::FuncOp findPrimaryGlobalKernel(ModuleOp moduleOp) {
  for (Operation &child : moduleOp.getBody()->getOperations()) {
    auto funcOp = dyn_cast<func::FuncOp>(child);
    if (funcOp && funcOp->hasAttr(ascendc::attr::global))
      return funcOp;
  }
  return {};
}

static bool isRankedMemrefOf(Type type, int64_t rank, Type elementType) {
  auto memrefType = dyn_cast<MemRefType>(type);
  return memrefType && memrefType.getRank() == rank &&
         memrefType.getElementType() == elementType;
}

enum class MixPartitionKind {
  Unknown,
  Cube,
  Vector,
  Boundary,
};

// Supported mix analysis helpers.
static MixPartitionKind getExplicitMixPartition(Operation *op) {
  auto unitAttr = op->getAttrOfType<StringAttr>("ascendc.unit");
  if (!unitAttr)
    return MixPartitionKind::Unknown;
  StringRef unit = unitAttr.getValue();
  if (unit == "AiCore.Cube")
    return MixPartitionKind::Cube;
  if (unit == "AiCore.Vector")
    return MixPartitionKind::Vector;
  return MixPartitionKind::Unknown;
}

struct MixPartitionSummary {
  SmallVector<Operation *> cubeOps;
  SmallVector<Operation *> vectorOps;
  SmallVector<Operation *> boundaryOps;

  bool hasCube() const { return !cubeOps.empty(); }
  bool hasVector() const { return !vectorOps.empty(); }
  bool hasBoundary() const { return !boundaryOps.empty(); }
};

struct MixBoundaryValue {
  Value value;
  MixPartitionKind producer;
  MixPartitionKind consumer;
};

struct MixRegionPlan {
  MixPartitionKind kind = MixPartitionKind::Unknown;
  SmallVector<Operation *> ops;
  SmallVector<MixBoundaryValue> inputs;
  SmallVector<MixBoundaryValue> outputs;
};

struct MixPartitionPlan {
  SmallVector<MixRegionPlan> regions;
  bool empty() const { return regions.empty(); }
};

enum class MixSingleChainFailureReason {
  MissingCubeRegion,
  MultipleCubeRegions,
  MissingBoundaryRegion,
  MultipleBoundaryRegions,
  MissingVectorRegion,
  MultipleVectorRegions,
  MissingBoundaryCrossing,
  InvalidOrdering,
};

struct MixSingleChainValidation {
  SmallVector<MixSingleChainFailureReason> failureReasons;

  bool succeeded() const { return failureReasons.empty(); }
};

static MixPartitionKind getStoragePartitionFromQueueLikeType(Type type);

static MixPartitionKind inferPartitionForQueueLikeUser(Operation *user) {
  if (auto enqueTensor = dyn_cast<ascendc::TQueBindEnqueTensorOp>(user))
    return getStoragePartitionFromQueueLikeType(enqueTensor.getQueue().getType());
  if (auto dequeTensor = dyn_cast<ascendc::TQueBindDequeTensorOp>(user))
    return getStoragePartitionFromQueueLikeType(dequeTensor.getQueue().getType());
  if (auto tbufTensor = dyn_cast<ascendc::TBufGetTensorOp>(user))
    return getStoragePartitionFromQueueLikeType(tbufTensor.getBuffer().getType());
  return MixPartitionKind::Unknown;
}

struct SupportedMixKernelConfig {
  enum class TaskKind {
    MixAic1To2,
  };

  enum class EpilogueKind {
    Unknown,
    Relu,
    LeakyRelu,
  };

  bool hasBiasAdd = false;
  TaskKind taskKind;
  EpilogueKind epilogueKind = EpilogueKind::Unknown;
  double leakyReluAlpha = 0.0;
};

struct MixTaskKindDescriptor {
  StringRef taskTypeSpelling;
  unsigned crossCoreMode;
  unsigned vectorTaskRatio;
  unsigned crossCoreFlagId;
};

static MixTaskKindDescriptor
getMixTaskKindDescriptor(SupportedMixKernelConfig::TaskKind taskKind) {
  switch (taskKind) {
  case SupportedMixKernelConfig::TaskKind::MixAic1To2:
    return {"KERNEL_TYPE_MIX_AIC_1_2", 0x2, 2, 3};
  }
  llvm_unreachable("unsupported mix task kind");
}

static MixPartitionKind getStoragePartitionForPosition(ascendc::TPosition position) {
  switch (position) {
  case ascendc::TPosition::A1:
  case ascendc::TPosition::A2:
  case ascendc::TPosition::B1:
  case ascendc::TPosition::B2:
  case ascendc::TPosition::CO1:
    return MixPartitionKind::Cube;
  case ascendc::TPosition::VECIN:
  case ascendc::TPosition::VECCALC:
  case ascendc::TPosition::VECOUT:
    return MixPartitionKind::Vector;
  default:
    return MixPartitionKind::Unknown;
  }
}

static MixPartitionKind getStoragePartitionFromQueueLikeType(Type type) {
  if (auto queueType = dyn_cast<ascendc::QueueType>(type))
    return getStoragePartitionForPosition(queueType.getPosition());
  if (auto tbufType = dyn_cast<ascendc::TBufType>(type))
    return getStoragePartitionForPosition(tbufType.getTPosition());
  if (auto queBindType = dyn_cast<ascendc::QueBindType>(type)) {
    MixPartitionKind src = getStoragePartitionForPosition(queBindType.getSrcPosition());
    MixPartitionKind dst = getStoragePartitionForPosition(queBindType.getDstPosition());
    if (src == dst)
      return src;
  }
  return MixPartitionKind::Unknown;
}

static MixPartitionKind getTensorStoragePartition(Value value) {
  if (auto direct = getStoragePartitionFromQueueLikeType(value.getType());
      direct != MixPartitionKind::Unknown)
    return direct;

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return MixPartitionKind::Unknown;

  if (auto allocTensor = dyn_cast<ascendc::TQueBindAllocTensorOp>(defOp))
    return getStoragePartitionFromQueueLikeType(allocTensor.getQueue().getType());
  if (auto dequeTensor = dyn_cast<ascendc::TQueBindDequeTensorOp>(defOp))
    return getStoragePartitionFromQueueLikeType(dequeTensor.getQueue().getType());
  if (auto tbufTensor = dyn_cast<ascendc::TBufGetTensorOp>(defOp))
    return getStoragePartitionFromQueueLikeType(tbufTensor.getBuffer().getType());

  return MixPartitionKind::Unknown;
}

static bool touchesStoragePartitions(Operation *op, MixPartitionKind lhs,
                                     MixPartitionKind rhs) {
  bool touchesLhs = false;
  bool touchesRhs = false;
  for (Value operand : op->getOperands()) {
    MixPartitionKind partition = getTensorStoragePartition(operand);
    touchesLhs |= partition == lhs;
    touchesRhs |= partition == rhs;
  }
  for (Value result : op->getResults()) {
    MixPartitionKind partition = getTensorStoragePartition(result);
    touchesLhs |= partition == lhs;
    touchesRhs |= partition == rhs;
  }
  return touchesLhs && touchesRhs;
}

static bool valueOriginatesFromPartition(Value value, MixPartitionKind target,
                                         llvm::DenseMap<Value, bool> &cache,
                                         llvm::SmallPtrSetImpl<Operation *> &visiting);

static bool opOriginatesFromPartition(Operation *op, MixPartitionKind target,
                                      llvm::DenseMap<Value, bool> &cache,
                                      llvm::SmallPtrSetImpl<Operation *> &visiting) {
  MixPartitionKind explicitPartition = getExplicitMixPartition(op);
  if (explicitPartition != MixPartitionKind::Unknown)
    return explicitPartition == target;

  if (!visiting.insert(op).second)
    return false;

  bool matches = llvm::any_of(op->getOperands(), [&](Value operand) {
    return valueOriginatesFromPartition(operand, target, cache, visiting);
  });
  visiting.erase(op);
  return matches;
}

static bool valueOriginatesFromPartition(Value value, MixPartitionKind target,
                                         llvm::DenseMap<Value, bool> &cache,
                                         llvm::SmallPtrSetImpl<Operation *> &visiting) {
  auto cached = cache.find(value);
  if (cached != cache.end())
    return cached->second;

  auto defOp = value.getDefiningOp();
  if (!defOp)
    return cache[value] = false;
  return cache[value] = opOriginatesFromPartition(defOp, target, cache, visiting);
}

static bool valueReachesPartition(Value value, MixPartitionKind target,
                                  Operation *skipUser,
                                  llvm::DenseMap<Value, bool> &cache,
                                  llvm::DenseSet<Value> &visiting) {
  auto cached = cache.find(value);
  if (cached != cache.end())
    return cached->second;

  if (!visiting.insert(value).second)
    return false;

  bool reaches = false;
  for (Operation *user : value.getUsers()) {
    if (user == skipUser)
      continue;
    MixPartitionKind explicitPartition = getExplicitMixPartition(user);
    if (explicitPartition == target) {
      reaches = true;
      break;
    }
    if (explicitPartition != MixPartitionKind::Unknown)
      continue;
    for (Value result : user->getResults()) {
      if (valueReachesPartition(result, target, nullptr, cache, visiting)) {
        reaches = true;
        break;
      }
    }
    if (reaches)
      break;
  }

  visiting.erase(value);
  cache[value] = reaches;
  return reaches;
}

static bool isBoundaryBetweenPartitions(Operation *op, bool hasCubeFlowIn,
                                        bool hasVectorFlowIn,
                                        bool hasCubeFlowOut,
                                        bool hasVectorFlowOut) {
  return touchesStoragePartitions(op, MixPartitionKind::Cube,
                                  MixPartitionKind::Vector) ||
         (hasCubeFlowIn && hasVectorFlowOut) ||
         (hasVectorFlowIn && hasCubeFlowOut);
}

static MixPartitionSummary buildMixPartitionSummary(func::FuncOp funcOp) {
  MixPartitionSummary summary;
  llvm::DenseMap<Value, bool> originCubeCache;
  llvm::DenseMap<Value, bool> originVectorCache;
  llvm::DenseMap<Value, bool> reachCubeCache;
  llvm::DenseMap<Value, bool> reachVectorCache;

  funcOp.walk([&](Operation *op) {
    MixPartitionKind explicitPartition = getExplicitMixPartition(op);
    if (explicitPartition == MixPartitionKind::Cube) {
      summary.cubeOps.push_back(op);
      return;
    }
    if (explicitPartition == MixPartitionKind::Vector) {
      summary.vectorOps.push_back(op);
      return;
    }

    llvm::SmallPtrSet<Operation *, 16> originVisiting;
    bool hasCubeFlowIn = llvm::any_of(op->getOperands(), [&](Value operand) {
      return valueOriginatesFromPartition(operand, MixPartitionKind::Cube,
                                         originCubeCache, originVisiting);
    });
    originVisiting.clear();
    bool hasVectorFlowIn = llvm::any_of(op->getOperands(), [&](Value operand) {
      return valueOriginatesFromPartition(operand, MixPartitionKind::Vector,
                                         originVectorCache, originVisiting);
    });

    llvm::DenseSet<Value> reachVisiting;
    bool hasCubeFlowOut = llvm::any_of(op->getResults(), [&](Value result) {
      return valueReachesPartition(result, MixPartitionKind::Cube, nullptr,
                                   reachCubeCache, reachVisiting);
    });
    reachVisiting.clear();
    bool hasVectorFlowOut = llvm::any_of(op->getResults(), [&](Value result) {
      return valueReachesPartition(result, MixPartitionKind::Vector, nullptr,
                                   reachVectorCache, reachVisiting);
    });

    if (!hasCubeFlowOut) {
      reachVisiting.clear();
      hasCubeFlowOut = llvm::any_of(op->getOperands(), [&](Value operand) {
        return valueReachesPartition(operand, MixPartitionKind::Cube, op,
                                     reachCubeCache, reachVisiting);
      });
    }
    if (!hasVectorFlowOut) {
      reachVisiting.clear();
      hasVectorFlowOut = llvm::any_of(op->getOperands(), [&](Value operand) {
        return valueReachesPartition(operand, MixPartitionKind::Vector, op,
                                     reachVectorCache, reachVisiting);
      });
    }

    if (isBoundaryBetweenPartitions(op, hasCubeFlowIn, hasVectorFlowIn,
                                    hasCubeFlowOut, hasVectorFlowOut))
      summary.boundaryOps.push_back(op);
  });

  return summary;
}

static bool hasSupportedMixFunctionSignature(func::FuncOp funcOp) {
  auto numInputsAttr = funcOp->getAttrOfType<IntegerAttr>("cann.num_inputs");
  if (!numInputsAttr || numInputsAttr.getInt() != 4)
    return false;

  auto args = funcOp.getArguments();
  if (args.size() != 7)
    return false;

  MLIRContext *ctx = funcOp.getContext();
  Type f16 = Float16Type::get(ctx);
  Type f32 = Float32Type::get(ctx);

  if (!isRankedMemrefOf(args[0].getType(), 2, f16) ||
      !isRankedMemrefOf(args[1].getType(), 2, f16) ||
      !isRankedMemrefOf(args[2].getType(), 1, f32) ||
      !isRankedMemrefOf(args[3].getType(), 2, f32))
    return false;

  auto outputType = dyn_cast<MemRefType>(args[4].getType());
  if (!outputType || outputType.getRank() != 2 ||
      outputType.getElementType() != f32)
    return false;

  auto workspaceType = dyn_cast<MemRefType>(args[5].getType());
  if (!workspaceType || !workspaceType.getElementType().isUnsignedInteger(8))
    return false;
  return isa<emitasc::PyStructType>(args[6].getType());
}

static bool hasSupportedMixPartitions(const MixPartitionSummary &summary) {
  return summary.hasCube() && summary.hasVector() && summary.hasBoundary();
}

static llvm::DenseMap<Operation *, MixPartitionKind>
buildMixPartitionMap(const MixPartitionSummary &summary) {
  llvm::DenseMap<Operation *, MixPartitionKind> partitionMap;
  for (Operation *op : summary.cubeOps)
    partitionMap.try_emplace(op, MixPartitionKind::Cube);
  for (Operation *op : summary.boundaryOps)
    partitionMap.try_emplace(op, MixPartitionKind::Boundary);
  for (Operation *op : summary.vectorOps)
    partitionMap.try_emplace(op, MixPartitionKind::Vector);
  return partitionMap;
}

static SmallVector<MixBoundaryValue>
collectMixBoundaryValues(const MixPartitionSummary &summary) {
  SmallVector<MixBoundaryValue> boundaryValues;
  llvm::DenseMap<Operation *, MixPartitionKind> partitionMap =
      buildMixPartitionMap(summary);

  auto recordCrossing = [&](Value value, MixPartitionKind producer,
                            MixPartitionKind consumer) {
    if (producer == MixPartitionKind::Unknown ||
        consumer == MixPartitionKind::Unknown || producer == consumer)
      return;
    for (const MixBoundaryValue &existing : boundaryValues) {
      if (existing.value == value && existing.producer == producer &&
          existing.consumer == consumer)
        return;
    }
    boundaryValues.push_back({value, producer, consumer});
  };

  auto getPartitionForSummaryOp = [&](Operation *op) {
    auto it = partitionMap.find(op);
    if (it != partitionMap.end())
      return it->second;
    return MixPartitionKind::Unknown;
  };

  auto getConsumerPartition = [&](Operation *user) {
    MixPartitionKind partition = getPartitionForSummaryOp(user);
    if (partition != MixPartitionKind::Unknown)
      return partition;
    return inferPartitionForQueueLikeUser(user);
  };

  for (Operation *op : summary.cubeOps) {
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers())
        recordCrossing(result, MixPartitionKind::Cube,
                       getConsumerPartition(user));
    }
  }

  for (Operation *op : summary.boundaryOps) {
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp)
        continue;
      recordCrossing(operand, getPartitionForSummaryOp(defOp),
                     MixPartitionKind::Boundary);
    }

    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers())
        recordCrossing(result, MixPartitionKind::Boundary,
                       getConsumerPartition(user));
    }
  }

  for (Operation *op : summary.vectorOps) {
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp)
        continue;
      recordCrossing(operand, getPartitionForSummaryOp(defOp),
                     MixPartitionKind::Vector);
    }
  }

  return boundaryValues;
}

static MixPartitionPlan buildInitialMixPartitionPlan(
    [[maybe_unused]] func::FuncOp funcOp, const MixPartitionSummary &summary) {
  MixPartitionPlan plan;
  SmallVector<MixBoundaryValue> boundaryValues = collectMixBoundaryValues(summary);

  auto addRegion = [&](MixPartitionKind kind,
                       ArrayRef<Operation *> ops) -> void {
    if (ops.empty())
      return;
    MixRegionPlan region;
    region.kind = kind;
    region.ops.append(ops.begin(), ops.end());
    if (kind == MixPartitionKind::Boundary) {
      for (const MixBoundaryValue &boundaryValue : boundaryValues) {
        if (boundaryValue.consumer == MixPartitionKind::Boundary)
          region.inputs.push_back(boundaryValue);
        if (boundaryValue.producer == MixPartitionKind::Boundary)
          region.outputs.push_back(boundaryValue);
      }
    }
    plan.regions.push_back(std::move(region));
  };

  addRegion(MixPartitionKind::Cube, summary.cubeOps);
  addRegion(MixPartitionKind::Boundary, summary.boundaryOps);
  addRegion(MixPartitionKind::Vector, summary.vectorOps);

  return plan;
}

// Supported mix configuration inference.
static bool isSupportedCurrentMixEmission(func::FuncOp funcOp,
                                          const MixPartitionSummary &summary) {
  if (!hasSupportedMixFunctionSignature(funcOp))
    return false;

  return hasSupportedMixPartitions(summary);
}

static bool hasFailureReason(ArrayRef<MixSingleChainFailureReason> reasons,
                             MixSingleChainFailureReason reason) {
  return llvm::is_contained(reasons, reason);
}

static MixSingleChainValidation
validateSingleChainGenericMixPlan(const MixPartitionPlan &plan) {
  MixSingleChainValidation validation;
  auto addFailureReason = [&](MixSingleChainFailureReason reason) {
    if (!hasFailureReason(validation.failureReasons, reason))
      validation.failureReasons.push_back(reason);
  };

  unsigned cubeCount = 0;
  unsigned boundaryCount = 0;
  unsigned vectorCount = 0;
  const MixRegionPlan *boundaryRegion = nullptr;
  std::optional<unsigned> cubeIndex;
  std::optional<unsigned> boundaryIndex;
  std::optional<unsigned> vectorIndex;

  for (const auto &[index, region] : llvm::enumerate(plan.regions)) {
    switch (region.kind) {
    case MixPartitionKind::Cube:
      ++cubeCount;
      if (!cubeIndex)
        cubeIndex = index;
      break;
    case MixPartitionKind::Boundary:
      ++boundaryCount;
      if (!boundaryRegion)
        boundaryRegion = &region;
      if (!boundaryIndex)
        boundaryIndex = index;
      break;
    case MixPartitionKind::Vector:
      ++vectorCount;
      if (!vectorIndex)
        vectorIndex = index;
      break;
    case MixPartitionKind::Unknown:
      addFailureReason(MixSingleChainFailureReason::InvalidOrdering);
      break;
    }
  }

  if (cubeCount == 0)
    addFailureReason(MixSingleChainFailureReason::MissingCubeRegion);
  else if (cubeCount > 1)
    addFailureReason(MixSingleChainFailureReason::MultipleCubeRegions);

  if (boundaryCount == 0)
    addFailureReason(MixSingleChainFailureReason::MissingBoundaryRegion);
  else if (boundaryCount > 1)
    addFailureReason(MixSingleChainFailureReason::MultipleBoundaryRegions);

  if (vectorCount == 0)
    addFailureReason(MixSingleChainFailureReason::MissingVectorRegion);
  else if (vectorCount > 1)
    addFailureReason(MixSingleChainFailureReason::MultipleVectorRegions);

  if (cubeIndex && boundaryIndex && vectorIndex &&
      !(*cubeIndex < *boundaryIndex && *boundaryIndex < *vectorIndex))
    addFailureReason(MixSingleChainFailureReason::InvalidOrdering);

  if (boundaryRegion) {
    bool hasCubeToBoundaryCrossing = llvm::any_of(
        boundaryRegion->inputs, [](const MixBoundaryValue &input) {
          return input.producer == MixPartitionKind::Cube &&
                 input.consumer == MixPartitionKind::Boundary;
        });
    bool hasBoundaryToVectorCrossing = llvm::any_of(
        boundaryRegion->outputs, [](const MixBoundaryValue &output) {
          return output.producer == MixPartitionKind::Boundary &&
                 output.consumer == MixPartitionKind::Vector;
        });
    if (!hasCubeToBoundaryCrossing || !hasBoundaryToVectorCrossing)
      addFailureReason(MixSingleChainFailureReason::MissingBoundaryCrossing);
  }

  return validation;
}

static StringRef
stringifyMixSingleChainFailureReason(MixSingleChainFailureReason reason) {
  switch (reason) {
  case MixSingleChainFailureReason::MissingCubeRegion:
    return "missing cube region";
  case MixSingleChainFailureReason::MultipleCubeRegions:
    return "multiple cube regions";
  case MixSingleChainFailureReason::MissingBoundaryRegion:
    return "missing boundary region";
  case MixSingleChainFailureReason::MultipleBoundaryRegions:
    return "multiple boundary regions";
  case MixSingleChainFailureReason::MissingVectorRegion:
    return "missing vector region";
  case MixSingleChainFailureReason::MultipleVectorRegions:
    return "multiple vector regions";
  case MixSingleChainFailureReason::MissingBoundaryCrossing:
    return "missing cube-to-boundary or boundary-to-vector crossing";
  case MixSingleChainFailureReason::InvalidOrdering:
    return "invalid region ordering";
  }
  llvm_unreachable("unexpected single-chain failure reason");
}

static std::string
describeMixSingleChainValidation(const MixSingleChainValidation &validation) {
  SmallString<128> description;
  llvm::raw_svector_ostream os(description);
  for (const auto &[index, reason] : llvm::enumerate(validation.failureReasons)) {
    if (index)
      os << ", ";
    os << stringifyMixSingleChainFailureReason(reason);
  }
  return std::string(description);
}

static bool canLowerLegacySupportedMix(func::FuncOp funcOp,
                                       const MixPartitionSummary &summary) {
  return isSupportedCurrentMixEmission(funcOp, summary);
}

static llvm::DenseSet<Value>
collectSupportedMixVectorBroadcastDsts(const MixPartitionSummary &summary) {
  llvm::DenseSet<Value> vectorBroadcastDsts;
  for (Operation *op : summary.vectorOps) {
    auto broadcastOp = dyn_cast<ascendc::BroadcastL2Op>(op);
    if (!broadcastOp)
      continue;
    vectorBroadcastDsts.insert(broadcastOp.getDst());
  }
  return vectorBroadcastDsts;
}

static bool inferSupportedMixHasBiasAdd(const MixPartitionSummary &summary) {
  llvm::DenseSet<Value> vectorBroadcastDsts =
      collectSupportedMixVectorBroadcastDsts(summary);
  for (Operation *op : summary.vectorOps) {
    auto addOp = dyn_cast<ascendc::AddL2Op>(op);
    if (!addOp)
      continue;
    if (llvm::any_of(addOp->getOperands(), [&](Value operand) {
          return vectorBroadcastDsts.contains(operand);
        }))
      return true;
  }
  return false;
}

static bool hasSupportedMixVectorMax(const MixPartitionSummary &summary) {
  return llvm::any_of(summary.vectorOps, [](Operation *op) {
    return isa<ascendc::MaxL2Op>(op);
  });
}

static bool isSupportedMixVectorMulUser(Operation *user) {
  auto mulOp = dyn_cast<ascendc::MulL2Op>(user);
  return mulOp &&
         getTensorStoragePartition(mulOp.getDst()) == MixPartitionKind::Vector;
}

static FailureOr<SupportedMixKernelConfig::EpilogueKind>
inferSupportedMixEpilogueKind(func::FuncOp funcOp,
                              const MixPartitionSummary &summary,
                              double &leakyReluAlpha) {
  bool hasVectorMax = hasSupportedMixVectorMax(summary);

  SupportedMixKernelConfig::EpilogueKind epilogueKind =
      SupportedMixKernelConfig::EpilogueKind::Unknown;
  funcOp.walk([&](Operation *op) {
    if (epilogueKind != SupportedMixKernelConfig::EpilogueKind::Unknown)
      return WalkResult::interrupt();
    auto dupOp = dyn_cast<ascendc::DuplicateL2Op>(op);
    if (!dupOp)
      return WalkResult::advance();
    if (getTensorStoragePartition(dupOp.getDst()) != MixPartitionKind::Vector)
      return WalkResult::advance();
    if (!hasVectorMax)
      return WalkResult::advance();
    bool usedByVectorMul =
        llvm::any_of(dupOp.getDst().getUsers(), isSupportedMixVectorMulUser);
    if (!usedByVectorMul)
      return WalkResult::advance();
    auto constOp = dupOp.getScalar().getDefiningOp<arith::ConstantOp>();
    if (!constOp)
      return WalkResult::advance();
    auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue());
    if (!floatAttr)
      return WalkResult::advance();
    leakyReluAlpha = floatAttr.getValue().convertToDouble();
    epilogueKind =
        (leakyReluAlpha == 0.0)
            ? SupportedMixKernelConfig::EpilogueKind::Relu
            : SupportedMixKernelConfig::EpilogueKind::LeakyRelu;
    return WalkResult::interrupt();
  });

  if (epilogueKind == SupportedMixKernelConfig::EpilogueKind::Unknown)
    return funcOp.emitOpError(
        "supported mix translation requires vector-region relu-style epilogue");
  return epilogueKind;
}

static SupportedMixKernelConfig::TaskKind
inferSupportedMixTaskKind(func::FuncOp funcOp,
                          const MixPartitionSummary &summary) {
  (void)funcOp;
  (void)summary;
  return SupportedMixKernelConfig::TaskKind::MixAic1To2;
}

static FailureOr<SupportedMixKernelConfig>
inferSupportedMixKernelConfig(func::FuncOp funcOp,
                              const MixPartitionSummary &summary) {
  SupportedMixKernelConfig config;
  config.taskKind = inferSupportedMixTaskKind(funcOp, summary);
  config.hasBiasAdd = inferSupportedMixHasBiasAdd(summary);
  auto epilogueKind = inferSupportedMixEpilogueKind(
      funcOp, summary, config.leakyReluAlpha);
  if (failed(epilogueKind))
    return failure();
  config.epilogueKind = *epilogueKind;
  return config;
}

// Supported mix emission helpers.
static void emitSupportedMixVectorEpilogue(raw_ostream &os,
                                           const SupportedMixKernelConfig &config) {
  if (config.epilogueKind == SupportedMixKernelConfig::EpilogueKind::Relu) {
    os << "    Relu(outLocal, inLocal, count);\n";
    return;
  }
  os << "    LeakyRelu(outLocal, inLocal, static_cast<float>("
     << llvm::formatv("{0:F6}", config.leakyReluAlpha).str()
     << "f), count);\n";
}

static void emitSupportedMixMatmulObjectDecl(raw_ostream &os) {
  os << "    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, half>,\n"
     << "           MatmulType<TPosition::GM, CubeFormat::ND, half>,\n"
     << "           MatmulType<TPosition::VECIN, CubeFormat::ND, float>,\n"
     << "           MatmulType<TPosition::GM, CubeFormat::ND, float>> mm;\n\n";
}

static void emitSupportedMixAicGlobalTensorSetup(
    raw_ostream &os, const SupportedMixKernelConfig &config) {
  os << "    GlobalTensor<half> aGM, bGM;\n"
     << "    GlobalTensor<float> cGM";
  if (config.hasBiasAdd)
    os << ", biasGM";
  os << ";\n"
     << "    aGM.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(a), tiling.M * tiling.Ka);\n"
     << "    bGM.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b), tiling.Kb * tiling.N);\n"
     << "    cGM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out), tiling.M * tiling.N);\n";
  if (config.hasBiasAdd)
    os << "    biasGM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bias), tiling.N);\n";
}

static void emitSupportedMixMatmulExecution(
    raw_ostream &os, const SupportedMixKernelConfig &config) {
  os << "\n"
     << "    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);\n"
     << "    mm.SetTensorA(aGM);\n"
     << "    mm.SetTensorB(bGM);\n"
     << (config.hasBiasAdd ? "    mm.SetBias(biasGM);\n" : "")
     << "    mm.template IterateAll(cGM);\n"
     << "    mm.End();\n";
}

static void emitSupportedMixCrossCoreSetFlag(
    raw_ostream &os, const MixTaskKindDescriptor &desc) {
  os << "    CrossCoreSetFlag<0x"
     << llvm::format_hex_no_prefix(desc.crossCoreMode, 1)
     << ", PIPE_FIX>("
     << desc.crossCoreFlagId << ");\n";
}

static void emitSupportedMixVectorCountDecl(raw_ostream &os,
                                            const MixTaskKindDescriptor &desc) {
  os << "    uint32_t count = static_cast<uint32_t>(tiling.singleCoreM * tiling.singleCoreN / "
     << desc.vectorTaskRatio << ");\n";
}

static void emitSupportedMixAivQueueSetup(raw_ostream &os,
                                          const MixTaskKindDescriptor &desc) {
  os << "    TQue<TPosition::VECIN, 1> reluInQueue;\n"
     << "    TQue<TPosition::VECOUT, 1> reluOutQueue;\n\n";
  emitSupportedMixVectorCountDecl(os, desc);
  os << "    GlobalTensor<float> cGM;\n"
     << "    cGM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out) + GetBlockIdx() * count, count);\n\n"
     << "    pipe.InitBuffer(reluInQueue, 1, count * sizeof(float));\n"
     << "    pipe.InitBuffer(reluOutQueue, 1, count * sizeof(float));\n\n";
}

static void emitSupportedMixAivInputCopy(raw_ostream &os) {
  os << "    LocalTensor<float> reluInLocal = reluInQueue.AllocTensor<float>();\n"
     << "    DataCopy(reluInLocal, cGM, count);\n"
     << "    reluInQueue.EnQue<float>(reluInLocal);\n\n"
     << "    LocalTensor<float> inLocal = reluInQueue.DeQue<float>();\n"
     << "    LocalTensor<float> outLocal = reluOutQueue.AllocTensor<float>();\n";
}

static void emitSupportedMixAivOutputCopy(raw_ostream &os) {
  os << "    reluOutQueue.EnQue<float>(outLocal);\n"
     << "    reluInQueue.FreeTensor(inLocal);\n\n"
     << "    LocalTensor<float> finalLocal = reluOutQueue.DeQue<float>();\n"
     << "    DataCopy(cGM, finalLocal, count);\n"
     << "    reluOutQueue.FreeTensor(finalLocal);\n";
}

static const MixRegionPlan *
findFirstMixRegionOfKind(ArrayRef<MixRegionPlan> regions, MixPartitionKind kind) {
  for (const MixRegionPlan &region : regions) {
    if (region.kind == kind)
      return &region;
  }
  return nullptr;
}

static void emitMixCubeRegion(raw_ostream &os, const MixRegionPlan &region,
                              const SupportedMixKernelConfig &config,
                              const MixTaskKindDescriptor &desc) {
  (void)region;
  (void)desc;
  os << "  if ASCEND_IS_AIC {\n";
  emitSupportedMixMatmulObjectDecl(os);
  emitSupportedMixAicGlobalTensorSetup(os, config);
  emitSupportedMixMatmulExecution(os, config);
}

static void emitMixBoundaryRegion(raw_ostream &os, const MixRegionPlan &region,
                                  const SupportedMixKernelConfig &config,
                                  const MixTaskKindDescriptor &desc) {
  (void)region;
  (void)config;
  emitSupportedMixCrossCoreSetFlag(os, desc);
  os << "  }\n\n"
     << "  if ASCEND_IS_AIV {\n";
  os << "    CrossCoreWaitFlag(" << desc.crossCoreFlagId << ");\n\n";
}

static void emitMixVectorRegion(raw_ostream &os, const MixRegionPlan &region,
                                const SupportedMixKernelConfig &config,
                                const MixTaskKindDescriptor &desc) {
  (void)region;
  emitSupportedMixAivQueueSetup(os, desc);
  emitSupportedMixAivInputCopy(os);
  emitSupportedMixVectorEpilogue(os, config);
  emitSupportedMixAivOutputCopy(os);
  os << "  }\n";
}

static void emitSupportedMixIncludesAndNamespaces(raw_ostream &os) {
  os << "#define __AFIR_RUNTIME_MIX_KERNEL_FUN_H__\n\n"
     << "#define ASCENDC_CUBE_ONLY\n"
     << "#include \"kernel_operator.h\"\n"
     << "#include \"lib/matmul_intf.h\"\n\n"
     << "using namespace AscendC;\n"
     << "using namespace matmul;\n\n";
}

static void emitSupportedMixCopyTilingHelper(raw_ostream &os) {
  os << "__aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM) {\n"
     << "  uint64_t *dst = reinterpret_cast<uint64_t *>(tiling);\n"
     << "  auto tiling64 = reinterpret_cast<__gm__ uint64_t *>(tilingGM);\n"
     << "  for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint64_t); ++i)\n"
     << "    dst[i] = tiling64[i];\n"
     << "}\n\n";
}

static void emitSupportedMixKernelSignature(raw_ostream &os,
                                            StringRef kernelName,
                                            const MixTaskKindDescriptor &desc) {
  os << "extern \"C\" __global__ __aicore__ void " << kernelName << "(\n"
     << "    GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR out, GM_ADDR workspace,\n"
     << "    GM_ADDR tilingGm) {\n"
     << "  KERNEL_TASK_TYPE_DEFAULT(" << desc.taskTypeSpelling << ");\n"
     << "  TPipe pipe;\n"
     << "  (void)workspace;\n\n"
     << "  TCubeTiling tiling;\n"
     << "  CopyTiling(&tiling, tilingGm);\n\n";
}

static void emitSupportedMixKernelPrologue(raw_ostream &os, StringRef kernelName,
                                           const MixTaskKindDescriptor &desc) {
  emitSupportedMixIncludesAndNamespaces(os);
  emitSupportedMixCopyTilingHelper(os);
  emitSupportedMixKernelSignature(os, kernelName, desc);
}

static void emitSupportedMixKernel(raw_ostream &os, func::FuncOp funcOp,
                                   const MixPartitionPlan &plan,
                                   const SupportedMixKernelConfig &config) {
  MixTaskKindDescriptor desc = getMixTaskKindDescriptor(config.taskKind);
  emitSupportedMixKernelPrologue(os, funcOp.getName(), desc);
  const MixRegionPlan *cubeRegion =
      findFirstMixRegionOfKind(plan.regions, MixPartitionKind::Cube);
  const MixRegionPlan *boundaryRegion = findFirstMixRegionOfKind(
      plan.regions, MixPartitionKind::Boundary);
  const MixRegionPlan *vectorRegion =
      findFirstMixRegionOfKind(plan.regions, MixPartitionKind::Vector);
  if (!cubeRegion || !boundaryRegion || !vectorRegion) {
    llvm_unreachable("supported mix emission requires cube, boundary, and vector regions");
  }
  emitMixCubeRegion(os, *cubeRegion, config, desc);
  emitMixBoundaryRegion(os, *boundaryRegion, config, desc);
  emitMixVectorRegion(os, *vectorRegion, config, desc);
  os << "}\n";
}

/// Emit the TilingData struct declaration from a PyStructType.
static LogicalResult emitTilingStructDecl(CodeEmitter &emitter, Location loc,
                                          emitasc::PyStructType pyType) {
  auto &os = emitter.ostream();
  StringRef structName = pyType.getNameAttr().getValue();
  os << "struct " << structName << " {\n";
  os.indent();

  auto typesAttr = pyType.getTypesAttr();
  auto namesAttr = pyType.getNamesAttr();
  auto types = typesAttr.getValue();
  auto names = namesAttr.getValue();

  if (types.size() != names.size())
    return emitError(loc, "PyStructType types/names size mismatch");

  for (size_t i = 0; i < types.size(); ++i) {
    Type fieldType = cast<TypeAttr>(types[i]).getValue();
    StringRef fieldName = cast<StringAttr>(names[i]).getValue();
    if (failed(emitter.emitType(loc, fieldType)))
      return failure();
    os << " " << fieldName << ";\n";
  }

  os.unindent() << "};\n\n";
  return success();
}

/// Write tiling_space.json skeleton to outPath.
/// dim_argN_D fields → fixed:true, shape_key:"argN_dimD".
/// Other fields (TB_M etc.) → fixed:false, values:[].
static void emitTilingSpaceJson(StringRef outPath,
                                StringRef kernelFile,
                                StringRef kernelName,
                                emitasc::PyStructType tilingType) {
  auto isDimField = [](StringRef name) {
    return name.starts_with("dim_arg");
  };
  // "dim_arg2_1" → drop "dim_" → "arg2_1" → rfind '_' → "arg2" + "_dim" + "1"
  auto makeShapeKey = [](StringRef name) -> std::string {
    StringRef rest = name.drop_front(4); // drop "dim_"
    auto pos = rest.rfind('_');
    if (pos == StringRef::npos) return rest.str(); // single-component: no dimension index
    return rest.substr(0, pos).str() + "_dim" + rest.substr(pos + 1).str();
  };

  auto names = tilingType.getNamesAttr().getValue();

  if (names.empty()) {
    llvm::errs() << "Warning: tiling struct has no fields; "
                    "tiling_space.json will have empty tiling_params\n";
  }

  llvm::json::Array params;
  for (auto &nameAttr : names) {
    StringRef name = cast<StringAttr>(nameAttr).getValue();
    llvm::json::Object p;
    p["name"] = name.str();
    p["type"] = "int64"; // TODO: derive from PyStructType field type when non-i64 fields exist
    if (isDimField(name)) {
      p["fixed"] = true;
      p["shape_key"] = makeShapeKey(name);
    } else {
      p["fixed"] = false;
      p["values"] = llvm::json::Array{};
    }
    params.push_back(std::move(p));
  }

  llvm::json::Object root;
  root["kernel"]         = kernelName.str();
  root["kernel_file"]    = kernelFile.str();
  root["soc"]            = "Ascend910B1";
  root["block_dim_expr"] = "";
  root["tiling_params"]  = std::move(params);

  std::error_code ec;
  llvm::raw_fd_ostream f(outPath, ec);
  if (ec) {
    llvm::errs() << "Warning: cannot write tiling_space.json to "
                 << outPath << ": " << ec.message() << "\n";
    return;
  }
  llvm::json::OStream jos(f, /*IndentSize=*/2);
  jos.value(llvm::json::Value(std::move(root)));
  f << "\n";
}

/// Emit the CANN-standard function signature and body.
static LogicalResult printCannFuncOp(CodeEmitter &emitter,
                                     func::FuncOp funcOp) {
  CodeEmitter::Scope scope(emitter);
  auto &os = emitter.ostream();

  // cann.num_inputs must be present (set by CanonicalizeCannSignaturePass).
  if (!funcOp->hasAttr("cann.num_inputs"))
    return funcOp.emitOpError("missing cann.num_inputs attribute; "
                               "run --canonicalize-cann-signature first");

  auto args = funcOp.getArguments();
  int numArgs = (int)args.size();

  // Layout: [0..N-3] = inputs+outputs (all GM_ADDR), [N-2] = workspace
  // (memref<ui8>), [N-1] = tiling (!emitasc.py_struct).
  if (numArgs < 4)
    return funcOp.emitOpError(
        "CANN function must have at least 4 args "
        "(inputs, outputs, workspace, tiling)");

  BlockArgument tilingArg = args[numArgs - 1];
  auto tilingType = dyn_cast<emitasc::PyStructType>(tilingArg.getType());
  if (!tilingType)
    return funcOp.emitOpError("last argument must be !emitasc.py_struct");

  // Validate workspace arg is memref<ui8>.
  auto wsType = dyn_cast<MemRefType>(args[numArgs - 2].getType());
  if (!wsType || !wsType.getElementType().isUnsignedInteger(8))
    return funcOp.emitOpError(
        "second-to-last argument must be memref<ui8> workspace");

  // Emit function header
  os << "extern \"C\" __global__ __aicore__ void " << funcOp.getName() << "(\n";
  os.indent();

  // Emit input/output GM_ADDR args (all memref args)
  for (int i = 0; i < numArgs - 1; ++i) {
    os << "GM_ADDR " << emitter.getOrCreateName(args[i]);
    os << ",\n";
  }

  // Emit tiling arg as struct by value
  StringRef tilingStructName = tilingType.getNameAttr().getValue();
  os << tilingStructName << " " << emitter.getOrCreateName(tilingArg) << "\n";

  os.unindent() << ") {\n";
  os.indent();

  // Emit body ops
  for (Block &block : funcOp.getBlocks()) {
    for (Operation &op : block.getOperations()) {
      // Skip func.return (void kernel, no return value needed)
      if (isa<func::ReturnOp>(op)) {
        os << "return;\n";
        continue;
      }
      if (failed(emitOperation(emitter, op, needsSemicolon(&op))))
        return failure();
    }
  }

  os.unindent() << "}\n";
  return success();
}

} // namespace

// ─── Pre-pass: replace broken PyAsc emitter ops with emitasc.verbatim ───────
//
// PyAsc's auto-generated printOperation() for BroadcastL2Op emits a wrong
// reinterpret_cast<uint64_t> and for ReduceSum2DL2Op uses the non-existent
// AscendC::ReduceLayout enum.  Both ops live in nested SCF regions that are
// recursively emitted by PyAsc's internal emitOperation() – meaning our
// top-level tryEmitAfirOp hook never reaches them.
//
// Solution: before calling the emitter, walk every occurrence of these ops and
// replace each one with an emitasc.verbatim that produces the correct C++.
// The verbatim op is in PyAsc's PrintableOpTypes and is emitted verbatim,
// so no further interception is needed.
//
// BroadcastL2Op operands:  (dst, src, dstShape..., srcShape...)
//   constRank attr tells us the shape array length.
//   Verbatim template (block-scoped to avoid name collisions):
//     {
//       uint32_t _ds[N] = {(uint32_t)$2, ...};
//       uint32_t _ss[N] = {(uint32_t)$K, ...};
//       AscendC::Broadcast<half,N,0>($0, $1, _ds, _ss);
//     }
//
// ReduceSum2DL2Op operands: (dst, src)
//   layout attr = AR (0) or RA (1).
//   Verbatim template:
//     {
//       uint32_t _s[2]={(uint32_t)($0.GetSize()/sizeof(half)),
//                       (uint32_t)($1.GetSize()/$0.GetSize())};
//       AscendC::LocalTensor<uint8_t> _t;
//       AscendC::PopStackBuffer<uint8_t,AscendC::TPosition::LCM>(_t);
//       AscendC::ReduceSum<half,AscendC::AR>($0,$1,_t,_s,false);
//     }
//
static void fixBrokenOpEmitters(Operation *moduleOp) {
  IRRewriter rewriter(moduleOp->getContext());

  // GlobalTensorSetGlobalBufferOp → verbatim with pointer arithmetic offset.
  //
  // PyAsc auto-generates: $tensor.SetGlobalBuffer($buffer_ptr, $offset)
  // But AscendC SetGlobalBuffer(ptr, uint64_t) treats the 2nd arg as a SIZE
  // hint, NOT an element offset.  As a result, DataCopy always reads/writes
  // from the base pointer, ignoring the offset.  This breaks multi-block
  // kernels where each block operates on a different slice of the buffer.
  //
  // Fix: emit pointer arithmetic to bake the offset into the pointer:
  //   $tensor.SetGlobalBuffer($buffer_ptr + $offset);
  //
  // When the offset is absent (op.getSize() is null), emit the 1-arg form.
  moduleOp->walk([&](ascendc::GlobalTensorSetGlobalBufferOp op) {
    Value sizeVal = op.getSize();
    if (!sizeVal)
      return; // no offset — let PyAsc emit the 1-arg form unchanged

    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    // Verbatim: $0 = tensor, $1 = buffer_ptr (__gm__ half*), $2 = offset (i32)
    // Emit: $0.SetGlobalBuffer($1 + $2);
    rewriter.create<emitasc::VerbatimOp>(
        loc,
        rewriter.getStringAttr("$0.SetGlobalBuffer($1 + $2)"),
        ValueRange({op.getTensor(), op.getBuffer(), sizeVal}));
    rewriter.eraseOp(op);
  });

  // BroadcastL2Op → verbatim
  moduleOp->walk([&](ascendc::BroadcastL2Op op) {
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    uint32_t rank = op.getConstRank();

    // Build verbatim string with $N placeholders.
    // Operand layout: $0=dst, $1=src, $2..$2+rank-1=dstShape, $2+rank..=srcShape
    std::string tmpl = "{\n";
    tmpl += "  uint32_t _afir_ds[" + std::to_string(rank) + "] = {";
    for (uint32_t i = 0; i < rank; ++i) {
      if (i) tmpl += ", ";
      tmpl += "(uint32_t)$" + std::to_string(2 + i);
    }
    tmpl += "};\n";
    tmpl += "  uint32_t _afir_ss[" + std::to_string(rank) + "] = {";
    for (uint32_t i = 0; i < rank; ++i) {
      if (i) tmpl += ", ";
      tmpl += "(uint32_t)$" + std::to_string(2 + rank + i);
    }
    tmpl += "};\n";
    // Determine axis: if srcShape[-1] == 1 (column broadcast), axis=1.
    // If srcShape[0] == 1 (row broadcast), axis=0.
    // Inspect the last srcShape value operand: if it is a constant 1, use axis=1.
    auto srcShapeVals = op.getSrcShape();
    int axis = 0;
    if (!srcShapeVals.empty()) {
      Value lastSrc = srcShapeVals[srcShapeVals.size() - 1];
      if (auto constOp = lastSrc.getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
          if (intAttr.getInt() == 1)
            axis = 1;
        }
      }
    }
    // Use actual element type of dst instead of hardcoded 'half'.
    auto dstElemType =
        cast<ascendc::LocalTensorType>(op.getDst().getType()).getElementType();
    std::string elemTypeStr;
    if (dstElemType.isF16())
      elemTypeStr = "half";
    else if (dstElemType.isF32())
      elemTypeStr = "float";
    else if (dstElemType.isF64())
      elemTypeStr = "double";
    else if (auto iType = dyn_cast<IntegerType>(dstElemType)) {
      bool isUnsigned = iType.isUnsigned();
      elemTypeStr = (isUnsigned ? "uint" : "int") +
                    std::to_string(iType.getWidth()) + "_t";
    } else {
      elemTypeStr = "half"; // fallback
    }
    tmpl += "  AscendC::Broadcast<" + elemTypeStr + ", " + std::to_string(rank) +
            ", " + std::to_string(axis) + ">($0, $1, _afir_ds, _afir_ss);\n}";

    SmallVector<Value> args;
    args.push_back(op.getDst());
    args.push_back(op.getSrc());
    for (Value v : op.getDstShape())
      args.push_back(v);
    for (Value v : op.getSrcShape())
      args.push_back(v);

    rewriter.create<emitasc::VerbatimOp>(
        loc, rewriter.getStringAttr(tmpl), ValueRange(args));
    rewriter.eraseOp(op);
  });

  // ReduceSum2DL2Op → verbatim
  //
  // AR layout: src[rows, cols] → dst[rows] by summing each row.
  // RA layout: not yet implemented.
  //
  // Uses AscendC::ReduceSum<half> per-row with a 32-byte scratch VECCALC TBuf.
  // GetValue/SetValue scalar loops over individual elements are avoided because
  // the simulator's LocalTensor::GetValue() does not correctly access elements
  // beyond the first 16 when called on a TBuf::Get() tensor allocated inside a
  // loop (the simulator does not update the LocalTensor's internal size field
  // for loop-iteration-dependent InitBuffer calls).
  //
  // We look up InitBuffer/InitQueue ops in the IR to get the byte-lengths as
  // explicit SSA operands ($2 = dst_queue_bytes = rows*2,
  // $3 = src_tbuf_bytes = rows*N*2), avoiding GetSize() entirely.
  //
  // Operand layout in the emitted verbatim:
  //   $0 = dst (VECOUT TQue LocalTensor)
  //   $1 = src (VECCALC accumulator LocalTensor)
  //   $2 = dst queue byte-length (= rows * sizeof(half)) from TPipeInitQueueOp
  //   $3 = src tbuf byte-length (= rows * N * sizeof(half)) from TPipeInitBufferOp
  moduleOp->walk([&](ascendc::ReduceSum2DL2Op op) {
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    bool isAR = (op.getLayout() == ascendc::ReduceLayout::AR);

    // $2: find TPipeInitQueueOp length for the dst TQue (rows * sizeof(half)).
    Value dstQueueLenVal;
    if (auto allocOp =
            op.getDst().getDefiningOp<ascendc::TQueBindAllocTensorOp>()) {
      Value queueVal = allocOp.getQueue();
      for (auto *user : queueVal.getUsers()) {
        if (auto initQ = dyn_cast<ascendc::TPipeInitQueueOp>(user)) {
          dstQueueLenVal = initQ.getLength();
          break;
        }
      }
    }

    // $3: find TPipeInitBufferOp length for the src TBuf (rows*N*sizeof(half)).
    Value srcTBufLenVal;
    if (auto getOp = op.getSrc().getDefiningOp<ascendc::TBufGetTensorOp>()) {
      Value tbufVal = getOp.getBuffer();
      for (auto *user : tbufVal.getUsers()) {
        if (auto initB = dyn_cast<ascendc::TPipeInitBufferOp>(user)) {
          srcTBufLenVal = initB.getLength();
          break;
        }
      }
    }

    std::string tmpl = "{\n";
    if (isAR) {
      // AR: dst[r] = sum(src[r*cols .. r*cols+cols-1])
      // Use ReduceSum<half> per row with a 32-byte scratch VECCALC TBuf.
      // The TPipe is passed as the last operand so we can InitBuffer the scratch.
      // $1[r * cols] slices the src tensor to the start of row r.
      std::string pipeRef; // placeholder name for pipe arg
      if (dstQueueLenVal && srcTBufLenVal) {
        // $2 = dst_bytes, $3 = src_bytes, $4 = pipe
        tmpl += "  uint32_t _afir_rows = (uint32_t)($2 / sizeof(half));\n";
        tmpl += "  uint32_t _afir_cols = (uint32_t)($3 / $2);\n";
        pipeRef = "$4";
      } else if (dstQueueLenVal) {
        // $2 = dst_bytes, $3 = pipe
        tmpl += "  uint32_t _afir_rows = (uint32_t)($2 / sizeof(half));\n";
        tmpl += "  uint32_t _afir_cols = (uint32_t)($1.GetSize() / $2);\n";
        pipeRef = "$3";
      } else {
        // $2 = pipe
        tmpl += "  uint32_t _afir_rows = (uint32_t)($0.GetSize() / sizeof(half));\n";
        tmpl += "  uint32_t _afir_cols = (uint32_t)($1.GetSize() / $0.GetSize());\n";
        pipeRef = "$2";
      }
      // Use TWO separate TBufs: _afir_tbuf_dst (result) and _afir_tbuf_ws (workspace).
      // ReduceSum requires dst != sharedTmpBuffer; aliasing them gives wrong results
      // on arch 3101 because the intermediate tree-reduction overwrites the output.
      tmpl += "  AscendC::TBuf<AscendC::TPosition::VECCALC> _afir_tbuf_dst;\n";
      tmpl += "  AscendC::TBuf<AscendC::TPosition::VECCALC> _afir_tbuf_ws;\n";
      tmpl += "  " + pipeRef + ".InitBuffer(_afir_tbuf_dst, 32);\n";
      tmpl += "  " + pipeRef + ".InitBuffer(_afir_tbuf_ws, 32);\n";
      tmpl += "  AscendC::LocalTensor<half> _afir_scalar = _afir_tbuf_dst.Get<half>();\n";
      tmpl += "  AscendC::LocalTensor<half> _afir_ws = _afir_tbuf_ws.Get<half>();\n";
      tmpl += "  for (uint32_t _afir_r = 0; _afir_r < _afir_rows; _afir_r++) {\n";
      tmpl += "    AscendC::ReduceSum<half>(_afir_scalar, $1[_afir_r * _afir_cols],\n";
      tmpl += "                            _afir_ws, (int32_t)_afir_cols);\n";
      tmpl += "    $0.SetValue(_afir_r, _afir_scalar.GetValue(0));\n";
      tmpl += "  }\n}";
    } else {
      tmpl += "  // RA layout not yet implemented\n}";
    }

    // Find the TPipe value: walk enclosing function for PipeOp.
    Value pipeVal;
    if (auto funcOp = op->getParentOfType<func::FuncOp>()) {
      funcOp.walk([&](ascendc::PipeOp pipeOp) {
        pipeVal = pipeOp.getResult();
        return WalkResult::interrupt();
      });
    }

    SmallVector<Value> args = {op.getDst(), op.getSrc()};
    if (dstQueueLenVal)
      args.push_back(dstQueueLenVal);
    if (srcTBufLenVal)
      args.push_back(srcTBufLenVal);
    if (pipeVal)
      args.push_back(pipeVal);

    rewriter.create<emitasc::VerbatimOp>(
        loc, rewriter.getStringAttr(tmpl), ValueRange(args));
    rewriter.eraseOp(op);
  });
}

LogicalResult mlir::translateToCannKernel(Operation *op, raw_ostream &os,
                                          StringRef tilingSpaceOutPath,
                                          StringRef kernelFile) {
  auto moduleOp = dyn_cast<ModuleOp>(op);
  if (!moduleOp)
    return op->emitOpError("expected a module op");

  func::FuncOp primaryKernel = findPrimaryGlobalKernel(moduleOp);
  if (primaryKernel &&
      getKernelKind(primaryKernel) == AscendCKernelKind::Mix) {
    MixPartitionSummary mixPartitionSummary =
        buildMixPartitionSummary(primaryKernel);
    MixPartitionPlan mixPartitionPlan =
        buildInitialMixPartitionPlan(primaryKernel, mixPartitionSummary);
    MixSingleChainValidation singleChainValidation =
        validateSingleChainGenericMixPlan(mixPartitionPlan);

    if (singleChainValidation.succeeded()) {
      FailureOr<SupportedMixKernelConfig> supportedMixConfig =
          inferSupportedMixKernelConfig(primaryKernel, mixPartitionSummary);
      if (failed(supportedMixConfig))
        return failure();
      emitSupportedMixKernel(os, primaryKernel, mixPartitionPlan,
                             *supportedMixConfig);
      return success();
    }

    if (canLowerLegacySupportedMix(primaryKernel, mixPartitionSummary)) {
      FailureOr<SupportedMixKernelConfig> supportedMixConfig =
          inferSupportedMixKernelConfig(primaryKernel, mixPartitionSummary);
      if (failed(supportedMixConfig))
        return failure();
      emitSupportedMixKernel(os, primaryKernel, mixPartitionPlan,
                             *supportedMixConfig);
      return success();
    }

    return primaryKernel.emitOpError(Twine(
        "mix translation requires a supported cube/vector partitioned kernel "
        "shape; generic single-chain analysis rejected plan because ") +
                                     (singleChainValidation.succeeded()
                                          ? Twine("legacy lowering also failed")
                                          : Twine(describeMixSingleChainValidation(
                                                singleChainValidation))));
  }

  // Replace ops whose PyAsc emitters generate wrong C++ with verbatim.
  fixBrokenOpEmitters(op);

  CodeEmitter emitter(os);
  CodeEmitter::Scope scope(emitter);

  os << "#include \"kernel_operator.h\"\n";
  // adv_api headers required by BroadcastL2Op and ReduceSum2DL2Op emitters.
  // These are not included by kernel_operator.h but are available via the
  // tikcfw/include search path added by the compiler driver.
  os << "#include \"adv_api/broadcast/broadcast.h\"\n";
  os << "#include \"adv_api/reduce/reduce.h\"\n";
  os << "\n";

  // First pass: emit TilingData struct declarations from aicore funcs
  bool jsonWritten = false;
  for (Operation &child : moduleOp.getBody()->getOperations()) {
    auto funcOp = dyn_cast<func::FuncOp>(child);
    if (!funcOp)
      continue;
    if (!funcOp->hasAttr(ascendc::attr::global))
      continue;

    auto args = funcOp.getArguments();
    if (args.empty())
      continue;
    auto tilingType =
        dyn_cast<emitasc::PyStructType>(args.back().getType());
    if (!tilingType)
      continue;

    if (failed(emitTilingStructDecl(emitter, funcOp.getLoc(), tilingType)))
      return failure();

    // Write JSON skeleton for the first aicore func only
    if (!tilingSpaceOutPath.empty() && !jsonWritten) {
      emitTilingSpaceJson(tilingSpaceOutPath, kernelFile,
                          funcOp.getName(), tilingType);
      jsonWritten = true;
    }
  }

  // Second pass: emit aicore kernel functions only.
  // Non-aicore ops (transform sequences, helper modules, etc.) are skipped —
  // they are pipeline infrastructure, not C++ kernel code.
  for (Operation &child : moduleOp.getBody()->getOperations()) {
    auto funcOp = dyn_cast<func::FuncOp>(child);
    if (!funcOp || !funcOp->hasAttr(ascendc::attr::global))
      continue;
    if (failed(printCannFuncOp(emitter, funcOp)))
      return failure();
  }

  return success();
}
