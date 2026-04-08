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

#include <limits>
#include <functional>
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
  Operation *producerOp = nullptr;
  Operation *consumerOp = nullptr;
};

struct MixRegionPlan {
  MixPartitionKind kind = MixPartitionKind::Unknown;
  SmallVector<Operation *> ops;
  SmallVector<MixBoundaryValue> inputs;
  SmallVector<MixBoundaryValue> outputs;
  unsigned firstOpOrder = 0;
  unsigned lastOpOrder = 0;
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
  MissingCubeToBoundaryCrossing,
  MissingBoundaryToVectorCrossing,
  MissingSupportedBoundaryPayload,
  BoundaryChainNotLinear,
  ExtraCubeOpsOutsideChain,
  ExtraVectorOpsOutsideChain,
  InvalidOrdering,
};

struct MixSingleChainValidation {
  SmallVector<MixSingleChainFailureReason> failureReasons;
  struct SelectedBoundaryCrossing {
    MixBoundaryValue input;
    MixBoundaryValue output;
  };
  std::optional<SelectedBoundaryCrossing> selectedBoundaryCrossing;

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

struct SupportedMixBoundaryPayload {
  Type elementType;
  ascendc::TQueBindEnqueTensorOp inputEnqueue;
  ascendc::DataCopyCO12DstOp transferCopy;
  ascendc::TQueBindAllocTensorOp outputAlloc;
};

struct SupportedMixBoundaryLayer {
  MixBoundaryValue input;
  MixBoundaryValue output;
  SupportedMixBoundaryPayload payload;
};

struct GenericMixSingleChainEmissionPlan {
  const MixRegionPlan *cubeRegion = nullptr;
  const MixRegionPlan *boundaryRegion = nullptr;
  const MixRegionPlan *vectorRegion = nullptr;
  MixSingleChainValidation::SelectedBoundaryCrossing selectedBoundaryCrossing;
};

struct GenericMixSingleChainSupportedLowering {
  SupportedMixBoundaryLayer boundaryLayer;
  SupportedMixKernelConfig config;
};

enum class GenericMixSingleChainEmissionFailureReason {
  MissingRequiredRegions,
  MissingSelectedBoundaryCrossing,
};

enum class SupportedMixLoweringFailureReason {
  UnsupportedLegacySignature,
  UnsupportedReluStyleEpilogue,
  MissingSelectedBoundaryCrossing,
  UnsupportedBoundaryPayload,
};

static SmallVector<MixBoundaryValue>
filterBoundaryValues(ArrayRef<MixBoundaryValue> values,
                     MixPartitionKind producer, MixPartitionKind consumer);

static const MixRegionPlan *
findFirstMixRegionOfKind(ArrayRef<MixRegionPlan> regions,
                         MixPartitionKind kind);

static Type getSupportedMixTensorElementType(Value value);

static FailureOr<SupportedMixBoundaryPayload>
buildSupportedMixBoundaryPayload(const MixBoundaryValue &input,
                                 const MixBoundaryValue &output,
                                 ArrayRef<Operation *> boundaryOps);

static FailureOr<SupportedMixBoundaryLayer>
inferLegacySupportedMixBoundaryLayer(ArrayRef<Operation *> boundaryOps);

static StringRef stringifyGenericMixSingleChainEmissionFailureReason(
    GenericMixSingleChainEmissionFailureReason reason);

static StringRef stringifySupportedMixLoweringFailureReason(
    SupportedMixLoweringFailureReason reason);

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

static std::optional<ascendc::TPosition>
getQueueLikePosition(Type type) {
  if (auto queueType = dyn_cast<ascendc::QueueType>(type))
    return queueType.getPosition();
  if (auto tbufType = dyn_cast<ascendc::TBufType>(type))
    return tbufType.getTPosition();
  return std::nullopt;
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
                            MixPartitionKind consumer, Operation *producerOp,
                            Operation *consumerOp) {
    if (producer == MixPartitionKind::Unknown ||
        consumer == MixPartitionKind::Unknown || producer == consumer)
      return;
    for (const MixBoundaryValue &existing : boundaryValues) {
      if (existing.value == value && existing.producer == producer &&
          existing.consumer == consumer && existing.producerOp == producerOp &&
          existing.consumerOp == consumerOp)
        return;
    }
    boundaryValues.push_back(
        {value, producer, consumer, producerOp, consumerOp});
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
                       getConsumerPartition(user), op, user);
    }
  }

  for (Operation *op : summary.boundaryOps) {
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      MixPartitionKind producer =
          defOp ? getPartitionForSummaryOp(defOp) : MixPartitionKind::Unknown;
      if (producer == MixPartitionKind::Unknown)
        producer = getTensorStoragePartition(operand);
      recordCrossing(operand, producer, MixPartitionKind::Boundary, defOp, op);
    }

    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers())
        recordCrossing(result, MixPartitionKind::Boundary,
                       getConsumerPartition(user), op, user);
    }

    if (auto copyOp = dyn_cast<ascendc::DataCopyCO12DstOp>(op)) {
      Operation *inputProducerOp = copyOp.getSrc().getDefiningOp();
      if (auto inputDeque =
              dyn_cast_or_null<ascendc::TQueBindDequeTensorOp>(inputProducerOp)) {
        for (Operation *queueUser : inputDeque.getQueue().getUsers()) {
          auto enqueTensor = dyn_cast<ascendc::TQueBindEnqueTensorOp>(queueUser);
          if (enqueTensor && enqueTensor.getQueue() == inputDeque.getQueue()) {
            inputProducerOp = enqueTensor;
            break;
          }
        }
      }
      recordCrossing(copyOp.getSrc(), getTensorStoragePartition(copyOp.getSrc()),
                     MixPartitionKind::Boundary, inputProducerOp, op);
      recordCrossing(copyOp.getDst(), MixPartitionKind::Boundary,
                     getTensorStoragePartition(copyOp.getDst()), op,
                     copyOp.getDst().getDefiningOp());
    }
  }

  for (Operation *op : summary.vectorOps) {
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp)
        continue;
      recordCrossing(operand, getPartitionForSummaryOp(defOp),
                     MixPartitionKind::Vector, defOp, op);
    }
  }

  return boundaryValues;
}

static MixPartitionPlan buildInitialMixPartitionPlan(
    func::FuncOp funcOp, const MixPartitionSummary &summary) {
  MixPartitionPlan plan;
  SmallVector<MixBoundaryValue> boundaryValues = collectMixBoundaryValues(summary);
  llvm::DenseMap<Operation *, unsigned> opOrder;
  unsigned nextOrder = 0;
  funcOp.walk([&](Operation *op) { opOrder[op] = nextOrder++; });

  auto addRegion = [&](MixPartitionKind kind,
                       ArrayRef<Operation *> ops) -> void {
    if (ops.empty())
      return;
    MixRegionPlan region;
    region.kind = kind;
    region.ops.append(ops.begin(), ops.end());
    region.firstOpOrder = std::numeric_limits<unsigned>::max();
    region.lastOpOrder = 0;
    for (Operation *op : ops) {
      auto it = opOrder.find(op);
      if (it == opOrder.end())
        continue;
      region.firstOpOrder = std::min(region.firstOpOrder, it->second);
      region.lastOpOrder = std::max(region.lastOpOrder, it->second);
    }
    if (region.firstOpOrder == std::numeric_limits<unsigned>::max())
      region.firstOpOrder = 0;
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

static bool hasFailureReason(ArrayRef<MixSingleChainFailureReason> reasons,
                             MixSingleChainFailureReason reason) {
  return llvm::is_contained(reasons, reason);
}

static bool shouldAttemptLegacyFallbackAfterValidationFailure(
    const MixSingleChainValidation &validation) {
  return !hasFailureReason(validation.failureReasons,
                           MixSingleChainFailureReason::ExtraCubeOpsOutsideChain) &&
         !hasFailureReason(validation.failureReasons,
                           MixSingleChainFailureReason::ExtraVectorOpsOutsideChain);
}

static Type getSupportedMixTensorElementType(Value value) {
  if (auto localTensorType = dyn_cast<ascendc::LocalTensorType>(value.getType()))
    return localTensorType.getElementType();
  if (auto globalTensorType = dyn_cast<ascendc::GlobalTensorType>(value.getType()))
    return globalTensorType.getElementType();
  return Type();
}

static SmallVector<MixBoundaryValue>
filterBoundaryValues(ArrayRef<MixBoundaryValue> values,
                     MixPartitionKind producer, MixPartitionKind consumer) {
  SmallVector<MixBoundaryValue> filtered;
  for (const MixBoundaryValue &value : values) {
    if (value.producer == producer && value.consumer == consumer)
      filtered.push_back(value);
  }
  return filtered;
}

static llvm::DenseSet<Operation *>
collectAncestorPartitionOps(Value seed,
                            const llvm::DenseMap<Operation *, MixPartitionKind>
                                &partitionMap,
                            MixPartitionKind targetKind) {
  llvm::DenseSet<Operation *> collected;
  llvm::DenseSet<Operation *> visiting;

  std::function<void(Value)> visitValue = [&](Value value) {
    for (const auto &entry : partitionMap) {
      Operation *candidate = entry.first;
      MixPartitionKind candidateKind = entry.second;
      if (candidateKind != targetKind || candidate->getNumOperands() == 0)
        continue;
      if (candidate->getOperand(0) != value || !collected.insert(candidate).second)
        continue;
      for (Value operand : candidate->getOperands().drop_front())
        visitValue(operand);
    }

    Operation *defOp = value.getDefiningOp();
    if (auto dequeTensor = dyn_cast_or_null<ascendc::TQueBindDequeTensorOp>(defOp)) {
      for (Operation *queueUser : dequeTensor.getQueue().getUsers()) {
        auto enqueTensor = dyn_cast<ascendc::TQueBindEnqueTensorOp>(queueUser);
        if (!enqueTensor || enqueTensor.getQueue() != dequeTensor.getQueue())
          continue;
        visitValue(enqueTensor.getTensor());
      }
    }
    if (!defOp || !visiting.insert(defOp).second)
      return;
    auto it = partitionMap.find(defOp);
    if (it != partitionMap.end() && it->second == targetKind)
      collected.insert(defOp);
    for (Value operand : defOp->getOperands())
      visitValue(operand);
    visiting.erase(defOp);
  };

  visitValue(seed);
  return collected;
}

static llvm::DenseSet<Operation *>
collectDescendantPartitionOps(Value seed,
                              const llvm::DenseMap<Operation *, MixPartitionKind>
                                  &partitionMap,
                              MixPartitionKind targetKind) {
  llvm::DenseSet<Operation *> collected;
  llvm::DenseSet<Value> visitingValues;

  auto getPrimaryPartitionWriteTarget = [&](Operation *op) -> Value {
    auto it = partitionMap.find(op);
    if (it == partitionMap.end() || it->second != targetKind ||
        op->getNumOperands() == 0)
      return Value();
    Value dst = op->getOperand(0);
    return getTensorStoragePartition(dst) == targetKind ? dst : Value();
  };

  std::function<void(Value)> visitValue = [&](Value value) {
    if (!visitingValues.insert(value).second)
      return;
    for (Operation *user : value.getUsers()) {
      auto enqueTensor = dyn_cast<ascendc::TQueBindEnqueTensorOp>(user);
      if (!enqueTensor || enqueTensor.getTensor() != value)
        continue;
      for (Operation *queueUser : enqueTensor.getQueue().getUsers()) {
        auto dequeTensor = dyn_cast<ascendc::TQueBindDequeTensorOp>(queueUser);
        if (!dequeTensor || dequeTensor.getQueue() != enqueTensor.getQueue())
          continue;
        visitValue(dequeTensor.getResult());
      }
    }
    for (Operation *user : value.getUsers()) {
      auto it = partitionMap.find(user);
      if (it != partitionMap.end() && it->second == targetKind) {
        collected.insert(user);
        if (Value written = getPrimaryPartitionWriteTarget(user);
            written && written != value)
          visitValue(written);
      }
      for (Value result : user->getResults())
        visitValue(result);
    }
    visitingValues.erase(value);
  };

  visitValue(seed);
  return collected;
}

static SmallVector<Value> collectSupportedMixVectorSeedValues(
    SupportedMixBoundaryPayload &payload, ArrayRef<Operation *> vectorOps) {
  SmallVector<Value> seeds;
  MixPartitionKind payloadSourcePartition =
      getTensorStoragePartition(payload.transferCopy.getSrc());
  if (payloadSourcePartition == MixPartitionKind::Unknown)
    return seeds;

  for (Operation *op : vectorOps) {
    for (Value operand : op->getOperands()) {
      if (getTensorStoragePartition(operand) != payloadSourcePartition)
        continue;
      if (!llvm::is_contained(seeds, operand))
        seeds.push_back(operand);
    }
  }
  return seeds;
}

static llvm::DenseSet<Operation *>
closePartitionOpsOverAncestors(
    llvm::DenseSet<Operation *> seeds, ArrayRef<Operation *> partitionOps,
    const llvm::DenseMap<Operation *, MixPartitionKind> &partitionMap,
    MixPartitionKind targetKind) {
  llvm::DenseSet<Operation *> partitionOpSet(partitionOps.begin(),
                                             partitionOps.end());
  SmallVector<Operation *> worklist(seeds.begin(), seeds.end());

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    for (Value operand : op->getOperands()) {
      for (const auto &entry : partitionMap) {
        Operation *candidate = entry.first;
        MixPartitionKind candidateKind = entry.second;
        if (candidateKind != targetKind || candidate->getNumOperands() == 0 ||
            candidate->getOperand(0) != operand ||
            !partitionOpSet.contains(candidate) || !seeds.insert(candidate).second)
          continue;
        worklist.push_back(candidate);
      }
      Operation *defOp = operand.getDefiningOp();
      if (!defOp)
        continue;
      auto it = partitionMap.find(defOp);
      if (it == partitionMap.end() || it->second != targetKind ||
          !partitionOpSet.contains(defOp) || !seeds.insert(defOp).second)
        continue;
      worklist.push_back(defOp);
    }
  }

  return seeds;
}

static void buildBoundaryRegionGraph(
    ArrayRef<Operation *> boundaryOps,
    const llvm::DenseMap<Operation *, MixPartitionKind> &partitionMap,
    llvm::DenseMap<Operation *, SmallVector<Operation *>> &successors,
    llvm::DenseMap<Operation *, SmallVector<Operation *>> &predecessors) {
  llvm::DenseSet<Operation *> boundarySet;
  for (Operation *op : boundaryOps)
    boundarySet.insert(op);
  for (Operation *op : boundaryOps) {
    successors.try_emplace(op, SmallVector<Operation *>());
    predecessors.try_emplace(op, SmallVector<Operation *>());
  }

  for (Operation *op : boundaryOps) {
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        auto it = partitionMap.find(user);
        if (it == partitionMap.end() || it->second != MixPartitionKind::Boundary ||
            !boundarySet.contains(user))
          continue;
        successors[op].push_back(user);
        predecessors[user].push_back(op);
      }
    }
  }
}

static llvm::DenseSet<Operation *>
collectReachableBoundaryOps(
    Operation *start,
    const llvm::DenseMap<Operation *, SmallVector<Operation *>> &adjacency) {
  llvm::DenseSet<Operation *> reachable;
  if (!start)
    return reachable;

  SmallVector<Operation *> worklist{start};
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!reachable.insert(op).second)
      continue;
    auto it = adjacency.find(op);
    if (it == adjacency.end())
      continue;
    worklist.append(it->second.begin(), it->second.end());
  }

  return reachable;
}

static bool isLinearBoundaryChain(
    Operation *start, Operation *end,
    const llvm::DenseSet<Operation *> &pathOps,
    const llvm::DenseMap<Operation *, SmallVector<Operation *>> &successors,
    const llvm::DenseMap<Operation *, SmallVector<Operation *>> &predecessors) {
  if (!start || !end || pathOps.empty() || !pathOps.contains(start) ||
      !pathOps.contains(end))
    return false;

  for (Operation *op : pathOps) {
    unsigned pathPredecessors = 0;
    unsigned pathSuccessors = 0;

    if (auto predIt = predecessors.find(op); predIt != predecessors.end()) {
      for (Operation *pred : predIt->second)
        pathPredecessors += pathOps.contains(pred);
    }
    if (auto succIt = successors.find(op); succIt != successors.end()) {
      for (Operation *succ : succIt->second)
        pathSuccessors += pathOps.contains(succ);
    }

    if (op == start && op == end) {
      if (pathPredecessors != 0 || pathSuccessors != 0)
        return false;
      continue;
    }
    if (op == start) {
      if (pathPredecessors != 0 || pathSuccessors != 1)
        return false;
      continue;
    }
    if (op == end) {
      if (pathPredecessors != 1 || pathSuccessors != 0)
        return false;
      continue;
    }
    if (pathPredecessors != 1 || pathSuccessors != 1)
      return false;
  }

  return true;
}

static MixSingleChainValidation
validateSingleChainGenericMixPlan(const MixPartitionPlan &plan) {
  MixSingleChainValidation validation;
  auto addFailureReason = [&](MixSingleChainFailureReason reason) {
    if (!hasFailureReason(validation.failureReasons, reason))
      validation.failureReasons.push_back(reason);
  };
  auto countRegionsOfKind = [&](MixPartitionKind kind) {
    return llvm::count_if(plan.regions, [&](const MixRegionPlan &region) {
      return region.kind == kind;
    });
  };
  auto findRegionOfKind = [&](MixPartitionKind kind) -> const MixRegionPlan * {
    for (const MixRegionPlan &region : plan.regions) {
      if (region.kind == kind)
        return &region;
    }
    return nullptr;
  };

  const MixRegionPlan *cubeRegion = findRegionOfKind(MixPartitionKind::Cube);
  const MixRegionPlan *boundaryRegion =
      findRegionOfKind(MixPartitionKind::Boundary);
  const MixRegionPlan *vectorRegion = findRegionOfKind(MixPartitionKind::Vector);
  unsigned cubeRegionCount = countRegionsOfKind(MixPartitionKind::Cube);
  unsigned boundaryRegionCount = countRegionsOfKind(MixPartitionKind::Boundary);
  unsigned vectorRegionCount = countRegionsOfKind(MixPartitionKind::Vector);

  if (!cubeRegion)
    addFailureReason(MixSingleChainFailureReason::MissingCubeRegion);
  else if (cubeRegionCount != 1)
    addFailureReason(MixSingleChainFailureReason::MultipleCubeRegions);
  if (!boundaryRegion)
    addFailureReason(MixSingleChainFailureReason::MissingBoundaryRegion);
  else if (boundaryRegionCount != 1)
    addFailureReason(MixSingleChainFailureReason::MultipleBoundaryRegions);
  if (!vectorRegion)
    addFailureReason(MixSingleChainFailureReason::MissingVectorRegion);
  else if (vectorRegionCount != 1)
    addFailureReason(MixSingleChainFailureReason::MultipleVectorRegions);
  if (!validation.succeeded())
    return validation;

  if (!(cubeRegion->lastOpOrder < boundaryRegion->firstOpOrder &&
        boundaryRegion->lastOpOrder < vectorRegion->firstOpOrder))
    addFailureReason(MixSingleChainFailureReason::InvalidOrdering);

  SmallVector<MixBoundaryValue> cubeToBoundary =
      filterBoundaryValues(boundaryRegion->inputs, MixPartitionKind::Cube,
                           MixPartitionKind::Boundary);
  SmallVector<MixBoundaryValue> boundaryToVector =
      filterBoundaryValues(boundaryRegion->outputs, MixPartitionKind::Boundary,
                           MixPartitionKind::Vector);

  if (cubeToBoundary.empty())
    addFailureReason(MixSingleChainFailureReason::MissingCubeToBoundaryCrossing);
  if (boundaryToVector.empty())
    addFailureReason(
        MixSingleChainFailureReason::MissingBoundaryToVectorCrossing);
  if (!validation.succeeded())
    return validation;

  MixPartitionSummary summary;
  summary.cubeOps = cubeRegion->ops;
  summary.boundaryOps = boundaryRegion->ops;
  summary.vectorOps = vectorRegion->ops;
  llvm::DenseMap<Operation *, MixPartitionKind> partitionMap =
      buildMixPartitionMap(summary);

  llvm::DenseMap<Operation *, SmallVector<Operation *>> boundarySuccessors;
  llvm::DenseMap<Operation *, SmallVector<Operation *>> boundaryPredecessors;
  buildBoundaryRegionGraph(boundaryRegion->ops, partitionMap, boundarySuccessors,
                           boundaryPredecessors);

  bool foundValidSingleChain = false;
  bool sawLinearBoundaryPath = false;
  bool sawFullCubeCoverage = false;
  bool sawFullVectorCoverage = false;
  bool sawChainShapeWithUnsupportedPayload = false;

  for (const MixBoundaryValue &inputCrossing : cubeToBoundary) {
    for (const MixBoundaryValue &outputCrossing : boundaryToVector) {
      llvm::DenseSet<Operation *> forwardBoundaryOps = collectReachableBoundaryOps(
          inputCrossing.consumerOp, boundarySuccessors);
      llvm::DenseSet<Operation *> backwardBoundaryOps = collectReachableBoundaryOps(
          outputCrossing.producerOp, boundaryPredecessors);
      llvm::DenseSet<Operation *> boundaryPathOps;
      for (Operation *op : forwardBoundaryOps) {
        if (backwardBoundaryOps.contains(op))
          boundaryPathOps.insert(op);
      }

      bool hasLinearBoundaryPath =
          boundaryPathOps.size() == boundaryRegion->ops.size() &&
          isLinearBoundaryChain(inputCrossing.consumerOp, outputCrossing.producerOp,
                                boundaryPathOps, boundarySuccessors,
                                boundaryPredecessors);
      sawLinearBoundaryPath |= hasLinearBoundaryPath;
      if (!hasLinearBoundaryPath)
        continue;

      llvm::DenseSet<Operation *> chainCubeOps =
          collectAncestorPartitionOps(inputCrossing.value, partitionMap,
                                      MixPartitionKind::Cube);
      bool hasFullCubeCoverage = chainCubeOps.size() == cubeRegion->ops.size();
      sawFullCubeCoverage |= hasFullCubeCoverage;
      if (!hasFullCubeCoverage)
        continue;

      FailureOr<SupportedMixBoundaryPayload> payload =
          buildSupportedMixBoundaryPayload(inputCrossing, outputCrossing,
                                           boundaryRegion->ops);
      llvm::DenseSet<Operation *> chainVectorOps =
          collectDescendantPartitionOps(outputCrossing.value, partitionMap,
                                        MixPartitionKind::Vector);
      if (succeeded(payload)) {
        for (Value seedValue :
             collectSupportedMixVectorSeedValues(*payload, vectorRegion->ops)) {
          llvm::DenseSet<Operation *> seedOps =
              collectDescendantPartitionOps(seedValue, partitionMap,
                                            MixPartitionKind::Vector);
          chainVectorOps.insert(seedOps.begin(), seedOps.end());
        }
      }
      if (!chainVectorOps.empty()) {
        chainVectorOps = closePartitionOpsOverAncestors(
            std::move(chainVectorOps), vectorRegion->ops, partitionMap,
            MixPartitionKind::Vector);
      }
      bool hasFullVectorCoverage =
          chainVectorOps.size() == vectorRegion->ops.size();
      sawFullVectorCoverage |= hasFullVectorCoverage;
      if (failed(payload))
        {
          if (hasFullVectorCoverage)
            sawChainShapeWithUnsupportedPayload = true;
          continue;
        }
      if (!hasFullVectorCoverage)
        continue;

      foundValidSingleChain = true;
      validation.selectedBoundaryCrossing =
          MixSingleChainValidation::SelectedBoundaryCrossing{inputCrossing,
                                                            outputCrossing};
      break;
    }
    if (foundValidSingleChain)
      break;
  }

  if (foundValidSingleChain)
    return validation;

  if (sawChainShapeWithUnsupportedPayload)
    addFailureReason(
        MixSingleChainFailureReason::MissingSupportedBoundaryPayload);
  if (!sawLinearBoundaryPath)
    addFailureReason(MixSingleChainFailureReason::BoundaryChainNotLinear);
  if (!sawFullCubeCoverage)
    addFailureReason(MixSingleChainFailureReason::ExtraCubeOpsOutsideChain);
  if (!sawFullVectorCoverage)
    addFailureReason(MixSingleChainFailureReason::ExtraVectorOpsOutsideChain);

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
  case MixSingleChainFailureReason::MissingCubeToBoundaryCrossing:
    return "missing cube-to-boundary crossing";
  case MixSingleChainFailureReason::MissingBoundaryToVectorCrossing:
    return "missing boundary-to-vector crossing";
  case MixSingleChainFailureReason::MissingSupportedBoundaryPayload:
    return "boundary chain lacks a supported explicit boundary payload";
  case MixSingleChainFailureReason::BoundaryChainNotLinear:
    return "boundary region does not form one linear chain";
  case MixSingleChainFailureReason::ExtraCubeOpsOutsideChain:
    return "cube region contains ops outside the single executable chain";
  case MixSingleChainFailureReason::ExtraVectorOpsOutsideChain:
    return "vector region contains ops outside the single executable chain";
  case MixSingleChainFailureReason::InvalidOrdering:
    return "invalid region ordering";
  }
  llvm_unreachable("unexpected single-chain failure reason");
}

static StringRef stringifyGenericMixSingleChainEmissionFailureReason(
    GenericMixSingleChainEmissionFailureReason reason) {
  switch (reason) {
  case GenericMixSingleChainEmissionFailureReason::MissingRequiredRegions:
    return "it could not recover cube/boundary/vector regions from the plan";
  case GenericMixSingleChainEmissionFailureReason::MissingSelectedBoundaryCrossing:
    return "it did not retain the selected boundary crossings";
  }
  llvm_unreachable("unexpected generic mix emission failure reason");
}

static StringRef stringifySupportedMixLoweringFailureReason(
    SupportedMixLoweringFailureReason reason) {
  switch (reason) {
  case SupportedMixLoweringFailureReason::UnsupportedLegacySignature:
    return "the legacy ABI/signature is unsupported";
  case SupportedMixLoweringFailureReason::UnsupportedReluStyleEpilogue:
    return "the vector region does not match the supported relu-style epilogue";
  case SupportedMixLoweringFailureReason::MissingSelectedBoundaryCrossing:
    return "the selected boundary crossings were not retained";
  case SupportedMixLoweringFailureReason::UnsupportedBoundaryPayload:
    return "the explicit boundary payload is unsupported";
  }
  llvm_unreachable("unexpected supported mix lowering failure reason");
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
    return failure();
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
  auto epilogueKind =
      inferSupportedMixEpilogueKind(funcOp, summary, config.leakyReluAlpha);
  if (failed(epilogueKind))
    return failure();
  config.epilogueKind = *epilogueKind;
  return config;
}

static FailureOr<SupportedMixBoundaryPayload>
buildSupportedMixBoundaryPayload(const MixBoundaryValue &input,
                                 const MixBoundaryValue &output,
                                 ArrayRef<Operation *> boundaryOps) {
  auto inputEnqueue = dyn_cast_or_null<ascendc::TQueBindEnqueTensorOp>(
      input.producerOp);
  if (!inputEnqueue)
    return failure();
  auto inputQueuePosition = getQueueLikePosition(inputEnqueue.getQueue().getType());
  if (!inputQueuePosition || *inputQueuePosition != ascendc::TPosition::CO1)
    return failure();

  Type inputElementType = getSupportedMixTensorElementType(input.value);
  Type outputElementType = getSupportedMixTensorElementType(output.value);
  if (!inputElementType || !outputElementType || inputElementType != outputElementType)
    return failure();

  for (Operation *op : boundaryOps) {
    auto copyOp = dyn_cast<ascendc::DataCopyCO12DstOp>(op);
    if (!copyOp)
      continue;
    if (copyOp.getSrc() != input.value || copyOp.getDst() != output.value ||
        copyOp.getOperation() != output.producerOp)
      continue;

    auto inputDeque =
        dyn_cast_or_null<ascendc::TQueBindDequeTensorOp>(copyOp.getSrc().getDefiningOp());
    if (!inputDeque || inputDeque.getQueue() != inputEnqueue.getQueue())
      continue;

    Type srcElementType = getSupportedMixTensorElementType(copyOp.getSrc());
    Type dstElementType = getSupportedMixTensorElementType(copyOp.getDst());
    if (!srcElementType || !dstElementType || srcElementType != dstElementType ||
        srcElementType != inputElementType)
      continue;

    Operation *dstDefOp = copyOp.getDst().getDefiningOp();
    auto dstAlloc = dyn_cast_or_null<ascendc::TQueBindAllocTensorOp>(dstDefOp);
    if (!dstAlloc)
      continue;
    auto dstQueuePosition = getQueueLikePosition(dstAlloc.getQueue().getType());
    if (!dstQueuePosition || *dstQueuePosition != ascendc::TPosition::VECIN)
      continue;

    return SupportedMixBoundaryPayload{
        inputElementType, inputEnqueue, copyOp, dstAlloc};
  }

  return failure();
}

static FailureOr<SupportedMixBoundaryLayer>
buildSupportedMixBoundaryLayer(ArrayRef<Operation *> boundaryOps,
                               const MixBoundaryValue &input,
                               const MixBoundaryValue &output) {
  if (boundaryOps.empty())
    return failure();

  FailureOr<SupportedMixBoundaryPayload> payload =
      buildSupportedMixBoundaryPayload(input, output, boundaryOps);
  if (failed(payload))
    return failure();

  return SupportedMixBoundaryLayer{input, output, *payload};
}

static FailureOr<GenericMixSingleChainEmissionPlan>
buildGenericMixSingleChainEmissionPlan(func::FuncOp funcOp,
                                       const MixPartitionPlan &plan,
                                       const MixSingleChainValidation &validation,
                                       GenericMixSingleChainEmissionFailureReason
                                           &failureReason) {
  const MixRegionPlan *cubeRegion =
      findFirstMixRegionOfKind(plan.regions, MixPartitionKind::Cube);
  const MixRegionPlan *boundaryRegion =
      findFirstMixRegionOfKind(plan.regions, MixPartitionKind::Boundary);
  const MixRegionPlan *vectorRegion =
      findFirstMixRegionOfKind(plan.regions, MixPartitionKind::Vector);
  (void)funcOp;
  if (!cubeRegion || !boundaryRegion || !vectorRegion) {
    failureReason =
        GenericMixSingleChainEmissionFailureReason::MissingRequiredRegions;
    return failure();
  }
  if (!validation.selectedBoundaryCrossing) {
    failureReason =
        GenericMixSingleChainEmissionFailureReason::MissingSelectedBoundaryCrossing;
    return failure();
  }

  return GenericMixSingleChainEmissionPlan{
      cubeRegion, boundaryRegion, vectorRegion,
      *validation.selectedBoundaryCrossing};
}

static FailureOr<GenericMixSingleChainSupportedLowering>
lowerGenericMixSingleChainToSupportedMix(
    func::FuncOp funcOp, const MixPartitionSummary &summary,
    const GenericMixSingleChainEmissionPlan &emissionPlan,
    SupportedMixLoweringFailureReason &failureReason) {
  (void)funcOp;
  if (!hasSupportedMixFunctionSignature(funcOp)) {
    failureReason = SupportedMixLoweringFailureReason::UnsupportedLegacySignature;
    return failure();
  }
  FailureOr<SupportedMixKernelConfig> config =
      inferSupportedMixKernelConfig(funcOp, summary);
  if (failed(config)) {
    failureReason =
        SupportedMixLoweringFailureReason::UnsupportedReluStyleEpilogue;
    return failure();
  }

  FailureOr<SupportedMixBoundaryLayer> boundaryLayer =
      buildSupportedMixBoundaryLayer(
          emissionPlan.boundaryRegion->ops,
          emissionPlan.selectedBoundaryCrossing.input,
          emissionPlan.selectedBoundaryCrossing.output);
  if (failed(boundaryLayer)) {
    failureReason = SupportedMixLoweringFailureReason::UnsupportedBoundaryPayload;
    return failure();
  }

  return GenericMixSingleChainSupportedLowering{*boundaryLayer, *config};
}

static FailureOr<GenericMixSingleChainSupportedLowering>
buildLegacySupportedMixLowering(const MixPartitionPlan &plan,
                                func::FuncOp funcOp,
                                const MixPartitionSummary &summary,
                                const MixSingleChainValidation &validation,
                                SupportedMixLoweringFailureReason &failureReason) {
  (void)validation;
  if (!hasSupportedMixFunctionSignature(funcOp)) {
    failureReason = SupportedMixLoweringFailureReason::UnsupportedLegacySignature;
    return failure();
  }
  FailureOr<SupportedMixKernelConfig> config =
      inferSupportedMixKernelConfig(funcOp, summary);
  if (failed(config)) {
    failureReason =
        SupportedMixLoweringFailureReason::UnsupportedReluStyleEpilogue;
    return failure();
  }
  const MixRegionPlan *boundaryRegion =
      findFirstMixRegionOfKind(plan.regions, MixPartitionKind::Boundary);
  if (!boundaryRegion) {
    failureReason = SupportedMixLoweringFailureReason::UnsupportedBoundaryPayload;
    return failure();
  }
  FailureOr<SupportedMixBoundaryLayer> boundaryLayer =
      inferLegacySupportedMixBoundaryLayer(boundaryRegion->ops);
  if (failed(boundaryLayer)) {
    failureReason = SupportedMixLoweringFailureReason::UnsupportedBoundaryPayload;
    return failure();
  }

  return GenericMixSingleChainSupportedLowering{*boundaryLayer, *config};
}

static FailureOr<SupportedMixBoundaryLayer>
inferLegacySupportedMixBoundaryLayer(ArrayRef<Operation *> boundaryOps) {
  for (Operation *op : boundaryOps) {
    auto copyOp = dyn_cast<ascendc::DataCopyCO12DstOp>(op);
    if (!copyOp)
      continue;

    auto inputDeque =
        dyn_cast_or_null<ascendc::TQueBindDequeTensorOp>(copyOp.getSrc().getDefiningOp());
    if (!inputDeque)
      continue;
    auto inputQueuePosition = getQueueLikePosition(inputDeque.getQueue().getType());
    if (!inputQueuePosition || *inputQueuePosition != ascendc::TPosition::CO1)
      continue;

    ascendc::TQueBindEnqueTensorOp inputEnqueue;
    for (Operation *queueUser : inputDeque.getQueue().getUsers()) {
      auto enqueTensor = dyn_cast<ascendc::TQueBindEnqueTensorOp>(queueUser);
      if (enqueTensor && enqueTensor.getQueue() == inputDeque.getQueue()) {
        inputEnqueue = enqueTensor;
        break;
      }
    }
    if (!inputEnqueue)
      continue;

    auto outputAlloc =
        dyn_cast_or_null<ascendc::TQueBindAllocTensorOp>(copyOp.getDst().getDefiningOp());
    if (!outputAlloc)
      continue;
    auto outputQueuePosition = getQueueLikePosition(outputAlloc.getQueue().getType());
    if (!outputQueuePosition || *outputQueuePosition != ascendc::TPosition::VECIN)
      continue;

    Type srcElementType = getSupportedMixTensorElementType(copyOp.getSrc());
    Type dstElementType = getSupportedMixTensorElementType(copyOp.getDst());
    if (!srcElementType || !dstElementType || srcElementType != dstElementType)
      continue;

    SupportedMixBoundaryPayload payload{srcElementType, inputEnqueue, copyOp,
                                        outputAlloc};
    MixBoundaryValue input{copyOp.getSrc(), MixPartitionKind::Cube,
                           MixPartitionKind::Boundary, inputEnqueue, copyOp};
    MixBoundaryValue output{copyOp.getDst(), MixPartitionKind::Boundary,
                            MixPartitionKind::Vector, copyOp, outputAlloc};
    return SupportedMixBoundaryLayer{input, output, payload};
  }

  return failure();
}

// Supported mix emission helpers.
static StringRef getSupportedMixElementTypeSpelling(Type type) {
  if (type.isF32())
    return "float";
  llvm_unreachable("unsupported supported-mix boundary element type");
}

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

static void emitSupportedMixBoundaryTransferSetup(
    raw_ostream &os, const SupportedMixBoundaryLayer &layer,
    const MixTaskKindDescriptor &desc) {
  auto inputEnqueue = layer.payload.inputEnqueue;
  auto transferCopy = layer.payload.transferCopy;
  auto outputAlloc = layer.payload.outputAlloc;
  StringRef elemType = getSupportedMixElementTypeSpelling(layer.payload.elementType);
  auto inputQueuePosition =
      getQueueLikePosition(inputEnqueue.getQueue().getType());
  auto outputQueuePosition =
      getQueueLikePosition(outputAlloc.getQueue().getType());
  if (!inputQueuePosition || *inputQueuePosition != ascendc::TPosition::CO1 ||
      !outputQueuePosition || *outputQueuePosition != ascendc::TPosition::VECIN ||
      transferCopy.getSrc() == Value() || transferCopy.getDst() == Value()) {
    llvm_unreachable("unsupported supported-mix boundary payload");
  }
  os << "    TQue<TPosition::VECIN, 1> reluInQueue;\n"
     << "    TQue<TPosition::VECOUT, 1> reluOutQueue;\n\n";
  emitSupportedMixVectorCountDecl(os, desc);
  os << "    GlobalTensor<" << elemType << "> cGM;\n"
     << "    cGM.SetGlobalBuffer(reinterpret_cast<__gm__ " << elemType
     << " *>(out) + GetBlockIdx() * count, count);\n\n"
     << "    pipe.InitBuffer(reluInQueue, 1, count * sizeof(" << elemType
     << "));\n"
     << "    pipe.InitBuffer(reluOutQueue, 1, count * sizeof(" << elemType
     << "));\n\n";
}

static void emitSupportedMixBoundaryInputTransfer(
    raw_ostream &os, const SupportedMixBoundaryLayer &layer) {
  auto inputEnqueue = layer.payload.inputEnqueue;
  auto transferCopy = layer.payload.transferCopy;
  StringRef elemType = getSupportedMixElementTypeSpelling(layer.payload.elementType);
  auto inputQueuePosition = getQueueLikePosition(inputEnqueue.getQueue().getType());
  if (!inputQueuePosition || *inputQueuePosition != ascendc::TPosition::CO1 ||
      transferCopy.getSrc() == Value() || transferCopy.getDst() == Value()) {
    llvm_unreachable("unsupported supported-mix boundary input payload");
  }
  os << "    LocalTensor<" << elemType
     << "> reluInLocal = reluInQueue.AllocTensor<" << elemType << ">();\n"
     << "    DataCopy(reluInLocal, cGM, count);\n"
     << "    reluInQueue.EnQue<" << elemType << ">(reluInLocal);\n\n"
     << "    LocalTensor<" << elemType
     << "> inLocal = reluInQueue.DeQue<" << elemType << ">();\n"
     << "    LocalTensor<" << elemType
     << "> outLocal = reluOutQueue.AllocTensor<" << elemType << ">();\n";
}

static void emitSupportedMixBoundaryOutputTransfer(
    raw_ostream &os, const SupportedMixBoundaryLayer &layer) {
  auto transferCopy = layer.payload.transferCopy;
  auto outputAlloc = layer.payload.outputAlloc;
  StringRef elemType = getSupportedMixElementTypeSpelling(layer.payload.elementType);
  auto outputQueuePosition =
      getQueueLikePosition(outputAlloc.getQueue().getType());
  if (!outputQueuePosition || *outputQueuePosition != ascendc::TPosition::VECIN ||
      transferCopy.getDst() == Value()) {
    llvm_unreachable("unsupported supported-mix boundary output payload");
  }
  os << "    reluOutQueue.EnQue<" << elemType << ">(outLocal);\n"
     << "    reluInQueue.FreeTensor(inLocal);\n\n"
     << "    LocalTensor<" << elemType
     << "> finalLocal = reluOutQueue.DeQue<" << elemType << ">();\n"
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

template <typename EmitVectorBodyFn>
static void emitMixBoundaryLayer(raw_ostream &os,
                                 const SupportedMixBoundaryLayer &layer,
                                 const MixTaskKindDescriptor &desc,
                                 EmitVectorBodyFn emitVectorBody) {
  emitSupportedMixCrossCoreSetFlag(os, desc);
  os << "  }\n\n"
     << "  if ASCEND_IS_AIV {\n";
  emitSupportedMixBoundaryTransferSetup(os, layer, desc);
  os << "    CrossCoreWaitFlag(" << desc.crossCoreFlagId << ");\n\n";
  emitSupportedMixBoundaryInputTransfer(os, layer);
  emitVectorBody();
  emitSupportedMixBoundaryOutputTransfer(os, layer);
  os << "  }\n";
}

static void emitMixVectorRegion(raw_ostream &os, const MixRegionPlan &region,
                                const SupportedMixKernelConfig &config) {
  (void)region;
  emitSupportedMixVectorEpilogue(os, config);
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
                                   const SupportedMixBoundaryLayer &boundaryLayer,
                                   const SupportedMixKernelConfig &config) {
  MixTaskKindDescriptor desc = getMixTaskKindDescriptor(config.taskKind);
  emitSupportedMixKernelPrologue(os, funcOp.getName(), desc);
  const MixRegionPlan *cubeRegion =
      findFirstMixRegionOfKind(plan.regions, MixPartitionKind::Cube);
  const MixRegionPlan *vectorRegion =
      findFirstMixRegionOfKind(plan.regions, MixPartitionKind::Vector);
  if (!cubeRegion || !vectorRegion)
    llvm_unreachable(
        "supported mix emission requires cube and vector regions");
  emitMixCubeRegion(os, *cubeRegion, config, desc);
  emitMixBoundaryLayer(os, boundaryLayer, desc, [&] {
    emitMixVectorRegion(os, *vectorRegion, config);
  });
  os << "}\n";
}

static void emitGenericMixKernelPrologue(raw_ostream &os, StringRef kernelName,
                                         const MixTaskKindDescriptor &desc) {
  emitSupportedMixIncludesAndNamespaces(os);
  emitSupportedMixCopyTilingHelper(os);
  emitSupportedMixKernelSignature(os, kernelName, desc);
}

static void emitGenericMixSingleChainKernel(
    raw_ostream &os, func::FuncOp funcOp,
    const GenericMixSingleChainEmissionPlan &emissionPlan,
    const GenericMixSingleChainSupportedLowering &supportedLowering) {
  MixTaskKindDescriptor desc =
      getMixTaskKindDescriptor(supportedLowering.config.taskKind);
  emitGenericMixKernelPrologue(os, funcOp.getName(), desc);
  emitMixCubeRegion(os, *emissionPlan.cubeRegion, supportedLowering.config,
                    desc);
  emitMixBoundaryLayer(os, supportedLowering.boundaryLayer, desc, [&] {
    emitMixVectorRegion(os, *emissionPlan.vectorRegion,
                        supportedLowering.config);
  });
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
      GenericMixSingleChainEmissionFailureReason genericEmissionFailureReason =
          GenericMixSingleChainEmissionFailureReason::MissingRequiredRegions;
      FailureOr<GenericMixSingleChainEmissionPlan> genericEmissionPlan =
          buildGenericMixSingleChainEmissionPlan(primaryKernel,
                                                mixPartitionPlan,
                                                singleChainValidation,
                                                genericEmissionFailureReason);
      if (succeeded(genericEmissionPlan)) {
        SupportedMixLoweringFailureReason genericLoweringFailureReason =
            SupportedMixLoweringFailureReason::UnsupportedLegacySignature;
        FailureOr<GenericMixSingleChainSupportedLowering> supportedLowering =
            lowerGenericMixSingleChainToSupportedMix(primaryKernel,
                                                    mixPartitionSummary,
                                                    *genericEmissionPlan,
                                                    genericLoweringFailureReason);
        if (succeeded(supportedLowering)) {
          emitGenericMixSingleChainKernel(os, primaryKernel, *genericEmissionPlan,
                                          *supportedLowering);
          return success();
        }

        SupportedMixLoweringFailureReason legacyFallbackFailureReason =
            SupportedMixLoweringFailureReason::UnsupportedLegacySignature;
        FailureOr<GenericMixSingleChainSupportedLowering> legacyFallbackLowering =
            buildLegacySupportedMixLowering(mixPartitionPlan, primaryKernel,
                                            mixPartitionSummary,
                                            singleChainValidation,
                                            legacyFallbackFailureReason);
        if (succeeded(legacyFallbackLowering)) {
          emitSupportedMixKernel(os, primaryKernel, mixPartitionPlan,
                                 legacyFallbackLowering->boundaryLayer,
                                 legacyFallbackLowering->config);
          return success();
        }

        return primaryKernel.emitOpError(
            Twine("mix translation found a valid single-chain cube/boundary/"
                  "vector plan, but the generic primary route could not lower "
                  "the current supported shell because ") +
            stringifySupportedMixLoweringFailureReason(
                genericLoweringFailureReason) +
            Twine("; the retained supported-mix fallback also failed because ") +
            stringifySupportedMixLoweringFailureReason(
                legacyFallbackFailureReason));
      }

      SupportedMixLoweringFailureReason legacyFallbackFailureReason =
          SupportedMixLoweringFailureReason::UnsupportedLegacySignature;
      FailureOr<GenericMixSingleChainSupportedLowering> legacyFallbackLowering =
          buildLegacySupportedMixLowering(mixPartitionPlan, primaryKernel,
                                          mixPartitionSummary,
                                          singleChainValidation,
                                          legacyFallbackFailureReason);
      if (succeeded(legacyFallbackLowering)) {
        emitSupportedMixKernel(os, primaryKernel, mixPartitionPlan,
                               legacyFallbackLowering->boundaryLayer,
                               legacyFallbackLowering->config);
        return success();
      }

      return primaryKernel.emitOpError(
          Twine("mix translation found a valid single-chain cube/boundary/"
                "vector plan, but the generic primary route could not "
                "materialize its emission plan because ") +
          stringifyGenericMixSingleChainEmissionFailureReason(
              genericEmissionFailureReason) +
          Twine("; the retained supported-mix fallback also failed because ") +
          stringifySupportedMixLoweringFailureReason(
              legacyFallbackFailureReason));
    }

    SupportedMixLoweringFailureReason legacyFallbackFailureReason =
        SupportedMixLoweringFailureReason::UnsupportedLegacySignature;
    if (shouldAttemptLegacyFallbackAfterValidationFailure(singleChainValidation)) {
      FailureOr<GenericMixSingleChainSupportedLowering> legacyFallbackLowering =
          buildLegacySupportedMixLowering(mixPartitionPlan, primaryKernel,
                                          mixPartitionSummary,
                                          singleChainValidation,
                                          legacyFallbackFailureReason);
      if (succeeded(legacyFallbackLowering)) {
        emitSupportedMixKernel(os, primaryKernel, mixPartitionPlan,
                               legacyFallbackLowering->boundaryLayer,
                               legacyFallbackLowering->config);
        return success();
      }
    }

    return primaryKernel.emitOpError(Twine(
        "mix translation requires a supported cube/vector partitioned kernel "
        "shape; generic single-chain analysis rejected plan because ") +
                                     Twine(describeMixSingleChainValidation(
                                         singleChainValidation)) +
                                     Twine("; the retained supported-mix "
                                           "fallback also failed because ") +
                                     stringifySupportedMixLoweringFailureReason(
                                         legacyFallbackFailureReason));
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
