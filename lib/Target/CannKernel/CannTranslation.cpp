//===- CannTranslation.cpp - CANN kernel C++ translation --------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/CannKernel/CannTranslation.h"
#include "Target/CannKernel/SocSpec.h"
#include "Target/CannKernel/UbCostExpr.h"
#include "Conversion/VectorPlan/TilePlan.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Path.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Target/Asc/CodeEmitter.h"
#include "ascir/Target/Asc/Common.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

#include <algorithm>
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

static std::string getAscendCScalarTypeName(Type elemType) {
  if (elemType.isF16())
    return "half";
  if (elemType.isF32())
    return "float";
  if (elemType.isF64())
    return "double";
  if (auto iType = dyn_cast<IntegerType>(elemType)) {
    bool isUnsigned = iType.isUnsigned();
    return (isUnsigned ? "uint" : "int") + std::to_string(iType.getWidth()) +
           "_t";
  }
  return "half";
}

static Value peelSourceValue(Value value) {
  if (!value)
    return value;
  if (auto castOp = value.getDefiningOp<emitasc::ReinterpretCastOp>())
    return castOp.getOperand();
  return value;
}

static Value peelIndexCast(Value value) {
  if (!value)
    return value;
  if (auto castOp = value.getDefiningOp<arith::IndexCastOp>())
    return castOp.getOperand();
  return value;
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
  SmallVector<Operation *> cubeOps;
};

struct GenericMixSingleChainSupportedLowering {
  SupportedMixBoundaryLayer boundaryLayer;
  SupportedMixKernelConfig config;
  SmallVector<Operation *> cubeOps;
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
  UnsupportedCubeOp,
  UnsupportedVectorOp,
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

static Operation *findUnsupportedMixVectorRegionOp(
    ArrayRef<Operation *> ops);

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
  if (!numInputsAttr)
    return false;
  int64_t numInputs = numInputsAttr.getInt();
  // Supported layouts:
  //   - matmul+bias+epilogue (num_inputs=4, 7 args).  Two arg orderings:
  //       legacy: A(f16,2D), B(f16,2D), bias(f32,1D), init(f32,2D), out, ws, tiling
  //       cube:   A(f16,2D), B(f16,2D), init(f32,2D), bias(f32,1D), out, ws, tiling
  //     (the vector-plan cube path produces the latter because the init
  //     operand of `linalg.matmul` precedes the bias operand of the trailing
  //     `linalg.generic` in func-arg order after bufferization)
  //   - matmul+epilogue, no-bias (num_inputs=3, 6 args):
  //       A(f16,2D), B(f16,2D), init(f32,2D), out(f32,2D), ws(ui8), tiling
  if (numInputs != 4 && numInputs != 3)
    return false;

  auto args = funcOp.getArguments();
  bool hasBias = (numInputs == 4);
  size_t expectedArgs = hasBias ? 7 : 6;
  if (args.size() != expectedArgs)
    return false;

  MLIRContext *ctx = funcOp.getContext();
  Type f16 = Float16Type::get(ctx);
  Type f32 = Float32Type::get(ctx);

  if (!isRankedMemrefOf(args[0].getType(), 2, f16) ||
      !isRankedMemrefOf(args[1].getType(), 2, f16))
    return false;

  size_t idx = 2;
  if (hasBias) {
    // Accept either order: (bias 1D, init 2D) or (init 2D, bias 1D).
    bool order1 = isRankedMemrefOf(args[idx].getType(), 1, f32) &&
                  isRankedMemrefOf(args[idx + 1].getType(), 2, f32);
    bool order2 = isRankedMemrefOf(args[idx].getType(), 2, f32) &&
                  isRankedMemrefOf(args[idx + 1].getType(), 1, f32);
    if (!order1 && !order2)
      return false;
    idx += 2;
  } else {
    // No-bias variant: just init (f32, 2D) at args[2].
    if (!isRankedMemrefOf(args[idx].getType(), 2, f32))
      return false;
    ++idx;
  }

  auto outputType = dyn_cast<MemRefType>(args[idx].getType());
  if (!outputType || outputType.getRank() != 2 ||
      outputType.getElementType() != f32)
    return false;
  ++idx;

  auto workspaceType = dyn_cast<MemRefType>(args[idx].getType());
  if (!workspaceType || !workspaceType.getElementType().isUnsignedInteger(8))
    return false;
  ++idx;

  return isa<emitasc::PyStructType>(args[idx].getType());
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

static SmallVector<Operation *>
collectOrderedAncestorPartitionOps(Value seed,
                                   const llvm::DenseMap<Operation *, MixPartitionKind>
                                       &partitionMap,
                                   ArrayRef<Operation *> regionOps,
                                   MixPartitionKind targetKind) {
  llvm::DenseSet<Operation *> selected =
      collectAncestorPartitionOps(seed, partitionMap, targetKind);
  SmallVector<Operation *> ordered;
  for (Operation *op : regionOps) {
    if (selected.contains(op))
      ordered.push_back(op);
  }
  return ordered;
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

  llvm::DenseMap<Value, bool> originCache;
  llvm::SmallPtrSet<Operation *, 16> originVisiting;
  for (Operation *op : vectorOps) {
    for (Value operand : op->getOperands()) {
      if (getTensorStoragePartition(operand) != payloadSourcePartition) {
        originVisiting.clear();
        if (!valueOriginatesFromPartition(operand, payloadSourcePartition,
                                          originCache, originVisiting))
          continue;
      }
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
      if (chainCubeOps.empty())
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
  case SupportedMixLoweringFailureReason::UnsupportedCubeOp:
    return "unsupported cube op in single-chain mix emitter";
  case SupportedMixLoweringFailureReason::UnsupportedVectorOp:
    return "unsupported vector op in single-chain mix emitter";
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

static bool isSupportedMixVectorMaxUser(Operation *user) {
  auto maxOp = dyn_cast<ascendc::MaxL2Op>(user);
  return maxOp &&
         getTensorStoragePartition(maxOp.getDst()) == MixPartitionKind::Vector;
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
    auto constOp = dupOp.getScalar().getDefiningOp<arith::ConstantOp>();
    if (!constOp)
      return WalkResult::advance();
    auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue());
    if (!floatAttr)
      return WalkResult::advance();
    double scalar = floatAttr.getValue().convertToDouble();
    bool usedByVectorMul =
        llvm::any_of(dupOp.getDst().getUsers(), isSupportedMixVectorMulUser);
    bool usedByVectorMax =
        llvm::any_of(dupOp.getDst().getUsers(), isSupportedMixVectorMaxUser);
    if (usedByVectorMul) {
      // dup(alpha) -> mul lane: LeakyRelu when alpha != 0, Relu when alpha == 0.
      leakyReluAlpha = scalar;
      epilogueKind =
          (scalar == 0.0)
              ? SupportedMixKernelConfig::EpilogueKind::Relu
              : SupportedMixKernelConfig::EpilogueKind::LeakyRelu;
      return WalkResult::interrupt();
    }
    if (usedByVectorMax && scalar == 0.0) {
      // dup(0.0) -> max lane only: pure Relu (no Mul means alpha == 0).
      leakyReluAlpha = 0.0;
      epilogueKind = SupportedMixKernelConfig::EpilogueKind::Relu;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
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
buildGenericMixSingleChainEmissionPlan(
    func::FuncOp funcOp, const MixPartitionPlan &plan,
    const MixPartitionSummary &summary,
    const MixSingleChainValidation &validation,
    GenericMixSingleChainEmissionFailureReason &failureReason) {
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

  llvm::DenseMap<Operation *, MixPartitionKind> partitionMap =
      buildMixPartitionMap(summary);
  SmallVector<Operation *> cubeOps = collectOrderedAncestorPartitionOps(
      validation.selectedBoundaryCrossing->input.value, partitionMap,
      cubeRegion->ops, MixPartitionKind::Cube);
  if (cubeOps.empty()) {
    failureReason =
        GenericMixSingleChainEmissionFailureReason::MissingRequiredRegions;
    return failure();
  }

  return GenericMixSingleChainEmissionPlan{
      cubeRegion, boundaryRegion, vectorRegion, *validation.selectedBoundaryCrossing,
      std::move(cubeOps)};
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

  return GenericMixSingleChainSupportedLowering{*boundaryLayer, *config,
                                               emissionPlan.cubeOps};
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
  const MixRegionPlan *cubeRegion =
      findFirstMixRegionOfKind(plan.regions, MixPartitionKind::Cube);
  if (!boundaryRegion) {
    failureReason = SupportedMixLoweringFailureReason::UnsupportedBoundaryPayload;
    return failure();
  }
  if (!cubeRegion) {
    failureReason = SupportedMixLoweringFailureReason::UnsupportedCubeOp;
    return failure();
  }
  llvm::DenseMap<Operation *, MixPartitionKind> partitionMap =
      buildMixPartitionMap(summary);
  FailureOr<SupportedMixBoundaryLayer> boundaryLayer =
      inferLegacySupportedMixBoundaryLayer(boundaryRegion->ops);
  if (failed(boundaryLayer)) {
    failureReason = SupportedMixLoweringFailureReason::UnsupportedBoundaryPayload;
    return failure();
  }
  SmallVector<Operation *> cubeOps = collectOrderedAncestorPartitionOps(
      boundaryLayer->input.value, partitionMap, cubeRegion->ops,
      MixPartitionKind::Cube);
  if (cubeOps.empty()) {
    failureReason = SupportedMixLoweringFailureReason::UnsupportedCubeOp;
    return failure();
  }

  return GenericMixSingleChainSupportedLowering{*boundaryLayer, *config,
                                               std::move(cubeOps)};
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

static Operation *getSelectedMixBoundaryTransferCopyOp(
    const SupportedMixBoundaryLayer &layer) {
  ascendc::DataCopyCO12DstOp transferCopy = layer.payload.transferCopy;
  return transferCopy.getOperation();
}

static const MixRegionPlan *
findFirstMixRegionOfKind(ArrayRef<MixRegionPlan> regions, MixPartitionKind kind) {
  for (const MixRegionPlan &region : regions) {
    if (region.kind == kind)
      return &region;
  }
  return nullptr;
}

static bool isSupportedMixVectorRegionOp(Operation *op) {
  return isa<ascendc::BroadcastL2Op, ascendc::AddL2Op, ascendc::DuplicateL2Op,
             ascendc::MulL2Op, ascendc::MaxL2Op, ascendc::DataCopyL2Op>(op);
}

static bool emitMixCubeRegionOpDispatch(raw_ostream &os, Operation *op,
                                        const SupportedMixKernelConfig &config) {
  if (!isa<ascendc::MmadOp>(op))
    return false;

  emitSupportedMixMatmulObjectDecl(os);
  emitSupportedMixAicGlobalTensorSetup(os, config);
  emitSupportedMixMatmulExecution(os, config);
  return true;
}

static bool emitMixCubeRegionOps(raw_ostream &os, ArrayRef<Operation *> cubeOps,
                                 const SupportedMixKernelConfig &config,
                                 SupportedMixLoweringFailureReason &failureReason) {
  Operation *supportedCubeOp = nullptr;
  for (Operation *op : cubeOps) {
    if (!isa<ascendc::MmadOp>(op)) {
      failureReason = SupportedMixLoweringFailureReason::UnsupportedCubeOp;
      return false;
    }
    if (supportedCubeOp) {
      failureReason = SupportedMixLoweringFailureReason::UnsupportedCubeOp;
      return false;
    }
    supportedCubeOp = op;
  }
  if (!supportedCubeOp) {
    failureReason = SupportedMixLoweringFailureReason::UnsupportedCubeOp;
    return false;
  }
  if (!emitMixCubeRegionOpDispatch(os, supportedCubeOp, config)) {
    failureReason = SupportedMixLoweringFailureReason::UnsupportedCubeOp;
    return false;
  }
  return true;
}

static bool emitMixBoundaryRegionSetupOpDispatch(
    raw_ostream &os, Operation *op, const SupportedMixBoundaryLayer &layer,
    const MixTaskKindDescriptor &desc) {
  if (op != getSelectedMixBoundaryTransferCopyOp(layer) ||
      !isa<ascendc::DataCopyCO12DstOp>(op))
    return false;
  emitSupportedMixBoundaryTransferSetup(os, layer, desc);
  return true;
}

static void emitMixBoundaryRegionSetupOps(
    raw_ostream &os, const MixRegionPlan &region,
    const SupportedMixBoundaryLayer &layer, const MixTaskKindDescriptor &desc) {
  if (region.kind != MixPartitionKind::Boundary)
    llvm_unreachable("boundary region setup received a non-boundary region");
  bool emitted = false;
  for (Operation *op : region.ops)
    emitted |= emitMixBoundaryRegionSetupOpDispatch(os, op, layer, desc);
  if (!emitted)
    emitSupportedMixBoundaryTransferSetup(os, layer, desc);
}

static bool emitMixBoundaryRegionInputOpDispatch(
    raw_ostream &os, Operation *op, const SupportedMixBoundaryLayer &layer) {
  if (op != getSelectedMixBoundaryTransferCopyOp(layer) ||
      !isa<ascendc::DataCopyCO12DstOp>(op))
    return false;
  emitSupportedMixBoundaryInputTransfer(os, layer);
  return true;
}

static void emitMixBoundaryRegionInputOps(raw_ostream &os,
                                          const MixRegionPlan &region,
                                          const SupportedMixBoundaryLayer &layer) {
  if (region.kind != MixPartitionKind::Boundary)
    llvm_unreachable(
        "boundary region input emission received a non-boundary region");
  bool emitted = false;
  for (Operation *op : region.ops)
    emitted |= emitMixBoundaryRegionInputOpDispatch(os, op, layer);
  if (!emitted)
    emitSupportedMixBoundaryInputTransfer(os, layer);
}

static bool emitMixBoundaryRegionOutputOpDispatch(
    raw_ostream &os, Operation *op, const SupportedMixBoundaryLayer &layer) {
  if (op != getSelectedMixBoundaryTransferCopyOp(layer) ||
      !isa<ascendc::DataCopyCO12DstOp>(op))
    return false;
  emitSupportedMixBoundaryOutputTransfer(os, layer);
  return true;
}

static void emitMixBoundaryRegionOutputOps(
    raw_ostream &os, const MixRegionPlan &region,
    const SupportedMixBoundaryLayer &layer) {
  if (region.kind != MixPartitionKind::Boundary)
    llvm_unreachable(
        "boundary region output emission received a non-boundary region");
  bool emitted = false;
  for (Operation *op : region.ops)
    emitted |= emitMixBoundaryRegionOutputOpDispatch(os, op, layer);
  if (!emitted)
    emitSupportedMixBoundaryOutputTransfer(os, layer);
}

static bool emitMixVectorRegionOps(raw_ostream &os, const MixRegionPlan &region,
                                   const SupportedMixKernelConfig &config,
                                   SupportedMixLoweringFailureReason &failureReason) {
  if (region.kind != MixPartitionKind::Vector)
    llvm_unreachable("vector region emission received a non-vector region");
  if (Operation *unsupportedOp = findUnsupportedMixVectorRegionOp(region.ops)) {
    (void)unsupportedOp;
    failureReason = SupportedMixLoweringFailureReason::UnsupportedVectorOp;
    return false;
  }

  unsigned supportedVectorOpCount =
      llvm::count_if(region.ops, isSupportedMixVectorRegionOp);
  os << "    const uint32_t vectorRegionOpCount = "
     << supportedVectorOpCount << ";\n"
     << "    (void)vectorRegionOpCount;\n";
  emitSupportedMixVectorEpilogue(os, config);
  return true;
}

static Operation *findUnsupportedMixVectorRegionOp(ArrayRef<Operation *> ops) {
  for (Operation *op : ops) {
    if (!isSupportedMixVectorRegionOp(op))
      return op;
  }
  return nullptr;
}

static bool emitMixKernelShellBody(
    raw_ostream &os, ArrayRef<Operation *> cubeOps,
    const MixRegionPlan &boundaryRegion, const MixRegionPlan &vectorRegion,
    const SupportedMixBoundaryLayer &boundaryLayer,
    const SupportedMixKernelConfig &config,
    const MixTaskKindDescriptor &desc,
    SupportedMixLoweringFailureReason &failureReason) {
  os << "  if ASCEND_IS_AIC {\n";
  if (!emitMixCubeRegionOps(os, cubeOps, config, failureReason))
    return false;
  emitSupportedMixCrossCoreSetFlag(os, desc);
  os << "  }\n\n"
     << "  if ASCEND_IS_AIV {\n";
  emitMixBoundaryRegionSetupOps(os, boundaryRegion, boundaryLayer, desc);
  os << "    CrossCoreWaitFlag(" << desc.crossCoreFlagId << ");\n\n";
  emitMixBoundaryRegionInputOps(os, boundaryRegion, boundaryLayer);
  if (!emitMixVectorRegionOps(os, vectorRegion, config, failureReason))
    return false;
  emitMixBoundaryRegionOutputOps(os, boundaryRegion, boundaryLayer);
  os << "  }\n";
  return true;
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
                                            const MixTaskKindDescriptor &desc,
                                            bool hasBias) {
  os << "extern \"C\" __global__ __aicore__ void " << kernelName << "(\n"
     << "    GM_ADDR a, GM_ADDR b, ";
  if (hasBias)
    os << "GM_ADDR bias, ";
  os << "GM_ADDR out, GM_ADDR workspace,\n"
     << "    GM_ADDR tilingGm) {\n"
     << "  KERNEL_TASK_TYPE_DEFAULT(" << desc.taskTypeSpelling << ");\n"
     << "  TPipe pipe;\n"
     << "  (void)workspace;\n\n"
     << "  TCubeTiling tiling;\n"
     << "  CopyTiling(&tiling, tilingGm);\n\n";
}

static void emitSupportedMixKernelShellPrologue(
    raw_ostream &os, StringRef kernelName, const MixTaskKindDescriptor &desc,
    bool hasBias) {
  emitSupportedMixIncludesAndNamespaces(os);
  emitSupportedMixCopyTilingHelper(os);
  emitSupportedMixKernelSignature(os, kernelName, desc, hasBias);
}

static void emitSupportedMixKernelShellEpilogue(raw_ostream &os) {
  os << "}\n";
}

static bool emitSupportedMixKernel(raw_ostream &os, func::FuncOp funcOp,
                                   const MixPartitionPlan &plan,
                                   const GenericMixSingleChainSupportedLowering &supportedLowering,
                                   SupportedMixLoweringFailureReason &failureReason) {
  const SupportedMixBoundaryLayer &boundaryLayer = supportedLowering.boundaryLayer;
  const SupportedMixKernelConfig &config = supportedLowering.config;
  MixTaskKindDescriptor desc = getMixTaskKindDescriptor(config.taskKind);
  emitSupportedMixKernelShellPrologue(os, funcOp.getName(), desc,
                                      config.hasBiasAdd);
  const MixRegionPlan *cubeRegion =
      findFirstMixRegionOfKind(plan.regions, MixPartitionKind::Cube);
  const MixRegionPlan *boundaryRegion =
      findFirstMixRegionOfKind(plan.regions, MixPartitionKind::Boundary);
  const MixRegionPlan *vectorRegion =
      findFirstMixRegionOfKind(plan.regions, MixPartitionKind::Vector);
  if (!cubeRegion || !boundaryRegion || !vectorRegion)
    llvm_unreachable("supported mix emission requires cube, boundary, and "
                     "vector regions");
  if (!emitMixKernelShellBody(os, supportedLowering.cubeOps, *boundaryRegion,
                              *vectorRegion, boundaryLayer, config, desc,
                              failureReason))
    return false;
  emitSupportedMixKernelShellEpilogue(os);
  return true;
}

static bool emitGenericMixSingleChainKernel(
    raw_ostream &os, func::FuncOp funcOp,
    const GenericMixSingleChainEmissionPlan &emissionPlan,
    const GenericMixSingleChainSupportedLowering &supportedLowering,
    SupportedMixLoweringFailureReason &failureReason) {
  MixTaskKindDescriptor desc =
      getMixTaskKindDescriptor(supportedLowering.config.taskKind);
  emitSupportedMixKernelShellPrologue(os, funcOp.getName(), desc,
                                      supportedLowering.config.hasBiasAdd);
  if (!emitMixKernelShellBody(os, supportedLowering.cubeOps,
                              *emissionPlan.boundaryRegion,
                              *emissionPlan.vectorRegion,
                              supportedLowering.boundaryLayer,
                              supportedLowering.config, desc, failureReason))
    return false;
  emitSupportedMixKernelShellEpilogue(os);
  return true;
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
/// block_dim_expr → the func's afir.block_dim_expr attr (set by TilePlanGen) if
/// present, else "".
static LogicalResult emitTilingSpaceJson(StringRef outPath,
                                         StringRef kernelFile,
                                         func::FuncOp funcOp,
                                         emitasc::PyStructType tilingType,
                                         StringRef socName) {
  StringRef kernelName = funcOp.getName();
  std::string blockDimExpr;
  if (auto a = funcOp->getAttrOfType<StringAttr>("afir.block_dim_expr"))
    blockDimExpr = a.getValue().str();
  std::string axisExtentExpr;
  if (auto a = funcOp->getAttrOfType<StringAttr>("afir.axis_extent_expr"))
    axisExtentExpr = a.getValue().str();
  auto isDimField = [](StringRef name) {
    return name.starts_with("dim_arg");
  };
  auto schema = mlir::vector_plan::lookupTilingInfoSchema(
      funcOp->getParentOfType<ModuleOp>(), funcOp.getName());

  // v2 path: translate `dim_arg<N>_<D>` via the schema's args[] table:
  //   - Input-arg-sourced field → "arg<call_arg_index>_dim<source_dim>".
  //   - Output-arg-sourced field → the precomputed shape_expr[source_dim].
  //
  // Failure paths return mlir::failure() and emit a hard diagnostic — empty
  // shape_keys must never be written into tiling_space.json, where the
  // downstream parsers (autotuner / network_runner) silently mis-resolve them.
  auto makeShapeKey =
      [&](StringRef fieldName) -> mlir::FailureOr<std::string> {
    if (!schema)
      return funcOp.emitError()
             << "CannTranslation: missing schema_version=2 for kernel '"
             << funcOp.getName() << "' — cannot derive shape_key for field '"
             << fieldName << "'";
    for (auto &f : schema->fields) {
      if (f.kind != mlir::vector_plan::SchemaFieldKind::ShapeDerived) continue;
      if (f.name != fieldName) continue;
      for (auto &a : schema->args) {
        if (a.mlirIndex != f.sourceArg) continue;
        if (a.role == mlir::vector_plan::SchemaArgRole::Input)
          return std::string("arg" + std::to_string(a.callArgIndex) +
                             "_dim" + std::to_string(f.sourceDim));
        if (a.role == mlir::vector_plan::SchemaArgRole::Output &&
            (size_t)f.sourceDim < a.shapeExpr.size())
          return a.shapeExpr[f.sourceDim];
        return funcOp.emitError()
               << "CannTranslation: schema arg mlirIndex=" << f.sourceArg
               << " for field '" << fieldName << "' in kernel '"
               << funcOp.getName()
               << "' has unsupported role/shapeExpr (source_dim="
               << f.sourceDim << ")";
      }
      return funcOp.emitError()
             << "CannTranslation: field '" << fieldName << "' in kernel '"
             << funcOp.getName() << "' references source_arg mlirIndex="
             << f.sourceArg << " but no such arg exists in schema";
    }
    return funcOp.emitError()
           << "CannTranslation: shape-derived field '" << fieldName
           << "' not found in schema for kernel '" << funcOp.getName() << "'";
  };

  // From vector_plan.tiling_infos (set by TilePlanGen): tunable field -> the
  // static extent of the axis it tiles (-1 if dynamic) and its default value.
  llvm::DenseMap<StringRef, std::pair<int64_t, int64_t>> tunableInfo; // name -> {axisSize, default}
  if (auto moduleOp = funcOp->getParentOfType<ModuleOp>()) {
    if (auto infos = moduleOp->getAttrOfType<ArrayAttr>("vector_plan.tiling_infos")) {
      for (Attribute ia : infos) {
        auto entry = dyn_cast<DictionaryAttr>(ia);
        if (!entry) continue;
        auto kid = dyn_cast_or_null<StringAttr>(entry.get("kernel_id"));
        if (!kid || kid.getValue() != kernelName) continue;
        if (auto fs = dyn_cast_or_null<ArrayAttr>(entry.get("fields"))) {
          for (Attribute fa : fs) {
            auto fd = dyn_cast<DictionaryAttr>(fa);
            if (!fd) continue;
            auto fn = dyn_cast_or_null<StringAttr>(fd.get("name"));
            auto as = dyn_cast_or_null<IntegerAttr>(fd.get("axis_size"));
            auto dv = dyn_cast_or_null<IntegerAttr>(fd.get("default_value"));
            if (fn)
              tunableInfo[fn.getValue()] = {as ? as.getInt() : -1,
                                            dv ? dv.getInt() : 0};
          }
        }
        break;
      }
    }
  }
  // Power-of-2 sweep for a tile-size param, capped at the axis size (when
  // known); always non-empty.  The cartesian product (and ordering constraints
  // like XBLOCK_SUB <= XBLOCK) are pruned by the autotuner.
  auto genTunableValues = [](int64_t axisSize, int64_t defaultVal) {
    llvm::SmallVector<int64_t, 8> cand{16, 32, 64, 128, 256};
    if (defaultVal > 0) cand.push_back(defaultVal);
    llvm::SmallVector<int64_t, 8> out;
    for (int64_t v : cand)
      if (v >= 1 && (axisSize <= 0 || v <= axisSize) && !llvm::is_contained(out, v))
        out.push_back(v);
    if (out.empty())
      out.push_back(axisSize > 0 ? axisSize : (defaultVal > 0 ? defaultVal : 16));
    llvm::sort(out);
    return out;
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
      auto sk = makeShapeKey(name);
      if (failed(sk)) return failure();
      p["shape_key"] = *sk;
    } else {
      p["fixed"] = false;
      auto it = tunableInfo.find(name);
      int64_t axisSize = it != tunableInfo.end() ? it->second.first : -1;
      int64_t defVal = it != tunableInfo.end() ? it->second.second : 0;
      llvm::json::Array vals;
      for (int64_t v : genTunableValues(axisSize, defVal))
        vals.push_back(v);
      p["values"] = std::move(vals);
    }
    params.push_back(std::move(p));
  }

  llvm::json::Object root;
  root["kernel"]         = kernelName.str();
  root["kernel_file"]    = kernelFile.str();
  StringRef socStr = socName.empty() ? StringRef("Ascend910B1") : socName;
  root["soc"]            = socStr.str();
  root["block_dim_expr"] = blockDimExpr;
  root["axis_extent_expr"] = axisExtentExpr;

  // v2 schema sidecar: emit the full per-MLIR-arg table for the runner.
  // The runner uses this to resolve shape_keys against arbitrary kernel-arg
  // topologies (kernels that consume non-leading network inputs, or whose
  // output stride depends on output-arg dims).  network.json is written at
  // GroupOutline time, before TilePlanGen has emitted the schema, so the
  // canonical sidecar lives here (post-codegen).  Shape matches the block
  // in NetworkJsonEmitter.cpp:206-242.
  if (schema) {
    llvm::json::Array schArgs;
    for (auto &a : schema->args) {
      llvm::json::Object e;
      e["mlir_index"] = static_cast<int64_t>(a.mlirIndex);
      switch (a.role) {
        case mlir::vector_plan::SchemaArgRole::Input:
          e["role"] = "input";
          e["call_arg_index"] = static_cast<int64_t>(a.callArgIndex);
          break;
        case mlir::vector_plan::SchemaArgRole::Output: {
          e["role"] = "output";
          e["result_index"] = static_cast<int64_t>(a.resultIndex);
          llvm::json::Array se;
          for (auto &s : a.shapeExpr) se.push_back(s);
          e["shape_expr"] = std::move(se);
          break;
        }
        case mlir::vector_plan::SchemaArgRole::TileParam:
          e["role"] = "tile_param";
          e["name"] = a.tileParamName;
          break;
        case mlir::vector_plan::SchemaArgRole::Workspace:
          e["role"] = "workspace";
          break;
        case mlir::vector_plan::SchemaArgRole::TilingDataStruct:
          e["role"] = "tiling_data_struct";
          break;
      }
      schArgs.push_back(std::move(e));
    }
    root["schema_args"] = std::move(schArgs);

    // shape_equalities: groups of (call_arg_index, dim) pairs whose input
    // shape dims must all resolve to the same integer at runtime.  Runner
    // validates before kernel launch.
    if (!schema->shapeEqualities.empty()) {
      llvm::json::Array eqs;
      for (const auto &g : schema->shapeEqualities) {
        llvm::json::Array group;
        for (auto [callIdx, dim] : g) {
          llvm::json::Array pair;
          pair.push_back(static_cast<int64_t>(callIdx));
          pair.push_back(static_cast<int64_t>(dim));
          group.push_back(std::move(pair));
        }
        eqs.push_back(std::move(group));
      }
      root["shape_equalities"] = std::move(eqs);
    }
  }

  // UB-aware tiling cost: stamp the SoC's TBuf/TQue pool size, plus a
  // symbolic byte cost that the picker / autotuner can compare against it
  // to prune over-budget candidates.  See plan
  // docs/superpowers/plans/2026-05-14-ub-aware-tiling-cost.zh.md.
  //
  // Cost model: cost = 2 * max(align32(init_buffer.size))
  //   - MAX (not SUM) over all init_buffer/init_queue ops in the function:
  //     the simulator's TPipe bump-pointer allocator only fails at the
  //     boundary where a *single* allocation runs off the pool end.  Sum
  //     of all InitBuffer sizes routinely exceeds the pool in working
  //     kernels (dead branches, dual-staged buffers) without observable
  //     failure, so SUM is empirically not the right metric.
  //   - 2x factor: peak live UB at any one instant typically holds an
  //     input-side TBuf and an output-side TBuf concurrently, so the
  //     effective cap on a single buffer is ~pool/2.  Validated against
  //     examples/dyn-bucketed-e2e d2 sweep:
  //       v32  =  32 KB (R=64,XBLOCK_SUB=128):  PASS — 2*32 = 64KB ≤ 184KB
  //       v32  =  96 KB (R=192,XBLOCK_SUB=128): PASS — 2*96 = 192KB > 184KB,
  //                                             picker drops to XBLOCK_SUB=64
  //       v32  = 128 KB (R=256,XBLOCK_SUB=128): FAIL — 2*128 = 256KB > 184KB,
  //                                             picker drops to XBLOCK_SUB=64
  //       v32  = 256 KB (R=512,XBLOCK_SUB=128): FAIL — picker drops to XBLOCK_SUB=16
  if (auto spec = afir::cannkernel::getSocSpec(socStr))
    root["ub_budget_bytes"] = (int64_t)spec->totalVecLocalSize;
  {
    // Emit a LIST of per-buffer aligned-size expressions (each pure +-*/);
    // consumers take max() in their host language and compare against
    // ub_budget_bytes / 2 (the 2x safety factor for input+output liveness
    // is applied at the consumer side, not baked into the expression, so
    // we don't have to nest a Max-tree in SymExpr — that nests `?:` at
    // each level and blows up emit-string size to 2^N with N buffers).
    // Dedup identical expressions so the list stays small.
    afir::cannkernel::NameSymTable names;
    std::vector<std::string> exprs;
    llvm::StringSet<> seen;
    bool ok = true;
    funcOp.walk([&](Operation *op) {
      Value sizeOperand;
      if (auto ib = dyn_cast<ascendc::TPipeInitBufferOp>(op))
        sizeOperand = ib.getLength();
      else if (auto iq = dyn_cast<ascendc::TPipeInitQueueOp>(op))
        sizeOperand = iq.getLength();
      else
        return;
      auto e = afir::cannkernel::liftSizeOperand(sizeOperand, names);
      if (!e) { ok = false; return; }
      auto aligned = afir::cannkernel::align32(*e);
      auto nameFor = [&](afir::symshape::SymId id) -> std::string {
        auto it = names.idToName.find(id);
        return it != names.idToName.end() ? it->second : "?";
      };
      std::string s = aligned.emitC(nameFor);
      if (seen.insert(s).second)
        exprs.push_back(std::move(s));
    });
    if (ok && !exprs.empty()) {
      llvm::json::Array arr;
      for (auto &s : exprs)
        arr.push_back(s);
      root["ub_cost_bytes_exprs"] = std::move(arr);
    }
  }

  root["tiling_params"]  = std::move(params);

  std::error_code ec;
  llvm::raw_fd_ostream f(outPath, ec);
  if (ec) {
    llvm::errs() << "Warning: cannot write tiling_space.json to "
                 << outPath << ": " << ec.message() << "\n";
    return failure();
  }
  llvm::json::OStream jos(f, /*IndentSize=*/2);
  jos.value(llvm::json::Value(std::move(root)));
  f << "\n";
  return success();
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
  // When the offset is absent (op.getSize() is null), emit the 1-arg form
  // with an explicit GM pointer cast because GM_ADDR is emitted as uint8_t*.
  moduleOp->walk([&](ascendc::GlobalTensorSetGlobalBufferOp op) {
    auto tensorType = dyn_cast<ascendc::GlobalTensorType>(op.getTensor().getType());
    if (!tensorType)
      return;

    std::string elemTypeStr =
        getAscendCScalarTypeName(tensorType.getElementType());
    Value baseBuffer = peelSourceValue(op.getBuffer());
    Value sizeVal = op.getSize();
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();

    // If baseBuffer comes from a memref.cast(memref.subview(base, [offset])),
    // bake the subview offset into the pointer so the emitter sees only plain
    // function-arg values that it already knows how to print.
    Value subviewOffset;
    if (auto castOp = baseBuffer.getDefiningOp<memref::CastOp>()) {
      if (auto subviewOp =
              castOp.getOperand().getDefiningOp<memref::SubViewOp>()) {
        SmallVector<OpFoldResult> mixedOffsets = subviewOp.getMixedOffsets();
        if (!mixedOffsets.empty()) {
          if (auto dynOff = dyn_cast<Value>(mixedOffsets[0])) {
            baseBuffer = subviewOp.getSource();
            subviewOffset = dynOff;
          } else if (auto attrOff = dyn_cast<Attribute>(mixedOffsets[0])) {
            int64_t constOff = cast<IntegerAttr>(attrOff).getInt();
            baseBuffer = subviewOp.getSource();
            if (constOff != 0)
              subviewOffset = rewriter.create<arith::ConstantIndexOp>(
                  loc, constOff);
          }
        }
      } else if (auto ba = dyn_cast<BlockArgument>(castOp.getOperand())) {
        // memref.cast directly off a *func-entry* BlockArgument (no subview
        // underneath) — e.g. the RCore output arg arrives as `memref<f16,
        // strided<[], offset:?>>` (from materialize_in_destination's
        // bufferization) and is cast to plain `memref<f16>` before
        // SetGlobalBuffer.  PyAsc's default printer for that memref.cast emits
        // `half *v = reinterpret_cast<half*>(gm_addr)` (missing __gm__), which
        // CANN rejects.  Peel the cast: use the BlockArgument directly; the
        // cleanup pass below erases the now-dead cast.  Restrict to func-entry
        // BlockArgs (not scf.for / scf.if region BlockArgs) so a future memref
        // iter_arg inside a loop doesn't accidentally match.
        Block *owner = ba.getOwner();
        if (auto funcOp = dyn_cast<func::FuncOp>(owner->getParentOp());
            funcOp && &funcOp.getBody().front() == owner)
          baseBuffer = castOp.getOperand();
      }
    }

    // Choose between 1-arg and 2-arg form based on whether we have an offset.
    Value elemOffset = subviewOffset ? subviewOffset
                                     : (sizeVal ? peelIndexCast(sizeVal) : Value{});
    if (!elemOffset) {
      std::string tmpl =
          "$0.SetGlobalBuffer(reinterpret_cast<__gm__ " + elemTypeStr +
          "*>($1))";
      rewriter.create<emitasc::VerbatimOp>(
          loc, rewriter.getStringAttr(tmpl),
          ValueRange({op.getTensor(), baseBuffer}));
      rewriter.eraseOp(op);
      return;
    }

    std::string tmpl =
        "$0.SetGlobalBuffer(reinterpret_cast<__gm__ " + elemTypeStr +
        "*>($1) + $2)";
    rewriter.create<emitasc::VerbatimOp>(
        loc, rewriter.getStringAttr(tmpl),
        ValueRange({op.getTensor(), baseBuffer, elemOffset}));
    rewriter.eraseOp(op);
  });

  // DataCopyL2Op store of a TBuf-backed accumulator (the RBLOCK reduction-split
  // path) → DataCopyPad, so a sub-32-byte element count (e.g. out[A_sub] with
  // A_sub·elem_bytes < 32) reaches GM correctly.  Plain DataCopy to GM requires
  // a block-aligned count and silently drops the tail otherwise; DataCopyPad
  // takes a byte length.  The queued VECOUT path always stores a full inner
  // row (≥ block-sized) and is left on the plain-DataCopy path.
  moduleOp->walk([&](ascendc::DataCopyL2Op op) {
    if (!isa<ascendc::GlobalTensorType>(op.getDst().getType()))
      return;
    if (!op.getSrc().getDefiningOp<ascendc::TBufGetTensorOp>())
      return;
    auto tt = cast<ascendc::LocalTensorType>(op.getSrc().getType());
    std::string elemTypeStr = getAscendCScalarTypeName(tt.getElementType());
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    std::string tmpl =
        "{\n  AscendC::DataCopyExtParams _afir_dcp{(uint16_t)1, "
        "(uint32_t)($2 * sizeof(" + elemTypeStr + ")), (uint32_t)0, "
        "(uint32_t)0, (uint32_t)0};\n"
        "  AscendC::DataCopyPad($0, $1, _afir_dcp);\n}";
    rewriter.create<emitasc::VerbatimOp>(
        loc, rewriter.getStringAttr(tmpl),
        ValueRange({op.getDst(), op.getSrc(), op.getCalCount()}));
    rewriter.eraseOp(op);
  });

  // DataCopyL2Op with GlobalTensorBracketOp source → verbatim
  //
  // PyAsc emits `GlobalTensor<T> row = base(offset);`, but AscendC's operator()
  // returns an element pointer/value rather than a sliced GlobalTensor. Rebuild a
  // temporary GlobalTensor from `GetPhyAddr(offset)` instead.
  moduleOp->walk([&](ascendc::DataCopyL2Op op) {
    auto bracketOp = op.getSrc().getDefiningOp<ascendc::GlobalTensorBracketOp>();
    if (!bracketOp)
      return;

    auto tensorType =
        dyn_cast<ascendc::GlobalTensorType>(bracketOp.getResult().getType());
    if (!tensorType)
      return;

    std::string elemTypeStr =
        getAscendCScalarTypeName(tensorType.getElementType());
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    std::string tmpl = "{\n";
    tmpl += "  AscendC::GlobalTensor<" + elemTypeStr + "> _afir_gt;\n";
    tmpl += "  _afir_gt.SetGlobalBuffer($1.GetPhyAddr($2));\n";
    tmpl += "  AscendC::DataCopy($0, _afir_gt, $3);\n}";
    rewriter.create<emitasc::VerbatimOp>(
        loc, rewriter.getStringAttr(tmpl),
        ValueRange({op.getDst(), bracketOp.getTensor(), bracketOp.getIndex(),
                    op.getCalCount()}));
    rewriter.eraseOp(op);
    if (bracketOp->use_empty())
      rewriter.eraseOp(bracketOp);
  });

  // BroadcastL2Op → verbatim
  //
  // AscendC::Broadcast only supports dim=1 or dim=2. For rank > 2, fold the
  // N-D shapes to 2D by locating the broadcast axis (the unique dim where
  // srcShape[i] is a compile-time constant 1 and dstShape[i] is not) and
  // computing:
  //
  //   prefix = product(srcShape[0..bcastAxis-1])  (same for dst)
  //   suffix = product(srcShape[bcastAxis+1..])    (same for dst)
  //   src2D  = {prefix,            suffix}
  //   dst2D  = {prefix*dstShape[i], suffix}
  //
  // This collapses all outer dims into one "row" count and all inner dims
  // into one "col" count.  The broadcast is then a row-broadcast (axis=0) if
  // bcastAxis < rank-1, or a column-broadcast (axis=1) otherwise.
  //
  // Multi-axis broadcasts (e.g. [1,D,1] → [D0,D,D2]) MUST be decomposed into
  // a chain of single-axis BroadcastL2Ops by an earlier pass before reaching
  // here; this handler emits an op error on multi-axis input.
  moduleOp->walk([&](ascendc::BroadcastL2Op op) {
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    uint32_t rank = op.getConstRank();

    // Operand layout: $0=dst, $1=src, $2..$2+rank-1=dstShape, $2+rank..=srcShape
    auto srcShapeVals = op.getSrcShape();
    auto dstShapeVals = op.getDstShape();

    // Helper: is `v` a compile-time constant integer with value 1?
    auto isStaticOne = [](Value v) -> bool {
      if (auto c = v.getDefiningOp<arith::ConstantOp>())
        if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
          return ia.getInt() == 1;
      return false;
    };

    // Use actual element type of dst instead of hardcoded 'half'.
    auto dstElemType =
        cast<ascendc::LocalTensorType>(op.getDst().getType()).getElementType();
    std::string elemTypeStr = getAscendCScalarTypeName(dstElemType);

    // Helper: build a placeholder product string for indices [lo, hi).
    // Returns e.g. "(uint32_t)$5 * (uint32_t)$6" or "(uint32_t)$5" for a
    // single index, or "1u" when the range is empty.
    auto placeholderProduct = [&](uint32_t lo, uint32_t hi) -> std::string {
      if (lo >= hi) return "1u";
      std::string s = "(uint32_t)$" + std::to_string(lo);
      for (uint32_t k = lo + 1; k < hi; ++k)
        s += " * (uint32_t)$" + std::to_string(k);
      return s;
    };

    std::string tmpl;

    // For rank > 2: fold to 2D.  Find the broadcast axis (where srcShape[i]
    // is statically 1 and dstShape[i] is not).  Also count broadcast axes;
    // multi-axis input is a programming error (decomposition pass missed it).
    int bcastAxis = -1;
    int bcastAxisCount = 0;
    if (rank > 2) {
      for (uint32_t i = 0; i < rank; ++i) {
        if (isStaticOne(srcShapeVals[i]) && !isStaticOne(dstShapeVals[i])) {
          if (bcastAxis < 0)
            bcastAxis = static_cast<int>(i);
          ++bcastAxisCount;
        }
      }
      if (bcastAxisCount > 1) {
        op.emitOpError("rank>2 broadcast with ")
            << bcastAxisCount
            << " broadcast axes reached CannTranslation; expected the "
               "DecomposeMultiAxisBroadcast pass to lower this to a chain of "
               "single-axis broadcasts";
        return;
      }
    }

    if (bcastAxis >= 0) {
      // Fold rank-N to 2D.  AscendC::Broadcast supports two 2D forms:
      //   axis=0 (row):    src{1, N}    -> dst{M, N}
      //   axis=1 (column): src{M, 1}    -> dst{M, N}
      // For a broadcast on axis ba of an N-D shape, the element mapping is
      // src[p, 0, s] -> dst[p, j, s] for j in [0, D), where p ranges over
      // prefix = prod(dims[0,ba)) and s over suffix = prod(dims(ba,N)).  This
      // folds to a 2D broadcast iff prefix==1 (row form: src{1, suffix} ->
      // dst{D, suffix}) OR suffix==1 (column form: src{prefix, 1} ->
      // dst{prefix, D}).  When both prefix and suffix are >1, the broadcast
      // interleaves and cannot be expressed as a single 2D AscendC::Broadcast
      // call.
      //
      // The check is STATIC: we treat a dim as "1" only when its shape operand
      // is an arith.constant 1.  Runtime values whose value happens to be 1
      // do not qualify — the kernel author / decompose pass must arrange the
      // IR so the foldable dims are emitted as arith.constant 1 (e.g. the
      // decompose pass walks broadcast axes in an order that keeps either
      // prefix or suffix all-constant-1).
      uint32_t ba = static_cast<uint32_t>(bcastAxis);
      uint32_t dstBase = 2, srcBase = 2 + rank;
      auto rangeAllStaticOne = [&](uint32_t lo, uint32_t hi) -> bool {
        for (uint32_t i = lo; i < hi; ++i)
          if (!isStaticOne(srcShapeVals[i]))
            return false;
        return true;
      };
      bool prefixIsOne = rangeAllStaticOne(0, ba);
      bool suffixIsOne = rangeAllStaticOne(ba + 1, rank);

      std::string prefixStr = placeholderProduct(srcBase, srcBase + ba);
      std::string suffixStr =
          placeholderProduct(srcBase + ba + 1, srcBase + rank);
      std::string bcastDimStr = "(uint32_t)$" + std::to_string(dstBase + ba);

      if (suffixIsOne) {
        // Column broadcast: src{prefix, 1} -> dst{prefix, D}
        std::string dst2DRow = prefixStr;
        std::string dst2DCol = bcastDimStr;
        std::string src2DRow = prefixStr;
        std::string src2DCol = "1u";
        tmpl = "{\n";
        tmpl += "  uint32_t _afir_ds[2] = {" + dst2DRow + ", " + dst2DCol + "};\n";
        tmpl += "  uint32_t _afir_ss[2] = {" + src2DRow + ", " + src2DCol + "};\n";
        tmpl += "  AscendC::Broadcast<" + elemTypeStr +
                ", 2, 1>($0, $1, _afir_ds, _afir_ss);\n}";
      } else if (prefixIsOne) {
        // Row broadcast: src{1, suffix} -> dst{D, suffix}
        std::string dst2DRow = bcastDimStr;
        std::string dst2DCol = suffixStr;
        std::string src2DRow = "1u";
        std::string src2DCol = suffixStr;
        tmpl = "{\n";
        tmpl += "  uint32_t _afir_ds[2] = {" + dst2DRow + ", " + dst2DCol + "};\n";
        tmpl += "  uint32_t _afir_ss[2] = {" + src2DRow + ", " + src2DCol + "};\n";
        tmpl += "  AscendC::Broadcast<" + elemTypeStr +
                ", 2, 0>($0, $1, _afir_ds, _afir_ss);\n}";
      } else {
        // Middle axis with both prefix>1 and suffix>1 statically — the
        // broadcast interleaves and is not a single 2D AscendC::Broadcast.
        // Emit a runtime loop: for p in [0, prefix), do a 2D row broadcast of
        // a (1, suffix) slice into a (D, suffix) slice, advancing the src/dst
        // pointers by suffix / (D*suffix) elements per iteration.  When
        // prefix==1 at runtime this is a single broadcast call (which is what
        // the previous formula relied on); when prefix>1 it is still correct.
        tmpl = "{\n";
        tmpl += "  uint32_t _afir_prefix = " + prefixStr + ";\n";
        tmpl += "  uint32_t _afir_suffix = " + suffixStr + ";\n";
        tmpl += "  uint32_t _afir_D = " + bcastDimStr + ";\n";
        tmpl += "  uint32_t _afir_ds[2] = {_afir_D, _afir_suffix};\n";
        tmpl += "  uint32_t _afir_ss[2] = {1u, _afir_suffix};\n";
        tmpl += "  for (uint32_t _afir_p = 0; _afir_p < _afir_prefix; ++_afir_p) {\n";
        tmpl += "    AscendC::Broadcast<" + elemTypeStr + ", 2, 0>(\n";
        tmpl += "        $0[_afir_p * _afir_D * _afir_suffix],\n";
        tmpl += "        $1[_afir_p * _afir_suffix],\n";
        tmpl += "        _afir_ds, _afir_ss);\n";
        tmpl += "  }\n}";
      }
    } else {
      // rank <= 2 or no constant-1 srcShape found: use native rank.
      // Determine axis: if last srcShape dim is constant 1, it is column
      // broadcast (axis=1); otherwise row broadcast (axis=0).
      int axis = 0;
      if (!srcShapeVals.empty()) {
        Value lastSrc = srcShapeVals[srcShapeVals.size() - 1];
        if (auto constOp = lastSrc.getDefiningOp<arith::ConstantOp>())
          if (cast<IntegerAttr>(constOp.getValue()).getInt() == 1)
            axis = 1;
      }
      tmpl = "{\n";
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
      tmpl += "  AscendC::Broadcast<" + elemTypeStr + ", " + std::to_string(rank) +
              ", " + std::to_string(axis) + ">($0, $1, _afir_ds, _afir_ss);\n}";
    }

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

  // GatherL2Op with i64 indices → verbatim
  //
  // AscendC::Gather requires LocalTensor<uint32_t> byte offsets. Gather
  // lowering currently feeds LocalTensor<int64_t> element indices when the
  // original indices memref is i64. Convert those indices once before the row
  // loop and reuse the uint32_t byte-offset tensor for each row. Keeping the
  // scratch TBuf outside the hot row loop avoids repeated InitBuffer calls that
  // can trip real-device UB bounds checks.
  unsigned gatherScratchId = 0;
  moduleOp->walk([&](ascendc::GatherL2Op op) {
    auto indicesType =
        dyn_cast<ascendc::LocalTensorType>(op.getSrcOffset().getType());
    if (!indicesType)
      return;
    auto idxElemType = dyn_cast<IntegerType>(indicesType.getElementType());
    if (!idxElemType || idxElemType.getWidth() != 64)
      return;

    Value pipeVal;
    if (auto funcOp = op->getParentOfType<func::FuncOp>()) {
      funcOp.walk([&](ascendc::PipeOp pipeOp) {
        pipeVal = pipeOp.getResult();
        return WalkResult::interrupt();
      });
    }
    if (!pipeVal)
      return;

    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    auto srcElemType =
        cast<ascendc::LocalTensorType>(op.getSrc().getType()).getElementType();
    unsigned srcElemBytes =
        std::max<unsigned>(1, srcElemType.getIntOrFloatBitWidth() / 8);
    unsigned maxGatherCount = srcElemBytes <= 2 ? 128 : (srcElemBytes <= 4 ? 64 : 32);

    unsigned scratchId = gatherScratchId++;
    std::string tbufName =
        "_afir_idx32_tbuf_" + std::to_string(scratchId);
    std::string tensorName = "_afir_idx32_" + std::to_string(scratchId);

    Operation *preludeInsertionPoint = op.getOperation();
    if (auto rowLoop = op->getParentOfType<scf::ForOp>())
      preludeInsertionPoint = rowLoop.getOperation();
    rewriter.setInsertionPoint(preludeInsertionPoint);
    std::string prelude;
    prelude += "AscendC::TBuf<AscendC::TPosition::VECCALC> " + tbufName +
               ";\n";
    prelude += "$0.InitBuffer(" + tbufName +
               ", (uint32_t)$1 * sizeof(uint32_t));\n";
    prelude += "AscendC::LocalTensor<uint32_t> " + tensorName + " = " +
               tbufName + ".Get<uint32_t>();\n";
    prelude +=
        "for (uint32_t _afir_i = 0; _afir_i < static_cast<uint32_t>($1); _afir_i++) {\n";
    prelude += "  " + tensorName +
               ".SetValue(_afir_i, static_cast<uint32_t>($2.GetValue(_afir_i)) * " +
               std::to_string(srcElemBytes) + ");\n";
    prelude += "}";
    prelude += "\nAscendC::PipeBarrier<PIPE_V>()";
    rewriter.create<emitasc::VerbatimOp>(
        loc, rewriter.getStringAttr(prelude),
        ValueRange({pipeVal, op.getCount(), op.getSrcOffset()}));

    rewriter.setInsertionPoint(op);
    std::string tmpl = "{\n";
    tmpl += "  uint32_t _afir_gather_count = static_cast<uint32_t>($3);\n";
    tmpl += "  for (uint32_t _afir_off = 0; _afir_off < _afir_gather_count; _afir_off += " +
            std::to_string(maxGatherCount) + ") {\n";
    tmpl += "    uint32_t _afir_chunk = ((_afir_gather_count - _afir_off) < " +
            std::to_string(maxGatherCount) + ") ? (_afir_gather_count - _afir_off) : " +
            std::to_string(maxGatherCount) + ";\n";
    tmpl += "    AscendC::Gather($0[_afir_off], $1, " + tensorName +
            "[_afir_off], $2, _afir_chunk);\n";
    tmpl += "  }\n}";
    tmpl += "\nAscendC::PipeBarrier<PIPE_V>()";

    SmallVector<Value> args = {op.getDst(), op.getSrc(), op.getSrcBaseAddr(),
                               op.getCount()};
    rewriter.create<emitasc::VerbatimOp>(
        loc, rewriter.getStringAttr(tmpl), ValueRange(args));
    rewriter.eraseOp(op);
  });

  // ReduceSum2DL2Op → verbatim
  //
  // AR layout: src[rows, cols] → dst[rows] by summing each row.
  // RA layout: not yet implemented.
  //
  // Uses the adv_api ReduceSum AR path for row-wise reductions.  CANN 9.1's
  // adv_api ReduceSum supports float, while the basic half ReduceSum count-mode
  // path and scalar SetValue/GetValue based stitching are not reliable in the
  // simulator for this generated kernel shape.  For f16 tensors, cast the
  // source tile to f32, reduce in f32 with isReuseSource=true, then cast the
  // row result back to f16.
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

    // dst byte-length: if dst is a TBuf-backed tensor (the RBLOCK reduction-
    // split accumulating-reduce path uses a fresh VECCALC TBuf for the per-chunk
    // result), find its TPipeInitBufferOp length the same way.
    if (!dstQueueLenVal) {
      if (auto getOp = op.getDst().getDefiningOp<ascendc::TBufGetTensorOp>()) {
        Value tbufVal = getOp.getBuffer();
        for (auto *user : tbufVal.getUsers()) {
          if (auto initB = dyn_cast<ascendc::TPipeInitBufferOp>(user)) {
            dstQueueLenVal = initB.getLength();
            break;
          }
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

    // Verbatim arg order is {dst, src, [dstLen], [srcLen], pipe}; the source
    // byte-length token shifts depending on whether dstLen was pushed, and the
    // pipe token depends on how many byte-length operands precede it.
    std::string srcLenTok = dstQueueLenVal ? "$3" : "$2";
    int pipeArgIdx = 2 + (dstQueueLenVal ? 1 : 0) + (srcTBufLenVal ? 1 : 0);
    std::string pipeTok = "$" + std::to_string(pipeArgIdx);

    // Use the actual element type of dst (was previously hardcoded to `half`,
    // which silently miscompiled f32 reduce kernels: cols counted in bytes/2
    // instead of bytes/4 and ReduceSum<half> reinterpreted f32 bytes as f16).
    auto dstElemType =
        cast<ascendc::LocalTensorType>(op.getDst().getType()).getElementType();
    std::string elemTypeStr = getAscendCScalarTypeName(dstElemType);

    std::string tmpl = "{\n";
    if (isAR) {
      // AR: dst[r] = sum(src[r*cols .. r*cols+cols-1])
      // Use ReduceSum<T> per row with a 32-byte scratch VECCALC TBuf.
      // The TPipe is passed as the last operand so we can InitBuffer the scratch.
      // $1[r * cols] slices the src tensor to the start of row r.
      std::string pipeRef = pipeTok;
      if (dstQueueLenVal && srcTBufLenVal) {
        tmpl += "  uint32_t _afir_rows = (uint32_t)($2 / sizeof(" + elemTypeStr + "));\n";
        tmpl += "  uint32_t _afir_cols = (uint32_t)(" + srcLenTok + " / $2);\n";
      } else if (dstQueueLenVal) {
        tmpl += "  uint32_t _afir_rows = (uint32_t)($2 / sizeof(" + elemTypeStr + "));\n";
        tmpl += "  uint32_t _afir_cols = (uint32_t)($1.GetSize() / $2);\n";
      } else if (srcTBufLenVal) {
        // dst byte-length unavailable; recover rows from $0.GetSize().
        tmpl += "  uint32_t _afir_rows = (uint32_t)$0.GetSize();\n";
        tmpl += "  uint32_t _afir_cols = (uint32_t)(" + srcLenTok +
                " / ($0.GetSize() * (uint32_t)sizeof(" + elemTypeStr + ")));\n";
      } else {
        tmpl += "  uint32_t _afir_rows = (uint32_t)($0.GetSize() / sizeof(" + elemTypeStr + "));\n";
        tmpl += "  uint32_t _afir_cols = (uint32_t)($1.GetSize() / $0.GetSize());\n";
      }
      // Need two distinct scratch tensors: _afir_scalar (per-row result, 1 elem)
      // and _afir_ws (ReduceSum's internal tree workspace, ≥ cols elems).
      // ReduceSum requires dst != sharedTmpBuffer (aliasing them corrupts the
      // output on arch 3101).
      if (op.getSharedTmpBuffer()) {
        // Pre-allocated scratch (the RBLOCK reduction-split path passes one so
        // we don't InitBuffer inside the loop): first 8 elems = result slot,
        // the rest = workspace.  Passed as the last verbatim operand ($N).
        tmpl += "  AscendC::LocalTensor<" + elemTypeStr + "> _afir_scalar = " +
                pipeTok + ";\n";
        tmpl += "  AscendC::LocalTensor<" + elemTypeStr + "> _afir_ws = " +
                pipeTok + "[8];\n";
      } else {
        // No pre-allocated scratch — InitBuffer two fresh VECCALC TBufs.  The
        // last operand ($N) is the TPipe.  Workspace must hold one source row
        // (cols elements); 32 B is too small for cols>8 fp32.
        tmpl += "  AscendC::TBuf<AscendC::TPosition::VECCALC> _afir_tbuf_dst;\n";
        tmpl += "  AscendC::TBuf<AscendC::TPosition::VECCALC> _afir_tbuf_ws;\n";
        tmpl += "  uint32_t _afir_ws_bytes = ((_afir_cols * (uint32_t)sizeof(" +
                elemTypeStr + ") + 31u) / 32u) * 32u;\n";
        tmpl += "  " + pipeRef + ".InitBuffer(_afir_tbuf_dst, 32);\n";
        tmpl += "  " + pipeRef + ".InitBuffer(_afir_tbuf_ws, _afir_ws_bytes);\n";
        tmpl += "  AscendC::LocalTensor<" + elemTypeStr +
                "> _afir_scalar = _afir_tbuf_dst.Get<" + elemTypeStr + ">();\n";
        tmpl += "  AscendC::LocalTensor<" + elemTypeStr +
                "> _afir_ws = _afir_tbuf_ws.Get<" + elemTypeStr + ">();\n";
      }
      tmpl += "  for (uint32_t _afir_r = 0; _afir_r < _afir_rows; _afir_r++) {\n";
      tmpl += "    AscendC::ReduceSum<" + elemTypeStr +
              ">(_afir_scalar, $1[_afir_r * _afir_cols],\n";
      tmpl += "                            _afir_ws, (int32_t)_afir_cols);\n";
      // ReduceSum writes _afir_scalar on PIPE_V; the GetValue below reads it on
      // PIPE_S — barrier so the scalar read sees the committed vector result.
      // (No PIPE_S barrier afterwards: SetValue targets a different tensor than
      // the next iteration's ReduceSum, and EnQue handles producer/consumer
      // sync for the result tensor.)
      tmpl += "    AscendC::PipeBarrier<PIPE_V>();\n";
      tmpl += "    $0.SetValue(_afir_r, _afir_scalar.GetValue(0));\n";
      tmpl += "  }\n";
      // The SetValue writes above happen on PIPE_S; a subsequent vector op that
      // reads $0 (e.g. the accumulating AddL2 in the RBLOCK reduction-split
      // path) runs on PIPE_V and must wait for the scalar writes to commit.
      // When $0 is a queued VECOUT tensor this is handled by EnQue/DeQue, but
      // when $0 is a plain TBuf tensor consumed directly we need an explicit
      // barrier; PIPE_ALL is safe in both cases.
      tmpl += "  AscendC::PipeBarrier<PIPE_ALL>();\n}";
    } else {
      // RA layout: src is laid out as [R, A] (R = reduction extent, the FIRST
      // axis; A = output extent, the SECOND axis), and we reduce R:
      //     dst[a] = sum_{r} src[r*A + a]
      // The reduction values for one output element are A-strided in the
      // buffer, which adv_api ReduceSum<float, Pattern::Reduce::RA> handles
      // directly.  This is the codegen path for a reduction whose iteration
      // axis is the first (or, after collapse, the outer) axis of the operand
      // — e.g. out[d0,d2] = sum_{d1} x[d0,d1,d2] processed one d0-row at a time
      // (the per-row buffer is [d1(R), d2(A)]).
      //
      // Byte-length operands (same scheme as AR):
      //   $2 = dst bytes = A * sizeof(T)       → A = $2 / sizeof(T)
      //   $3 = src bytes = R * A * sizeof(T)   → R = $3 / $2
      //   last operand   = TPipe (for the scratch TBuf)
      //
      // adv_api ReduceSum<*, RA> supports float only.  For a half input we
      // upcast src to float in a VECCALC scratch TBuf, ReduceSum<float, RA>
      // into a float result TBuf, then downcast back to half $0.  (R2 in the
      // reduce-codegen-status notes.)
      bool needUpcast = (elemTypeStr == "half");
      std::string opTypeStr = needUpcast ? "float" : elemTypeStr;
      std::string pipeRef = pipeTok;
      bool haveLens = (bool)dstQueueLenVal && (bool)srcTBufLenVal;
      if (haveLens) {
        tmpl += "  uint32_t _afir_A = (uint32_t)($2 / sizeof(" + elemTypeStr + "));\n";
        tmpl += "  uint32_t _afir_R = (uint32_t)(" + srcLenTok + " / $2);\n";
      } else if (dstQueueLenVal) {
        tmpl += "  uint32_t _afir_A = (uint32_t)($2 / sizeof(" + elemTypeStr + "));\n";
        tmpl += "  uint32_t _afir_R = (uint32_t)($1.GetSize() / _afir_A);\n";
      } else {
        tmpl += "  uint32_t _afir_A = (uint32_t)($0.GetSize());\n";
        tmpl += "  uint32_t _afir_R = (uint32_t)($1.GetSize() / _afir_A);\n";
      }
      // Scratch for ReduceSum's internal tree reduction, sized in the *op*
      // type (float when upcasting from half).  Over-estimate as the float
      // input byte size, 32 B-aligned, minimum 256 B.
      tmpl += "  uint32_t _afir_ws_bytes = _afir_R * _afir_A * (uint32_t)sizeof(" +
              opTypeStr + ");\n";
      tmpl += "  _afir_ws_bytes = (_afir_ws_bytes < 256u) ? 256u : _afir_ws_bytes;\n";
      tmpl += "  _afir_ws_bytes = ((_afir_ws_bytes + 31u) / 32u) * 32u;\n";
      tmpl += "  AscendC::TBuf<AscendC::TPosition::VECCALC> _afir_tbuf_ws;\n";
      tmpl += "  " + pipeRef + ".InitBuffer(_afir_tbuf_ws, _afir_ws_bytes);\n";
      tmpl += "  AscendC::LocalTensor<uint8_t> _afir_ws = _afir_tbuf_ws.Get<uint8_t>();\n";

      std::string srcRef = "$1";
      std::string dstRef = "$0";
      if (needUpcast) {
        // Float src TBuf (size = R*A*4 bytes), Cast half→float.
        tmpl += "  uint32_t _afir_src_bytes = _afir_R * _afir_A * (uint32_t)sizeof(float);\n";
        tmpl += "  _afir_src_bytes = ((_afir_src_bytes + 31u) / 32u) * 32u;\n";
        tmpl += "  AscendC::TBuf<AscendC::TPosition::VECCALC> _afir_tbuf_srcf;\n";
        tmpl += "  " + pipeRef + ".InitBuffer(_afir_tbuf_srcf, _afir_src_bytes);\n";
        tmpl += "  AscendC::LocalTensor<float> _afir_srcf = _afir_tbuf_srcf.Get<float>();\n";
        // Float dst TBuf (size = A*4 bytes).
        tmpl += "  uint32_t _afir_dst_bytes = _afir_A * (uint32_t)sizeof(float);\n";
        tmpl += "  _afir_dst_bytes = ((_afir_dst_bytes + 31u) / 32u) * 32u;\n";
        tmpl += "  AscendC::TBuf<AscendC::TPosition::VECCALC> _afir_tbuf_dstf;\n";
        tmpl += "  " + pipeRef + ".InitBuffer(_afir_tbuf_dstf, _afir_dst_bytes);\n";
        tmpl += "  AscendC::LocalTensor<float> _afir_dstf = _afir_tbuf_dstf.Get<float>();\n";
        tmpl += "  AscendC::Cast<float, half>(_afir_srcf, $1, AscendC::RoundMode::CAST_NONE, _afir_R * _afir_A);\n";
        tmpl += "  AscendC::PipeBarrier<PIPE_V>();\n";
        srcRef = "_afir_srcf";
        dstRef = "_afir_dstf";
      }
      tmpl += "  uint32_t _afir_srcShape[2] = {_afir_R, _afir_A};\n";
      tmpl += "  AscendC::ReduceSum<" + opTypeStr +
              ", AscendC::Pattern::Reduce::RA>(" + dstRef + ", " + srcRef +
              ", _afir_ws, _afir_srcShape, false);\n";
      if (needUpcast) {
        tmpl += "  AscendC::PipeBarrier<PIPE_V>();\n";
        tmpl += "  AscendC::Cast<half, float>($0, _afir_dstf, AscendC::RoundMode::CAST_RINT, _afir_A);\n";
      }
      tmpl += "}";
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
    // Last operand ($N): a pre-allocated scratch tensor when one was provided
    // (AR reduction-split path), otherwise the TPipe (so the verbatim can
    // InitBuffer its own scratch).
    if (Value lastArg = op.getSharedTmpBuffer() ? op.getSharedTmpBuffer() : pipeVal)
      args.push_back(lastArg);

    rewriter.create<emitasc::VerbatimOp>(
        loc, rewriter.getStringAttr(tmpl), ValueRange(args));
    rewriter.eraseOp(op);
  });

  // Erase dead memref.cast and memref.subview ops left behind by the
  // SetGlobalBuffer subview-peeling above.  Collect in two passes so we
  // erase cast before subview (cast's operand is the subview result).
  SmallVector<Operation *> deadCasts, deadSubviews;
  moduleOp->walk([&](memref::CastOp op) {
    if (op.getResult().use_empty())
      deadCasts.push_back(op);
  });
  for (Operation *op : deadCasts)
    op->erase();
  moduleOp->walk([&](memref::SubViewOp op) {
    if (op.getResult().use_empty())
      deadSubviews.push_back(op);
  });
  for (Operation *op : deadSubviews)
    op->erase();
}

LogicalResult mlir::translateToCannKernel(Operation *op, raw_ostream &os,
                                          StringRef tilingSpaceOutPath,
                                          StringRef kernelFile,
                                          StringRef socName) {
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
                                                mixPartitionSummary,
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
          SupportedMixLoweringFailureReason emissionFailureReason =
              SupportedMixLoweringFailureReason::UnsupportedLegacySignature;
          if (emitGenericMixSingleChainKernel(
                  os, primaryKernel, *genericEmissionPlan, *supportedLowering,
                  emissionFailureReason))
            return success();
          return primaryKernel.emitOpError(
              Twine("mix translation found a valid single-chain cube/boundary/"
                    "vector plan, but the generic primary route could not "
                    "lower the current supported shell because ") +
              stringifySupportedMixLoweringFailureReason(
                  emissionFailureReason));
        }

        SupportedMixLoweringFailureReason legacyFallbackFailureReason =
            SupportedMixLoweringFailureReason::UnsupportedLegacySignature;
        FailureOr<GenericMixSingleChainSupportedLowering> legacyFallbackLowering =
            buildLegacySupportedMixLowering(mixPartitionPlan, primaryKernel,
                                            mixPartitionSummary,
                                            singleChainValidation,
                                            legacyFallbackFailureReason);
        if (succeeded(legacyFallbackLowering)) {
          SupportedMixLoweringFailureReason emissionFailureReason =
              SupportedMixLoweringFailureReason::UnsupportedLegacySignature;
          if (emitSupportedMixKernel(os, primaryKernel, mixPartitionPlan,
                                     *legacyFallbackLowering,
                                     emissionFailureReason))
            return success();
          return primaryKernel.emitOpError(
              Twine("mix translation found a valid single-chain cube/boundary/"
                    "vector plan, but the retained supported-mix fallback "
                    "could not lower the current supported shell because ") +
              stringifySupportedMixLoweringFailureReason(
                  emissionFailureReason));
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
        SupportedMixLoweringFailureReason emissionFailureReason =
            SupportedMixLoweringFailureReason::UnsupportedLegacySignature;
        if (emitSupportedMixKernel(os, primaryKernel, mixPartitionPlan,
                                   *legacyFallbackLowering,
                                   emissionFailureReason))
          return success();
        return primaryKernel.emitOpError(
            Twine("mix translation found a valid single-chain cube/boundary/"
                  "vector plan, but the retained supported-mix fallback "
                  "could not lower the current supported shell because ") +
            stringifySupportedMixLoweringFailureReason(
                emissionFailureReason));
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
        SupportedMixLoweringFailureReason emissionFailureReason =
            SupportedMixLoweringFailureReason::UnsupportedLegacySignature;
        if (emitSupportedMixKernel(os, primaryKernel, mixPartitionPlan,
                                   *legacyFallbackLowering,
                                   emissionFailureReason))
          return success();
        return primaryKernel.emitOpError(
            Twine("mix translation requires a supported cube/vector "
                  "partitioned kernel shape; generic single-chain analysis "
                  "rejected plan because ") +
            Twine(describeMixSingleChainValidation(singleChainValidation)) +
            Twine("; the retained supported-mix fallback could not lower the "
                  "current supported shell because ") +
            stringifySupportedMixLoweringFailureReason(
                emissionFailureReason));
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

  // First pass: emit TilingData struct declarations + per-func space.json.
  //
  // P2 (multi-plan tiling variants): when the module contains funcs named
  // <familyId>__v<idx>, write one space.json per func at the same dir as
  // `tilingSpaceOutPath`, named <funcName>_space.json, plus a sibling
  // <familyId>_family.json index. For back-compat with legacy direct
  // afir-translate callers that read the original tilingSpaceOutPath path,
  // if there is exactly one aicore func ALSO write that path (same content
  // as the per-func file).
  struct VariantInfo { std::string id; std::string funcName; std::string spaceFile; };
  llvm::StringMap<llvm::SmallVector<VariantInfo>> familyVariants;
  llvm::SmallVector<func::FuncOp> aicoreFuncs;
  for (Operation &child : moduleOp.getBody()->getOperations()) {
    auto funcOp = dyn_cast<func::FuncOp>(child);
    if (!funcOp || !funcOp->hasAttr(ascendc::attr::global)) continue;
    auto args = funcOp.getArguments();
    if (args.empty()) continue;
    if (!dyn_cast<emitasc::PyStructType>(args.back().getType())) continue;
    aicoreFuncs.push_back(funcOp);
  }

  // Helper: split "<family>__v<idx>" into (family, variantId). If no suffix,
  // family == funcName and variantId == "" (legacy / no-variant codepath).
  auto splitVariant = [](llvm::StringRef name)
      -> std::pair<std::string, std::string> {
    auto pos = name.rfind("__v");
    if (pos == llvm::StringRef::npos) return {name.str(), ""};
    auto tail = name.substr(pos + 3);
    for (char c : tail) if (!llvm::isDigit(c)) return {name.str(), ""};
    return {name.substr(0, pos).str(), ("v" + tail).str()};
  };

  // Emit the TilingData struct decl just once. P1b's multi-variant codegen
  // produces N aicore funcs in this module, all sharing the same struct
  // schema (they're clones of one source kernel before TilePlanGen). The
  // emitter writes the C++ name "TilingData", so emitting per-func would
  // cause `redefinition of TilingData` at compile time. If two genuinely
  // different schemas ever end up in one .cpp, dedup needs to key on
  // struct content rather than just "first-one-wins".
  bool tilingStructEmitted = false;
  for (func::FuncOp funcOp : aicoreFuncs) {
    auto args = funcOp.getArguments();
    auto tilingType = cast<emitasc::PyStructType>(args.back().getType());
    if (!tilingStructEmitted) {
      if (failed(emitTilingStructDecl(emitter, funcOp.getLoc(), tilingType)))
        return failure();
      tilingStructEmitted = true;
    }

    if (tilingSpaceOutPath.empty()) continue;

    auto [familyId, variantId] = splitVariant(funcOp.getName());
    std::string funcNameStr = funcOp.getName().str();
    SmallString<256> baseDir(tilingSpaceOutPath);
    llvm::sys::path::remove_filename(baseDir);
    SmallString<256> perFuncPath = baseDir;
    llvm::sys::path::append(perFuncPath, funcNameStr + "_space.json");
    if (failed(emitTilingSpaceJson(perFuncPath, kernelFile, funcOp, tilingType,
                                   socName)))
      return failure();

    if (!variantId.empty())
      familyVariants[familyId].push_back({variantId, funcNameStr,
                                          perFuncPath.str().str()});

    // Back-compat: always overwrite the requested tilingSpaceOutPath with the
    // first variant's (v0) content, so direct callers (run.sh examples) see
    // fresh schema even when multiple variants exist for a family.  Autotuner
    // reads <family>_family.json + per-variant files and doesn't depend on
    // this path, so the back-compat copy doesn't affect it.  Without this,
    // multi-variant codegen left stale content from pre-multi-variant runs
    // because the old single-variant guard skipped the write.
    bool isFirstVariant = variantId.empty() || variantId == "v0";
    if (isFirstVariant && tilingSpaceOutPath != perFuncPath)
      if (failed(emitTilingSpaceJson(tilingSpaceOutPath, kernelFile, funcOp,
                                     tilingType, socName)))
        return failure();
  }

  // Emit family.json per family (only when at least one variant present).
  for (auto &kv : familyVariants) {
    llvm::StringRef familyId = kv.first();
    SmallString<256> baseDir(tilingSpaceOutPath);
    llvm::sys::path::remove_filename(baseDir);
    SmallString<256> familyPath = baseDir;
    llvm::sys::path::append(familyPath, familyId.str() + "_family.json");

    llvm::json::Array variantArr;
    for (auto &v : kv.second) {
      llvm::json::Object entry;
      entry["id"] = v.id;
      entry["func_name"] = v.funcName;
      entry["space_file"] = llvm::sys::path::filename(v.spaceFile).str();
      variantArr.push_back(std::move(entry));
    }
    llvm::json::Object root;
    root["kernel_id"] = familyId.str();
    root["variants"] = std::move(variantArr);

    std::error_code ec;
    llvm::raw_fd_ostream f(familyPath, ec);
    if (ec) {
      llvm::errs() << "Warning: cannot write family.json to " << familyPath
                   << ": " << ec.message() << "\n";
      continue;
    }
    llvm::json::OStream jos(f, /*IndentSize=*/2);
    jos.value(llvm::json::Value(std::move(root)));
    f << "\n";
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
