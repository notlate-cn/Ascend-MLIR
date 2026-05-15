//===- CannTranslation.cpp - CANN kernel C++ translation --------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/CannKernel/CannTranslation.h"
#include "Target/CannKernel/CannRuntimeArtifacts.h"
#include "Conversion/Ascend/Common/Attributes.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Target/Asc/CodeEmitter.h"
#include "ascir/Target/Asc/Common.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <functional>
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
  auto kindAttr = funcOp->getAttrOfType<StringAttr>(
      mlir::afir::ascend::kAscendCKernelKindAttr);
  if (!kindAttr)
    return AscendCKernelKind::Unknown;
  StringRef kind = kindAttr.getValue();
  if (kind == mlir::afir::ascend::kAscendCKernelKindVec)
    return AscendCKernelKind::Vec;
  if (kind == mlir::afir::ascend::kAscendCKernelKindCube)
    return AscendCKernelKind::Cube;
  if (kind == mlir::afir::ascend::kAscendCKernelKindMix)
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
  while (value) {
    if (auto castOp = value.getDefiningOp<emitasc::ReinterpretCastOp>()) {
      value = castOp.getOperand();
      continue;
    }
    if (auto castOp = value.getDefiningOp<memref::CastOp>()) {
      value = castOp.getSource();
      continue;
    }
    break;
  }
  return value;
}

static Value peelIndexCast(Value value) {
  if (!value)
    return value;
  if (auto castOp = value.getDefiningOp<arith::IndexCastOp>())
    return castOp.getOperand();
  return value;
}

struct IndexedValueStore {
  Value memref;
  int64_t index = 0;
  Value stored;
};

struct MemRefDimBound {
  Value memref;
  int64_t dim = 0;
  Value upperBound;
};

static std::optional<int64_t> getConstantIndexValue(Value value) {
  APInt intValue;
  if (matchPattern(value, m_ConstantInt(&intValue)))
    return intValue.getSExtValue();
  return std::nullopt;
}

static Value ensureIndexValue(Value value, IRRewriter &rewriter, Location loc) {
  if (value.getType().isIndex())
    return value;
  return rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                             value);
}

static SmallVector<SmallVector<int64_t>>
parseReassociationIndices(ArrayAttr reassociation) {
  SmallVector<SmallVector<int64_t>> groups;
  for (Attribute rawGroup : reassociation) {
    SmallVector<int64_t> group;
    if (auto denseGroup = dyn_cast<DenseI64ArrayAttr>(rawGroup)) {
      group.append(denseGroup.asArrayRef().begin(),
                   denseGroup.asArrayRef().end());
    } else if (auto arrayGroup = dyn_cast<ArrayAttr>(rawGroup)) {
      for (Attribute rawIndex : arrayGroup)
        group.push_back(cast<IntegerAttr>(rawIndex).getInt());
    }
    groups.push_back(std::move(group));
  }
  return groups;
}

static bool hasOnlyZeroStaticOffsets(memref::SubViewOp op) {
  if (!op.getOffsets().empty())
    return false;
  return llvm::all_of(op.getStaticOffsets(), [](int64_t offset) {
    return !ShapedType::isDynamic(offset) && offset == 0;
  });
}

static FailureOr<Value> buildStaticSubViewLinearOffset(memref::SubViewOp op,
                                                       IRRewriter &rewriter) {
  if (!op.getOffsets().empty())
    return failure();

  ArrayRef<int64_t> staticOffsets = op.getStaticOffsets();
  if (llvm::any_of(staticOffsets, ShapedType::isDynamic))
    return failure();

  auto sourceType = dyn_cast<MemRefType>(op.getSource().getType());
  if (!sourceType || sourceType.getRank() !=
                         static_cast<int64_t>(staticOffsets.size()))
    return failure();

  Location loc = op.getLoc();
  auto makeIndex = [&](int64_t value) -> Value {
    return rewriter.create<arith::ConstantIndexOp>(loc, value);
  };
  auto mul = [&](Value lhs, Value rhs) -> Value {
    return rewriter.create<arith::MulIOp>(loc, lhs, rhs);
  };
  auto add = [&](Value lhs, Value rhs) -> Value {
    return rewriter.create<arith::AddIOp>(loc, lhs, rhs);
  };

  SmallVector<Value> dynamicSizes(op.getSizes().begin(), op.getSizes().end());
  ArrayRef<int64_t> staticSizes = op.getStaticSizes();
  ArrayRef<int64_t> sourceShape = sourceType.getShape();
  SmallVector<Value> dimExtents;
  dimExtents.reserve(sourceType.getRank());

  unsigned dynamicSizeIndex = 0;
  for (auto [dim, sourceExtent] : llvm::enumerate(sourceShape)) {
    Value dynamicSize;
    int64_t staticSize = staticSizes[dim];
    if (ShapedType::isDynamic(staticSize)) {
      if (dynamicSizeIndex >= dynamicSizes.size())
        return failure();
      dynamicSize = dynamicSizes[dynamicSizeIndex++];
    }

    if (!ShapedType::isDynamic(sourceExtent))
      dimExtents.push_back(makeIndex(sourceExtent));
    else if (dynamicSize)
      dimExtents.push_back(dynamicSize);
    else
      dimExtents.push_back(makeIndex(staticSize));
  }

  Value linearOffset = makeIndex(0);
  for (int64_t dim = 0, rank = sourceType.getRank(); dim < rank; ++dim) {
    int64_t offset = staticOffsets[dim];
    if (offset == 0)
      continue;

    Value stride = makeIndex(1);
    for (int64_t strideDim = dim + 1; strideDim < rank; ++strideDim)
      stride = mul(stride, dimExtents[strideDim]);

    Value term = stride;
    if (offset != 1)
      term = mul(makeIndex(offset), stride);
    linearOffset = add(linearOffset, term);
  }
  return linearOffset;
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
  auto unitAttr = op->getAttrOfType<StringAttr>(
      mlir::afir::ascend::kAscendCUnitAttr);
  if (!unitAttr)
    return MixPartitionKind::Unknown;
  StringRef unit = unitAttr.getValue();
  if (unit == mlir::afir::ascend::kAscendCUnitCube)
    return MixPartitionKind::Cube;
  if (unit == mlir::afir::ascend::kAscendCUnitVector)
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

static void emitSupportedMixKernelShellPrologue(
    raw_ostream &os, StringRef kernelName, const MixTaskKindDescriptor &desc) {
  emitSupportedMixIncludesAndNamespaces(os);
  emitSupportedMixCopyTilingHelper(os);
  emitSupportedMixKernelSignature(os, kernelName, desc);
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
  emitSupportedMixKernelShellPrologue(os, funcOp.getName(), desc);
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
  emitSupportedMixKernelShellPrologue(os, funcOp.getName(), desc);
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

static bool isSameConstant(arith::ConstantOp lhs, arith::ConstantOp rhs) {
  return lhs.getType() == rhs.getType() && lhs.getValue() == rhs.getValue();
}

static void deduplicateConstantsInBlock(Block &block) {
  SmallVector<arith::ConstantOp> uniqueConstants;
  SmallVector<arith::ConstantOp> constantsToErase;

  for (Operation &op : block) {
    if (auto constantOp = dyn_cast<arith::ConstantOp>(&op)) {
      auto it = llvm::find_if(uniqueConstants, [&](arith::ConstantOp seen) {
        return isSameConstant(seen, constantOp);
      });
      if (it != uniqueConstants.end()) {
        constantOp.getResult().replaceAllUsesWith(it->getResult());
        constantsToErase.push_back(constantOp);
        continue;
      }

      uniqueConstants.push_back(constantOp);
      continue;
    }

    for (Region &region : op.getRegions())
      for (Block &nestedBlock : region)
        deduplicateConstantsInBlock(nestedBlock);
  }

  for (arith::ConstantOp constantOp : constantsToErase)
    constantOp.erase();
}

static void deduplicateConstantsForEmission(Operation *op) {
  for (Region &region : op->getRegions())
    for (Block &block : region)
      deduplicateConstantsInBlock(block);
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
static Value findQueuedDataCopyGlobalSource(Value tensor) {
  auto dequeOp = tensor.getDefiningOp<ascendc::TQueBindDequeTensorOp>();
  if (!dequeOp)
    return {};

  Value queue = dequeOp.getQueue();
  for (Operation *queueUser : queue.getUsers()) {
    auto enqueOp = dyn_cast<ascendc::TQueBindEnqueTensorOp>(queueUser);
    if (!enqueOp || enqueOp.getQueue() != queue)
      continue;

    Value enqueuedTensor = enqueOp.getTensor();
    for (Operation *tensorUser : enqueuedTensor.getUsers()) {
      auto copyOp = dyn_cast<ascendc::DataCopyL2Op>(tensorUser);
      if (copyOp && copyOp.getDst() == enqueuedTensor &&
          isa<ascendc::GlobalTensorType>(copyOp.getSrc().getType()))
        return copyOp.getSrc();

      // The GM->local DataCopyL2 pre-lowering may already have replaced the
      // copy with a verbatim block in this pass. That block keeps operands as
      // (localDst, globalSrc, count), so preserve the source trace here.
      auto verbatimOp = dyn_cast<emitasc::VerbatimOp>(tensorUser);
      if (verbatimOp && verbatimOp->getNumOperands() >= 2 &&
          verbatimOp->getOperand(0) == enqueuedTensor &&
          isa<ascendc::GlobalTensorType>(verbatimOp->getOperand(1).getType()))
        return verbatimOp->getOperand(1);
    }
  }

  return {};
}

static Value findLocalTensorDataCopyCountBefore(Operation *anchor,
                                                Value tensor) {
  for (Operation *it = anchor->getPrevNode(); it; it = it->getPrevNode()) {
    auto copyOp = dyn_cast<ascendc::DataCopyL2Op>(it);
    if (copyOp && copyOp.getDst() == tensor)
      return copyOp.getCalCount();
  }
  return {};
}

static Value findLocalTensorDataCopyGlobalSourceBefore(Operation *anchor,
                                                       Value tensor) {
  for (Operation *scope = anchor; scope; scope = scope->getParentOp()) {
    if (!scope->getBlock())
      continue;
    for (Operation *it = scope->getPrevNode(); it; it = it->getPrevNode()) {
      auto copyOp = dyn_cast<ascendc::DataCopyL2Op>(it);
      if (copyOp && copyOp.getDst() == tensor &&
          isa<ascendc::GlobalTensorType>(copyOp.getSrc().getType()))
        return copyOp.getSrc();
    }
  }
  return {};
}

static Value findLocalTensorByteLength(Value tensor) {
  if (auto getOp = tensor.getDefiningOp<ascendc::TBufGetTensorOp>()) {
    Value tbuf = getOp.getBuffer();
    for (Operation *user : tbuf.getUsers())
      if (auto initBuffer = dyn_cast<ascendc::TPipeInitBufferOp>(user))
        return initBuffer.getLength();
  }

  Value queue;
  if (auto allocOp = tensor.getDefiningOp<ascendc::TQueBindAllocTensorOp>())
    queue = allocOp.getQueue();
  else if (auto dequeOp =
               tensor.getDefiningOp<ascendc::TQueBindDequeTensorOp>())
    queue = dequeOp.getQueue();

  if (queue)
    for (Operation *user : queue.getUsers())
      if (auto initQueue = dyn_cast<ascendc::TPipeInitQueueOp>(user))
        return initQueue.getLength();

  return {};
}

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
    if (!sizeVal) {
      std::string tmpl =
          "$0.SetGlobalBuffer(reinterpret_cast<__gm__ " + elemTypeStr +
          "*>($1))";
      rewriter.create<emitasc::VerbatimOp>(
          loc, rewriter.getStringAttr(tmpl),
          ValueRange({op.getTensor(), baseBuffer}));
      rewriter.eraseOp(op);
      return;
    }

    Value elemOffset = peelIndexCast(sizeVal);
    std::string tmpl =
        "$0.SetGlobalBuffer(reinterpret_cast<__gm__ " + elemTypeStr +
        "*>($1) + $2)";
    rewriter.create<emitasc::VerbatimOp>(
        loc, rewriter.getStringAttr(tmpl),
        ValueRange({op.getTensor(), baseBuffer, elemOffset}));
    rewriter.eraseOp(op);
  });

  SmallVector<memref::CastOp> deadMemrefCasts;
  moduleOp->walk([&](memref::CastOp op) {
    if (op->use_empty())
      deadMemrefCasts.push_back(op);
  });
  for (memref::CastOp op : deadMemrefCasts)
    rewriter.eraseOp(op);

  // Frontend shape guards survive Normalize as cf.assert. AscendC kernels have
  // no cf.assert printer, so lower them to fail-closed early returns.
  moduleOp->walk([&](cf::AssertOp op) {
    rewriter.setInsertionPoint(op);
    rewriter.create<emitasc::VerbatimOp>(
        op.getLoc(), rewriter.getStringAttr("if (!$0) {\n  return;\n}"),
        ValueRange({op.getArg()}));
    rewriter.eraseOp(op);
  });

  SmallVector<IndexedValueStore> indexedStores;
  SmallVector<MemRefDimBound> dimBounds;
  moduleOp->walk([&](memref::StoreOp op) {
    if (op.getIndices().size() == 1) {
      std::optional<int64_t> index = getConstantIndexValue(op.getIndices()[0]);
      if (index)
        indexedStores.push_back(
            {peelSourceValue(op.getMemref()), *index, op.getValueToStore()});
    }

    Value root = peelSourceValue(op.getMemref());
    for (auto [dim, index] : llvm::enumerate(op.getIndices())) {
      auto blockArg = dyn_cast<BlockArgument>(index);
      if (!blockArg)
        continue;
      auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
      if (!forOp || forOp.getInductionVar() != index)
        continue;
      dimBounds.push_back(
          {root, static_cast<int64_t>(dim), forOp.getUpperBound()});
    }
  });

  auto lookupIndexedStore = [&](Value memref,
                                int64_t index) -> std::optional<Value> {
    Value root = peelSourceValue(memref);
    for (const IndexedValueStore &store : indexedStores)
      if (store.memref == root && store.index == index)
        return store.stored;
    return std::nullopt;
  };

  auto lookupDimBound = [&](Value memref, int64_t dim) -> std::optional<Value> {
    Value root = peelSourceValue(memref);
    for (const MemRefDimBound &bound : llvm::reverse(dimBounds))
      if (bound.memref == root && bound.dim == dim)
        return bound.upperBound;
    return std::nullopt;
  };

  std::function<FailureOr<Value>(Value, int64_t, unsigned)> resolveDim;
  resolveDim = [&](Value source, int64_t dim,
                   unsigned depth) -> FailureOr<Value> {
    if (depth > 8)
      return failure();

    auto sourceType = dyn_cast<MemRefType>(source.getType());
    if (!sourceType || dim < 0 || dim >= sourceType.getRank())
      return failure();

    Location loc = source.getLoc();
    auto makeIndex = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantIndexOp>(loc, value);
    };
    auto mul = [&](Value lhs, Value rhs) -> Value {
      return rewriter.create<arith::MulIOp>(loc, lhs, rhs);
    };

    int64_t staticExtent = sourceType.getShape()[dim];
    if (!ShapedType::isDynamic(staticExtent))
      return makeIndex(staticExtent);

    if (std::optional<Value> bound = lookupDimBound(source, dim))
      return ensureIndexValue(*bound, rewriter, loc);

    if (auto castOp = source.getDefiningOp<memref::CastOp>())
      return resolveDim(castOp.getSource(), dim, depth + 1);

    if (auto castOp = source.getDefiningOp<emitasc::ReinterpretCastOp>()) {
      if (isa<MemRefType>(castOp.getSource().getType()))
        return resolveDim(castOp.getSource(), dim, depth + 1);
    }

    if (auto reshapeOp = source.getDefiningOp<memref::ReshapeOp>()) {
      if (std::optional<Value> stored =
              lookupIndexedStore(reshapeOp.getShape(), dim))
        return ensureIndexValue(*stored, rewriter, loc);

      Value index = makeIndex(dim);
      Value loaded = rewriter.create<memref::LoadOp>(
          loc, reshapeOp.getShape(), ValueRange({index}));
      return ensureIndexValue(loaded, rewriter, loc);
    }

    if (auto expandOp = source.getDefiningOp<memref::ExpandShapeOp>()) {
      ArrayRef<int64_t> staticOutputShape = expandOp.getStaticOutputShape();
      if (dim >= static_cast<int64_t>(staticOutputShape.size()))
        return failure();
      int64_t staticDim = staticOutputShape[dim];
      if (!ShapedType::isDynamic(staticDim))
        return makeIndex(staticDim);

      unsigned dynamicIndex = 0;
      for (int64_t i = 0; i < dim; ++i)
        if (ShapedType::isDynamic(staticOutputShape[i]))
          ++dynamicIndex;
      auto outputShape = expandOp.getOutputShape();
      if (dynamicIndex >= outputShape.size())
        return failure();
      return ensureIndexValue(outputShape[dynamicIndex], rewriter, loc);
    }

    if (auto collapseOp = source.getDefiningOp<memref::CollapseShapeOp>()) {
      SmallVector<SmallVector<int64_t>> groups =
          parseReassociationIndices(collapseOp.getReassociation());
      if (dim >= static_cast<int64_t>(groups.size()) || groups[dim].empty())
        return failure();

      Value product = makeIndex(1);
      for (int64_t sourceDim : groups[dim]) {
        FailureOr<Value> extent =
            resolveDim(collapseOp.getSrc(), sourceDim, depth + 1);
        if (failed(extent))
          return failure();
        product = mul(product, *extent);
      }
      return product;
    }

    if (auto subViewOp = source.getDefiningOp<memref::SubViewOp>()) {
      ArrayRef<int64_t> staticSizes = subViewOp.getStaticSizes();
      if (dim >= static_cast<int64_t>(staticSizes.size()))
        return failure();
      int64_t staticSize = staticSizes[dim];
      if (!ShapedType::isDynamic(staticSize))
        return makeIndex(staticSize);

      unsigned dynamicIndex = 0;
      for (int64_t i = 0; i < dim; ++i)
        if (ShapedType::isDynamic(staticSizes[i]))
          ++dynamicIndex;
      auto sizes = subViewOp.getSizes();
      if (dynamicIndex >= sizes.size())
        return failure();
      return ensureIndexValue(sizes[dynamicIndex], rewriter, loc);
    }

    return failure();
  };

  SmallVector<memref::DimOp> dimOps;
  moduleOp->walk([&](memref::DimOp op) { dimOps.push_back(op); });
  for (memref::DimOp op : dimOps) {
    std::optional<int64_t> dim = getConstantIndexValue(op.getIndex());
    if (!dim)
      continue;
    rewriter.setInsertionPoint(op);
    FailureOr<Value> resolved = resolveDim(op.getSource(), *dim, /*depth=*/0);
    if (failed(resolved))
      continue;
    rewriter.replaceOp(op, *resolved);
  }

  // memref.reshape is a view-like operation. PyAsc has printers for
  // emitasc.reinterpret_cast but not memref.reshape, so materialize the view as
  // an explicit pointer reinterpretation before emission.
  moduleOp->walk([&](memref::ReshapeOp op) {
    rewriter.setInsertionPoint(op);
    auto castOp = rewriter.create<emitasc::ReinterpretCastOp>(
        op.getLoc(), op.getResult().getType(), peelSourceValue(op.getSource()));
    rewriter.replaceOp(op, castOp.getResult());
  });

  auto lowerShapeViewToReinterpret = [&](auto op) {
    rewriter.setInsertionPoint(op);
    auto castOp = rewriter.create<emitasc::ReinterpretCastOp>(
        op.getLoc(), op.getResult().getType(), peelSourceValue(op.getSrc()));
    rewriter.replaceOp(op, castOp.getResult());
  };
  moduleOp->walk([&](memref::ExpandShapeOp op) {
    lowerShapeViewToReinterpret(op);
  });
  moduleOp->walk([&](memref::CollapseShapeOp op) {
    lowerShapeViewToReinterpret(op);
  });
  moduleOp->walk([&](memref::SubViewOp op) {
    if (hasOnlyZeroStaticOffsets(op)) {
      rewriter.setInsertionPoint(op);
      auto castOp = rewriter.create<emitasc::ReinterpretCastOp>(
          op.getLoc(), op.getResult().getType(),
          peelSourceValue(op.getSource()));
      rewriter.replaceOp(op, castOp.getResult());
      return;
    }

    rewriter.setInsertionPoint(op);
    FailureOr<Value> linearOffset = buildStaticSubViewLinearOffset(op, rewriter);
    if (failed(linearOffset))
      return;
    auto offsetOp = rewriter.create<emitasc::PtrOffsetOp>(
        op.getLoc(), op.getResult().getType(), peelSourceValue(op.getSource()),
        /*staticOffset=*/IntegerAttr{}, /*dynamicOffset=*/*linearOffset);
    rewriter.replaceOp(op, offsetOp.getResult());
  });

  auto lowerIntegerMinMax = [&](auto op, arith::CmpIPredicate predicate) {
    rewriter.setInsertionPoint(op);
    Value cmp = rewriter.create<arith::CmpIOp>(
        op.getLoc(), predicate, op.getLhs(), op.getRhs());
    Value selected = rewriter.create<arith::SelectOp>(
        op.getLoc(), cmp, op.getLhs(), op.getRhs());
    rewriter.replaceOp(op, selected);
  };
  moduleOp->walk([&](arith::MaxSIOp op) {
    lowerIntegerMinMax(op, arith::CmpIPredicate::sgt);
  });
  moduleOp->walk([&](arith::MaxUIOp op) {
    lowerIntegerMinMax(op, arith::CmpIPredicate::ugt);
  });
  moduleOp->walk([&](arith::MinSIOp op) {
    lowerIntegerMinMax(op, arith::CmpIPredicate::slt);
  });
  moduleOp->walk([&](arith::MinUIOp op) {
    lowerIntegerMinMax(op, arith::CmpIPredicate::ult);
  });

  // AscendC vector Add does not reliably consume a VECCALC tensor that was
  // populated directly from GM in the simulator. For this narrow gather+bias
  // pattern, load each bias scalar from GM and apply it with Adds on a
  // one-element LocalTensor slice so the local source still stays on the vector
  // path.
  moduleOp->walk([&](ascendc::AddL2Op op) {
    Value gmSource =
        findLocalTensorDataCopyGlobalSourceBefore(op, op.getSrc1());
    Value localSource = op.getSrc0();
    if (!gmSource) {
      gmSource = findLocalTensorDataCopyGlobalSourceBefore(op, op.getSrc0());
      localSource = op.getSrc1();
    }
    if (!gmSource)
      return;

    auto dstType = dyn_cast<ascendc::LocalTensorType>(op.getDst().getType());
    if (!dstType)
      return;

    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    std::string elemTypeStr = getAscendCScalarTypeName(dstType.getElementType());
    std::string tmpl = "{\n";
    tmpl += "  for (uint32_t _afir_i = 0; _afir_i < static_cast<uint32_t>($2); ++_afir_i) {\n";
    tmpl += "    " + elemTypeStr + " _afir_r = $3.GetValue(_afir_i);\n";
    tmpl += "    AscendC::Adds($0[_afir_i], $1[_afir_i], _afir_r, (int32_t)1);\n";
    tmpl += "  }\n";
    tmpl += "  $0.SetSize((uint32_t)$2);\n}";
    rewriter.create<emitasc::VerbatimOp>(
        loc, rewriter.getStringAttr(tmpl),
        ValueRange({op.getDst(), localSource, op.getCalCount(), gmSource}));
    rewriter.eraseOp(op);
  });

  // DataCopyL2Op with GlobalTensorBracketOp source → verbatim
  //
  // PyAsc emits `GlobalTensor<T> row = base(offset);`, but AscendC's operator()
  // returns an element pointer/value rather than a sliced GlobalTensor. Rebuild a
  // temporary GlobalTensor from `GetPhyAddr(offset)` instead.  MTE DataCopy is
  // 32-byte block oriented, so dynamic tails fall back to scalar GM loads.
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
    tmpl += "  uint32_t _afir_count = (uint32_t)$3;\n";
    tmpl += "  if ((_afir_count * sizeof(" + elemTypeStr + ")) % 32u == 0u) {\n";
    tmpl += "    AscendC::DataCopy($0, _afir_gt, _afir_count);\n";
    tmpl += "  } else {\n";
    tmpl += "    for (uint32_t _afir_i = 0; _afir_i < _afir_count; ++_afir_i)\n";
    tmpl += "      $0.SetValue(_afir_i, _afir_gt.GetValue(_afir_i));\n";
    tmpl += "  }\n";
    tmpl += "  $0.SetSize(_afir_count);\n}";
    rewriter.create<emitasc::VerbatimOp>(
        loc, rewriter.getStringAttr(tmpl),
        ValueRange({op.getDst(), bracketOp.getTensor(), bracketOp.getIndex(),
                    op.getCalCount()}));
    rewriter.eraseOp(op);
    if (bracketOp->use_empty())
      rewriter.eraseOp(bracketOp);
  });

  // DataCopyL2Op from GlobalTensor to LocalTensor must leave the local tensor
  // with an explicit element count for subsequent vector ops and scalar reads.
  // Non-32B dynamic tails use scalar GM loads instead of MTE DataCopy.
  moduleOp->walk([&](ascendc::DataCopyL2Op op) {
    if (!isa<ascendc::LocalTensorType>(op.getDst().getType()) ||
        !isa<ascendc::GlobalTensorType>(op.getSrc().getType()))
      return;

    auto dstType = cast<ascendc::LocalTensorType>(op.getDst().getType());
    std::string elemTypeStr = getAscendCScalarTypeName(dstType.getElementType());
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    std::string tmpl = "{\n";
    tmpl += "  uint32_t _afir_count = (uint32_t)$2;\n";
    tmpl += "  if ((_afir_count * sizeof(" + elemTypeStr + ")) % 32u == 0u) {\n";
    tmpl += "    AscendC::DataCopy($0, $1, _afir_count);\n";
    tmpl += "  } else {\n";
    tmpl += "    for (uint32_t _afir_i = 0; _afir_i < _afir_count; ++_afir_i)\n";
    tmpl += "      $0.SetValue(_afir_i, $1.GetValue(_afir_i));\n";
    tmpl += "  }\n";
    tmpl += "  $0.SetSize(_afir_count);\n}";
    rewriter.create<emitasc::VerbatimOp>(
        loc, rewriter.getStringAttr(tmpl),
        ValueRange({op.getDst(), op.getSrc(), op.getCalCount()}));
    rewriter.eraseOp(op);
  });

  // DataCopyL2Op from LocalTensor to GlobalTensor mirrors the same tail policy:
  // aligned full blocks use MTE DataCopy, scalar tails avoid over-writing GM.
  moduleOp->walk([&](ascendc::DataCopyL2Op op) {
    if (!isa<ascendc::GlobalTensorType>(op.getDst().getType()) ||
        !isa<ascendc::LocalTensorType>(op.getSrc().getType()))
      return;

    auto srcType = cast<ascendc::LocalTensorType>(op.getSrc().getType());
    std::string elemTypeStr = getAscendCScalarTypeName(srcType.getElementType());
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    std::string tmpl = "{\n";
    tmpl += "  uint32_t _afir_count = (uint32_t)$2;\n";
    tmpl += "  if ((_afir_count * sizeof(" + elemTypeStr + ")) % 32u == 0u) {\n";
    tmpl += "    AscendC::DataCopy($0, $1, _afir_count);\n";
    tmpl += "  } else {\n";
    tmpl += "    for (uint32_t _afir_i = 0; _afir_i < _afir_count; ++_afir_i)\n";
    tmpl += "      $0.SetValue(_afir_i, $1.GetValue(_afir_i));\n";
    tmpl += "  }\n}";
    rewriter.create<emitasc::VerbatimOp>(
        loc, rewriter.getStringAttr(tmpl),
        ValueRange({op.getDst(), op.getSrc(), op.getCalCount()}));
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
    std::string elemTypeStr = getAscendCScalarTypeName(dstElemType);
    Value gmScalarSource =
        (rank == 2 && axis == 1) ? findQueuedDataCopyGlobalSource(op.getSrc())
                                 : Value{};
    if (gmScalarSource) {
      unsigned gmOperand = 2 + (2 * rank);
      tmpl += "  if (_afir_ds[0] < 16u && _afir_ss[1] == 1u) {\n";
      tmpl += "    " + elemTypeStr + " _afir_zero = 0;\n";
      tmpl += "    AscendC::Duplicate($0, _afir_zero, _afir_ds[0] * _afir_ds[1]);\n";
      tmpl += "    for (uint32_t _afir_r = 0; _afir_r < _afir_ds[0]; ++_afir_r) {\n";
      tmpl += "      auto _afir_v = $" + std::to_string(gmOperand) +
              ".GetValue(_afir_r);\n";
      tmpl += "      AscendC::Adds($0[_afir_r * _afir_ds[1]], "
              "$0[_afir_r * _afir_ds[1]], _afir_v, (int32_t)_afir_ds[1]);\n";
      tmpl += "    }\n";
      tmpl += "  } else {\n";
      tmpl += "    AscendC::Broadcast<" + elemTypeStr + ", " +
              std::to_string(rank) + ", " + std::to_string(axis) +
              ">($0, $1, _afir_ds, _afir_ss);\n";
      tmpl += "  }\n}";
    } else {
      tmpl += "  AscendC::Broadcast<" + elemTypeStr + ", " +
              std::to_string(rank) + ", " + std::to_string(axis) +
              ">($0, $1, _afir_ds, _afir_ss);\n}";
    }

    SmallVector<Value> args;
    args.push_back(op.getDst());
    args.push_back(op.getSrc());
    for (Value v : op.getDstShape())
      args.push_back(v);
    for (Value v : op.getSrcShape())
      args.push_back(v);
    if (gmScalarSource)
      args.push_back(gmScalarSource);

    rewriter.create<emitasc::VerbatimOp>(
        loc, rewriter.getStringAttr(tmpl), ValueRange(args));
    rewriter.eraseOp(op);
  });

  // GatherL2Op with i64 element indices -> byte offsets -> verbatim
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
    auto sourceType = dyn_cast<ascendc::LocalTensorType>(op.getSrc().getType());
    if (!sourceType)
      return;
    unsigned sourceElemBytes =
        sourceType.getElementType().getIntOrFloatBitWidth() / 8;

    Value pipeVal;
    if (auto funcOp = op->getParentOfType<func::FuncOp>()) {
      funcOp.walk([&](ascendc::PipeOp pipeOp) {
        pipeVal = pipeOp.getResult();
        return WalkResult::interrupt();
      });
    }
    if (!pipeVal)
      return;
    Value sourceElementCount =
        findLocalTensorDataCopyCountBefore(op.getOperation(), op.getSrc());
    std::string sourceSetSizeExpr = "$4";
    if (!sourceElementCount) {
      sourceElementCount = findLocalTensorByteLength(op.getSrc());
      if (sourceElementCount)
        sourceSetSizeExpr =
            "($4 / " + std::to_string(sourceElemBytes) + "u)";
    }
    if (!sourceElementCount)
      sourceElementCount = op.getCount();

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
    prelude += "uint32_t _afir_idx32_bytes = (uint32_t)$1 * sizeof(uint32_t);\n";
    prelude +=
        "uint32_t _afir_idx32_aligned_bytes = _afir_idx32_bytes == 0 ? 0 : "
        "((_afir_idx32_bytes + 31u) / 32u) * 32u;\n";
    prelude +=
        "if (_afir_idx32_bytes != 0u && _afir_idx32_aligned_bytes < 32u)\n";
    prelude += "  _afir_idx32_aligned_bytes = 32u;\n";
    prelude += "$0.InitBuffer(" + tbufName +
               ", _afir_idx32_aligned_bytes);\n";
    prelude += "AscendC::LocalTensor<uint32_t> " + tensorName + " = " +
               tbufName + ".Get<uint32_t>();\n";
    prelude += "$2.SetSize((uint32_t)$1);\n";
    prelude +=
        "for (uint32_t _afir_i = 0; _afir_i < static_cast<uint32_t>($1); _afir_i++) {\n";
    prelude += "  " + tensorName +
               ".SetValue(_afir_i, static_cast<uint32_t>($2.GetValue(_afir_i)) * " +
               std::to_string(srcElemBytes) + "u);\n";
    prelude += "}";
    prelude += "\n" + tensorName + ".SetSize((uint32_t)$1);";
    prelude += "\nAscendC::PipeBarrier<PIPE_V>()";
    rewriter.create<emitasc::VerbatimOp>(
        loc, rewriter.getStringAttr(prelude),
        ValueRange({pipeVal, op.getCount(), op.getSrcOffset()}));

    rewriter.setInsertionPoint(op);
    std::string tmpl = "{\n";
    tmpl += "  uint32_t _afir_gather_count = static_cast<uint32_t>($3);\n";
    tmpl += "  $0.SetSize(_afir_gather_count);\n";
    tmpl += "  $1.SetSize((uint32_t)" + sourceSetSizeExpr + ");\n";
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
                               sourceElementCount};
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
      auto srcElemType =
          cast<ascendc::LocalTensorType>(op.getSrc().getType()).getElementType();
      if (srcElemType.isF16()) {
        tmpl += "  uint32_t _afir_src_elems = _afir_rows * _afir_cols;\n";
        tmpl += "  AscendC::TBuf<AscendC::TPosition::VECCALC> _afir_src_f32_tbuf;\n";
        tmpl += "  AscendC::TBuf<AscendC::TPosition::VECCALC> _afir_dst_f32_tbuf;\n";
        tmpl += "  " + pipeRef + ".InitBuffer(_afir_src_f32_tbuf, _afir_src_elems * sizeof(float));\n";
        tmpl += "  " + pipeRef + ".InitBuffer(_afir_dst_f32_tbuf, _afir_rows * sizeof(float));\n";
        tmpl += "  AscendC::LocalTensor<float> _afir_src_f32 = _afir_src_f32_tbuf.Get<float>();\n";
        tmpl += "  AscendC::LocalTensor<float> _afir_dst_f32 = _afir_dst_f32_tbuf.Get<float>();\n";
        tmpl += "  AscendC::Cast(_afir_src_f32, $1, AscendC::RoundMode::CAST_NONE,\n";
        tmpl += "                _afir_src_elems);\n";
        tmpl += "  uint32_t _afir_shape[2] = {_afir_rows, _afir_cols};\n";
        tmpl += "  AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(\n";
        tmpl += "      _afir_dst_f32, _afir_src_f32, _afir_shape, false);\n";
        tmpl += "  AscendC::Cast($0, _afir_dst_f32, AscendC::RoundMode::CAST_NONE,\n";
        tmpl += "                _afir_rows);\n";
      } else {
        tmpl += "  uint32_t _afir_shape[2] = {_afir_rows, _afir_cols};\n";
        tmpl += "  AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(\n";
        tmpl += "      $0, $1, _afir_shape, false);\n";
      }
      tmpl += "}";
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

  auto alignedByteLengthBlock = [](unsigned lengthOperand) {
    std::string tmpl = "{\n";
    tmpl += "  uint32_t _afir_bytes = static_cast<uint32_t>($" +
            std::to_string(lengthOperand) + ");\n";
    tmpl += "  uint32_t _afir_aligned_bytes = _afir_bytes == 0 ? 0 : "
            "((_afir_bytes + 31u) / 32u) * 32u;\n";
    tmpl += "  if (_afir_aligned_bytes < 32u)\n";
    tmpl += "    _afir_aligned_bytes = 32u;\n";
    return tmpl;
  };

  // TPipe buffer sizes are physical byte capacities. Keep IR lengths logical
  // for shape reasoning, but round emitted runtime allocations to AscendC's
  // 32-byte minimum/alignment so small tail tiles have valid LocalTensors.
  moduleOp->walk([&](ascendc::TPipeInitBufferOp op) {
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    std::string tmpl = alignedByteLengthBlock(/*lengthOperand=*/2);
    tmpl += "  $0.InitBuffer($1, _afir_aligned_bytes);\n}";
    rewriter.create<emitasc::VerbatimOp>(
        loc, rewriter.getStringAttr(tmpl),
        ValueRange({op.getPipe(), op.getBuffer(), op.getLength()}));
    rewriter.eraseOp(op);
  });

  moduleOp->walk([&](ascendc::TPipeInitQueueOp op) {
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    std::string tmpl = alignedByteLengthBlock(/*lengthOperand=*/3);
    tmpl += "  $0.InitBuffer($1, $2, _afir_aligned_bytes);\n}";
    rewriter.create<emitasc::VerbatimOp>(
        loc, rewriter.getStringAttr(tmpl),
        ValueRange({op.getPipe(), op.getQueue(), op.getNum(),
                    op.getLength()}));
    rewriter.eraseOp(op);
  });
}

static LogicalResult
emitRequestedRuntimeArtifacts(ModuleOp moduleOp,
                              const CannTranslationOptions &options) {
  afir::cann::CannRuntimeArtifactOptions artifactOptions;
  artifactOptions.kernelFile = options.kernelFile;
  artifactOptions.soc = options.soc;

  if (!options.tilingSpaceOutPath.empty() &&
      failed(afir::cann::emitTilingSpaceJson(
          moduleOp, options.tilingSpaceOutPath, artifactOptions)))
    return failure();
  if (!options.runtimeManifestOutPath.empty() &&
      failed(afir::cann::emitRuntimeManifestJson(
          moduleOp, options.runtimeManifestOutPath, artifactOptions)))
    return failure();
  if (!options.hostTilingOutPath.empty() &&
      failed(afir::cann::emitHostTilingCpp(
          moduleOp, options.hostTilingOutPath, artifactOptions)))
    return failure();

  return success();
}

LogicalResult mlir::translateToCannKernel(Operation *op, raw_ostream &os,
                                          const CannTranslationOptions &options) {
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
            return emitRequestedRuntimeArtifacts(moduleOp, options);
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
            return emitRequestedRuntimeArtifacts(moduleOp, options);
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
          return emitRequestedRuntimeArtifacts(moduleOp, options);
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
          return emitRequestedRuntimeArtifacts(moduleOp, options);
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
  deduplicateConstantsForEmission(op);

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

  return emitRequestedRuntimeArtifacts(moduleOp, options);
}

LogicalResult mlir::translateToCannKernel(Operation *op, raw_ostream &os,
                                          StringRef tilingSpaceOutPath,
                                          StringRef kernelFile) {
  CannTranslationOptions options;
  options.tilingSpaceOutPath = tilingSpaceOutPath;
  options.kernelFile = kernelFile;
  return translateToCannKernel(op, os, options);
}
