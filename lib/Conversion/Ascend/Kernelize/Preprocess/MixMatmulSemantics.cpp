//===- MixMatmulSemantics.cpp - Stable mix matmul attrs -------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/KernelizeInternalPasses.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <optional>
#include <string>
#include <vector>

using namespace mlir;

namespace mlir::ascend {

namespace {

struct MatmulLikeOpInfo {
  Value lhs;
  Value rhs;
  Value out;
  Operation *op = nullptr;
  std::string opKind = "matmul";
  bool transA = false;
  bool transB = false;
  std::vector<int64_t> batchShape;
};

static MemRefType getRankedMemRefType(Value value, int64_t rank) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || !type.hasRank() || type.getRank() != rank)
    return {};
  return type;
}

static bool isRankedMemRef(Value value, int64_t rank) {
  return static_cast<bool>(getRankedMemRefType(value, rank));
}

static bool isIdentity2DMemRef(Value value) {
  auto type = getRankedMemRefType(value, 2);
  return type && type.getLayout().isIdentity();
}

static bool isIdentity3DMemRef(Value value) {
  auto type = getRankedMemRefType(value, 3);
  return type && type.getLayout().isIdentity();
}

static std::optional<int64_t> getStaticBatchDim(Value value) {
  auto type = getRankedMemRefType(value, 3);
  if (!type)
    return std::nullopt;
  const int64_t batch = type.getShape().front();
  if (batch == ShapedType::kDynamic || batch <= 0)
    return std::nullopt;
  return batch;
}

static bool hasUnitAttr(Operation *op, StringRef expected) {
  auto unitAttr = op->getAttrOfType<StringAttr>(ascend::kAscendCUnitAttr);
  return unitAttr && unitAttr.getValue() == expected;
}

static std::optional<MatmulLikeOpInfo> getMatmulLikeOpInfo(Operation *op) {
  if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
    return MatmulLikeOpInfo{
        matmul.getDpsInputOperand(0)->get(),
        matmul.getDpsInputOperand(1)->get(),
        matmul.getDpsInitOperand(0)->get(),
        matmul,
        "matmul",
        false,
        false,
        {},
    };
  }
  if (auto matmul = dyn_cast<linalg::MatmulTransposeAOp>(op)) {
    return MatmulLikeOpInfo{
        matmul.getDpsInputOperand(0)->get(),
        matmul.getDpsInputOperand(1)->get(),
        matmul.getDpsInitOperand(0)->get(),
        matmul,
        "matmul",
        true,
        false,
        {},
    };
  }
  if (auto matmul = dyn_cast<linalg::MatmulTransposeBOp>(op)) {
    return MatmulLikeOpInfo{
        matmul.getDpsInputOperand(0)->get(),
        matmul.getDpsInputOperand(1)->get(),
        matmul.getDpsInitOperand(0)->get(),
        matmul,
        "matmul",
        false,
        true,
        {},
    };
  }
  if (auto matmul = dyn_cast<linalg::BatchMatmulOp>(op)) {
    auto batch = getStaticBatchDim(matmul.getDpsInitOperand(0)->get());
    std::vector<int64_t> batchShape;
    if (batch)
      batchShape.push_back(*batch);
    return MatmulLikeOpInfo{
        matmul.getDpsInputOperand(0)->get(),
        matmul.getDpsInputOperand(1)->get(),
        matmul.getDpsInitOperand(0)->get(),
        matmul,
        "batch_matmul",
        false,
        false,
        std::move(batchShape),
    };
  }
  if (auto matmul = dyn_cast<linalg::BatchMatmulTransposeAOp>(op)) {
    auto batch = getStaticBatchDim(matmul.getDpsInitOperand(0)->get());
    std::vector<int64_t> batchShape;
    if (batch)
      batchShape.push_back(*batch);
    return MatmulLikeOpInfo{
        matmul.getDpsInputOperand(0)->get(),
        matmul.getDpsInputOperand(1)->get(),
        matmul.getDpsInitOperand(0)->get(),
        matmul,
        "batch_matmul",
        true,
        false,
        std::move(batchShape),
    };
  }
  if (auto matmul = dyn_cast<linalg::BatchMatmulTransposeBOp>(op)) {
    auto batch = getStaticBatchDim(matmul.getDpsInitOperand(0)->get());
    std::vector<int64_t> batchShape;
    if (batch)
      batchShape.push_back(*batch);
    return MatmulLikeOpInfo{
        matmul.getDpsInputOperand(0)->get(),
        matmul.getDpsInputOperand(1)->get(),
        matmul.getDpsInitOperand(0)->get(),
        matmul,
        "batch_matmul",
        false,
        true,
        std::move(batchShape),
    };
  }
  return std::nullopt;
}

static bool isParallelGeneric(linalg::GenericOp genericOp) {
  return llvm::all_of(genericOp.getIteratorTypesArray(),
                      [](utils::IteratorType type) {
                        return type == utils::IteratorType::parallel;
                      });
}

static bool isBiasAddGeneric(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1 ||
      !isParallelGeneric(genericOp) ||
      !hasUnitAttr(genericOp, ascend::kAscendCUnitVector))
    return false;
  if (!isRankedMemRef(genericOp.getDpsInputOperand(0)->get(), 2) ||
      !isRankedMemRef(genericOp.getDpsInputOperand(1)->get(), 1) ||
      !isRankedMemRef(genericOp.getDpsInitOperand(0)->get(), 2))
    return false;

  Block &body = genericOp.getRegion().front();
  if (body.getNumArguments() != 3)
    return false;

  arith::AddFOp addOp;
  linalg::YieldOp yieldOp;
  for (Operation &op : body.getOperations()) {
    if (auto add = dyn_cast<arith::AddFOp>(op)) {
      if (addOp)
        return false;
      addOp = add;
      continue;
    }
    if (auto yield = dyn_cast<linalg::YieldOp>(op)) {
      yieldOp = yield;
      continue;
    }
    return false;
  }
  return addOp && yieldOp && yieldOp.getNumOperands() == 1 &&
         yieldOp.getOperand(0) == addOp.getResult();
}

static bool isFullRank3AddGeneric(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1 ||
      !isParallelGeneric(genericOp) ||
      !hasUnitAttr(genericOp, ascend::kAscendCUnitVector))
    return false;
  if (!isRankedMemRef(genericOp.getDpsInputOperand(0)->get(), 3) ||
      !isRankedMemRef(genericOp.getDpsInputOperand(1)->get(), 3) ||
      !isRankedMemRef(genericOp.getDpsInitOperand(0)->get(), 3))
    return false;

  Block &body = genericOp.getRegion().front();
  if (body.getNumArguments() != 3)
    return false;

  arith::AddFOp addOp;
  linalg::YieldOp yieldOp;
  for (Operation &op : body.getOperations()) {
    if (auto add = dyn_cast<arith::AddFOp>(op)) {
      if (addOp)
        return false;
      addOp = add;
      continue;
    }
    if (auto yield = dyn_cast<linalg::YieldOp>(op)) {
      yieldOp = yield;
      continue;
    }
    return false;
  }
  return addOp && yieldOp && yieldOp.getNumOperands() == 1 &&
         yieldOp.getOperand(0) == addOp.getResult();
}

static bool isRank3ByRank1BiasAddGeneric(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1 ||
      !isParallelGeneric(genericOp) ||
      !hasUnitAttr(genericOp, ascend::kAscendCUnitVector))
    return false;
  if (!isRankedMemRef(genericOp.getDpsInputOperand(0)->get(), 3) ||
      !isRankedMemRef(genericOp.getDpsInputOperand(1)->get(), 1) ||
      !isRankedMemRef(genericOp.getDpsInitOperand(0)->get(), 3))
    return false;

  Block &body = genericOp.getRegion().front();
  if (body.getNumArguments() != 3)
    return false;

  arith::AddFOp addOp;
  linalg::YieldOp yieldOp;
  for (Operation &op : body.getOperations()) {
    if (auto add = dyn_cast<arith::AddFOp>(op)) {
      if (addOp)
        return false;
      addOp = add;
      continue;
    }
    if (auto yield = dyn_cast<linalg::YieldOp>(op)) {
      yieldOp = yield;
      continue;
    }
    return false;
  }
  return addOp && yieldOp && yieldOp.getNumOperands() == 1 &&
         yieldOp.getOperand(0) == addOp.getResult();
}

static bool isLeakyReluGeneric(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1 ||
      !isParallelGeneric(genericOp) ||
      !hasUnitAttr(genericOp, ascend::kAscendCUnitVector))
    return false;
  if (!isRankedMemRef(genericOp.getDpsInputOperand(0)->get(), 2) ||
      !isRankedMemRef(genericOp.getDpsInitOperand(0)->get(), 2))
    return false;

  Block &body = genericOp.getRegion().front();
  if (body.getNumArguments() != 2)
    return false;

  arith::MulFOp mulOp;
  arith::MaximumFOp maxOp;
  linalg::YieldOp yieldOp;
  for (Operation &op : body.getOperations()) {
    if (auto mul = dyn_cast<arith::MulFOp>(op)) {
      if (mulOp)
        return false;
      mulOp = mul;
      continue;
    }
    if (auto max = dyn_cast<arith::MaximumFOp>(op)) {
      if (maxOp)
        return false;
      maxOp = max;
      continue;
    }
    if (auto yield = dyn_cast<linalg::YieldOp>(op)) {
      yieldOp = yield;
      continue;
    }
    if (!isa<arith::ConstantOp>(op))
      return false;
  }
  return mulOp && maxOp && yieldOp && yieldOp.getNumOperands() == 1 &&
         yieldOp.getOperand(0) == maxOp.getResult();
}

static bool isSimple2DNdMatmulLike(const MatmulLikeOpInfo &info) {
  if (!info.batchShape.empty())
    return false;
  return hasUnitAttr(info.op, ascend::kAscendCUnitCube) &&
         isIdentity2DMemRef(info.lhs) &&
         isIdentity2DMemRef(info.rhs) && isIdentity2DMemRef(info.out);
}

static bool isSimple3DNdBatchMatmulLike(const MatmulLikeOpInfo &info) {
  if (info.opKind != "batch_matmul")
    return false;
  return hasUnitAttr(info.op, ascend::kAscendCUnitCube) &&
         isIdentity3DMemRef(info.lhs) &&
         isIdentity3DMemRef(info.rhs) && isIdentity3DMemRef(info.out);
}

static linalg::GenericOp findUniqueChainedGeneric(Value current,
                                                  ArrayRef<linalg::GenericOp> generics,
                                                  function_ref<bool(linalg::GenericOp)> predicate) {
  linalg::GenericOp matched;
  for (linalg::GenericOp generic : generics) {
    if (generic.getDpsInputOperand(0)->get() != current || !predicate(generic))
      continue;
    if (matched)
      return {};
    matched = generic;
  }
  return matched;
}

static SmallVector<Value> collectCopyForwardedValues(Value source) {
  SmallVector<Value> values{source};
  llvm::SmallPtrSet<Value, 8> seen;
  seen.insert(source);
  for (unsigned index = 0; index < values.size(); ++index) {
    Value current = values[index];
    for (Operation *user : current.getUsers()) {
      auto copyOp = dyn_cast<memref::CopyOp>(user);
      if (!copyOp || copyOp.getSource() != current)
        continue;
      Value target = copyOp.getTarget();
      if (seen.insert(target).second)
        values.push_back(target);
    }
  }
  return values;
}

static linalg::GenericOp findUniqueChainedGenericThroughCopies(
    Value current, ArrayRef<linalg::GenericOp> generics,
    function_ref<bool(linalg::GenericOp)> predicate) {
  linalg::GenericOp matched;
  for (Value forwarded : collectCopyForwardedValues(current)) {
    linalg::GenericOp candidate =
        findUniqueChainedGeneric(forwarded, generics, predicate);
    if (!candidate)
      continue;
    if (matched && matched != candidate)
      return {};
    matched = candidate;
  }
  return matched;
}

} // namespace

LogicalResult annotateMixMatmulSemantics(func::FuncOp funcOp) {
  auto kernelKind =
      funcOp->getAttrOfType<StringAttr>(ascend::kAscendCKernelKindAttr);
  if (kernelKind && kernelKind.getValue() != ascend::kAscendCKernelKindMix)
    return success();
  if (funcOp->hasAttr("abi_matmul_op_kind"))
    return success();

  SmallVector<MatmulLikeOpInfo> matmuls;
  SmallVector<linalg::GenericOp> generics;
  funcOp.walk([&](Operation *op) {
    if (auto matmul = getMatmulLikeOpInfo(op))
      matmuls.push_back(*matmul);
    else if (auto generic = dyn_cast<linalg::GenericOp>(op))
      generics.push_back(generic);
  });

  if (matmuls.size() != 1)
    return success();
  const MatmulLikeOpInfo &matmulOp = matmuls.front();
  if (!isSimple2DNdMatmulLike(matmulOp) &&
      !isSimple3DNdBatchMatmulLike(matmulOp))
    return success();

  bool hasBias = false;
  bool hasLeakyRelu = false;
  StringRef epilogueKind = "None";

  if (matmulOp.opKind == "batch_matmul") {
    if (generics.size() > 1)
      return success();
    if (!generics.empty()) {
      linalg::GenericOp fullBiasAdd =
          findUniqueChainedGenericThroughCopies(matmulOp.out, generics,
                                                isFullRank3AddGeneric);
      linalg::GenericOp rank1BiasAdd =
          findUniqueChainedGenericThroughCopies(matmulOp.out, generics,
                                                isRank3ByRank1BiasAddGeneric);
      if (rank1BiasAdd) {
        hasBias = true;
        epilogueKind = "BiasAdd";
      } else if (!fullBiasAdd) {
        return success();
      }
    }
  } else {
    Value current = matmulOp.out;
    linalg::GenericOp biasGeneric =
        findUniqueChainedGenericThroughCopies(current, generics,
                                              isBiasAddGeneric);
    hasBias = static_cast<bool>(biasGeneric);
    if (biasGeneric)
      current = biasGeneric.getDpsInitOperand(0)->get();

    linalg::GenericOp leakyReluGeneric =
        findUniqueChainedGenericThroughCopies(current, generics,
                                              isLeakyReluGeneric);
    hasLeakyRelu = static_cast<bool>(leakyReluGeneric);

    if (hasBias && hasLeakyRelu)
      epilogueKind = "BiasAddLeakyRelu";
    else if (hasBias)
      epilogueKind = "BiasAdd";
    else if (hasLeakyRelu)
      return success();
  }

  MLIRContext *ctx = funcOp.getContext();
  funcOp->setAttr("abi_matmul_op_kind",
                  StringAttr::get(ctx, matmulOp.opKind));
  funcOp->setAttr("abi_matmul_trans_a", BoolAttr::get(ctx, matmulOp.transA));
  funcOp->setAttr("abi_matmul_trans_b", BoolAttr::get(ctx, matmulOp.transB));
  funcOp->setAttr("abi_matmul_has_bias", BoolAttr::get(ctx, hasBias));
  funcOp->setAttr("abi_matmul_layout_a", StringAttr::get(ctx, "ND"));
  funcOp->setAttr("abi_matmul_layout_b", StringAttr::get(ctx, "ND"));
  funcOp->setAttr("abi_matmul_layout_c", StringAttr::get(ctx, "ND"));
  funcOp->setAttr("abi_matmul_epilogue_kind",
                  StringAttr::get(ctx, epilogueKind));
  if (!matmulOp.batchShape.empty())
    funcOp->setAttr("abi_matmul_batch_shape",
                    Builder(ctx).getI64ArrayAttr(matmulOp.batchShape));
  return success();
}

} // namespace mlir::ascend
