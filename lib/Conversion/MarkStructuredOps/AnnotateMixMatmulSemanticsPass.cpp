//===- AnnotateMixMatmulSemanticsPass.cpp - Stable mix matmul attrs -------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/MarkStructuredOps/MarkStructuredOpsPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#define GEN_PASS_DECL_ANNOTATEMIXMATMULSEMANTICSPASS
#define GEN_PASS_DEF_ANNOTATEMIXMATMULSEMANTICSPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

namespace {

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

static bool hasUnitAttr(Operation *op, StringRef expected) {
  auto unitAttr = op->getAttrOfType<StringAttr>("ascendc.unit");
  return unitAttr && unitAttr.getValue() == expected;
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
      !hasUnitAttr(genericOp, "AiCore.Vector"))
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

static bool isLeakyReluGeneric(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1 ||
      !isParallelGeneric(genericOp) ||
      !hasUnitAttr(genericOp, "AiCore.Vector"))
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

static bool isSimple2DNdMatmul(linalg::MatmulOp matmulOp) {
  return hasUnitAttr(matmulOp, "AiCore.Cube") &&
         isIdentity2DMemRef(matmulOp.getDpsInputOperand(0)->get()) &&
         isIdentity2DMemRef(matmulOp.getDpsInputOperand(1)->get()) &&
         isIdentity2DMemRef(matmulOp.getDpsInitOperand(0)->get());
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

} // namespace

struct AnnotateMixMatmulSemanticsPass
    : public ::impl::AnnotateMixMatmulSemanticsPassBase<
          AnnotateMixMatmulSemanticsPass> {
  using Base =
      ::impl::AnnotateMixMatmulSemanticsPassBase<
          AnnotateMixMatmulSemanticsPass>;
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    auto kernelKind = funcOp->getAttrOfType<StringAttr>("ascendc.kernel_kind");
    if (!kernelKind || kernelKind.getValue() != "mix")
      return;
    if (funcOp->hasAttr("abi_matmul_op_kind"))
      return;

    SmallVector<linalg::MatmulOp> matmuls;
    SmallVector<linalg::GenericOp> generics;
    funcOp.walk([&](Operation *op) {
      if (auto matmul = dyn_cast<linalg::MatmulOp>(op))
        matmuls.push_back(matmul);
      else if (auto generic = dyn_cast<linalg::GenericOp>(op))
        generics.push_back(generic);
    });

    if (matmuls.size() != 1)
      return;
    linalg::MatmulOp matmulOp = matmuls.front();
    if (!isSimple2DNdMatmul(matmulOp))
      return;

    Value current = matmulOp.getDpsInitOperand(0)->get();
    linalg::GenericOp biasGeneric =
        findUniqueChainedGeneric(current, generics, isBiasAddGeneric);
    bool hasBias = static_cast<bool>(biasGeneric);
    if (biasGeneric)
      current = biasGeneric.getDpsInitOperand(0)->get();

    linalg::GenericOp leakyReluGeneric =
        findUniqueChainedGeneric(current, generics, isLeakyReluGeneric);
    bool hasLeakyRelu = static_cast<bool>(leakyReluGeneric);

    StringRef epilogueKind = "None";
    if (hasBias && hasLeakyRelu)
      epilogueKind = "BiasAddLeakyRelu";
    else if (hasBias)
      epilogueKind = "BiasAdd";
    else if (hasLeakyRelu)
      return;

    MLIRContext *ctx = funcOp.getContext();
    funcOp->setAttr("abi_matmul_op_kind", StringAttr::get(ctx, "matmul"));
    funcOp->setAttr("abi_matmul_trans_a", BoolAttr::get(ctx, false));
    funcOp->setAttr("abi_matmul_trans_b", BoolAttr::get(ctx, false));
    funcOp->setAttr("abi_matmul_has_bias", BoolAttr::get(ctx, hasBias));
    funcOp->setAttr("abi_matmul_layout_a", StringAttr::get(ctx, "ND"));
    funcOp->setAttr("abi_matmul_layout_b", StringAttr::get(ctx, "ND"));
    funcOp->setAttr("abi_matmul_layout_c", StringAttr::get(ctx, "ND"));
    funcOp->setAttr("abi_matmul_epilogue_kind",
                    StringAttr::get(ctx, epilogueKind));
  }
};

std::unique_ptr<Pass> createAnnotateMixMatmulSemanticsPass() {
  return std::make_unique<AnnotateMixMatmulSemanticsPass>();
}

} // namespace mlir::afir
