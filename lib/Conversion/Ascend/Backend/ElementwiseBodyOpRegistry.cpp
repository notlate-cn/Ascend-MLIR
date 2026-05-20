//===- ElementwiseBodyOpRegistry.cpp - Elementwise body op registry -------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Backend/ElementwiseBodyOpRegistry.h"

#include "ascir/Dialect/Asc/IR/Asc.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ManagedStatic.h"

#include <mutex>

namespace mlir::afir::ascend::backend {
namespace {

struct RegistrySingleton {
  std::mutex mu;
  llvm::StringMap<ElementwiseBodyOpEntry> entries;
  bool builtinsRegistered = false;
};

llvm::ManagedStatic<RegistrySingleton> gRegistry;

} // namespace

void registerElementwiseBodyOp(ElementwiseBodyOpEntry entry) {
  auto &reg = *gRegistry;
  std::lock_guard<std::mutex> lock(reg.mu);
  reg.entries.try_emplace(entry.dialectOpName, std::move(entry));
}

void registerBuiltinElementwiseBodyOps() {
  auto &reg = *gRegistry;
  std::lock_guard<std::mutex> lock(reg.mu);
  if (reg.builtinsRegistered)
    return;
  using namespace mlir::ascendc;
  auto add = [&](ElementwiseBodyOpEntry entry) {
    reg.entries.try_emplace(entry.dialectOpName, std::move(entry));
  };

  // Binary ops
  add({"arith.addf", ComputeKind::ElementwiseAdd,
    nullptr,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src0, mlir::Value src1, mlir::Value cnt) {
      b.create<AddL2Op>(loc, dst, src0, src1, cnt);
    }});
  add({"arith.mulf", ComputeKind::ElementwiseMul,
    nullptr,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src0, mlir::Value src1, mlir::Value cnt) {
      b.create<MulL2Op>(loc, dst, src0, src1, cnt);
    }});
  add({"arith.maximumf", ComputeKind::ElementwiseMax,
    nullptr,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src0, mlir::Value src1, mlir::Value cnt) {
      b.create<MaxL2Op>(loc, dst, src0, src1, cnt);
    }});
  add({"arith.minimumf", ComputeKind::ElementwiseMin,
    nullptr,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src0, mlir::Value src1, mlir::Value cnt) {
      b.create<MinL2Op>(loc, dst, src0, src1, cnt);
    }});
  add({"arith.subf", ComputeKind::ElementwiseSub,
    nullptr,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src0, mlir::Value src1, mlir::Value cnt) {
      b.create<SubL2Op>(loc, dst, src0, src1, cnt);
    }});
  add({"arith.divf", ComputeKind::ElementwiseDiv,
    nullptr,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src0, mlir::Value src1, mlir::Value cnt) {
      b.create<DivL2Op>(loc, dst, src0, src1, cnt);
    }});

  // Unary ops
  add({"arith.negf", ComputeKind::ElementwiseNeg,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src, mlir::Value cnt) {
      b.create<NegL2Op>(loc, dst, src, cnt);
    }, nullptr});
  add({"math.exp", ComputeKind::ElementwiseExp,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src, mlir::Value cnt) {
      b.create<ExpL2Op>(loc, dst, src, cnt);
    }, nullptr});
  add({"math.log", ComputeKind::ElementwiseLog,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src, mlir::Value cnt) {
      b.create<LnL2Op>(loc, dst, src, cnt);
    }, nullptr});
  add({"math.sqrt", ComputeKind::ElementwiseSqrt,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src, mlir::Value cnt) {
      b.create<SqrtL2Op>(loc, dst, src, cnt);
    }, nullptr});
  add({"math.rsqrt", ComputeKind::ElementwiseRsqrt,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src, mlir::Value cnt) {
      b.create<RsqrtL2Op>(loc, dst, src, cnt);
    }, nullptr});
  add({"math.absf", ComputeKind::ElementwiseAbs,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src, mlir::Value cnt) {
      b.create<AbsL2Op>(loc, dst, src, cnt);
    }, nullptr});
  reg.builtinsRegistered = true;
}

const ElementwiseBodyOpEntry *
lookupElementwiseBodyOp(llvm::StringRef dialectOpName) {
  if (dialectOpName.empty())
    return nullptr;
  registerBuiltinElementwiseBodyOps();
  auto &reg = *gRegistry;
  std::lock_guard<std::mutex> lock(reg.mu);
  auto it = reg.entries.find(dialectOpName);
  return it != reg.entries.end() ? &it->second : nullptr;
}

} // namespace mlir::afir::ascend::backend
