//===- ElementwiseBodyOpRegistry.cpp - Elementwise body op registry -------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Translate/KernelIR/Capabilities/ElementwiseBodyOpRegistry.h"

#include "ascir/Dialect/Asc/IR/Asc.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ManagedStatic.h"

#include <mutex>

namespace mlir::ascend::backend {
namespace {

struct RegistrySingleton {
  std::mutex mu;
  llvm::StringMap<ElementwiseBodyOpEntry> entries;
  bool builtinsRegistered = false;
};

llvm::ManagedStatic<RegistrySingleton> gRegistry;

template <typename OpTy>
void addUnaryL2Op(llvm::function_ref<void(ElementwiseBodyOpEntry)> add,
                 llvm::StringRef dialectOpName, ComputeKind kind) {
  add({dialectOpName, kind,
       [](mlir::OpBuilder &b, mlir::Location loc,
          mlir::Value dst, mlir::Value src, mlir::Value cnt) {
         b.create<OpTy>(loc, dst, src, cnt);
       },
       nullptr});
}

template <typename OpTy>
void addBinaryL2Op(llvm::function_ref<void(ElementwiseBodyOpEntry)> add,
                  llvm::StringRef dialectOpName, ComputeKind kind) {
  add({dialectOpName, kind, nullptr,
       [](mlir::OpBuilder &b, mlir::Location loc,
          mlir::Value dst, mlir::Value src0, mlir::Value src1,
          mlir::Value cnt) {
         b.create<OpTy>(loc, dst, src0, src1, cnt);
       }});
}

template <typename OpTy>
void addUnaryMathLibraryOp(llvm::function_ref<void(ElementwiseBodyOpEntry)> add,
                           llvm::StringRef dialectOpName, ComputeKind kind) {
  add({dialectOpName, kind,
       [](mlir::OpBuilder &b, mlir::Location loc,
          mlir::Value dst, mlir::Value src, mlir::Value cnt) {
         mlir::Value reuseSource =
             b.create<mlir::arith::ConstantIntOp>(loc, b.getI1Type(), 0);
         b.create<OpTy>(loc, dst, src, /*sharedTmpBuffer=*/mlir::Value{},
                        /*calCount=*/cnt, reuseSource);
       },
       nullptr});
}

template <typename OpTy>
void addBinaryMathLibraryOp(llvm::function_ref<void(ElementwiseBodyOpEntry)> add,
                            llvm::StringRef dialectOpName, ComputeKind kind) {
  add({dialectOpName, kind, nullptr,
       [](mlir::OpBuilder &b, mlir::Location loc,
          mlir::Value dst, mlir::Value src0, mlir::Value src1,
          mlir::Value cnt) {
         mlir::Value reuseSource =
             b.create<mlir::arith::ConstantIntOp>(loc, b.getI1Type(), 0);
         b.create<OpTy>(loc, dst, src0, src1,
                        /*sharedTmpBuffer=*/mlir::Value{},
                        /*calCount=*/cnt, reuseSource);
       }});
}

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
  addBinaryL2Op<AddL2Op>(add, "arith.addf", ComputeKind::ElementwiseAdd);
  addBinaryL2Op<MulL2Op>(add, "arith.mulf", ComputeKind::ElementwiseMul);
  addBinaryL2Op<MaxL2Op>(add, "arith.maximumf", ComputeKind::ElementwiseMax);
  addBinaryL2Op<MinL2Op>(add, "arith.minimumf", ComputeKind::ElementwiseMin);
  addBinaryL2Op<SubL2Op>(add, "arith.subf", ComputeKind::ElementwiseSub);
  addBinaryL2Op<DivL2Op>(add, "arith.divf", ComputeKind::ElementwiseDiv);
  addBinaryL2Op<AndL2Op>(add, "arith.andi",
                         ComputeKind::ElementwisePyAscBitwise);
  addBinaryL2Op<OrL2Op>(add, "arith.ori",
                        ComputeKind::ElementwisePyAscBitwise);

  // Unary ops
  addUnaryL2Op<NegL2Op>(add, "arith.negf", ComputeKind::ElementwiseNeg);
  addUnaryL2Op<ExpL2Op>(add, "math.exp", ComputeKind::ElementwiseExp);
  addUnaryL2Op<LnL2Op>(add, "math.log", ComputeKind::ElementwiseLog);
  addUnaryL2Op<SqrtL2Op>(add, "math.sqrt", ComputeKind::ElementwiseSqrt);
  addUnaryL2Op<RsqrtL2Op>(add, "math.rsqrt", ComputeKind::ElementwiseRsqrt);
  addUnaryL2Op<AbsL2Op>(add, "math.absf", ComputeKind::ElementwiseAbs);

  // PyAsc Adv Math UnaryMathOp entries with existing fine-grained kinds.
  addUnaryMathLibraryOp<ErfOp>(add, "math.erf", ComputeKind::ElementwiseErf);
  addUnaryMathLibraryOp<TanhOp>(add, "math.tanh", ComputeKind::ElementwiseTanh);
  addUnaryMathLibraryOp<SinOp>(add, "math.sin", ComputeKind::ElementwiseSin);
  addUnaryMathLibraryOp<CosOp>(add, "math.cos", ComputeKind::ElementwiseCos);

  // Remaining PyAsc Adv Math unary entries that have no dedicated ComputeKind.
  // The source names are MLIR math dialect names where available. Entries whose
  // source op is not present in the current MLIR build are harmless registry
  // entries and become active when the source dialect grows that op.
  addUnaryMathLibraryOp<AcoshOp>(add, "math.acosh",
                                ComputeKind::ElementwisePyAscMath);
  addUnaryMathLibraryOp<AcosOp>(add, "math.acos",
                               ComputeKind::ElementwisePyAscMath);
  addUnaryMathLibraryOp<AsinhOp>(add, "math.asinh",
                                ComputeKind::ElementwisePyAscMath);
  addUnaryMathLibraryOp<AsinOp>(add, "math.asin",
                               ComputeKind::ElementwisePyAscMath);
  addUnaryMathLibraryOp<AtanhOp>(add, "math.atanh",
                                ComputeKind::ElementwisePyAscMath);
  addUnaryMathLibraryOp<AtanOp>(add, "math.atan",
                               ComputeKind::ElementwisePyAscMath);
  addUnaryMathLibraryOp<CeilOp>(add, "math.ceil",
                               ComputeKind::ElementwisePyAscMath);
  addUnaryMathLibraryOp<CoshOp>(add, "math.cosh",
                               ComputeKind::ElementwisePyAscMath);
  addUnaryMathLibraryOp<DigammaOp>(add, "math.digamma",
                                  ComputeKind::ElementwisePyAscMath);
  addUnaryMathLibraryOp<ErfcOp>(add, "math.erfc",
                               ComputeKind::ElementwisePyAscMath);
  addUnaryMathLibraryOp<FloorOp>(add, "math.floor",
                                ComputeKind::ElementwisePyAscMath);
  addUnaryMathLibraryOp<FracOp>(add, "math.frac",
                               ComputeKind::ElementwisePyAscMath);
  addUnaryMathLibraryOp<LgammaOp>(add, "math.lgamma",
                                 ComputeKind::ElementwisePyAscMath);
  addUnaryMathLibraryOp<RoundOp>(add, "math.round",
                                ComputeKind::ElementwisePyAscMath);
  addUnaryMathLibraryOp<SignOp>(add, "math.sign",
                               ComputeKind::ElementwisePyAscMath);
  addUnaryMathLibraryOp<SinhOp>(add, "math.sinh",
                               ComputeKind::ElementwisePyAscMath);
  addUnaryMathLibraryOp<TanOp>(add, "math.tan",
                              ComputeKind::ElementwisePyAscMath);
  addUnaryMathLibraryOp<TruncOp>(add, "math.trunc",
                                ComputeKind::ElementwisePyAscMath);

  addBinaryMathLibraryOp<PowerOp>(add, "math.powf",
                                 ComputeKind::ElementwisePyAscMath);
  addBinaryMathLibraryOp<XorOp>(add, "arith.xori",
                                ComputeKind::ElementwisePyAscBitwise);
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

} // namespace mlir::ascend::backend
