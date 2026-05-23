//===- BackendWrapperPasses.cpp - Ascend backend wrapper passes -----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Backend/Wrappers/BackendWrapperPasses.h"
#include "../Codegen/CodegenPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#define GEN_PASS_DECL_ASCENDCANONICALIZECANNSIGNATUREPASS
#define GEN_PASS_DECL_ASCENDPARALLELIZEPASS
#define GEN_PASS_DECL_ASCENDPREPAREFOREMITPASS
#define GEN_PASS_DEF_ASCENDCANONICALIZECANNSIGNATUREPASS
#define GEN_PASS_DEF_ASCENDPARALLELIZEPASS
#define GEN_PASS_DEF_ASCENDPREPAREFOREMITPASS
#include "Conversion/Ascend/Passes.h.inc"

namespace mlir::afir {
namespace {

struct AscendParallelizePass
    : public ::impl::AscendParallelizePassBase<AscendParallelizePass> {
  void runOnOperation() override {
    OpPassManager pm("builtin.module");
    pm.nest<func::FuncOp>().addPass(createAscendCodegenParallelizePass());
    if (failed(runPipeline(pm, getOperation())))
      signalPassFailure();
  }
};

struct AscendPrepareForEmitPass
    : public ::impl::AscendPrepareForEmitPassBase<AscendPrepareForEmitPass> {
  void runOnOperation() override {
    OpPassManager pm("builtin.module");
    pm.addPass(createAscendCodegenPrepareForEmitPass());
    pm.nest<func::FuncOp>().addPass(
        createAscendCodegenAnnotateKernelKindPass());
    if (failed(runPipeline(pm, getOperation())))
      signalPassFailure();
  }
};

struct AscendCanonicalizeCannSignaturePass
    : public ::impl::AscendCanonicalizeCannSignaturePassBase<
          AscendCanonicalizeCannSignaturePass> {
  void runOnOperation() override {
    OpPassManager pm("builtin.module");
    pm.addPass(createAscendCodegenCanonicalizeCannSignaturePass());
    if (failed(runPipeline(pm, getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createAscendParallelizePass() {
  return std::make_unique<AscendParallelizePass>();
}

std::unique_ptr<Pass> createAscendPrepareForEmitPass() {
  return std::make_unique<AscendPrepareForEmitPass>();
}

std::unique_ptr<Pass> createAscendCanonicalizeCannSignaturePass() {
  return std::make_unique<AscendCanonicalizeCannSignaturePass>();
}

} // namespace mlir::afir
