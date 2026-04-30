#include "Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#define GEN_PASS_DECL_ASCENDCFINALIZEKERNELPASS
#define GEN_PASS_DEF_ASCENDCFINALIZEKERNELPASS
#include "Conversion/Passes.h.inc"
using namespace mlir;
namespace mlir::afir {
struct AscendCFinalizeKernelPass
    : public ::impl::AscendCFinalizeKernelPassBase<AscendCFinalizeKernelPass> {
  using AscendCFinalizeKernelPassBase::AscendCFinalizeKernelPassBase;
  void runOnOperation() override {}
};
std::unique_ptr<Pass> createAscendCFinalizeKernelPass() {
  return std::make_unique<AscendCFinalizeKernelPass>();
}
} // namespace mlir::afir
