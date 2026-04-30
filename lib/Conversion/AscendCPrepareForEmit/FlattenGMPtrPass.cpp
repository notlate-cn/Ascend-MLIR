#include "Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#define GEN_PASS_DECL_ASCENDCFLATTENGMPTRPASS
#define GEN_PASS_DEF_ASCENDCFLATTENGMPTRPASS
#include "Conversion/Passes.h.inc"
using namespace mlir;
namespace mlir::afir {
struct AscendCFlattenGMPtrPass
    : public ::impl::AscendCFlattenGMPtrPassBase<AscendCFlattenGMPtrPass> {
  using AscendCFlattenGMPtrPassBase::AscendCFlattenGMPtrPassBase;
  void runOnOperation() override {}
};
std::unique_ptr<Pass> createAscendCFlattenGMPtrPass() {
  return std::make_unique<AscendCFlattenGMPtrPass>();
}
} // namespace mlir::afir
