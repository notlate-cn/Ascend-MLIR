#include "Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#define GEN_PASS_DECL_ASCENDCPACKTILINGDATAPASS
#define GEN_PASS_DEF_ASCENDCPACKTILINGDATAPASS
#include "Conversion/Passes.h.inc"
using namespace mlir;
namespace mlir::afir {
struct AscendCPackTilingDataPass
    : public ::impl::AscendCPackTilingDataPassBase<AscendCPackTilingDataPass> {
  using AscendCPackTilingDataPassBase::AscendCPackTilingDataPassBase;
  void runOnOperation() override {}
};
std::unique_ptr<Pass> createAscendCPackTilingDataPass() {
  return std::make_unique<AscendCPackTilingDataPass>();
}
} // namespace mlir::afir
