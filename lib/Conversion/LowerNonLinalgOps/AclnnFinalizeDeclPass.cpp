#include "Conversion/LowerNonLinalgOps/LowerNonLinalgOpsPass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"

#define GEN_PASS_DECL_ACLNNFINALIZEDECL
#define GEN_PASS_DEF_ACLNNFINALIZEDECL
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

namespace {

//===----------------------------------------------------------------------===//
// Registry: aclnn.kind -> {aclnn.op, aclnn.layout}
// To add a new aclnn op: append one entry here and implement the C++ wrapper
// in lib/Runtime/AclnnOps.cpp.
//===----------------------------------------------------------------------===//

struct AclnnOpMeta {
  StringRef op;
  StringRef layout;
};

static const llvm::StringMap<AclnnOpMeta> &getRegistry() {
  static llvm::StringMap<AclnnOpMeta> table = {
      {"flash_attention", {"FlashAttentionScore", "BNSD"}},
      {"layer_norm", {"LayerNorm", "ND"}},
      {"batch_norm", {"BatchNorm", "NCHW"}},
  };
  return table;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct AclnnFinalizeDeclPass
    : public ::impl::AclnnFinalizeDeclBase<AclnnFinalizeDeclPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder b(module.getContext());

    module.walk([&](func::FuncOp func) {
      if (!func.isPrivate())
        return;
      auto kind = func->getAttrOfType<StringAttr>("aclnn.kind");
      if (!kind)
        return;

      const auto &reg = getRegistry();
      auto it = reg.find(kind.getValue());
      if (it == reg.end()) {
        func->emitError("AclnnFinalizeDeclPass: unknown aclnn.kind '")
            << kind.getValue() << "' — not in registry";
        return signalPassFailure();
      }

      func->removeAttr("aclnn.kind");
      func->setAttr("aclnn.op",     b.getStringAttr(it->second.op));
      func->setAttr("aclnn.layout", b.getStringAttr(it->second.layout));
    });
  }
};

} // namespace

std::unique_ptr<Pass> createAclnnFinalizeDeclPass() {
  return std::make_unique<AclnnFinalizeDeclPass>();
}

} // namespace mlir::afir
