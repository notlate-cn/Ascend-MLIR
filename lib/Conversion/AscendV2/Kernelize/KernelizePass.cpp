//===- KernelizePass.cpp - Ascend V2 kernelize pass -----------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AscendV2/Kernelize/KernelizePass.h"

#include "Conversion/AscendV2/Debug/DebugOptions.h"
#include "Conversion/AscendV2/Kernelize/KernelizeTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

#define GEN_PASS_DECL_ASCENDKERNELIZEPASS
#define GEN_PASS_DEF_ASCENDKERNELIZEPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;
using namespace mlir::afir::ascend::v2::kernelize;

namespace {

struct KernelizeReportEntry {
  std::string opRole;
  std::string kernelPattern;
};

bool isFuncOp(Operation *op) {
  return op->getName().getStringRef() == "func.func";
}

bool isLinalgStructuredOp(Operation *op) {
  StringRef opName = op->getName().getStringRef();
  return op->getName().getDialectNamespace() == "linalg" &&
         opName != "linalg.yield" && opName != "linalg.index";
}

bool isSupportedLinalgOpName(StringRef opName) {
  return opName == "linalg.generic" || opName == "linalg.matmul" ||
         opName == "linalg.batch_matmul";
}

bool isIteratorType(Attribute attr, StringRef expected) {
  if (auto stringAttr = dyn_cast<StringAttr>(attr))
    return stringAttr.getValue() == expected;

  SmallString<32> storage;
  llvm::raw_svector_ostream os(storage);
  attr.print(os);
  return StringRef(storage).contains(expected);
}

std::optional<StringRef> classifyLinalgOp(Operation *op) {
  StringRef opName = op->getName().getStringRef();
  if (opName == "linalg.matmul" || opName == "linalg.batch_matmul")
    return StringRef("cube");

  if (opName != "linalg.generic")
    return StringRef("unsupported");

  auto iteratorTypes = op->getAttrOfType<ArrayAttr>("iterator_types");
  if (!iteratorTypes)
    return std::nullopt;

  bool hasReduction = false;
  for (Attribute iteratorType : iteratorTypes) {
    if (isIteratorType(iteratorType, "parallel"))
      continue;
    if (isIteratorType(iteratorType, "reduction")) {
      hasReduction = true;
      continue;
    }
    return std::nullopt;
  }

  return hasReduction ? StringRef("reduction") : StringRef("vector");
}

void emitKernelizeReport(ArrayRef<KernelizeReportEntry> entries) {
  llvm::errs() << "Kernelize report\n";
  for (const KernelizeReportEntry &entry : entries) {
    llvm::errs() << "  op_role = \"" << entry.opRole << "\"\n";
    llvm::errs() << "  kernel_pattern = \"" << entry.kernelPattern << "\"\n";
    llvm::errs() << "  primary_ops = 1\n";
  }
}

} // namespace

namespace mlir::afir {

struct AscendKernelizePass
    : public ::impl::AscendKernelizePassBase<AscendKernelizePass> {
  using AscendKernelizePassBase::AscendKernelizePassBase;

  void runOnOperation() override {
    ::mlir::ascend::v2::DebugOptions options{
        ::mlir::ascend::v2::parseDebugStage(debugStage), dumpReport};
    if (::mlir::ascend::v2::shouldDump(
            options, ::mlir::ascend::v2::DebugStage::Kernelize))
      ::mlir::ascend::v2::emitStageHeader(
          llvm::errs(), ::mlir::ascend::v2::DebugStage::Kernelize,
          getArgument());

    ModuleOp module = getOperation();
    if (module
            .walk([&](Operation *op) {
              if (!isFuncOp(op))
                return WalkResult::advance();

              auto normalized = op->getAttrOfType<BoolAttr>(kNormalizedAttr);
              if (normalized && normalized.getValue())
                return WalkResult::advance();

              op->emitError() << "requires ascend.v2.normalized";
              return WalkResult::interrupt();
            })
            .wasInterrupted()) {
      signalPassFailure();
      return;
    }

    MLIRContext *context = module.getContext();
    SmallVector<KernelizeReportEntry> reportEntries;
    unsigned nextKernelId = 0;
    if (module
            .walk([&](Operation *op) {
              if (!isLinalgStructuredOp(op))
                return WalkResult::advance();

              std::optional<StringRef> role = classifyLinalgOp(op);
              if (!role) {
                if (isSupportedLinalgOpName(op->getName().getStringRef())) {
                  op->emitError() << "failed to classify supported linalg op";
                  return WalkResult::interrupt();
                }
                role = StringRef("unsupported");
              }

              op->setAttr(kOpRoleAttr, StringAttr::get(context, *role));
              if (*role == "unsupported")
                return WalkResult::advance();

              std::string kernelId =
                  (llvm::Twine("kernel_") + llvm::Twine(nextKernelId++))
                      .str();
              op->setAttr(kKernelAttr, StringAttr::get(context, kernelId));
              op->setAttr(kPrimaryAttr, BoolAttr::get(context, true));
              reportEntries.push_back(
                  KernelizeReportEntry{role->str(), kernelId});
              return WalkResult::advance();
            })
            .wasInterrupted()) {
      signalPassFailure();
      return;
    }

    if (::mlir::ascend::v2::shouldDump(
            options, ::mlir::ascend::v2::DebugStage::Kernelize))
      emitKernelizeReport(reportEntries);
  }
};

std::unique_ptr<Pass> createAscendKernelizePass() {
  return std::make_unique<AscendKernelizePass>();
}

} // namespace mlir::afir
