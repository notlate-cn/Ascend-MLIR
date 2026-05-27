//===- OpRoleClassifier.cpp - Classify a linalg op to a role --------------===//
//
// See OpRoleClassifier.h for coverage rules.
//
//===----------------------------------------------------------------------===//

#include "Conversion/AutoFuse/GroupOutline/OpRoleClassifier.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::auto_fuse {

namespace {

// Strip dialect prefix ("arith.addf" → "addf", "math.exp" → "exp").
std::string stripDialect(llvm::StringRef opName) {
  auto dot = opName.find('.');
  if (dot == llvm::StringRef::npos)
    return opName.str();
  return opName.drop_front(dot + 1).str();
}

// Map an arith/math op name to a friendly short name. Returns the stripped
// dialect name as fallback so unknown ops still get something readable.
//   arith.addf → "add"
//   arith.maximumf → "max"
//   math.exp → "exp"
std::string friendlyOpName(llvm::StringRef opName) {
  std::string s = stripDialect(opName);
  // Drop trailing 'f' / 'i' / 'si' / 'ui' type tags so add/sub/mul/div/max/min
  // collapse across element types.
  static constexpr llvm::StringRef kBases[] = {
      "add", "sub", "mul", "div", "rem", "neg",
      "maximum", "minimum", "max", "min"};
  for (auto base : kBases) {
    if (llvm::StringRef(s).starts_with(base)) {
      auto rest = llvm::StringRef(s).drop_front(base.size());
      if (rest == "f" || rest == "i" || rest == "si" || rest == "ui") {
        std::string out = base.str();
        // "maximum"/"minimum" → "max"/"min"
        if (out == "maximum") return "max";
        if (out == "minimum") return "min";
        return out;
      }
    }
  }
  return s;
}

// True if a linalg.generic has any reduction iterator.
bool hasReductionIter(mlir::linalg::GenericOp generic) {
  for (mlir::utils::IteratorType it : generic.getIteratorTypesArray())
    if (it == mlir::utils::IteratorType::reduction)
      return true;
  return false;
}

// Find the arith/math op whose result feeds linalg.yield. For a reduction body
//   ^bb0(%a, %acc):
//     %v = arith.addf %a, %acc
//     linalg.yield %v
// returns the arith.addf.
mlir::Operation *findCombiner(mlir::linalg::GenericOp generic) {
  mlir::Operation *yield = generic.getBody()->getTerminator();
  if (!yield || yield->getNumOperands() == 0)
    return nullptr;
  return yield->getOperand(0).getDefiningOp();
}

// Map combiner op name to canonical reduce flavor.
//   arith.addf → "sum"; arith.maximumf → "max"; arith.minimumf → "min";
//   arith.mulf → "prod"; otherwise → friendly short name.
std::string reduceFlavor(mlir::Operation *combiner) {
  if (!combiner) return "unknown";
  std::string s = friendlyOpName(combiner->getName().getStringRef());
  if (s == "add") return "sum";
  if (s == "mul") return "prod";
  if (s == "max" || s == "min") return s;
  return s;
}

// Count compute ops (arith.*, math.*) in a linalg.generic body, skipping
// linalg.yield and trivial casts.
unsigned countComputeOps(mlir::linalg::GenericOp generic,
                         mlir::Operation *&single) {
  unsigned n = 0;
  single = nullptr;
  for (mlir::Operation &op : generic.getBody()->without_terminator()) {
    llvm::StringRef dialect = op.getName().getDialectNamespace();
    if (dialect != "arith" && dialect != "math")
      continue;
    if (n == 0) single = &op;
    else        single = nullptr;
    ++n;
  }
  return n;
}

} // namespace

std::string classifyOpRole(mlir::Operation *op) {
  if (!op) return "unknown";

  // Rule 1-3: named linalg ops.
  if (llvm::isa<mlir::linalg::MatmulOp, mlir::linalg::BatchMatmulOp>(op))
    return "matmul";
  if (llvm::isa<mlir::linalg::TransposeOp>(op))
    return "transpose";
  if (llvm::isa<mlir::linalg::CopyOp>(op))     return "copy";
  if (llvm::isa<mlir::linalg::FillOp>(op))     return "fill";
  if (llvm::isa<mlir::linalg::BroadcastOp>(op)) return "broadcast";

  // Rule 4-6: linalg.generic introspection.
  if (auto generic = llvm::dyn_cast<mlir::linalg::GenericOp>(op)) {
    if (hasReductionIter(generic))
      return "reduce_" + reduceFlavor(findCombiner(generic));

    mlir::Operation *single = nullptr;
    unsigned n = countComputeOps(generic, single);
    if (n == 1 && single)
      return friendlyOpName(single->getName().getStringRef());
    if (n >= 2)
      return "elementwise_chain";
    // 0 compute ops: pure copy/yield body — common for identity generics.
    return "copy";
  }

  // Rule 7: fallback.
  return op->getName().getStringRef().str();
}

} // namespace mlir::auto_fuse
