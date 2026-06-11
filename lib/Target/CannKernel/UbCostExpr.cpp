//===- UbCostExpr.cpp -----------------------------------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/CannKernel/UbCostExpr.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
namespace ss = mlir::afir::symshape;

namespace mlir::afir::cannkernel {

ss::SymId NameSymTable::getOrCreate(llvm::StringRef name) {
  auto it = nameToId.find(name);
  if (it != nameToId.end())
    return it->second;
  ss::SymId id = static_cast<ss::SymId>(nameToId.size());
  nameToId.try_emplace(name, id);
  idToName[id] = name.str();
  return id;
}

std::optional<ss::SymExpr> liftSizeOperand(Value v, NameSymTable &names) {
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return std::nullopt;

  if (auto c = dyn_cast<arith::ConstantOp>(defOp)) {
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ss::SymExpr::constant(ia.getInt());
    return std::nullopt;
  }
  if (auto mb = dyn_cast<emitasc::MemberOp>(defOp))
    return ss::SymExpr::sym(names.getOrCreate(mb.getField()));
  if (auto ic = dyn_cast<arith::IndexCastOp>(defOp))
    return liftSizeOperand(ic.getIn(), names);

  auto lift2 = [&](auto folder) -> std::optional<ss::SymExpr> {
    auto lhs = liftSizeOperand(defOp->getOperand(0), names);
    if (!lhs)
      return std::nullopt;
    auto rhs = liftSizeOperand(defOp->getOperand(1), names);
    if (!rhs)
      return std::nullopt;
    return folder(*lhs, *rhs);
  };
  if (isa<arith::MulIOp>(defOp))
    return lift2([](ss::SymExpr a, ss::SymExpr b) { return ss::SymExpr::mul(a, b); });
  if (isa<arith::AddIOp>(defOp))
    return lift2([](ss::SymExpr a, ss::SymExpr b) { return ss::SymExpr::add(a, b); });
  if (isa<arith::SubIOp>(defOp))
    return lift2([](ss::SymExpr a, ss::SymExpr b) { return ss::SymExpr::sub(a, b); });
  return std::nullopt;
}

ss::SymExpr align32(ss::SymExpr s) {
  ss::SymExpr c32 = ss::SymExpr::constant(32);
  return ss::SymExpr::mul(ss::SymExpr::ceilDiv(s, c32), c32);
}

} // namespace mlir::afir::cannkernel
