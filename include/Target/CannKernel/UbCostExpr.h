//===- UbCostExpr.h - lift init_buffer size operand to SymExpr -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
// Walks an ascendc.pipe.init_buffer / init_queue size operand's def-use chain
// and rebuilds it as a SymExpr keyed on TilingData field names (XBLOCK_SUB,
// dim_arg0_2, ...).  Used by CannTranslation to emit ub_cost_bytes_expr into
// tiling_space.json.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_TARGET_CANNKERNEL_UBCOSTEXPR_H
#define AFIR_TARGET_CANNKERNEL_UBCOSTEXPR_H

#include "Analysis/SymbolicShape/SymExpr.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include <optional>

namespace mlir::afir::cannkernel {

/// Bidirectional name <-> SymId table for lifting emitasc.member-typed leaves
/// (each unique field name gets a fresh SymId).  Caller owns the lifetime.
struct NameSymTable {
  llvm::StringMap<symshape::SymId> nameToId;
  llvm::DenseMap<symshape::SymId, std::string> idToName;
  symshape::SymId getOrCreate(llvm::StringRef name);
};

/// Walks the def-use chain of `sizeOperand` and lifts it to a SymExpr.
/// Returns nullopt on the first unknown op kind; callers downgrade to
/// "no UB budget known" and skip pruning for the kernel.
///
/// Supported leaves / ops:
///   - emitasc.member (leaf, keyed by `field` name)
///   - arith.constant (integer)
///   - arith.muli, arith.addi, arith.subi (binary)
///   - arith.index_cast (transparent)
///
/// Division (divsi/divui) intentionally NOT supported: real-world
/// init_buffer size chains are monomial products of tiling params and
/// sizeof, no division ever appears; modeling floor-vs-ceil semantics
/// is error-prone, so we conservatively return nullopt when we see one.
std::optional<symshape::SymExpr> liftSizeOperand(mlir::Value sizeOperand,
                                                 NameSymTable &names);

/// align_up(s, 32) expressed in SymExpr:  ceilDiv(s, 32) * 32.
symshape::SymExpr align32(symshape::SymExpr s);

} // namespace mlir::afir::cannkernel

#endif // AFIR_TARGET_CANNKERNEL_UBCOSTEXPR_H
