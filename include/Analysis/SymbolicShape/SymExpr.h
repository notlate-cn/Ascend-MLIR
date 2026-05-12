//===- SymExpr.h - Tiny symbolic-shape expression type -----------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
// A deliberately minimal symbolic arithmetic expression used to carry the shape
// of dynamic tensors through the linalg-level pipeline (see
// docs/superpowers/specs/2026-05-12-mlir-shape-symbolization-design.md).
//
// Nine node kinds, immutable value type, shared_ptr-linked nodes.  Smart
// constructors do constant folding + identity elimination only -- there is no
// algebraic simplifier.  The only operations are eval(), emitC(), print/parse,
// and structural equality/hash.  This file depends only on LLVM ADT.
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_ANALYSIS_SYMBOLICSHAPE_SYMEXPR_H
#define AFIR_ANALYSIS_SYMBOLICSHAPE_SYMEXPR_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace mlir::afir::symshape {

/// Index into a SymbolTable -- identifies a symbolic dimension.
using SymId = uint32_t;

namespace detail {
struct SymExprNode;
} // namespace detail

class SymExpr {
public:
  enum class Kind : uint8_t {
    Sym,     // a symbolic variable, payload = SymId
    Const,   // an integer constant, payload = int64_t
    Add,     // lhs + rhs
    Sub,     // lhs - rhs
    Mul,     // lhs * rhs
    CeilDiv, // (lhs + rhs - 1) / rhs   (rhs assumed > 0)
    Mod,     // lhs % rhs                (rhs assumed > 0)
    Min,     // min(lhs, rhs)
    Max,     // max(lhs, rhs)
  };

private:
  std::shared_ptr<const detail::SymExprNode> node;
  explicit SymExpr(std::shared_ptr<const detail::SymExprNode> n)
      : node(std::move(n)) {}
  static SymExpr make(Kind k, SymId s, int64_t c, SymExpr l, SymExpr r);

public:
  /// Builds a null/invalid SymExpr.  Only meaningful as a placeholder before
  /// assignment; calling any method other than isValid() on it is UB.
  SymExpr() = default;
  bool isValid() const { return static_cast<bool>(node); }

  // -- leaves --
  static SymExpr sym(SymId id);
  static SymExpr constant(int64_t v);

  // -- binary, with folding (see .cpp) --
  static SymExpr add(SymExpr a, SymExpr b);
  static SymExpr sub(SymExpr a, SymExpr b);
  static SymExpr mul(SymExpr a, SymExpr b);
  static SymExpr ceilDiv(SymExpr a, SymExpr b);
  static SymExpr mod(SymExpr a, SymExpr b);
  static SymExpr min(SymExpr a, SymExpr b);
  static SymExpr max(SymExpr a, SymExpr b);

  // -- accessors --
  Kind getKind() const;
  SymId getSym() const;          // requires Kind::Sym
  int64_t getCstValue() const;   // requires Kind::Const
  SymExpr getLhs() const;        // requires a binary kind
  SymExpr getRhs() const;        // requires a binary kind

  /// If this expression is a literal constant (after folding), returns it.
  std::optional<int64_t> getConst() const;

  // -- operations --

  /// Evaluates to an integer.  Every symbol that appears must be present in
  /// `env`; otherwise this asserts.  Use walkSymbols() to check completeness.
  int64_t eval(const llvm::DenseMap<SymId, int64_t> &env) const;

  /// Emits a C expression string.  `name` maps a SymId to its C identifier.
  /// Not wired into codegen in v1; provided so the interface is settled.
  std::string emitC(llvm::function_ref<std::string(SymId)> name) const;

  /// Calls `cb` once for every SymId that appears (duplicates possible).
  void walkSymbols(llvm::function_ref<void(SymId)> cb) const;

  /// Returns a copy with every Sym leaf `id` replaced by `f(id)` (structure and
  /// constants unchanged; folding still applies, e.g. if `f` maps two distinct
  /// ids to the same one).
  SymExpr mapSymbols(llvm::function_ref<SymId(SymId)> f) const;

  /// Prints in the grammar understood by parseSymExpr():
  ///   int | 's' int | '(' e op e ')' | 'ceildiv(' e ',' e ')'
  ///       | 'mod(' e ',' e ')' | 'min(' e ',' e ')' | 'max(' e ',' e ')'
  /// with op in {+, -, *}.  Examples: "s3", "(s0*s1)", "(s2+8)", "ceildiv(s0,s1)".
  void print(llvm::raw_ostream &os) const;
  std::string str() const;

  /// Structural equality on the (already folded) representation.  ORDER
  /// SENSITIVE -- add(s0,s1) != add(s1,s0).  Not semantic equality.
  bool operator==(const SymExpr &other) const;
  bool operator!=(const SymExpr &other) const { return !(*this == other); }
  llvm::hash_code hashValue() const;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const SymExpr &e) {
  e.print(os);
  return os;
}

/// Parses the grammar described above.  Returns std::nullopt on any syntax
/// error or on a trailing-garbage input.  Whitespace is permitted between
/// tokens.
std::optional<SymExpr> parseSymExpr(llvm::StringRef text);

/// Parses a comma-separated list of SymExpr (the per-result-dim encoding used
/// in the `afir.symbolic_shapes` attribute).  An empty/whitespace string yields
/// an empty vector (rank-0 result).  Returns std::nullopt if any element fails.
std::optional<llvm::SmallVector<SymExpr, 4>>
parseSymExprList(llvm::StringRef text);

/// Prints a list of SymExpr comma-separated (no spaces), the inverse of
/// parseSymExprList.
void printSymExprList(llvm::ArrayRef<SymExpr> exprs, llvm::raw_ostream &os);
std::string symExprListStr(llvm::ArrayRef<SymExpr> exprs);

} // namespace mlir::afir::symshape

#endif // AFIR_ANALYSIS_SYMBOLICSHAPE_SYMEXPR_H
