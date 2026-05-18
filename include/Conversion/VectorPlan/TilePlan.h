#pragma once
#include "Conversion/VectorPlan/GroupInfo.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>

namespace mlir::vector_plan {

enum class TileLevel : uint8_t { Outer, Inner, Full };

enum class TileFieldKind : uint8_t {
  TunableTile,
  FixedTile,
  ShapeDim,
  Derived,
};

// AxisKind / AxisClass / AxisGrouping live in GroupInfo.h (computed by Collapse,
// carried in CollapsedGroupInfo::grouping).

// One enumerated tiling-case draft (≈ AutoFuse `TilingCase`).  P1 produces a
// single default draft; full y×x×r enumeration + RCore variant comes later.
struct TilePlanDraft {
  int  ubTilingAxisY = -1;
  int  ubTilingAxisX = -1;
  int  ubTilingAxisR = -1;
  int  blockTilingId = 0;
  bool reduceIsBlock = false;
  // FullLoad (≈ AF kAllLoad): every R axis kept whole AND treated as part of
  // the vectorized region — the reduce happens inside the vector op, no
  // ReduceSum intrinsic. Only feasible when total `Σ R · elemBytes` fits
  // on-chip (≈ AF's reduction_tile_bytes ≤ UB_budget). Today enumerated by
  // enumerateTilingCases but always ∞-scored: codegen for the in-vector
  // reduce isn't implemented yet (GroupEmitter / ComputeConversion would
  // need a new template; deferred).
  bool isFullLoad   = false;
  // Non-contiguous multi-reduce ("displaced R"): out[a]=sum_{r1,r2} x[r1,a,r2]
  // — iter [reduction, parallel, reduction] where Collapse cannot merge the two
  // R axes (a parallel axis separates them).  When >= 0, the named axis is the
  // outermost displaced R; the GroupEmitter peels it into a step=1 outer scf.for
  // and rank-reduces it out of the inner linalg.generic so the inner op shape
  // matches the well-tested single-P/single-R reduce path.  Mirrors AutoFuse's
  // `IsNeedMultiReduce` outer-loop emission (reduce_api_call.cpp:96).
  int  peelOuterR   = -1;
};

// A schedulability/legality constraint over the tunable params (≈ ATT's
// tiling-data constraints).  Emitted into `vector_plan.tiling_infos` in P6.
//
// Both `lhs` and `rhs` are free-form arithmetic expressions over tile-param
// names (e.g. "XBLOCK_SUB", "XBLOCK") and integer literals, using the same
// pure +-*/ grammar as block_dim_expr / ub_cost_bytes_exprs (CeilDiv encoded
// as `((a + b - 1) / b)`).  The autotuner re-uses its evalBlockExpr to
// evaluate them under candidate tile values.
//   Divides : lhs must evenly divide rhs (i.e. rhs % lhs == 0)
//   LeBytes : lhs (in bytes) must be ≤ rhs (in bytes)
struct TileConstraint {
  enum Kind : uint8_t { Divides, LeBytes };
  Kind        kind;
  std::string lhs;
  std::string rhs;
};

struct TileParam {
  std::string  name;
  Value        ssa;
  OpFoldResult defaultValue;
  int32_t      axisIdx;
  TileLevel    level;
  AxisRole     role;

  // Flags carried from the axis's AxisClass (consulted by downstream passes;
  // default-initialized so existing aggregate-initializing call sites compile).
  bool bindMultiCore    = false;
  bool enableTail       = true;
  bool enablePad        = false;
  bool isReduceSplit    = false;
  bool isBroadcastSplit = false;
};

struct TilePlan {
  enum class ReduceTemplate : uint8_t { None, Common, FullLoad, RCore };

  const CollapsedGroupInfo                        *group = nullptr;
  llvm::SmallVector<llvm::SmallVector<TileParam>>  tileable;
  llvm::SmallVector<TileParam>                     full;
  llvm::SmallVector<OpFoldResult, 2>               blockDimExprs;

  ReduceTemplate            reduceTemplate = ReduceTemplate::None;
  bool                      reduceIsBlock  = false;
  llvm::SmallVector<int>    blockFusedAxes;        // collapsed-axis ids fused into the block axis (in order)
  int                       ubTilingAxisY = -1;
  int                       ubTilingAxisX = -1;
  int                       ubTilingAxisR = -1;
  // Mirrors TilePlanDraft::peelOuterR — see TilePlanDraft for semantics.
  int                       peelOuterR    = -1;
  // Per-operand vectorized iteration dims (the dims that stay whole inside the
  // tiled linalg op, ≈ tensor.attr.vectorized_axis).  Filled by P3.
  llvm::DenseMap<Value, llvm::SmallVector<int>> vectorizedDims;
  llvm::SmallVector<TileConstraint>             constraints;
};

// ──── Tiling-info schema v2 ──────────────────────────────────────────────
// One TilingInfoSchema per kernel func.  Serialized into the
// `vector_plan.tiling_infos` ModuleOp attribute (as a DictAttr), read by
// PackTilingData / CannTranslation / NetworkJsonEmitter / runner.
//
// See docs/superpowers/specs/2026-05-18-tiling-info-schema-design.md.

enum class SchemaFieldKind { Tunable, ShapeDerived };

struct SchemaField {
  std::string name;
  SchemaFieldKind kind;
  // Tunable: axis size (-1 dyn) + default + arg_index (the MLIR arg holding
  // the index-typed tile-param value injected by TilePlanGen).
  int64_t axisSize = -1;
  int64_t defaultValue = 0;
  int32_t argIndex = -1;  // MLIR arg index for the tile-param SSA value
  // ShapeDerived: which (MLIR arg, dim) this field's value comes from.
  int32_t sourceArg = -1;
  int32_t sourceDim = -1;
};

enum class SchemaArgRole {
  Input,
  Output,
  TileParam,
  Workspace,
  TilingDataStruct,
};

struct SchemaArg {
  int32_t mlirIndex;            // position in the kernel func signature
  SchemaArgRole role;
  // Input: coordinator-call operand position.
  int32_t callArgIndex = -1;
  // Output: result_index in the kernel's `results` array + shape_expr per
  // output dim (each entry is a host-evaluable string like "arg0_dim1").
  int32_t resultIndex = -1;
  llvm::SmallVector<std::string, 4> shapeExpr;
  // TileParam: the field name this arg holds.
  std::string tileParamName;
};

struct TilingInfoSchema {
  static constexpr int kSchemaVersion = 2;
  std::string kernelId;
  std::string blockDimExpr;
  std::string axisExtentExpr;
  llvm::SmallVector<SchemaField, 8> fields;
  llvm::SmallVector<SchemaArg, 8> args;
  // Constraints: reuse the existing {kind, lhs, rhs} struct.  Carried through
  // unchanged.
};

// Serialize a TilingInfoSchema to a DictionaryAttr suitable for embedding in
// `vector_plan.tiling_infos`.  Round-trips with deserialize().
mlir::DictionaryAttr serializeTilingInfoSchema(
    mlir::MLIRContext *ctx, const TilingInfoSchema &s,
    mlir::ArrayAttr constraintsAttr);

// Decode a vector_plan.tiling_infos entry into a TilingInfoSchema.  Returns
// std::nullopt when the entry is not v2 (caller must fall back to legacy).
std::optional<TilingInfoSchema> deserializeTilingInfoSchema(
    mlir::DictionaryAttr entry);

// Look up the schema entry for a given kernel func from a module's
// vector_plan.tiling_infos attr.  Returns std::nullopt when absent or v1.
std::optional<TilingInfoSchema> lookupTilingInfoSchema(
    mlir::ModuleOp moduleOp, llvm::StringRef kernelName);

} // namespace mlir::vector_plan
