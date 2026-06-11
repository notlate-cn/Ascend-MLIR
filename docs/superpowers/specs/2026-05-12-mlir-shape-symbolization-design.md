# MLIR Shape Symbolization (linalg-level) — Design

**Date:** 2026-05-12
**Status:** approved, implementing
**Driver:** AF-scheduler-port plan §6 / P6 — "tiling-data 约束 + UB 峰值内存量化". To prune
tiling cases whose on-chip footprint exceeds UB capacity we need a symbolic-shape representation
on tensors. Today the pipeline carries no symbolic shape: dynamic dims stay `?` end-to-end and
buffer byte sizes are emitted as runtime `arith.muli` chains, so footprint is only known at
runtime.

Reference (heavyweight, *not* what we build): AF's `compiler/graph/optimize/symbolic/` +
`graph_metadef/graph/expression/` (SymEngine-backed `ge::Expression`, `ShapeEnvAttr`, guards,
~44 per-op symbolic kernels, cache-key/merge-key codegen). See
`docs/AF知识地图/专题-符号化推导.md`. We deliberately port only a thin subset.

## 1. Scope

**In v1:**
- A standalone expression library `SymExpr` (9 node kinds, no algebraic simplifier).
- A linalg-level pass `afir-symbolize-shapes` that mints symbols for `func.func @kernel` `?`
  arg dims and propagates symbolic shapes through our op set, writing them as attributes.
- A lightweight verifier pass for those attributes (lit-only).
- TilePlanGen consumes the attributes to compute a conservative UB-peak footprint per tiling
  case and prunes cases that exceed UB capacity.

**Out of v1 (follow-up P's):**
- codegen-infershape rewritten to read the attrs; tiling-function codegen using `SymExpr::emitC`.
- Carrying `torch.bind_symbolic_shape` annotations through the frontend (no torch dependency at
  all in v1).
- Symbolic shapes on the tiled loop nest / tiled ops (attrs live on the pre-tiled op only).
- Tail-accurate peak (v1 uses a full-tile upper bound, no `min`).
- Persisting symbol equalities / guards (`s == const`, `s_a == s_b` are discovered and used
  transiently; only union-find roots are serialized).

## 2. Decisions (final)

| # | Decision |
|---|---|
| Source | Pure linalg-level `afir-symbolize-shapes` pass, **zero torch dependency**. Symbols originate **only** at `func.func @kernel` block-arg `?` dims. Dedup happens via linalg structural unification (multiple operand pins on the same `linalg` iteration dim are unified). `afir.dim_symbols` is output-only — never read as input. |
| Carrier | Op attr `afir.symbolic_shapes` (ArrayAttr of StringAttr; one StringAttr per result; each is a comma-separated list of `SymExpr` strings, one per result dim) + func attr `afir.dim_symbols` (ArrayAttr of DictionaryAttr; output-only; no `name` field; lists only union-find root symbols). Builtin attrs only — no new dialect, no TableGen attr type. |
| Expression | Custom `SymExpr`, 9 node kinds: `Sym`, `Const`, `Add`, `Sub`, `Mul`, `CeilDiv`, `Mod`, `Min`, `Max`. shared_ptr-linked immutable nodes; smart constructors do constant folding + identity elimination only (no distribution / factoring). Operations: `eval(DenseMap<SymId,int64_t>)`, `emitC(symId→name)` (skeleton, not wired in v1), `print` / `parseSymExpr`, structural `operator==` / `hash`, `walkSymbols`. Location: `lib/Analysis/SymbolicShape/` + `include/Analysis/SymbolicShape/`; depends only on LLVM ADT, no dialect — so both `Dialect/AFIR` and `Conversion/VectorPlan` can link it. |
| Symbol id | `SymId` = `uint32` index into a per-`func::FuncOp` `SymbolTable` carrying union-find. `getOrCreateForArgDim(argIdx,dimIdx)`, `alias(a,b)`, `find`, `sourceOf`, `toAttr`/`fromAttr`. |
| Pipeline position | In `vector-plan-codegen`: `generalize-named-ops → elementwise-fusion → [new] afir-symbolize-shapes → tile-fuse → ...`. After generalize+fusion so each outlined kernel func is in stable form (one fused `linalg.generic` per group + reshape ops, named ops generalized); before tile-fuse so TilePlanGen reads attrs on the pre-tiled op. Nested on `func::FuncOp`. |
| v1 consumer | Only TilePlanGen `PruneTilingCase` (UB-peak). Conservative full-tile upper bound, no tail `min`. No `afir.symbolic_shapes` on the op ⇒ skip UB-peak pruning for that kernel (nothing pruned, all cases pass). Tiling params (`XBLOCK` / `XBLOCK_SUB` ...) get transient `SymId`s but are **not** serialized into `afir.dim_symbols`. |

## 3. `SymExpr` library

`lib/Analysis/SymbolicShape/SymExpr.{h,cpp}`, `lib/Analysis/SymbolicShape/SymbolTable.{h,cpp}`.

```cpp
using SymId = uint32_t;

class SymExpr {
  enum class Kind { Sym, Const, Add, Sub, Mul, CeilDiv, Mod, Min, Max };
  // payload: Sym -> SymId; Const -> int64_t; binary kinds -> two SymExpr children.
  // node held by shared_ptr<const Node>; SymExpr is a value wrapper around it.
public:
  static SymExpr sym(SymId);
  static SymExpr constant(int64_t);
  // binary constructors fold: add(C0,x)=x; sub(x,C0)=x; mul(C1,x)=x; mul(C0,_)=C0;
  //   ceilDiv(x,C1)=x; mod(_,C1)=C0; both-const -> evaluated constant; ...
  static SymExpr add(SymExpr,SymExpr);    static SymExpr sub(SymExpr,SymExpr);
  static SymExpr mul(SymExpr,SymExpr);    static SymExpr ceilDiv(SymExpr,SymExpr);
  static SymExpr mod(SymExpr,SymExpr);    static SymExpr min(SymExpr,SymExpr);
  static SymExpr max(SymExpr,SymExpr);

  int64_t eval(const llvm::DenseMap<SymId,int64_t> &env) const;          // total; missing sym -> assert
  std::string emitC(llvm::function_ref<std::string(SymId)> name) const;  // skeleton; not wired in v1
  void walkSymbols(llvm::function_ref<void(SymId)>) const;
  void print(llvm::raw_ostream &) const;                                 // "(s0*s1)" "s3" "(s2+8)" "ceildiv(s0,s1)"
  std::optional<int64_t> getConst() const;
  bool operator==(const SymExpr &) const;  // structural equality on folded form; ORDER-SENSITIVE
  llvm::hash_code hash() const;
};
// grammar: int | 's' int | '(' expr op expr ')' | 'min(' expr ',' expr ')' | 'max(' expr ',' expr ')'
//          | 'ceildiv(' expr ',' expr ')' | 'mod(' expr ',' expr ')'        ; op in + - *
std::optional<SymExpr> parseSymExpr(llvm::StringRef);
```

`SymExpr` does **not** normalize commutatively — `add(s0,s1) != add(s1,s0)`. That's fine: we
only `eval` and `emitC`; structural `==`/`hash` is used for cheap "same symbol / same expr"
checks, not semantic equality.

```cpp
class SymbolTable {
  struct Entry { unsigned argIdx, dimIdx; SymId parent; };
  llvm::SmallVector<Entry> entries;  // index == SymId
public:
  SymId getOrCreateForArgDim(unsigned argIdx, unsigned dimIdx);
  void  alias(SymId a, SymId b);     // union-find: find(b).parent = find(a); canonical = first-created
  SymId find(SymId) const;
  std::pair<unsigned,unsigned> sourceOf(SymId) const;   // canonical's (argIdx,dimIdx)
  unsigned numRoots() const;
  mlir::ArrayAttr toAttr(mlir::MLIRContext *) const;    // only roots, re-indexed 0..numRoots-1
  static std::optional<SymbolTable> fromAttr(mlir::ArrayAttr);
};
```

`toAttr` re-indexes so serialized `id`s are dense `0..numRoots-1` and match array position;
the pass keeps an internal `SymId -> serialized-id` map so attr strings reference serialized ids.

## 4. `afir-symbolize-shapes` pass

`lib/Dialect/AFIR/Transforms/AFIRSymbolizeShapes.cpp`, registered `afir-symbolize-shapes`, per
`func::FuncOp`. Algorithm:

1. **Arg symbols.** For each block arg: static dim → `SymExpr::constant`; `?` dim →
   `table.getOrCreateForArgDim(argIdx, dimIdx)` → `SymExpr::sym`. Store the arg's shape vector in
   an in-pass `DenseMap<Value, SmallVector<SymExpr>>`. (Args have no place for an attr; their
   shape is rebuildable from `afir.dim_symbols`, so it's not written to IR.)
2. **Topological op walk.** For each op with results, dispatch to a transfer function by op kind.
   The transfer reads operand shapes from the in-pass map, computes result shapes, stores them in
   the map, and writes `afir.symbolic_shapes` (one comma-separated `SymExpr` string per result).
3. **Bail.** Op with no transfer, or transfer failure, or an operand missing from the in-pass map
   → results get **no** attr and are **not** put in the map. "Unknown" propagates forward along
   use-def (downstream transfers reading it also bail). Never mint a fresh "orphan" symbol for an
   uncomputable dynamic dim — such a symbol has no runtime value and is useless for `eval`.
4. **Finalize.** Write `func.func`'s `afir.dim_symbols = table.toAttr(ctx)`.

Transfer functions (our op set ≈ 10):

| op | rule | bail when |
|---|---|---|
| `func.func` block arg | step 1 | — |
| `linalg::LinalgOp` (generic + generalized matmul/reduce/broadcast/transpose) | **first verify all indexing maps are projected permutations**; for each iteration dim `d_k` collect the `SymExpr` from every operand (incl. DPS inits) that maps some operand dim to `d_k` — these are "pins"; **unify** pins: two distinct syms → `table.alias`; sym vs const → use const (sym `== const` discovered, not persisted in v1); incompatible exprs → keep first; then read each result's shape via its **output** indexing map | any map not a projected permutation |
| `tensor.collapse_shape` | result dim for group `[i..j]` = `Mul`-fold of operand dims `i..j` (static dims fold into the constant) | — |
| `tensor.expand_shape` | splitting a **static** dim → result dims all constants | splitting a dynamic dim |
| `tensor.pad` | result dim = `operand_dim + low + high` (static low/high → `Add` with const) | dynamic low/high SSA |
| `tensor.empty` / `linalg.fill` / `bufferization.alloc_tensor` | result dims from the dynamic size operands; each dynamic size SSA expected to be a `tensor.dim` of a known tensor, or an `arith.{addi,subi,muli}` / `affine.apply` over such — interpret it into a `SymExpr` (localized version of "interpret reify output") | size SSA not interpretable |
| `tensor.extract_slice` / `insert_slice` | static sizes → result = those constants | dynamic sizes + size SSA not interpretable |
| `tensor.cast` | pass operand shape through (refine `?`→const ok) | — |
| `func.return` | no result | — |
| anything else | no transfer | always |

Note on staleness: tile-fuse rewrites the op (creates loop nest + tiled clones) but TilePlanGen
reads the attr on the pre-tiled op *before* transforming. Attrs on replaced ops are irrelevant in
v1; symbolic shapes on the loop nest are a future P.

## 5. Attribute spec + verify

`afir.dim_symbols` on `func.func` — ArrayAttr of DictionaryAttr, **roots only**:
```mlir
afir.dim_symbols = [
  {id = 0 : i64, arg = 0 : i64, dim = 0 : i64},
  {id = 1 : i64, arg = 0 : i64, dim = 1 : i64}
]
```
`id` == array index (redundant, for readability). `arg`/`dim` = the root symbol's defining
block-arg dim. Aliased arg-dims are not listed (their "equals root `sN`" fact is not persisted in
v1).

`afir.symbolic_shapes` on any op with results — ArrayAttr of StringAttr, length == #results;
each StringAttr is a comma-separated list of `SymExpr` strings, length == that result's rank:
```mlir
%1 = linalg.generic ... { afir.symbolic_shapes = ["(s0*s1)"] }
%2 = linalg.generic ... { afir.symbolic_shapes = ["(s0*s1), s1"] }   // multi-result
```
If any result of an op is uncomputable, the op gets **no** attr (no partial writes).

Verify (`afir-verify-symbolic-shapes`, lit-only pass; or `verifyAnalysis`):
- `afir.symbolic_shapes` array length == #results; each list length == that result's rank.
- every `sN` in a string has `N < len(afir.dim_symbols)`.
- result dims that are static in the type must `eval` to a symbol-free value equal to the static
  size (consistency self-check).
- dynamic dims may legitimately be a constant `SymExpr` (post-refinement) — not flagged.

## 6. TilePlanGen consumer — UB-peak pruning

Hook: `TilePlanGen.cpp` `enumerateTilingCases` → `PruneTilingCase` (the comments already reserve
the "P6's UB-peak accounting" slot). Per fused `linalg.generic` × per enumerated tiling case:

1. Parse the op's `afir.symbolic_shapes` + `func.func`'s `afir.dim_symbols`. **No attr → skip
   UB-peak pruning** (conservative: prune nothing).
2. From the tiling case (per-axis level: block `XBLOCK` / inner `XBLOCK_SUB` / untiled), build the
   set of on-chip buffers live in the innermost body = each VECIN input tile + each VECOUT output
   tile (+ broadcast temporaries if any). Reuse TilePlanGen's existing "who is live" logic.
3. Each buffer's byte-size `SymExpr` = `elementBytes (const) × Π_axis (tile size along axis)`:
   - untiled axis → that buffer's tensor's symbolic size along that axis (from step 1);
   - block/inner-tiled axis → **conservative full-tile size** (`XBLOCK_SUB` etc. as a `SymExpr`
     symbol with a transient `SymId`). No tail `min` in v1.
4. `footprint` `SymExpr` = Σ buffer byte-sizes.
5. **Eval.** In the autotuner scenario the sample shape's arg-dim values are known → `afir.dim_symbols`
   maps each root symbol → value; tiling params = the case being enumerated. `footprint.eval(env)` →
   an int. `> UB capacity` (read from target config, e.g. 192 KiB) → **prune this case**.
6. Surviving cases (+ their estimated footprint) flow into the existing `vector_plan.tiling_infos`.

Tiling params get transient `SymId`s (so `footprint` is a real symbolic expr, leaving the door
open for a future "emit footprint into the runtime tiling function for an OOM check"), but those
`SymId`s are not serialized into `afir.dim_symbols`.

## 7. Testing

**Unit (`SymExpr` library, no MLIR):** constructor folding (`add(C0,x)=x`, `mul(C0,_)=C0`,
`ceilDiv(mul(s0,s1),C8)` not mis-folded, ...); `eval` correctness + missing-symbol assert;
`print`↔`parseSymExpr` round-trip + rejection of malformed strings; `==`/`hash` is order-sensitive
(explicitly tested so no one mistakes it for semantic equality); `emitC` of a few exprs;
`SymbolTable` idempotent `getOrCreateForArgDim`, `alias`/`find` convergence, `toAttr`/`fromAttr`
round-trip (roots only).

**lit (`afir-symbolize-shapes`, `test/Dialect/AFIR/symbolize-shapes/`):** pure elementwise (shared
dim aliased away from `dim_symbols`); broadcast (post-fold-unit-extent form); reduce
`[d0,d1,d2]→[d0,d1]`; `tensor.collapse_shape [[0,1]]` → `["(s0*s1)"]` and static+dynamic →
`["(s0*4)"]`; the full worked example `[d0,d1,d2]` → collapse `[(s0*s1), s2]` → reduce `[(s0*s1)]`;
bail cases (non-projected-permutation `linalg.generic`; `tensor.expand_shape` splitting a dynamic
dim) → op and its downstream have no `afir.symbolic_shapes`; mixed static/dynamic arg.

**lit (verify):** intentionally-broken IR (wrong array length, out-of-range `sN`, static dim whose
`SymExpr` disagrees with the type) → `afir-verify-symbolic-shapes -verify-diagnostics`.

**Integration (TilePlanGen UB-peak):** lit test that an obviously-over-192KiB tiling case is absent
from `vector_plan.tiling_infos` and a reasonable one is present; reuse an existing dynamic-shape
`tests/torch_e2e/` test (e.g. `test_broadcast_add_reduce` with named dims `M`/`N`) to confirm the
autotuner still finds a PASS config and the numerics still match (pruning must not drop a correct
config); confirm the no-attr fallback (a kernel where symbolize bails) still tiles fine with no
pruning and unchanged results.

CI: unit tests in the `afir-opt` unit-test target; lit in the existing `test/` suite; integration
reuses `tests/torch_e2e/`. No new CI job.
