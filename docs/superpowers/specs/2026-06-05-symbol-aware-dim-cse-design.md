# Design: Symbol-aware `tensor.dim` CSE

Date: 2026-06-05
Status: approved (pending spec review)

## Problem

In dynamic-shape kernels, the same dynamic dimension is queried via `tensor.dim`
on several different SSA values. These queries compute the *same integer* but the
stock MLIR `--cse` cannot merge them, because CSE only collapses ops that are
syntactically identical (same operands). Two redundancy classes exist:

1. **Same-source duplicates** — e.g. `tensor.dim %arg0, %c1` appearing 5× because
   nothing ran CSE at this stage. Plain `--cse` *could* merge these.
2. **Cross-value, same-symbol** — e.g. `tensor.dim %arg0, %c1` and
   `tensor.dim %1, %c1` where `%1` is an intermediate whose dim 1 is the same
   symbol `s0`. Plain CSE **cannot** merge these — only the symbolic system knows
   they are equal.

Concrete measurement (`examples/gpt2-dyn-e2e/build_e2e/model_symbolized.mlir`):
9 `tensor.dim` ops, `afir.dim_symbols = [{arg=0, dim=1, id=0}]` (one symbol).
All 9 equal `s0`. 5 are `tensor.dim %arg0, %c1`; 4 query intermediate results.
Symbol-aware CSE collapses all 9 → **one** `tensor.dim %arg0, %c1`.

`two-elewise-dyn-e2e` is too simple to show this (every dim already on `%arg0`),
so `gpt2-dyn` is the demonstrating/regression case.

## Goal & non-goals

**Goals** (both confirmed with user):
- Cleaner mid-stage IR — one `tensor.dim` per symbol.
- Fewer `tensor.dim` ops for downstream passes to carry/reason about.

**Non-goals:**
- No effect on the *final* kernel: `PackTilingDataPass` already folds every
  `memref.dim` into one TilingData field per `(arg,dim)` root, so the emitted
  kernel has zero dim redundancy today. This pass only cleans the IR *between*
  symbolize-shapes and bufferization.
- Not constant-folding: a `tensor.dim` whose symbolic value is a literal constant
  is left for plain `--canonicalize`.
- Not handling compound symbolic dims (`s0*s1`, `ceildiv(...)`): those cannot be
  expressed as a single `tensor.dim`, so they are skipped.

## Inputs (already present in the IR)

Produced by `--afir-symbolize-shapes`, verified present:
- `func` attr `afir.dim_symbols`: array of `{arg, dim, id}` — serialized symbol
  `id` → its root `(arg, dim)`. Roots are always a block arg's dynamic dim.
- arg attr `afir.symbolic_shape` (singular): per-arg serialized `SymExpr` list,
  e.g. `"s0,s1,s2"`.
- op attr `afir.symbolic_shapes` (plural): per-result serialized `SymExpr` lists.

Parsing: `parseSymExprList` (in `Analysis/SymbolicShape/SymExpr.h`) turns a
serialized string back into `SmallVector<SymExpr>`; `SymExpr::getKind()/getSym()`
read off the symbol id. The ids in these attrs are the *serialized* ids, dense
`0..numRoots-1`, matching `afir.dim_symbols` array positions — so equal id ⇒
provably equal dim, and `afir.dim_symbols[id]` gives the canonical root directly.
No `DimSymbolTable` rebuild needed.

## Approach (chosen: standalone pass + fresh-anchor canonicalization)

New pass `afir-symbolic-dim-cse`, `Pass<"afir-symbolic-dim-cse", "func::FuncOp">`,
run **immediately after `--afir-symbolize-shapes`** in the same places that pass
runs (registered `--auto-fuse` pipeline and any `network_runner` phase-1
invocation). Lives at
`lib/Dialect/AFIR/Transforms/AFIRSymbolicDimCSE.cpp`, registered via
`include/Dialect/AFIR/Transforms/Passes.td`, mirroring `AFIRSymbolizeShapes`.

### Algorithm

```
0. Read func attr afir.dim_symbols → idToRoot: map<id, (argIdx, dimIdx)>.
   If absent (no dynamic dims / not symbolized) → return (nothing to do).

1. Resolve each tensor.dim. Walk every tensor::DimOp `d = tensor.dim %v, %c`:
   - require constant index c (skip otherwise).
   - get %v's serialized shape:
       * %v is a block arg of this func → func.getArgAttr(idx, "afir.symbolic_shape")
       * %v = op result            → defOp->getAttr("afir.symbolic_shapes")[resultNo]
     (skip if missing.)
   - parseSymExprList → element [c]. If Kind::Sym → key = getSym(). Else skip
     (constant or compound expr).
   - record d in classes[key].

2. Canonicalize each class with ≥2 members (or ≥1 member that is not already the
   canonical op):
   - root = idToRoot[key]; canonical value = `tensor.dim %arg{root.arg}, %c{root.dim}`.
   - Reuse: if some member is already exactly that op AND sits at/above all other
     members (entry block ⇒ trivially true when placed first), use it.
     Otherwise materialize one fresh `tensor.dim` at the top of the entry block
     (after the index constants). Index constant for root.dim is created/reused.
   - RAUW every other member to the canonical value; erase the now-dead members.
```

Entry-block placement is unconditionally safe: block args dominate the whole
body, so a `tensor.dim` on a root arg placed at entry dominates every original
use. This avoids needing `DominanceInfo`.

### Why fresh-anchor over reuse-earliest

Considered alternative B: never create a new op, pick the earliest existing
member that dominates the rest. Rejected: needs `DominanceInfo`, and a class with
no single dominating member can't be fully merged. Fresh-anchor fully collapses
every class to one op with simpler code; it introduces a new op only when ≥2
members exist (always a net reduction) and reuses an existing root-dim op when
one is already there.

## Correctness / safety

- **Provenance independence.** Rewriting `tensor.dim %arg3, 0` → `tensor.dim %arg0, 0`
  is safe because (a) `afir.dim_symbols` asserts both are the same symbol, and
  (b) after bufferization `PackTilingDataPass` canonicalizes `memref.dim` by
  `(arg,dim)→root` via the same `dim_symbols`, landing on one field regardless of
  which arg the dim came from. To verify during implementation: confirm tile-fuse
  consumes the `afir.symbolic_shapes`/`afir.iter_extents` attrs (not raw dim
  provenance) — grep its inputs.
- **Attribute staleness.** This pass does not change any op's *result types* or
  symbolic-shape attrs; it only redirects `tensor.dim` value uses. `afir.*`
  attrs remain valid.
- **Idempotent.** A second run finds each class already collapsed to one op → no
  change.

## Testing / success criteria

1. **lit** (`afir-opt --afir-symbolic-dim-cse`): a fixture with
   `afir.dim_symbols` + several same-symbol `tensor.dim` on different args/values
   ⇒ FileCheck that exactly one `tensor.dim` per symbol remains and uses count
   drops. Include a negative: a compound `s0*s1` dim is left untouched.
2. **gpt2-dyn regression** — `model_symbolized.mlir` 9 `tensor.dim` → 1 after the
   pass; full phase-5 e2e still PASS (sim), `max_diff` unchanged (7.15e-7).
3. **two-elewise-dyn / gelu-dyn** phase-5 PASS (no regression; nothing to merge).
4. **static graphs unaffected** — pass returns early when `afir.dim_symbols`
   absent.

## Files touched

- `include/Dialect/AFIR/Transforms/Passes.td` — pass def + `createAFIR…` decl.
- `lib/Dialect/AFIR/Transforms/AFIRSymbolicDimCSE.cpp` — new (~120 LOC).
- `lib/Dialect/AFIR/Transforms/CMakeLists.txt` — add source; link
  `AFIRSymbolicShape`.
- pipeline wiring: insert after `--afir-symbolize-shapes` (registered pipeline
  builder + `network_runner.py` phase-1 pass list — locate both during impl).
- `test/.../afir-symbolic-dim-cse.mlir` — new lit fixture.

## Open items to resolve during implementation

- Exact pipeline insertion points (registered `--auto-fuse` builder vs
  `network_runner` phase-1) — grep where `afir-symbolize-shapes` is listed.
- Confirm tile-fuse has no raw-dim-provenance dependency (safety check above).
