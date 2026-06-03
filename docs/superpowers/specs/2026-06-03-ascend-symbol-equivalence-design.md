# Ascend Symbol Equivalence Design

## Problem

The V2 design requires Layer 1 Normalize to prove symbolic dimension equality
once and expose that fact as `AscendSymbolConstraintAttr`. The current
implementation effectively treats shape facts as static ranks or string
constraints later in Schedule. That is not equivalent to the design:

- Dynamic ranked tensor dimensions that are equal by IR structure are not
  recorded.
- Kernelize cannot build `OpSemanticSummary.resultShape` as `DimExpr`.
- Kernelize cannot build `GlobalAxisSpace` or `OpAxisMap` from proven shape
  identity.
- Schedule cannot turn proven equal dimensions into
  `ShapeConstraint::DimEquality`.
- Guard elimination and dynamic-shape scheduling are forced to be conservative
  or static-shape-only.

This design restores the Layer 1 contract first. Later slices can upgrade
Kernelize and Schedule to consume it.

## Design Source

This implementation follows the V2 documents as the source of truth:

- Layer 1 architecture: weighted Union-Find symbolic equivalence analysis
  produces `AscendSymbolConstraintAttr`.
- V2-2 section 2.3.4: `DimRef(Value, dim)`, `DimExpr`, R1-R6 merge rules, and
  function-level `AscendSymbolConstraintAttr`.
- V2-2 section 2.6: `EntryNormalizationVerifier` must check symbol constraint
  completeness before Layer 2.
- V2-3 sections 3.3.3-3.3.4: Kernelize derives `OpSemanticSummary.resultShape`,
  `GlobalAxisSpace`, and `OpAxisMap` from the attr.
- V2-4 section 4.4.6: Schedule reads the attr and lifts proven equal dimensions
  to `ShapeConstraint::DimEquality`.

## Scope

This first implementation slice adds the producer-side semantic contract:

- A Normalize-stage symbol equivalence analysis over each `func.func`.
- A function-level `ascend.symbol_constraints` attribute that represents
  `AscendSymbolConstraintAttr`.
- Parser/helper APIs that resolve the serialized attribute back to `DimRef`
  semantics.
- An entry verifier check for structural validity and coverage of the R1-R6
  cases defined in the V2 design.
- Focused lit tests for the attr and verifier behavior.

This slice does not fully rewrite Kernelize, AxisCoalescer, or
ScheduleProblemBuilder. Those are follow-up slices that consume this contract.
The first slice may add read-only helpers in common headers so downstream work
does not invent a second format.

## Semantic Model

The compiler semantic model is exactly the V2 model:

```c++
struct DimRef {
  mlir::Value value;
  int64_t dim;
};
```

Two `DimRef`s are equivalent only when their dimension sizes are equal for all
valid runtime inputs. Static constant dimensions do not enter equivalence
classes; they are represented as constant `DimExpr` by consumers.

The analysis uses a weighted Union-Find over internal `DimRef` nodes. The
serialized attr is only a persistence format. It must not become the semantic
source of truth while the analysis is running.

## Attribute Contract

MLIR attributes cannot directly store `Value`, so the function-level attr uses a
deterministic function-local value reference. The helper API owns this encoding
and resolves it back to `Value + dim` before any downstream analysis consumes
the data.

Attribute name:

```text
ascend.symbol_constraints
```

Shape:

```mlir
ascend.symbol_constraints = [
  {
    sym_name = "arg0_dim0",
    members = [
      { value = 0 : i64, dim = 0 : i64 },
      { value = 2 : i64, dim = 0 : i64 }
    ]
  }
]
```

The `value` field is a stable function-local value ordinal:

1. Function block arguments are numbered first, in argument order.
2. Ranked tensor op results are numbered next, in block walk order, preserving
   result order within each op.
3. Non-ranked-tensor values are not assigned ordinals.

This encoding is intentionally local to the current function snapshot. Any pass
that mutates the tensor value graph or shape-affecting maps after Normalize must
rerun symbol equivalence before reusing the attr.

Helper APIs must provide:

- Build a `ValueOrdinalMap` for a `func.func`.
- Convert `DimRef` to `{value, dim}` and back.
- Look up the equivalence class for a `DimRef`.
- Verify uniqueness, rank bounds, non-empty classes, and unique `sym_name`s.

Downstream code must not parse raw dictionary attributes directly.

## Merge Rules

The analysis walks ops in function topology order and applies the V2 R1-R6
rules. IR shapes outside those rules, or ops rejected by Normalize, stay
independent; the analysis must never merge dimensions by guess.

### R1: Generic Indexing Maps

For `linalg.generic`, compare each pair of tensor operands/results. When their
indexing maps both project the same iterator dimension to tensor dimensions,
merge the corresponding `DimRef`s.

Broadcasted dimensions are not merged. Constant or non-projected expressions
are ignored for this rule.

### R2: Named Contraction Ops

For contraction-like named ops such as `linalg.matmul`, merge semantic
contraction dimensions by op interface or a narrow static rule table. For
matmul, merge `(lhs, 1)` with `(rhs, 0)`. Output dimensions are merged with
their corresponding input dimensions when the op semantics makes them equal.

This rule does not lower named ops to `linalg.generic` and does not depend on
generic indexing-map inference when a stronger op semantic rule exists.

### R3: Producer-Consumer SSA Edges

When an operand is exactly the same SSA value as a producer result, merge all
matching dimensions of that value. This records identity across use contexts and
keeps later result/operand summaries stable.

### R4: Extract Slice

For `tensor.extract_slice`, merge source and result dimensions when:

- stride is statically one, and
- the slice size is either `tensor.dim(source, d)` or statically equal to the
  source dimension, and
- the dimension is not a degenerate size-one collapse.

Stride changes, size-one degenerate slices, and unproven dynamic sizes do not
merge.

### R5: Tensor Dim Def-Use

Track `tensor.dim(v, d)` SSA results used as dynamic sizes or offsets for
shape-building ops such as `tensor.empty` and `tensor.extract_slice`. When the
def-use chain proves a target result dimension is the queried source dimension,
merge the source and target `DimRef`s.

This rule is only a proof rule for shape-carrying uses. Arbitrary index
arithmetic does not imply equality.

### R6: Linalg Broadcast

For `linalg.broadcast`, merge output non-broadcast dimensions with their
corresponding input dimensions according to the op `dimensions` attribute.
Dimensions that exist only in the output are broadcast axes and remain
independent.

## Naming

After Union-Find completes, assign one `sym_name` per non-static connected
component:

1. Prefer a function argument member name: `arg<index>_dim<dim>`.
2. Otherwise use deterministic generated names: `sym_<ordinal>`.

Names are unique within a function. The semantic identity of an equivalence
class is its member set, not the string name. Future fingerprints must ignore
symbol names and use the equivalence structure.

## Entry Verifier

`EntryNormalizationVerifier` validates:

- `ascend.symbol_constraints` exists on each normalized function.
- Each class has a unique non-empty `sym_name`.
- Each class has at least one member.
- Each member resolves to a ranked tensor value in the same function.
- Member dimensions are in range.
- A `DimRef` appears in at most one class.
- Static constant dimensions are not serialized as dynamic equivalence members.
- R1-R6 patterns in the function are covered by the attr.

The verifier is read-only. It reports Normalize-stage diagnostics and fails
closed when the contract is missing or inconsistent.

## Downstream Boundary

Layer 1 owns proving equality. Later layers are consumers:

- Kernelize fills `OpSemanticSummary.resultShape` by looking up each result
  `DimRef` in the attr and producing a `DimExpr`.
- Kernelize builds `GlobalAxisSpace` and `OpAxisMap` from equivalence classes.
- AxisCoalescer uses the attr to prove logical extent equivalence.
- ScheduleProblemBuilder lifts attr pairs into
  `ShapeConstraint::DimEquality`.
- Guard collection can eliminate guards already proven by the attr.

Schedule must not replace a missing attr by re-deriving ad hoc string equality.
If the attr is absent after Normalize, later stages should fail or report an
explicit unsupported dynamic-shape path.

## Diagnostics And Failure Policy

The analysis is conservative:

- Proven equality is serialized.
- Unproven equality remains independent.
- Inconsistent or malformed attr content is an error.
- Ops or shape expressions outside R1-R6 do not force false merges.

The first implementation slice must cover R1-R6 as specified. Any conservative
non-merge must be explained by being outside those rules, not by an incomplete
implementation of those rules.

## Tests

Add focused tests under `test/Conversion` or the existing Normalize test area:

- `linalg.matmul` + elementwise + reduction produces `M`, `N`, and `K` classes.
- `linalg.generic` elementwise indexing maps merge matching parallel axes.
- Broadcast output-only axes are not merged.
- `tensor.extract_slice` full-size stride-one slices merge dimensions.
- Degenerate or strided slices do not merge dimensions.
- `tensor.dim` dynamic-size def-use merges source and constructed result dims
  only when the use proves equality.
- Missing or malformed `ascend.symbol_constraints` fails the Normalize entry
  verifier.
- Static dimensions are represented as constants by consumers and are absent
  from equivalence classes.

The first red tests should assert the attr text form and verifier failure modes
before production code is added.

## Acceptance Criteria

- Normalize produces `ascend.symbol_constraints` for ranked dynamic tensor
  functions.
- R1-R6 supported cases have lit coverage.
- The attr round-trips through MLIR printing/parsing.
- Helper APIs resolve serialized members back to `DimRef(Value, dim)`.
- Entry verifier fails closed on missing, malformed, duplicated, or incomplete
  symbol constraints.
- Existing static-shape tests remain green.
- xvm focused lit verification passes before this contract is used for any
  candidate Runtime/NPU validation.
