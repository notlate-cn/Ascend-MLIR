# Conversion Module Code Review — 2026-05-30

**Scope:** `lib/Conversion/` (~20.5K LOC, 16 submodules). Static code audit performed by 7 parallel reviewer agents, one per coherent slice. Findings are grounded in cited `file:line` + quoted code; **not every finding has been independently re-verified** — triage before acting.

**Method:** `superpowers:requesting-code-review` adapted for a static module audit (no diff). Each slice reviewed for correctness bugs, design issues, dead code, complexity, and split candidates.

**Architecture verdict:** The backbone (two linear pass pipelines — `torch-frontend`, `auto-fuse` + `auto-fuse-codegen` — communicating via IR attributes, not direct calls) is **healthy, readable, and well-documented**. The risk is concentrated in the *implementation* layer: correctness rests heavily on **unchecked implicit contracts** and **release-mode-disappearing `assert`s**, so violations **fail silently (wrong output, no error)** rather than diagnosing.

---

## Cross-cutting systemic patterns (the highest-value synthesis)

### ① Silent-drop / silent-fallback — the most dangerous recurring class
- `LinalgToAscendC/ComputeConversion.cpp:957+` — body walker `if(!lhs||!rhs) continue;`: when `resolve()` returns null the op is silently skipped **and its result is never registered**, so dependent ops chain-skip and the whole body tail vanishes / accumulator keeps a stale value. The existing `emitError` guards do **not** cover this path.
- **Recognize\* false-matches** (frontend) — the single biggest correctness risk in the module:
  - `RecognizeAttentionPass` decides "this is attention" from *"some `math.exp` reachable + one upstream bmm"* (no softmax/scale/`K^T` structure check); a kernelized/linear-attention or any `bmm2(exp(f(bmm1)))` folds to `FlashAttentionScore` → different computation.
  - Attention mask: when the mask fails the strict `[S,S]` filter it falls through to `maskArg = q4dyn`, **silently dropping a real causal mask → causal attention becomes bidirectional**. A `[B,1,S,S]` HF mask hits this.
  - `RecognizeLayerNorm`/`RecognizeBatchNorm` take gamma/beta as "the first user generic containing mulf" + "the other input" with no role/shape check at selection time → a residual branch (`layernorm(x)+residual`) mis-captures `residual` as `beta`.
- Many late passes `continue`/early-return on unhandled IR with only `LLVM_DEBUG`, surfacing as opaque downstream failures.

### ② Release-mode `assert` is the only correctness barrier (NDEBUG → silent miscompile)
- `TileFuse/TilePlanGen.cpp:1398` — `assert(bestScore < kInfeasible)`: in release, builds a kernel from an infeasible tiling draft with no diagnostic.
- `LinalgToAscendC/DataMoveConversion.cpp:130` — `assert(rank==2)` guards the dynamic-stride path; rank>2 dynamic-stride store computes addresses from the wrong stride in release.
- `TileFuse/TileFuseUtils.cpp:199` — terminal `llvm_unreachable` (release crash).

### ③ Address arithmetic relies on unchecked contracts (contiguity / 32B alignment / overflow)
- GM-flatten uses `dim(arg,1)` as the row stride — valid only for contiguous row-major; a non-contiguous GM view yields a wrong address (`AscendCBufferPlacement` C3, `AscendCPrepareForEmitPass.cpp:507`, `FlattenGMPtrPass` C4).
- `DataMoveConversion.cpp:174+` per-row `DataCopy` requires `cols*elemBytes % 32 == 0` — enforced only by an external generator + a comment; a tiling change silently corrupts adjacent rows.
- `TilePlanGen.cpp` footprint byte math (`numBufs*elems*elemBytes`) has no overflow guard → very large static dims wrap int64 negative → over-budget tile judged feasible.

### ④ Duplicated code that has drifted (fixes land in one copy only)
- **Two ASCIR text emitters** (`AFIRToASCIR/AFIRToASCIRText.cpp` 670 LOC + `AFIRToASCIRText/AFIRToASCIRText.cpp` 907 LOC) — both compiled and linked into afir-opt, both `#define GEN_PASS_DEF_CONVERTAFIRTOASCIRTEXTPASS` and define the external factory → **ODR violation; which one runs is link-order-dependent**; outputs differ. Headers also share the same include guard. **Most egregious; clear cleanup.**
- **Two GM-flatten implementations** (`AscendCPrepareForEmitPass` vs `FlattenGMPtrPass`) — capabilities have diverged (one handles chained 2D subviews, the other doesn't).
- **Three matmul recognizers** (`CubeEmitter`, `RestoreMatmul`, `Collapse`) with inconsistent predicates.
- Recognize\* matching primitives (`bodyHas`/`userGenericWithBody`/`otherInput`) copy-pasted across 4 passes.
- Two CfAssert passes (`TorchFrontend/RemoveCfAssert` vs `EliminateCfAssert`).

### ⑤ Positional ABI / role assumptions that break silently on signature change
- `AscendCBufferPlacement findRoleSubview` hard-codes lhs=arg0/rhs=arg1/bias=arg2/result=last.
- `TilePlanGen buildSchemaArgs` classifies args as workspace (i8) / input (identity layout) / output — mis-classifies quantized-i8 inputs and >1 real output.
- `CanonicalizeCannSignature` infers `cann.num_inputs` heuristically; a miscount shifts the runtime tensor-binding boundary (cf. `feedback_e2e_manifest_num_inputs` — mismatch hangs sim).

---

## Per-slice findings

### GroupAnalysis + Verify — 🟢 healthy (battle-hardened)
- **C2** `GroupAnalysisPass.cpp:216` creates `arith::ConstantOp`, `:125` clones `tensor::EmptyOp`, but `Passes.td` declares only Linalg+Func as `dependentDialects` → latent registration crash if Arith/Tensor not otherwise loaded. **(low-risk fix)**
- **C3** `CanFuse.cpp:323` E5 single-consumer guard only triggers for `linalg::MatmulOp`/`ascendc.unit`, but `isCubeOp` also classifies BatchMatmul/Conv2D/reduce-generic as Cube → bmm epilogue bypasses the fan-out guard.
- **C1** `GroupAnalysisPass.cpp:214` transpose const-fold lacks rank/perm validation before indexing `srcStrides[perm[d]]`.
- **I2** `wouldCreateCycle` recomputed O(N) per candidate pair inside an O(N²) fixpoint → scaling risk on GPT-2/BERT-size graphs.
- **I3/I4** `AxisLattice` `collectBoundaryIn(func)` ignores its `func` param; `computeCanonicalAxes` joins axes positionally with `staticSize=kDynamic` always — honest stubs that work only because their one consumer asks almost nothing.
- **M5** `GroupAnalysisPass::runOnOperation` ~300 LOC mixing 3 step-0 helpers + driver → extract `PreprocessHelpers`.

### GroupOutline — 🟢 healthy (excellent comments)
- **C1** `GroupOutlinePass.cpp:170` `boundaryOut` has no dedup (unlike `boundaryIn`); `:347` zips positionally with call results with **no `assert(size==numResults)`** — load-bearing undocumented contract.
- **C2** `:242-302` aclnn stamping: three independent `setAttr("aclnn.op",...)` blocks (Cube/pool/transpose) gated only by `size==1`, run without `else` → fragile, correct only because op-type sets happen not to intersect.
- **C3** `NetworkJsonEmitter.cpp:241` inline `DenseElementsAttr` tensor constant emits `"const"` with no value/resource key → runner can't materialize.
- **I3** `SplitRCoreGroup` seeds partial with `getZeroAttr` — wrong identity for max/min/prod full-reduce.
- **M1** `NetworkJsonEmitter.cpp` mixes `emitNetworkJson` (pipeline) + `emitNetworkProvenanceJson` (debug) in 520 LOC → split.

### TileFuse — 🟡 medium (largest god-function cluster)
- **C1** `LoopNestBuilder.cpp:97` overlap-tail uses *global* axis extent; `outerOfTailIV` is captured (`:124`) but **never read** → per-block tail composition was designed and dropped; interior-block tails miscompile when `extent%XBLOCK!=0 && XBLOCK%XBLOCK_SUB!=0`.
- **C2** `TileFuseUtils.cpp:166` `getAxisExtentValue` returns first matching operand dim — can return a broadcast operand's `1` instead of the loop extent (used for reductionExtent/peelExt/bcast bounds).
- **C3** `TilePlanGen.cpp:1398` infeasible-tiling `assert` only (see ②).
- **C4** footprint byte math overflow (see ③).
- **I5/I6** `costEstimate` (god-function: 7 early-return branches mixing infeasible-gate + soft-cost) and `buildPlan` (~190 LOC, 3 separate codegen bodies) are the two extract-first hotspots.
- **Split:** `TilePlanGen.cpp` (1952 LOC) → serialize / scheduler+cost / buildPlan / schema-emit.

### LinalgToAscendC — 🟡 medium
- **C1** reduce-path body walker (`ComputeConversion.cpp:954`) supports a *narrower* op set than the parallel path (`:2082`) — same arith op compiles in one generic shape, errors in another.
- **C2** `resolve()→continue` silent-drop (see ①).
- **C3** `subviewByteOffset:152` uses `getShape()[1]` as row stride (ignores parent `StridedLayoutAttr`); rank≠2 silently returns whole tensor.
- **C4** `DataMoveConversion.cpp:130` rank>2 dynamic-stride wrong (see ②).
- **C5** 32B-alignment contract unchecked (see ③).
- **Split:** `ComputeConversion.cpp` (2466 LOC, one ~2400-line function) → per-op-family files + shared Utils; unifying the body walker during the split closes C1+C2.

### AscendC backend passes — 🟡 medium (correctness-by-test)
- **C1** `AscendCRCoreCombinePass.cpp:226` `block_dim=ceildiv(ub,step)` assumes host launches exactly that many cores for `SyncAll(usedCores)` — no verification of the launch-grid invariant.
- **C3** `AscendCPrepareForEmitPass.cpp:507` 2D flat-offset uses col-count as row stride (contiguity, see ③).
- **C4** `FlattenGMPtrPass` ≥2-offset subview is treated as terminal (doesn't walk outer subviews) — `PrepareForEmit` does; divergent capability.
- **I1/C4** two `resolveGMChain` implementations drifted — unify into one `GMAddressing` util (closes C4 + most of I1).
- **I3** RCoreCombine returns early for non-f32 **after** doing nothing → non-f32 RCore kernel keeps the racy per-core write, no SyncAll → silent miscompile in release.
- **Split:** `AscendCBufferPlacement` (1650 LOC, but clean section banners) → (1) TPosition inference (Rule A/B/C), (2) memory-space rewrite, (3) copy insertion.

### Frontend / Recognize\* — 🔴 weakest (biggest correctness risk)
- **C1–C5** Recognize\* false-matches (see ①): attention exp/softmax/mask predicates too weak; mask silently dropped (causal→bidirectional); layernorm/batchnorm gamma/beta role mis-capture.
- **I1** `CanonicalizeCannSignature` num_inputs heuristic (see ⑤) — high-impact (this pass reorders args).
- **I6** `FuseGatherElementwise:94` selects first elementwise user without requiring `gatherResult.hasOneUse()` → a second consumer references the erased gather → verifier failure/miscompile.
- **M2/M3** Recognize\* primitives copy-pasted ×4; two CfAssert passes.

### ASCIR path (AFIRToASCIR + AFIRToASCIRText) — 🔴 stale / unmaintained
- **C1/C2** two divergent text emitters, both linked → ODR / duplicate-symbol; shared include guard drops one header. **Delete the stale older emitter.**
- **C4** new emitter `exec_order` hand-rolled off-by-one; Output node hardcodes `exec_order: 8` (frozen from an 8-node sample).
- **C5** `parseDataType` uses `isUnsigned()` on signless AFIR tensors → all unsigned tensors mistyped.
- **I1** `opInfoMap` op table drifted behind the dialect's own enum (no transpose/gather/cube/split entries → emitted as `compute_type 11 Invalid`, no diagnostic).
- **I4** `calculateStrides` ignores `vectorized_strides` attr → wrong strides for non-contiguous tensors.
- Only the dialect-conversion half (`AFIRToASCIR.cpp` + `Math/Elementwise.cpp`) has lit coverage and is clean; the text emitters have none.
- **Dead:** `generateSchedAttr` (old emitter, never called); `DialectBuilder.cpp` mostly unused.

---

## Recommended fix ordering (risk-tagged)

**A. Low-risk, behavior-neutral on valid input (fits the refactor session; keeps the 3 networks bit-equal):**
1. GroupAnalysis C2 — add `arith`/`tensor` to `dependentDialects`.
2. ASCIR C1/C2 — delete the stale duplicate emitter + fix the ODR/include-guard (ASCIR not in the e2e gate).
3. Turn the silent `assert` correctness-barriers into `report_fatal_error`/pass-failure (TileFuse C3, DataMove C4, RCore I3) — fail-loud, doesn't change passing behavior.
4. Dead-code removal (`generateSchedAttr`, `isEmbeddingGeneric` `(void)`-cast, dead params).

**B. Behavior-changing correctness fixes (need TDD + careful validation; belongs to bring-up/debug session):**
5. Recognize\* predicate tightening (attention softmax/mask, layernorm/batchnorm roles) — add a regression case per fix.
6. Unify the two ComputeConversion body walkers → any unresolved operand becomes a hard error (closes C1+C2).
7. Unify the two `resolveGMChain` implementations (closes C4 + most of I1).

**C. Structural splits (real but mid-flight — coordinate with CV-fusion owner):**
8. `ComputeConversion.cpp` (transpose bug lives here — §6 off-limits), `TilePlanGen.cpp`, `AscendCBufferPlacement`, `NetworkJsonEmitter.cpp`.

**Verification gate for every fix:** `ninja afir-opt` + full lit + the 3 reference networks (two-elewise / BERT-tiny / GPT-2) must stay bit-equal. See `docs/superpowers/notes/2026-05-30-refactor-handoff.md` §5.
