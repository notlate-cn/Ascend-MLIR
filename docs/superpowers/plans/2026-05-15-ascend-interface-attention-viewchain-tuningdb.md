# Ascend Interface, Attention, View-Chain, And Tuning DB Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` when implementing this plan. Track progress with the checkbox items below.

**Goal:** Complete the next four commercial-hardening tracks together:

1. Kernelize registry evolves into a real `OpInterface` / trait model.
2. Attention / FlashAttention-like grouped path is closed beyond Kernelize.
3. Realize dynamic view-chain movement is hardened and broadened.
4. Schedule persistent cache becomes a versioned tuning DB.

**Architecture:** Keep `KernelizeOpSemanticInfo` as the single semantic contract. Make native `KernelizeSemanticOpInterface` and fallback registry share the same semantic helpers. Carry handwritten attention metadata from Kernelize into Schedule and Realize instead of relying on a generic cube family. Make Realize view-chain materialization preflighted and reportable. Replace line-based tuning signatures with a schema-versioned schedule tuning database while preserving old cache options as compatibility shims.

**Current Baseline:**

- `KernelizeSemanticOpInterface` exists, but `KernelizeExternalModels.cpp` only registers empty dialect extensions; default semantic logic still lives in `KernelizeOpRegistry.cpp`.
- Attention-like SDPA detection exists in `FusionCandidateAnalysis.cpp`, but it only produces a Kernelize `HandwrittenPattern` candidate and currently reports family `cube`.
- `TemplateRegistry.cpp` only has `vector_generic`, `reduction_static`, `cube_static_matmul`, and `memory_copy`.
- Realize already has selected dynamic view-chain support and `ascend-realize-dynamic-view-chain-movement.mlir`; the next step is hardening, refactoring, and extending unsupported/deferred coverage.
- Schedule persistent cache I/O is a line-based list of tuning signatures, not a target-aware tuning DB.

Do not modify or stage `docs/Ascend-MLIR-V2-Problem-Formulation.zh.md`; it is user-owned dirty state in the current worktree.

---

## Delivery Shape

Implement as one integrated delivery, but keep internal commits/review slices separate:

1. **Contract Slice:** Kernelize semantic interface and shared helpers.
2. **Attention Slice:** Kernelize metadata, Schedule template, Realize/backend integration, full-pipeline smoke.
3. **Realize Slice:** Dynamic view-chain preflight/materialization hardening.
4. **Tuning DB Slice:** Persistent schedule DB schema, parser/writer, cache model integration.
5. **Integration Slice:** tracking doc, examples, xvm verification, final review.

The slices can be worked in parallel after the Contract Slice lands. Attention depends on Contract. Realize and Tuning DB are mostly independent.

---

## File Ownership

### Contract Slice

- Modify: `include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h`
- Modify: `include/Conversion/Ascend/Kernelize/KernelizeOpInterfaces.td`
- Modify: `include/Conversion/Ascend/Kernelize/KernelizeExternalModels.h`
- Modify: `lib/Conversion/Ascend/Kernelize/KernelizeExternalModels.cpp`
- Create: `lib/Conversion/Ascend/Kernelize/KernelizeSemanticUtils.h`
- Create: `lib/Conversion/Ascend/Kernelize/KernelizeSemanticUtils.cpp`
- Modify: `lib/Conversion/Ascend/Kernelize/KernelizeOpRegistry.cpp`
- Modify: `lib/Conversion/Ascend/Kernelize/DependencyAnalysis.cpp`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`
- Modify: `test/unittests/Conversion/AscendKernelizeOpInterfaceTest.cpp`
- Create/modify: `test/Conversion/ascend-kernelize-op-interface-*.mlir`

### Attention Slice

- Modify: `include/Conversion/Ascend/Common/Attributes.h`
- Modify: `include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h`
- Modify: `lib/Conversion/Ascend/Kernelize/FusionCandidateAnalysis.cpp`
- Modify: `lib/Conversion/Ascend/Kernelize/KernelPattern.cpp`
- Modify: `lib/Conversion/Ascend/Schedule/KernelPatternView.cpp`
- Modify: `lib/Conversion/Ascend/Schedule/ScheduleProblemBuilder.cpp`
- Modify: `lib/Conversion/Ascend/Schedule/TemplateRegistry.cpp`
- Modify: `lib/Conversion/Ascend/Schedule/ScheduleSearch.cpp`
- Modify: `lib/Conversion/Ascend/Realize/MemoryRealizationDriver.cpp`
- Modify backend files only if full-pipeline smoke exposes a real lowering gap.
- Modify: `test/Conversion/ascend-kernelize-attention-handwritten-pattern.mlir`
- Create: `test/Conversion/ascend-schedule-attention-handwritten-pattern.mlir`
- Create: `test/Conversion/ascend-full-pipeline-attention-handwritten-smoke.mlir`

### Realize Slice

- Modify: `lib/Conversion/Ascend/Realize/BufferizationDriver.cpp`
- Modify: `lib/Conversion/Ascend/Realize/MemoryRealizationDriver.cpp`
- Modify: `lib/Conversion/Ascend/Realize/RealizeTypes.h`
- Modify: `lib/Conversion/Ascend/Realize/RealizeReport.cpp`
- Modify/create: `test/Conversion/ascend-realize-dynamic-view-chain-movement.mlir`
- Create: `test/Conversion/ascend-realize-dynamic-view-chain-deferred.mlir`
- Modify: `test/unittests/Conversion/AscendRealizePlannerTest.cpp`

### Tuning DB Slice

- Modify: `include/Conversion/Passes.td`
- Modify: `lib/Conversion/Ascend/Schedule/ScheduleTypes.h`
- Modify: `lib/Conversion/Ascend/Schedule/ScheduleCache.h`
- Modify: `lib/Conversion/Ascend/Schedule/ScheduleCache.cpp`
- Replace/extend: `lib/Conversion/Ascend/Schedule/SchedulePersistentCacheIO.h`
- Replace/extend: `lib/Conversion/Ascend/Schedule/SchedulePersistentCacheIO.cpp`
- Create: `lib/Conversion/Ascend/Schedule/ScheduleTuningDB.h`
- Create: `lib/Conversion/Ascend/Schedule/ScheduleTuningDB.cpp`
- Modify: `lib/Conversion/Ascend/Schedule/SchedulePass.cpp`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`
- Modify: `test/Conversion/ascend-schedule-persistent-cache.mlir`
- Create: `test/Conversion/ascend-schedule-tuning-db.mlir`
- Create: `test/Conversion/ascend-schedule-tuning-db-invalid.mlir`

### Shared

- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`
- Run: `test/tools/check_ascend_no_v2_code_naming.sh`
- Run: `test/tools/check_ascend_public_headers.sh`

---

## Task 1: Kernelize OpInterface / Trait Model

**Intent:** New op semantics should be added by interface or external model registration, not by editing `DependencyAnalysis.cpp` or cloning local `isa<>` checks.

- [ ] **Step 1: Add RED tests for real external models**

  Add or extend unit tests so `KernelizeOpModelRegistry::resolve()` first queries `KernelizeSemanticOpInterface` when an op implements it. Add a LIT that runs `afir-opt` with default `afir-opt` registry and proves linalg/tensor semantics come from registered external models.

  Required checks:

  - linalg contraction / reduction / layout / broadcast semantics are unchanged.
  - tensor view ops are `Transparent`.
  - arith constant remains `Ignore`.
  - fallback registry still works when no external model is registered.

- [ ] **Step 2: Extract shared semantic helpers**

  Move semantic logic out of `KernelizeOpRegistry.cpp` into `KernelizeSemanticUtils.{h,cpp}`:

  - iterator conversion
  - result-rank collection
  - indexing-map collection
  - parallel indexing classification
  - contraction classification
  - linalg semantic population
  - tensor-view semantic population
  - arith constant semantic population

  `KernelizeOpRegistry.cpp` becomes a fallback adapter, not the semantic owner.

- [ ] **Step 3: Implement external models**

  In `KernelizeExternalModels.cpp`, attach external models for:

  - `linalg::LinalgOp`
  - `arith::ConstantOp`
  - `tensor::CastOp`
  - `tensor::CollapseShapeOp`
  - `tensor::ExpandShapeOp`
  - `tensor::ExtractSliceOp`
  - `tensor::ReshapeOp`

  These models must call the shared helpers from Step 2.

- [ ] **Step 4: Update resolver precedence**

  `KernelizeOpModelRegistry::resolve()` should use this order:

  1. Native or external `KernelizeSemanticOpInterface`.
  2. Explicit registry model fallback.
  3. Unsupported semantic result with diagnostic reason.

  Unsupported ops must not silently look like analyzed zero-dependency ops. Keep fail-closed behavior but make the reason observable in debug report.

- [ ] **Step 5: Verification**

  Run:

  ```bash
  ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt AscendKernelizeOpInterfaceTest'
  ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ctest --test-dir build -R "AscendKernelizeOpInterfaceTest" --output-on-failure'
  ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-kernelize-op-interface-*.mlir'
  ```

---

## Task 2: Attention / FlashAttention-Like Path

**Intent:** The SDPA-like pattern must not stop at Kernelize. It must carry a stable handwritten pattern contract into Schedule, Realize, and full-pipeline lowering. The initial target is a correct grouped attention path, not a peak-performance FlashAttention kernel.

- [ ] **Step 1: Add explicit handwritten pattern metadata**

  Add centralized attrs/constants:

  - `ascend.kernelize.handwritten_kind`
  - value `attention_sdpa`
  - optional value `flash_attention_candidate`

  `FusionCandidateAnalysis.cpp` should set the kind when it recognizes:

  - first cube op: QK or score matmul
  - reduction row-max or row-sum
  - vector chain for score normalization
  - second cube op: probability/value matmul

  Keep `MustCoLocate` edge generation for this candidate so partition cannot split it.

- [ ] **Step 2: Preserve the pattern contract into Schedule**

  Extend `KernelPatternView` and `ScheduleProblem` so the handwritten kind is visible to Schedule. `buildScheduleProblem()` should append template tags:

  - `cube`
  - `reduction`
  - `vector`
  - `attention_sdpa`

  Add structure constraints such as:

  - `attention_sdpa_chain`
  - `handwritten_group`

- [ ] **Step 3: Add Schedule template and search support**

  Add a template registry entry:

  ```text
  family = attention_sdpa
  name = grouped_tile_per_block
  tags = [attention_sdpa, cube, reduction, vector]
  minRank = 2
  maxRank = 4
  priority < cube_static_matmul
  ```

  `ScheduleSearch.cpp` should generate conservative valid instances for this template:

  - tile all visible batch/head/query axes consistently
  - keep reduction/K axis visible in guard/tail plans
  - emit `schedule_family = "attention_sdpa"`
  - never fall back to plain `cube_static_matmul` when `attention_sdpa` is present

- [ ] **Step 4: Close Realize/backend behavior**

  Realize must treat the attention group as one kernel contract:

  - no partial output bridge that breaks the group
  - selected movement paths must be counted per actual materialized movement
  - dynamic view-chain copies inside the group use the same materializer as Task 3

  Backend lowering must either:

  - lower all constituent linalg ops through existing supported body classifiers, or
  - emit a precise diagnostic anchored on the unsupported op/kind.

  Silent fallback to separate kernels is not allowed.

- [ ] **Step 5: Add end-to-end tests**

  Required tests:

  - `ascend-kernelize-attention-handwritten-pattern.mlir`: now checks `handwritten_kind = "attention_sdpa"` and family `attention_sdpa`.
  - `ascend-schedule-attention-handwritten-pattern.mlir`: checks template match `attention_sdpa/grouped_tile_per_block`.
  - `ascend-full-pipeline-attention-handwritten-smoke.mlir`: runs normalize -> kernelize -> schedule -> realize -> compute-lower and checks no `linalg.` survives, or checks a deliberate diagnostic if a backend operation is intentionally unsupported.

  The preferred acceptance is full-pipeline pass with no `linalg.` left. A diagnostic-only result is acceptable only if it is explicitly recorded in the tracking doc as a backend non-goal for this slice.

---

## Task 3: Realize Dynamic View-Chain Hardening

**Intent:** Dynamic view-chain movement should be correct, preflighted, and observable. It should not leave half-mutated IR when one use cannot be rewritten.

- [ ] **Step 1: Add RED/guard tests**

  Extend `ascend-realize-dynamic-view-chain-movement.mlir` and add `ascend-realize-dynamic-view-chain-deferred.mlir` for:

  - dynamic `tensor.extract_slice` -> bufferized `memref.subview`
  - `tensor.cast`
  - `tensor.expand_shape`
  - `tensor.collapse_shape`
  - `tensor.reshape`
  - multi-use producer with two dynamic view users
  - negative/deferred case for an unsupported non-view producer such as concat

- [ ] **Step 2: Refactor view-chain collection and materialization**

  Keep one path for both workspace movement and Phase5 bridge:

  - `collectViewChainToSource`
  - `materializeMovementUseViewChain`
  - dynamic operand detection
  - deferred reason reporting

  If this makes `MemoryRealizationDriver.cpp` too dense, split into a private `RealizeViewChainMaterializer.{h,cpp}` under `lib/Conversion/Ascend/Realize`.

- [ ] **Step 3: Add full preflight**

  Before rewriting any IR for a movement group:

  - collect every consumer rewrite
  - verify dominance and block ordering
  - verify each view op can be reconstructed on the target memory space
  - verify dynamic sizes can be sourced from original alloc/view operands
  - verify no unsupported region-crossing use exists

  If any preflight fails, do not mutate that group. Increment `deferred_view_chain_rewrites` and emit debug report detail.

- [ ] **Step 4: Keep counts tied to IR mutation**

  `dynamic_view_chain_rewrites` increments only for dynamic view uses actually rewritten.

  `deferred_view_chain_rewrites` increments only for dynamic view uses intentionally skipped after preflight.

  `materialized_allocs` and `materialized_copies` must reflect inserted IR, not plan demand.

- [ ] **Step 5: Verification**

  Run:

  ```bash
  ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-realize-dynamic-view-chain-*.mlir'
  ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ctest --test-dir build -R "AscendRealizePlannerTest" --output-on-failure'
  ```

---

## Task 4: Schedule Persistent Cache -> Tuning DB

**Intent:** Persistent schedule reuse must be target-aware, schema-versioned, and able to represent selected decisions and negative entries. A raw line list of signatures is not enough.

- [ ] **Step 1: Define DB data model**

  Add `ScheduleTuningDB.{h,cpp}` with:

  - `constexpr unsigned kScheduleTuningDBSchemaVersion = 1`
  - `ScheduleTuningRecord`
  - `ScheduleNegativeRecord`
  - `ScheduleTuningDatabase`
  - parse/write helpers

  Record identity includes:

  - schema version
  - target identity: `soc`, target tile policy, optional CANN profile id
  - schedule problem signature: family, template tags, result shape, element type bit width, axis constraints, structure constraints
  - selected decision: family, template, tile shape, tail policy, runtime top-k position
  - optional score fields for future autotuning

- [ ] **Step 2: Add pass options**

  In `Passes.td`, add:

  - `tuning-db-in`
  - `tuning-db-out`

  Keep `tuning-cache-in/out` for compatibility. Old cache files seed only legacy signature hits and are written only when the old output option is used.

- [ ] **Step 3: Integrate with `ScheduleCacheModel`**

  `ScheduleCacheModel` should seed records from `ScheduleTuningDatabase` instead of raw strings.

  Required behavior:

  - target mismatch = miss
  - schema mismatch = hard diagnostic
  - matching selected decision = persistent hit
  - matching negative record = negative cache hit
  - guard-budget pruning is not counted as negative cache entry

- [ ] **Step 4: Round-trip writer**

  `tuning-db-out` writes the complete DB after the pass:

  - previously loaded records
  - newly selected records
  - newly observed negative records
  - stable sorted order for FileCheck

  The writer should not depend on unordered JSON key order. Prefer a deterministic line-oriented schema:

  ```text
  # ascend.schedule.tuning_db schema=1
  record schema=1 target=Ascend910B2 policy=legacy-default problem=... family=... template=... tile=... tail=... score=...
  negative schema=1 target=Ascend910B2 policy=legacy-default problem=... reason=no_template
  ```

- [ ] **Step 5: Tests**

  Add:

  - `ascend-schedule-tuning-db.mlir`: first run writes DB, second run reads it and reports persistent hit.
  - `ascend-schedule-tuning-db-invalid.mlir`: schema mismatch and malformed line diagnostics.
  - update `ascend-schedule-persistent-cache.mlir`: old cache options still work.
  - update `ascend-schedule-cache.mlir`: negative hit counters are separated from guard-budget pruning.

---

## Integration And Verification

- [ ] **Step 1: Update tracking doc**

  Update `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md` with four status rows:

  - Kernelize OpInterface / trait model: closed when external models are real.
  - Attention / FlashAttention-like path: closed or explicitly backend-deferred with diagnostic.
  - Realize dynamic view-chain: closed when pass/deferred counts are covered.
  - Schedule tuning DB: closed when schema DB round-trips.

- [ ] **Step 2: Run focused LIT**

  ```bash
  ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-kernelize-op-interface-*.mlir build/test/Conversion/ascend-kernelize-attention-handwritten-pattern.mlir build/test/Conversion/ascend-schedule-attention-handwritten-pattern.mlir build/test/Conversion/ascend-realize-dynamic-view-chain-*.mlir build/test/Conversion/ascend-schedule-tuning-db*.mlir'
  ```

- [ ] **Step 3: Run broader conversion LIT**

  ```bash
  ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion'
  ```

- [ ] **Step 4: Run unit tests and guards**

  ```bash
  ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt AscendKernelizeOpInterfaceTest AscendRealizePlannerTest AscendKernelPatternTest'
  ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ctest --test-dir build -R "Ascend(KernelizeOpInterface|RealizePlanner|KernelPattern)Test" --output-on-failure'
  ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && test/tools/check_ascend_no_v2_code_naming.sh'
  ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && test/tools/check_ascend_public_headers.sh'
  ```

- [ ] **Step 5: Verify examples**

  Run all maintained demos under `examples/`, with at least:

  ```bash
  ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && examples/relu-broadcast-transpose/run-mainline.sh'
  ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && examples/matmul-add-leakyrelu/run-mainline.sh --log'
  ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && examples/transformer/run-mainline.sh'
  ```

  If a script name has drifted, discover current maintained demo scripts with:

  ```bash
  find examples -maxdepth 2 -type f -name 'run*.sh' -print | sort
  ```

- [ ] **Step 6: Final review**

  Request/perform two reviews:

  - spec compliance: no new upper-layer op-name branch hardcoding, no silent fallback, docs/tracking aligned.
  - code quality: public/private header boundary, deterministic DB output, mutation preflight safety, tests scoped to real behavior.

---

## Acceptance Criteria

- Kernelize semantics can be supplied by `KernelizeSemanticOpInterface` / external models; fallback registry remains compatibility only.
- Attention SDPA-like group carries a stable `attention_sdpa` contract through Kernelize and Schedule.
- Attention path either fully lowers in the main pipeline or fails with a precise backend diagnostic; it must not silently split or drop the group.
- Realize dynamic view-chain materialization is preflighted and its report counts match actual inserted IR.
- Schedule tuning DB is schema-versioned, target-aware, deterministic on disk, and backwards compatible with old line-based cache options.
- `examples/` maintained demos still run after the changes.
- `docs/Ascend-MLIR-V2-Problem-Formulation.zh.md` remains untouched.
