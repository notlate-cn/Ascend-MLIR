# Refactor session handoff — 2026-05-30

**For:** the next session, whose role is **structural refactor / dead-code cleanup / big-file splits / interface extraction**. Strictly NOT the same role as the upstream session (debug tools + bring-up bugs).
**Branch start point:** `develop @ e4f7068f` (push pending; verify origin/develop is at e4f7068f or later before starting).
**Read first:** this file, then `docs/network-runner-e2e-walkthrough.zh.md`, then `docs/superpowers/notes/2026-05-29-debug-system-session-summary.md`.

---

## 1. Working-mode constraints (non-negotiable)

Per CLAUDE.md + project MEMORY (`feedback_discuss_before_editing`, `feedback_session_role_realnpu_runner`):

1. **Discuss every change before editing**. Even "obvious" cleanups need a 1-line confirmation from the user. Per-item.
2. **Surgical**. Each commit touches one file (or one logical group); no drive-by changes.
3. **Verify per commit**:
   - `ninja afir-opt` rebuild
   - `llvm-lit -sv build/test` full suite
   - Re-run the 3 reference networks: `two-elewise-e2e`, `bert-e2e`, `gpt2-e2e` phase 3 (~3 min total). Numbers MUST be identical (max_diff bit-equal). See §5 for the recipe.
4. **Never delete without proof of dead**. "Looks unused" is not enough. See §3 for the evidence-collection methods.
5. **Don't change behavior**. Refactor = same outputs, different code structure. If a refactor changes ANY observable output (kernel count, log lines, file paths, schema fields), it's not a refactor — it's a behavior change and needs separate justification.
6. **Match existing style**. The repo has MLIR conventions; resist the urge to introduce new patterns.
7. **No backwards-compat shims for things you remove**. If a function is truly dead, remove it cleanly; don't leave a deprecated stub.
8. **Externals submodules**: `externals/llvm-project` / `externals/pyasc` / `externals/stablehlo` are user-maintained symlink workarounds (MEMORY `externals_dir_vs_symlink`). NEVER modify, commit, or `git checkout --` them.

---

## 2. Module inventory (use this as your map)

### 2.1 `lib/` — 42 K LOC C++

| Submodule | LOC | Status |
|---|---|---|
| `lib/Conversion/AutoFuse/TileFuse/` | **5,168** / 12 files | Core fusion + codegen. **Not touched by upstream session.** Biggest unknown. |
| `lib/Conversion/AutoFuse/GroupOutline/` | 1,859 / 5 files | Heavily touched (provenance + pad + transpose stamps). |
| `lib/Conversion/AutoFuse/GroupAnalysis/` | 915 / 3 files | Heavily touched (3 step-0-pre helpers). |
| `lib/Conversion/AutoFuse/Verify/` | 191 / 1 file | Tiling-info schema verifier. |
| `lib/Conversion/LinalgToAscendC/` | 3,567 / 3 files | `ComputeConversion.cpp` is the elementwise/reduce/transpose lowering; `DataMoveConversion.cpp` is GM↔UB. |
| `lib/Conversion/AscendCPrepareForEmit/` | 1,650 / 5 files | Pre-emit transforms (alias resolution etc.). |
| `lib/Conversion/LowerNonLinalgOps/` | 926 / 5 files | Includes `AclnnFinalizeDeclPass` (aclnn.op stamp). |
| `lib/Conversion/MarkStructuredOps/` | 610 / 3 files | Mark cube/vec on linalg ops. |
| `lib/Conversion/AscendCRCoreCombine/` | 343 / 1 file | RCore phase-2 SyncAll combine (per MEMORY). |
| `lib/Conversion/AscendCFoldConcatAlloc/` | 325 / 1 file |  |
| `lib/Conversion/FuseGatherElementwise/` | 309 / 1 file |  |
| `lib/Conversion/CanonicalizeCannSignature/` | 239 / 1 file |  |
| `lib/Conversion/TorchFrontend/` | 201 / 3 files | Torch dialect → linalg bridge. |
| `lib/Conversion/AscendCParallelize/` | 176 / 1 file |  |
| `lib/Conversion/EliminateCfAssert/` | 67 / 1 file | Smallest. |
| `lib/Runtime/Mix/` | **5,284** / 20 files | **MEMORY: 950-specific NDDMA / v35**. Possibly not on current main path. Investigate before touching. |
| `lib/Runtime/Execution/` | 4,273 / 16 files | Session / scheduler / sim & npu backends. Main runtime. |
| `lib/Runtime/Artifact/` | 1,158 / 5 files |  |
| `lib/Runtime/Support/` | 823 / 5 files |  |
| `lib/Runtime/AclnnBackend/` | 747 / 1 file | Host C++ emitter (one big file — split candidate). |
| `lib/Runtime/Profile/` | 503 / 2 files |  |
| `lib/Dialect/AFIR/` | 1,957 / 11 files | Custom dialect. `Transforms/` has 9 passes. |
| `lib/Dialect/TmTensor/` | 90 / 1 file | Tiny. |
| `lib/Target/CannKernel/` | 3,757 / 3 files | AscendC MLIR → C++ translation (`afir-translate` core). |
| `lib/Utils/` | 45 / 1 file |  |

### 2.2 `tools/` — 1,928 LOC

| Binary | LOC | Note |
|---|---|---|
| `tools/autotuner/autotuner_main.cpp` | **1,131 / 1 file** | Big single-file. Phase-4 main loop. Split candidate but UNDERSTAND first. |
| `tools/runtime-session/` | 397 |  |
| `tools/mix-tiling-helper/` | 156 |  |
| `tools/mix-compiler/` | 88 |  |
| `tools/aclnn-backend/` | 56 | Thin wrapper around `lib/Runtime/AclnnBackend`. |
| `tools/afir-opt/` | 53 | Standard `MlirOptMain`. |
| `tools/afir-translate/` | 47 | Standard `MlirTranslateMain`. |

### 2.3 `python/` — 10.8 K LOC + 7.5 K MLIR

| File / dir | LOC | Status |
|---|---|---|
| `python/network_runner.py` | 963 | Main 5-phase driver. |
| `python/runner_utils/logger.py` | 220 | Recently touched. |
| `python/runner_utils/run_subprocess.py` | 66 | Recently touched. |
| `python/runner_utils/build_host.py` | 143 | g++ link step. |
| `python/runner_utils/harness.cpp` | 309 | phase-5 host entry. |
| `python/runner_utils/network_json.py` | 30 | Schema reader. |
| `python/tools/ascend_diff.py` | 337 | L0 + locate. |
| `python/tools/ascend_kernel_dag_viz/` | 324 | gen + viewer. |
| `python/afir-to-ascir/` | ? | **Possibly stale**. Old lowering tool? |
| `python/ascir-to-afir/` | ? | **Possibly stale**. Reverse direction. |
| `python/inductor_backend/` | ? | Torch inductor stub? Status unclear. |
| `python/dialects/` | ? | pybind dialect (maybe upstream MLIR python bindings local copy). |
| `python/torch/` | ? | Torch frontend? |
| `python/runtime/` | ? |  |
| `python/test/` + `python/tests/` | ? | **Two test dirs** — one may be stale. |
| `python/proto/` | ? | Protobuf? |

**The `?` rows are exactly the candidates for "is this dead?" investigation.**

### 2.4 `test/` — lit suite (104 tests, 102 pass + 2 unsupported)

```
test/Conversion/    — by-pass tests
test/Dialect/       — dialect op tests
test/Target/        — afir-translate
test/tools/         — examples / mix-tiling-helper
test/python/        — pybind
test/unittests/     — C++ gtest
```

### 2.5 `examples/` — 20+ directories

Two categories:

- **e2e-runnable** (have `run.sh`, drive `network_runner.py`): `two-elewise-e2e`, `bert-e2e`, `gpt2-e2e`, `resnet18-e2e`, `dyn-bucketed-e2e`, `mixed-attn-e2e`, `autotune-*`, `bcast-*`, `reduce-*`, `relu-*`, `combo-*`, `multi-r*`, `transpose-*-e2e`, `full-reduce-e2e`, `leading-reduce-e2e`, `aclnn-attn-e2e`.

- **hand-coded step0..step8 series** (likely legacy demo pipelines, NOT driven by network_runner): `broadcast-add-reduce/`, `gather-elementwise-fusion/`, `relu-broadcast-transpose/`, `matmul-add-leakyrelu/`, `matmul-add-relu-sum/`. These have `step0_input.mlir` through `step8_kernel.cpp` as snapshots of a pre-network_runner walkthrough. **Strong candidates for deletion or move to `docs/legacy-walkthroughs/`** — but verify nothing in `test/` or `CMakeLists.txt` references them first.

### 2.6 `docs/` — 75 K LOC markdown

- `docs/auto-fuse/` — design docs (`debug.md` §7 is the spec we built P0-P2 against)
- `docs/superpowers/notes/` — handoffs (many; chronological)
- `docs/superpowers/plans/` — plans
- `docs/network-runner-walkthrough.zh.md` — v1 internals
- `docs/network-runner-e2e-walkthrough.zh.md` — current walkthrough
- `docs/vector-plan.bak/` — name suggests `.bak` ⇒ legacy

**No active code dependency on docs** — safe to reorganize but ask before deleting.

---

## 3. High-confidence cleanup candidates (with evidence methods)

### 3.1 Dead-code identification — recipe to use

For ANY candidate file `X.cpp`, before claiming dead, run all of these:

```bash
# 1. CMakeLists reference?
grep -rn "$(basename X .cpp)" $(find . -name CMakeLists.txt) 

# 2. Header included by anyone?
grep -rn "$(basename X .cpp).h" lib/ include/ tools/ python/

# 3. Symbol used at link time? (after build)
nm build/lib/lib*.a 2>/dev/null | grep "$(some symbol from X)"

# 4. Recently touched?
git log --oneline --since="1 year ago" -- X.cpp | head

# 5. Any lit test depend on it?
grep -rn "$(some attribute from X)" test/
```

If 1-3 all empty and #4 is also empty, it's a high-confidence dead candidate.

### 3.2 Likely-dead candidates (need verification per above)

**Examples (low risk to remove if verified):**

- [ ] `examples/broadcast-add-reduce/step0..step8*` — legacy walkthrough
- [ ] `examples/gather-elementwise-fusion/step0..step8*`
- [ ] `examples/relu-broadcast-transpose/step0..step8*`
- [ ] `examples/matmul-add-leakyrelu/step0..step8*`
- [ ] `examples/matmul-add-relu-sum/step0..step8*`
- [ ] `docs/vector-plan.bak/` — `.bak` suffix strongly suggests legacy

**Python (need real investigation):**

- [ ] `python/afir-to-ascir/` — investigate scope first
- [ ] `python/ascir-to-afir/` — likely paired with above
- [ ] `python/inductor_backend/` — torch inductor backend; check if `examples/` or `test/` use it
- [ ] `python/test/` vs `python/tests/` — figure out which (if either) is current
- [ ] `python/proto/` — protobuf for what?

**lib/ (high risk — INVESTIGATE before assuming):**

- [ ] `lib/Runtime/Mix/` 5.3 K LOC — MEMORY says "950-specific NDDMA / v35, NOT in scope" for current AscendC vector codegen. But verify nothing in `lib/Conversion/` or runtime path links to it. If truly dead for current target, this is the single biggest cleanup.

**Build artifacts (must NOT commit):**

- `build/`, `cmake-build-debug/`, `install/`, `output/` — should be in `.gitignore`; verify and add if missing.

### 3.3 Untracked clutter at repo root (clean OR add to .gitignore)

`git status` shows lots of untracked: `architecture-simple.svg`, `architecture.svg`, etc. Inventory + classify:

- doc artifacts → move to `docs/` or delete
- e2e build products → `.gitignore` (already happens for some)
- generated `__v0_space.json` / `_kernel.cpp` in `examples/*` (not in `build_*/`) → `.gitignore`

---

## 4. Big-file split candidates

### 4.1 `tools/autotuner/autotuner_main.cpp` (1,131 LOC)

Single file containing argparse + phase-4 driving + tile-candidate generation + comparison + reporting. **Likely natural splits**:

```
tools/autotuner/
  autotuner_main.cpp           — argparse + dispatch (~150 LOC)
  CandidateGenerator.{h,cpp}   — tile-candidate enumeration
  CandidateRunner.{h,cpp}      — runtime-session-per-candidate
  GoldenCompare.{h,cpp}        — accuracy gating
  Reporter.{h,cpp}             — output / best-pick selection
```

**Process**: read entire file end-to-end → identify natural boundaries → propose split to user before doing.

### 4.2 `lib/Runtime/AclnnBackend/AclnnBackend.cpp` (747 LOC)

Contains `CoordEmitter` (walks coordinator, emits host C++) + helpers. Natural split:

```
lib/Runtime/AclnnBackend/
  AclnnBackend.cpp             — public API entry point
  CoordEmitter.cpp             — the walk
  HostEmitHelpers.cpp          — emitExtractSlice / emitPad / emitReturn etc.
```

Note: this file has been touched by multiple sessions (mine for pad, others for BN/Conv/Pool). Coordinate before re-org.

### 4.3 `lib/Runtime/Execution/NativeExecutionRunner.cpp` (804 LOC)

Big but might already be cohesive. **Investigate before splitting** — if it's one tightly-coupled state machine, splitting may hurt readability.

### 4.4 `lib/Runtime/Execution/GlobalScheduler.cpp` (715 LOC)

Same as above — investigate cohesion first.

### 4.5 `lib/Conversion/AutoFuse/GroupAnalysis/GroupAnalysisPass.cpp` (~700 LOC)

Recently grown by 3 step-0-pre helpers (unshare-empty / const-fold-transpose / fill-detach). Reasonable split:

```
lib/Conversion/AutoFuse/GroupAnalysis/
  GroupAnalysisPass.cpp        — main pass
  PreprocessHelpers.cpp        — the 3 step-0-pre helpers (each ~30 LOC)
  AxisLattice.cpp              — already split
  CanFuse.cpp                  — already split
```

(I'd do this myself but the work was scoped to "fix the bugs", not "refactor".)

### 4.6 `lib/Conversion/AutoFuse/GroupOutline/NetworkJsonEmitter.cpp` (555 LOC)

Two distinct emit functions (`emitNetworkJson` + `emitNetworkProvenanceJson`) sharing helpers. Natural split:

```
lib/Conversion/AutoFuse/GroupOutline/
  NetworkJsonEmitter.cpp           — emitNetworkJson + shared helpers
  NetworkProvenanceEmitter.cpp     — emitNetworkProvenanceJson
```

---

## 5. Verification recipe (run per commit)

```bash
cd /home/gser/code/Ascend-MLIR
export ASCEND_HOME_PATH=/home/gser/Ascend/cann
source examples/env.sh > /dev/null

# Build
cd build && ASCEND_HOME_PATH=/home/gser/Ascend/cann ninja afir-opt
cd ..

# Lit (fast)
/home/gser/anaconda3/envs/torch-mlir/bin/python3 \
  externals/llvm-project/build/bin/llvm-lit -sv build/test 2>&1 | tail -5
# expect: 102 PASS + 2 unsupported

# Three reference networks (must produce IDENTICAL numbers — change a digit, refactor is wrong)

# 1) two-elewise (smoke, ~10s)
WORK=/tmp/regress-twe && rm -rf $WORK
WORK=$WORK bash examples/two-elewise-e2e/run.sh 2>&1 | grep "network.output"
# expect: PASS max_diff=0 / PASS max_diff=0

# 2) BERT-tiny (~30s)
WORK=/tmp/regress-bert && rm -rf $WORK
conda run -n torch-mlir python examples/bert-e2e/export_bert.py \
  --hidden 64 --heads 1 --seq 8 --dtype fp32 --outdir $WORK > /dev/null 2>&1
PYTHONPATH=python python3 python/network_runner.py \
  --input-linalg $WORK/step0_linalg.mlir \
  --inputs $WORK/input_0.npy --expected $WORK/expected_0.npy \
  --workdir $WORK/work --max-phase 3 > /dev/null 2>&1
python3 python/tools/ascend_diff.py \
  --actual $WORK/work/output_default_0.npy \
  --expected $WORK/expected_0.npy --atol 1e-2 --rtol 1e-2
# expect: PASS max_diff=1.407e-05 (bit-equal)

# 3) GPT-2 small (~90s)
WORK=/tmp/regress-gpt2 && rm -rf $WORK
bash examples/gpt2-e2e/run.sh 3 $WORK > /dev/null 2>&1
python3 python/tools/ascend_diff.py \
  --actual $WORK/work/output_default_0.npy \
  --expected $WORK/expected_0.npy --atol 1e-2 --rtol 1e-2
# expect: PASS max_diff=8.345e-07 (bit-equal)

# Kernel counts (must be identical to baseline)
jq '.kernels | length' $WORK/work/groups/network.json
# expect: 197

# Cleanup
rm -rf /tmp/regress-*
```

**If ANY of these change** (PASS→FAIL, or max_diff drifts even slightly, or kernel count moves), the refactor changed observable behavior and is wrong — revert and investigate.

---

## 6. What to NOT touch

| Item | Why |
|---|---|
| `externals/*` | symlink workarounds per MEMORY |
| `lib/Conversion/AutoFuse/TileFuse/` | 5.2 K LOC tightly-coupled tile/fuse/emit; only refactor if you have a specific bug fix that needs it |
| `lib/Conversion/LinalgToAscendC/ComputeConversion.cpp` | MEMORY has the transpose codegen value-dependent bug here; left for a separate session |
| Provenance / DAG viz / ascend_diff / logger | Just-landed (last 3 days); let them sit |
| Anything touched by an in-flight commit on develop | grep `git log --since=...` to see what's hot |

---

## 7. Suggested first 3 commits (low-risk warmups)

In recommended order. STOP after each, run §5 verification, only proceed if clean.

### Commit 1: `.gitignore` polish

Add entries for the untracked build artifacts that clutter `git status` (`build_*` patterns under examples, `__v*_space.json`, `_kernel.cpp`, etc.). **No code change**.

### Commit 2: Move `docs/vector-plan.bak/` → delete or `docs/legacy/`

Pure doc movement. Zero code impact (verify no link references first via grep).

### Commit 3: Delete `examples/{broadcast-add-reduce,gather-elementwise-fusion,relu-broadcast-transpose,matmul-add-leakyrelu,matmul-add-relu-sum}/step{0..8}*`

But ONLY after running §3.1 evidence check per file and confirming with user. Likely 50+ files removed; could be impressive cleanup if confirmed dead.

After these warmups: open the bigger questions (Mix dir, autotuner split, AclnnBackend split) one at a time.

---

## 8. Reference state

- **HEAD at handoff**: `e4f7068f` (push pending — ALSO `edfe850b`)
- **2 commits to push first**: `git push origin develop` should fast-forward
- **Lit baseline**: 102 PASS + 2 unsupported
- **Network baselines** (must preserve bit-for-bit):
  - two-elewise: max_diff = 0 (both outputs)
  - BERT-tiny:   max_diff = 1.407e-05
  - GPT-2 small: max_diff = 8.345e-07, 197 kernels

If you're unsure about ANY change, the verification recipe in §5 takes ~3 minutes. Use it. The cost of a bad refactor that silently changes behavior is much higher than the cost of running a verification.

---

## 9. Things upstream session noticed but didn't pursue

- `Mix/` dir likely dead-for-current-target — investigation TBD
- `python/inductor_backend` / `afir-to-ascir` / `ascir-to-afir` status TBD
- `python/test` vs `python/tests` — duplication or rename in progress?
- `tools/autotuner/autotuner_main.cpp` 1,131 LOC — natural split exists
- `lib/Runtime/AclnnBackend/AclnnBackend.cpp` 747 LOC — natural split exists
- AscendC transpose codegen bug (MEMORY warns) — separate session

## 10. Anti-patterns to avoid

- **Don't rename for the sake of renaming**. If a name is bad but consistent, leave it. Inconsistent naming costs more.
- **Don't extract interfaces preemptively**. Need 2+ implementations OR a clear cross-cut use case.
- **Don't make `static` into class methods**. Free functions are fine when state isn't needed.
- **Don't introduce new abstraction layers**. The repo is direct-style; respect that.
- **Don't refactor while a feature is mid-flight**. Check `git log --since="3 days ago"` first.
