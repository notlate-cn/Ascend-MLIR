# Ascend Debug Stage Timeline Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Stage Timeline the single homepage timeline in `ascend-debug open`, with Debug Contract and artifact links attached to the relevant stage lane.

**Architecture:** Add a focused `timeline_model` module that converts existing run manifests and optional Debug Contract JSON files into a stable UI model. Keep `open_view.py` responsible for HTML rendering. Stage Timeline owns the visible flow: stage rows keep the existing per-pass browsing experience, a new `Artifacts / Contracts` column shows lane attachments, and manifest/runtime-only artifacts appear as synthetic rows in the same table. Stage MLIR remains source/fallback evidence, not the primary semantic interface.

**Tech Stack:** Python standard library, existing `ascend_debug` package, shell-based `test/tools/diagnostics/test_ascend_debug_cli.sh`, xvm verification workflow.

---

### Task 1: Add Timeline Model Builder

**Files:**
- Create: `tools/ascend-debug/ascend_debug/timeline_model.py`
- Test: `test/tools/diagnostics/test_ascend_debug_cli.sh`

- [x] **Step 1: Write failing checks**

Add checks after an `ascend-debug open` invocation for a full-codegen run:

```bash
if grep -Fq '<h2>End-to-End Timeline</h2>' "${TMP_DIR}/debug-run-full-codegen/index.html"; then
  echo "index should merge End-to-End Timeline into Stage Timeline" >&2
  exit 1
fi
if grep -Fq '<h2>Kernel / Runtime Artifacts</h2>' "${TMP_DIR}/debug-run-full-codegen/index.html"; then
  echo "index should merge Kernel / Runtime Artifacts into Stage Timeline" >&2
  exit 1
fi
grep -Fq '<thead><tr><th>Stage</th><th>Step / Per pass</th><th>View</th><th>Artifacts / Contracts</th><th>Command</th><th>Report</th></tr></thead>' "${TMP_DIR}/debug-run-full-codegen/index.html"
grep -Fq 'class="timeline-attachment-cell" data-lane-id="translate"' "${TMP_DIR}/debug-run-full-codegen/index.html"
grep -Fq 'class="timeline-attachment-cell" data-lane-id="artifacts"' "${TMP_DIR}/debug-run-full-codegen/index.html"
grep -Fq '<span class="timeline-source-badge legacy_adapter">legacy_adapter</span>' "${TMP_DIR}/debug-run-full-codegen/index.html"
grep -Fq 'kernel.cpp' "${TMP_DIR}/debug-run-full-codegen/index.html"
grep -Fq 'host_tiling.cpp' "${TMP_DIR}/debug-run-full-codegen/index.html"
grep -Fq 'artifact_manifest.json' "${TMP_DIR}/debug-run-full-codegen/index.html"
```

- [x] **Step 2: Run checks and verify failure**

Run:

```bash
bash test/tools/diagnostics/test_ascend_debug_cli.sh
```

Expected: fails because the homepage does not contain `End-to-End Timeline` or lane ids.

- [x] **Step 3: Implement model builder**

Create `timeline_model.py` with:

```python
LANE_ORDER = ("source", "normalize", "kernelize", "schedule", "realize", "translate", "artifacts", "runtime")

def build_timeline_model(run_dir: pathlib.Path, manifest: dict[str, Any]) -> dict[str, Any]:
    ...
```

The builder:

- groups stages by existing `phase` or name inference;
- attaches current artifacts by explicit `producer_stage` if present;
- falls back to kind/path mapping for current manifests;
- attaches optional contract files when present;
- marks compatibility-derived entries with `source = "legacy_adapter"`;
- preserves unknown artifact fields in `raw`.

- [x] **Step 4: Add direct Python assertions**

Add a Python assertion block in `test_ascend_debug_cli.sh` that imports `ascend_debug.timeline_model`, loads the generated full-codegen manifest, and asserts:

```python
translate = lane("translate")
assert any(item["path"] == "kernel.cpp" for item in translate["artifacts"])
assert any(item["path"] == "host_tiling.cpp" for item in translate["artifacts"])
artifacts = lane("artifacts")
assert any(item["path"] == "artifact_manifest.json" for item in artifacts["artifacts"])
assert model["source"] in ("legacy_adapter", "mixed")
```

- [x] **Step 5: Run test and verify pass**

Run:

```bash
bash test/tools/diagnostics/test_ascend_debug_cli.sh
```

Expected: pass locally if fake-tool diagnostics are self-contained.

### Task 2: Merge Attachments Into Stage Timeline

**Files:**
- Modify: `tools/ascend-debug/ascend_debug/open_view.py`
- Test: `test/tools/diagnostics/test_ascend_debug_cli.sh`

- [x] **Step 1: Add attachment renderer**

Add Stage Timeline helpers in `open_view.py`:

- map stage groups to timeline lanes;
- render contract/artifact/diagnostic attachments in a row-spanned table cell;
- prefer visualized artifact/json/report views as the primary link and keep raw paths as secondary links;
- render `artifacts` and `runtime` lanes as synthetic Stage Timeline rows when they have attachments but no stage dump.

- [x] **Step 2: Wire renderer into homepage**

In `render_index`, build the timeline model after artifact/report/stage/json views are available and pass it to `_stage_rows(...)`.

- [x] **Step 3: Remove duplicate homepage sections**

Do not render a standalone end-to-end section. Do not render a standalone Kernel / Runtime Artifacts section on the homepage. Keep artifact dashboard pages and raw artifacts available through Stage Timeline links.

- [x] **Step 4: Run shell diagnostics**

Run:

```bash
bash test/tools/diagnostics/test_ascend_debug_cli.sh
```

Expected: existing Stage Timeline assertions still pass and the new attachment assertions pass.

### Task 3: Verify With Real Debug Output

**Files:**
- No source edits expected.

- [x] **Step 1: Build on xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/dev/null 2>&1 && ninja -C build ascend-debug && bash test/tools/diagnostics/test_ascend_debug_cli.sh test/tools/diagnostics/ascend-debug-cli.mlir'
```

Expected: `ALL ASCEND DEBUG CLI TESTS PASSED`.

- [x] **Step 2: Generate debug output on xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/dev/null 2>&1 && rm -rf /tmp/codex-ascend-debug-e2e-timeline && ascend-debug collect test/tools/diagnostics/ascend-debug-cli.mlir --out /tmp/codex-ascend-debug-e2e-timeline --mode deep && ascend-debug open /tmp/codex-ascend-debug-e2e-timeline --no-browser'
```

Expected: output includes `ascend-debug.open.index=.../index.html`.

- [x] **Step 3: Browser check**

Serve the generated output locally and verify:

- homepage opens;
- Stage Timeline has the `Artifacts / Contracts` column;
- no standalone end-to-end or Kernel / Runtime Artifacts section appears;
- `Translate` attachment cell links to `kernel.cpp` and `host_tiling.cpp`;
- `Artifacts` synthetic row links to `artifact_manifest.json`;
- existing debug graph link still opens.

Result note: generated HTML was verified with local HTTP `curl` after serving the xvm output.

- [x] **Step 4: Final hygiene**

Run:

```bash
git diff --check
python3 - <<'PY'
import pathlib
import re

pattern = re.compile(r"\b" + "a" + "fir" + r"\b|" + "A" + "FIR|" + "a" + "fir-", re.IGNORECASE)
paths = [
    pathlib.Path("tools/ascend-debug"),
    pathlib.Path("docs/superpowers/specs/2026-06-06-ascend-debug-end-to-end-timeline-design.md"),
    pathlib.Path("docs/superpowers/plans/2026-06-06-ascend-debug-end-to-end-timeline-implementation.md"),
]
for root in paths:
    files = root.rglob("*") if root.is_dir() else [root]
    for path in files:
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            if pattern.search(text):
                raise SystemExit(f"unexpected legacy frontend name in {path}")
PY
```

Expected: no whitespace errors; no current-surface legacy frontend naming.
