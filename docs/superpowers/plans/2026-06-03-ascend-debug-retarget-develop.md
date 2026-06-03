# ascend-debug Re-target to `develop` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port dev-nyh's `ascend-debug` visual debugger to `develop` as a standalone tool, MVP = `collect` + `open` with the per-stage structural graph, re-targeted to `develop`'s `afir-opt` pipeline. Zero C++.

**Architecture:** A self-contained Python package under `tools/ascend-debug/` exposing an `ascend-debug` CLI. `collect` drives `develop`'s lowering in 6 coarse stages via sequential `afir-opt` invocations (each consuming the prior stage's IR), writing numbered IR artifacts + `manifest.json`. `open` renders a read-only HTML dashboard (stage navigation + per-stage IR views + per-stage structural graph). The structural graph's generic structure layer works on any IR; its semantic badges are remapped from dev-nyh's `ascend.*` attrs to `develop`'s `auto_fuse.*`/`aclnn.*`/`afir.*` attrs.

**Tech Stack:** Python 3 stdlib, `afir-opt` (built MLIR tool), CMake install wrapper, LLVM lit/FileCheck, static HTML/JSON.

**Source of truth for ported files:** `upstream/dev-nyh:tools/ascend-debug/...`. Fetch verbatim with `git show upstream/dev-nyh:<path> > <dest>`, then apply the specific edits each task names. Reference spec: `docs/superpowers/specs/2026-06-03-ascend-debug-retarget-develop-design.md`.

---

## File Structure

```
tools/ascend-debug/
  CMakeLists.txt                 # installs build/bin/ascend-debug wrapper
  ascend-debug.py                # CLI entry; subcommands trimmed to collect + open
  ascend_debug/
    __init__.py                  # __version__
    runner.py                    # find_tool/run_command (default tool -> afir-opt)
    layout.py                    # stage numbering, manifest/provenance writers
    collect.py                   # REWRITTEN 6-stage afir-opt pipeline
    open_view.py                 # manifest-driven HTML dashboard
    stage_graph.py               # per-stage structural graph; badges remapped to develop attrs
    ui_text.py                   # coarse stage UI copy
    memory.py                    # imported by open_view; degrades when no data
    debug_graph.py               # imported by open_view; degrades (unified workspace = follow-up)
tools/CMakeLists.txt             # + add_subdirectory(ascend-debug)
test/CMakeLists.txt              # + ascend-debug in AFIR_TEST_DEPENDS
test/tools/diagnostics/
  ascend-debug-cli.mlir          # lit entry
  test_ascend_debug_cli.sh       # trimmed collect+open smoke test
tests/tools/ascend-debug/        # pure-Python unit tests (no build needed)
  test_cli_parser.py
  test_collect_stages.py
  test_stage_graph_badges.py
```

**Note on collisions:** `develop`'s existing debug tooling lives under `python/tools/` (`ascend_diff.py`, `ascend_kernel_dag_viz/`); this tool lives under `tools/ascend-debug/`. No path or binary-name collision. We do NOT touch develop's existing debug system.

---

### Task 1: Scaffold standalone package + CLI (collect/open only)

**Files:**
- Create: `tools/ascend-debug/ascend_debug/__init__.py`
- Create: `tools/ascend-debug/ascend-debug.py`
- Test: `tests/tools/ascend-debug/test_cli_parser.py`

- [ ] **Step 1: Write the failing unit test**

Create `tests/tools/ascend-debug/test_cli_parser.py`:

```python
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "tools" / "ascend-debug"
sys.path.insert(0, str(PKG_DIR))


def _load_entry():
    spec = importlib.util.spec_from_file_location(
        "ascend_debug_cli", PKG_DIR / "ascend-debug.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_has_only_collect_and_open():
    entry = _load_entry()
    parser = entry.build_parser()
    sub = next(a for a in parser._actions if hasattr(a, "choices") and a.choices)
    assert set(sub.choices) == {"collect", "open"}


def test_collect_requires_input_and_out():
    entry = _load_entry()
    parser = entry.build_parser()
    args = parser.parse_args(["collect", "in.mlir", "--out", "run"])
    assert str(args.input) == "in.mlir"
    assert str(args.out) == "run"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/tools/ascend-debug/test_cli_parser.py -v`
Expected: FAIL — `ModuleNotFoundError` / file not found (entry script does not exist yet).

- [ ] **Step 3: Create `__init__.py`**

Create `tools/ascend-debug/ascend_debug/__init__.py`:

```python
__version__ = "0.1"
```

- [ ] **Step 4: Create the CLI entry (collect + open only)**

Create `tools/ascend-debug/ascend-debug.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys

from ascend_debug import __version__
from ascend_debug.collect import collect_run
from ascend_debug.open_view import open_run
from ascend_debug.runner import CommandError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ascend-debug",
        description="Collect and inspect Ascend-MLIR debug artifacts.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Collect a debug run")
    collect.add_argument("input", type=pathlib.Path)
    collect.add_argument("--out", type=pathlib.Path, required=True)
    collect.set_defaults(handler=collect_run)

    open_cmd = subparsers.add_parser("open", help="Generate the debug dashboard")
    open_cmd.add_argument("run_dir", type=pathlib.Path)
    open_cmd.add_argument("--no-browser", action="store_true")
    open_cmd.set_defaults(handler=open_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except CommandError as error:
        print(f"ascend-debug: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

> NOTE: imports of `collect_run` / `open_run` resolve in later tasks. The parser test (Step 1) only exercises `build_parser`, but Python imports the module top-to-bottom, so create minimal stubs now to let the import succeed: create `tools/ascend-debug/ascend_debug/runner.py`, `collect.py`, `open_view.py` with just the referenced symbol as a stub (`def collect_run(args): ...`, `def open_run(args): ...`, and a `class CommandError(RuntimeError): pass`). These stubs are fully replaced in Tasks 2-4.

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/tools/ascend-debug/test_cli_parser.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add tools/ascend-debug/ tests/tools/ascend-debug/test_cli_parser.py
git commit -m "feat(ascend-debug): scaffold standalone CLI (collect+open)"
```

---

### Task 2: Port runner / layout / ui_text verbatim, re-target tool name

**Files:**
- Create (port): `tools/ascend-debug/ascend_debug/runner.py`
- Create (port): `tools/ascend-debug/ascend_debug/layout.py`
- Create (port): `tools/ascend-debug/ascend_debug/ui_text.py`

- [ ] **Step 1: Port the three modules verbatim from dev-nyh**

```bash
cd /home/gser/code/Ascend-MLIR
git show upstream/dev-nyh:tools/ascend-debug/ascend_debug/runner.py  > tools/ascend-debug/ascend_debug/runner.py
git show upstream/dev-nyh:tools/ascend-debug/ascend_debug/layout.py  > tools/ascend-debug/ascend_debug/layout.py
git show upstream/dev-nyh:tools/ascend-debug/ascend_debug/ui_text.py > tools/ascend-debug/ascend_debug/ui_text.py
```

This overwrites the `runner.py` stub from Task 1 with the real implementation (`CommandError`, `CommandResult`, `find_tool`, `run_command`).

- [ ] **Step 2: Verify `runner.py` already has no dev-nyh tool-name coupling**

`runner.find_tool(name)` takes the tool name as an argument (no hardcoded `ascend-mlir-opt`). The default `afir-opt` is chosen by the *caller* (`collect.py`, Task 3). No edit needed here. Confirm:

Run: `grep -n "ascend-mlir-opt" tools/ascend-debug/ascend_debug/runner.py`
Expected: no output.

- [ ] **Step 3: Check `layout.py` for any dev-nyh pipeline assumptions**

Run: `grep -nE "ascend-mlir-opt|--ascend-|debug-dump-dir" tools/ascend-debug/ascend_debug/layout.py`
Expected: no output (layout is pure numbering/manifest IO). If anything appears, note it and stop — layout was assumed pipeline-agnostic.

- [ ] **Step 4: Smoke-import the modules**

Run:
```bash
cd tools/ascend-debug && python3 -c "import sys; sys.path.insert(0,'.'); from ascend_debug import runner, layout, ui_text; print('ok')"
```
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add tools/ascend-debug/ascend_debug/runner.py tools/ascend-debug/ascend_debug/layout.py tools/ascend-debug/ascend_debug/ui_text.py
git commit -m "feat(ascend-debug): port runner/layout/ui_text (pipeline-agnostic)"
```

---

### Task 3: Rewrite `collect.py` for develop's 6-stage afir-opt pipeline

**Files:**
- Create: `tools/ascend-debug/ascend_debug/collect.py` (rewrite; not a verbatim port)
- Test: `tests/tools/ascend-debug/test_collect_stages.py`

The stage table mirrors dev-nyh's `pass_steps` shape `(stage, input_key, output_key, [pass_flags])` but uses develop's `afir-opt` flags and drops all `dump-report=/debug-stage=/debug-dump-dir=` options (no C++). Flags resolved from `include/Conversion/Passes.td` and `lib/Conversion/AutoFuse/Pipeline.cpp`.

- [ ] **Step 1: Write the failing stage-table unit test**

Create `tests/tools/ascend-debug/test_collect_stages.py`:

```python
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "ascend-debug"))

from ascend_debug import collect


def test_six_stages_in_order():
    names = [s[0] for s in collect.PASS_STEPS]
    assert names == [
        "normalize", "kernelize", "schedule",
        "realize", "parallelize", "finalize",
    ]


def test_stage_io_chains():
    # each stage's input_key is the previous stage's output_key
    steps = collect.PASS_STEPS
    assert steps[0][1] == "source"
    for prev, cur in zip(steps, steps[1:]):
        assert cur[1] == prev[2], (prev, cur)


def test_no_devnyh_pass_flags():
    for _, _, _, flags in collect.PASS_STEPS:
        joined = " ".join(flags)
        assert "--ascend-normalize" not in joined
        assert "debug-dump-dir" not in joined
        assert "dump-report" not in joined


def test_normalize_uses_develop_flags():
    flags = dict((s[0], s[3]) for s in collect.PASS_STEPS)
    assert "--auto-fuse-group-analysis" in flags["kernelize"]
    assert "--linalg-to-ascendc" in flags["realize"]
    assert "--canonicalize-cann-signature" in flags["finalize"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/tools/ascend-debug/test_collect_stages.py -v`
Expected: FAIL — `AttributeError: module 'ascend_debug.collect' has no attribute 'PASS_STEPS'` (collect.py is still a stub).

- [ ] **Step 3: Write `collect.py`**

Create `tools/ascend-debug/ascend_debug/collect.py`:

```python
from __future__ import annotations

import pathlib

from ascend_debug import __version__, layout
from ascend_debug.runner import CommandError, find_tool, run_command

# (stage_name, input_stage_key, output_stage_key, [afir-opt pass flags])
# Stages are a logical 6-way split of develop's `--auto-fuse-codegen`
# pipeline (lib/Conversion/AutoFuse/Pipeline.cpp). Order is preserved
# exactly; only dump points are inserted. Flags are afir-opt CLI names
# from include/Conversion/Passes.td. NO C++ dump options.
PASS_STEPS: list[tuple[str, str, str, list[str]]] = [
    (
        "normalize", "source", "010-normalize-out",
        [
            "--linalg-generalize-named-ops",
            "--linalg-fuse-elementwise-ops",
            "--linalg-fold-unit-extent-dims",
            "--canonicalize",
        ],
    ),
    (
        "kernelize", "010-normalize-out", "020-kernelize-out",
        [
            "--auto-fuse-group-analysis",
            "--auto-fuse-group-outline",
        ],
    ),
    (
        "schedule", "020-kernelize-out", "030-schedule-out",
        [
            "--auto-fuse-restore-matmul",
            "--auto-fuse-isolate-kernel-outputs",
            "--afir-symbolize-shapes",
            "--auto-fuse-tile-fuse",
            "--canonicalize",
        ],
    ),
    (
        "realize", "030-schedule-out", "040-realize-out",
        [
            "--annotate-ascendc-kernel-kind",
            "--auto-fuse-fold-shadow-alloc",
            "--auto-fuse-insert-tile-buffers",
            "--ascendc-buffer-placement",
            "--linalg-to-ascendc",
        ],
    ),
    (
        "parallelize", "040-realize-out", "050-parallelize-out",
        [
            "--ascendc-decompose-multi-axis-broadcast",
            "--ascendc-parallelize",
            "--ascendc-flatten-gm-ptr",
            "--canonicalize",
            "--cse",
        ],
    ),
    (
        "finalize", "050-parallelize-out", "060-finalize-out",
        [
            "--auto-fuse-verify-tiling-info-schema",
            "--ascendc-pack-tiling-data",
            "--ascendc-finalize-kernel",
            "--canonicalize-cann-signature",
            "--ascendc-rcore-combine",
        ],
    ),
]

_DEFAULT_TOOL = "afir-opt"


def _stage_rel(output_key: str) -> str:
    return f"stages/{output_key}.mlir"


def collect_run(args) -> int:
    run_dir: pathlib.Path = args.out
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stages").mkdir(exist_ok=True)

    opt = find_tool(_DEFAULT_TOOL)

    # stage key -> absolute artifact path
    paths: dict[str, pathlib.Path] = {"source": run_dir / "stages" / "000-source.mlir"}
    layout.copy_stage(args.input, paths["source"])

    commands: list[dict] = []
    stage_records: list[dict] = []
    for stage, in_key, out_key, flags in PASS_STEPS:
        out_path = run_dir / _stage_rel(out_key)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        argv = [opt, str(paths[in_key]), *flags, "-o", str(out_path)]
        run_command(argv, stderr_report_path=run_dir / "stages" / f"{out_key}.report.txt")
        paths[out_key] = out_path
        commands.append({"stage": stage, "tool": _DEFAULT_TOOL, "argv": argv})
        stage_records.append(
            {"order": out_key.split("-")[0], "name": stage, "path": _stage_rel(out_key)}
        )

    # Provenance skeleton (one source file) + manifest. StageArtifact is the
    # frozen dataclass from layout.py: (order:int, name, path, phase, step).
    # `step` is looked up against layout.STEP_INFO_BY_STEP for per-stage
    # explanation text; for develop's stage names it may have no entry (the
    # dashboard then shows no explanation — acceptable for MVP).
    stages = tuple(
        layout.StageArtifact(order=i, name=r["name"], path=r["path"], step=r["name"])
        for i, r in enumerate(stage_records, start=1)
    )
    layout.write_manifest(
        run_dir,
        mode="develop-codegen",
        preset="",
        pipeline="auto-fuse-codegen",
        stages=stages,
        version=__version__,
        commands=commands,
        reports=[],
        graphs=[],
    )
    layout.write_provenance_skeleton(
        run_dir, original_input=args.input, version=__version__
    )
    print(f"ascend-debug.collect.out={run_dir}")
    return 0
```

The `stage_records` entries built in the loop are plain dicts `{"name": stage, "path": _stage_rel(out_key)}`; drop the `"order"` string key (order is the int from `enumerate`). Adjust the loop's `stage_records.append(...)` to:

```python
        stage_records.append({"name": stage, "path": _stage_rel(out_key)})
```

> VERIFIED API (from `git show upstream/dev-nyh:tools/ascend-debug/ascend_debug/layout.py`): `StageArtifact(order:int, name:str, path:str, phase=None, step=None)`; `write_manifest(run_dir, *, mode, preset, pipeline, stages, version, ...)`; `copy_stage(src, dst)`; `write_provenance_skeleton(run_dir, *, original_input, version)`. Conform to these exactly; do NOT edit `layout.py`.

> OPTIONAL (per-stage explanation text): to populate the dashboard's per-stage descriptions, add 6 keys (`normalize`, `kernelize`, `schedule`, `realize`, `parallelize`, `finalize`) to `layout.STEP_INFO_BY_STEP` with develop-accurate copy. This is the one allowed additive edit to the ported `layout.py`. Defer if MVP scope is tight — empty descriptions are acceptable.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/tools/ascend-debug/test_collect_stages.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Verify each stage flag is a real afir-opt pass**

Run (requires a built `afir-opt` in PATH; skip if not yet built — Task 6 covers the integration run):
```bash
afir-opt --help 2>&1 | grep -oE -- "--(linalg-generalize-named-ops|linalg-fuse-elementwise-ops|linalg-fold-unit-extent-dims|auto-fuse-group-analysis|auto-fuse-group-outline|auto-fuse-restore-matmul|auto-fuse-isolate-kernel-outputs|afir-symbolize-shapes|auto-fuse-tile-fuse|annotate-ascendc-kernel-kind|auto-fuse-fold-shadow-alloc|auto-fuse-insert-tile-buffers|ascendc-buffer-placement|linalg-to-ascendc|ascendc-decompose-multi-axis-broadcast|ascendc-parallelize|ascendc-flatten-gm-ptr|canonicalize-cann-signature|auto-fuse-verify-tiling-info-schema|ascendc-pack-tiling-data|ascendc-finalize-kernel|ascendc-rcore-combine)" | sort -u | wc -l
```
Expected: `22` (all flags registered). If fewer, the missing flag names are wrong — reconcile each against `afir-opt --help` and fix `PASS_STEPS`.

- [ ] **Step 6: Commit**

```bash
git add tools/ascend-debug/ascend_debug/collect.py tests/tools/ascend-debug/test_collect_stages.py
git commit -m "feat(ascend-debug): collect drives develop 6-stage afir-opt pipeline"
```

---

### Task 4: Port `open_view` + dashboard deps with graceful degradation

**Files:**
- Create (port): `tools/ascend-debug/ascend_debug/open_view.py`
- Create (port): `tools/ascend-debug/ascend_debug/stage_graph.py`
- Create (port): `tools/ascend-debug/ascend_debug/memory.py`
- Create (port): `tools/ascend-debug/ascend_debug/debug_graph.py`

- [ ] **Step 1: Port the four modules verbatim**

```bash
cd /home/gser/code/Ascend-MLIR
for m in open_view stage_graph memory debug_graph; do
  git show upstream/dev-nyh:tools/ascend-debug/ascend_debug/$m.py > tools/ascend-debug/ascend_debug/$m.py
done
```

This overwrites the `open_view.py` stub from Task 1.

- [ ] **Step 2: Identify imports `open_view` needs that we did NOT port**

Run:
```bash
grep -nE "^from ascend_debug|^import ascend_debug|from ascend_debug import" tools/ascend-debug/ascend_debug/open_view.py
```
Expected import line: `from ascend_debug import debug_graph, layout, memory, stage_graph, ui_text`. All five are now present. If `open_view` imports any module NOT in our file list (e.g. `diff`, `locate`, `kernel_dag`, `failure`), either (a) port that module too if it is needed for dashboard render, or (b) guard the import. Decide per-module: dashboard-render dependency → port; diff/locate-only → remove the import and the code path that uses it.

- [ ] **Step 3: Smoke-import open_view**

Run:
```bash
cd tools/ascend-debug && python3 -c "import sys; sys.path.insert(0,'.'); from ascend_debug import open_view; print('ok')"
```
Expected: `ok`. If `ImportError` on a missing sibling module, resolve per Step 2.

- [ ] **Step 4: Make `memory` / `debug_graph` degrade when their data is absent**

The dashboard must render even though `develop` collect produces no memory timeline / unified-workspace data. In `open_view.py`, locate where it calls into `memory` and `debug_graph`. Wrap each render call so a missing-data return (`None`/empty) yields an empty section instead of an exception. Concretely, find calls like `memory.render(...)` / `debug_graph.render(...)` and ensure the surrounding code already tolerates `None`; if it raises, add a guard:

```python
try:
    memory_section = memory.render_section(manifest, run_dir)  # exact name per memory.py
except Exception:
    memory_section = ""  # no runtime memory data in MVP collect
```

Use the ACTUAL function names found in `memory.py` / `debug_graph.py` (read them first). This is the only behavioral edit to ported dashboard code; keep it minimal.

- [ ] **Step 5: Commit**

```bash
git add tools/ascend-debug/ascend_debug/open_view.py tools/ascend-debug/ascend_debug/stage_graph.py tools/ascend-debug/ascend_debug/memory.py tools/ascend-debug/ascend_debug/debug_graph.py
git commit -m "feat(ascend-debug): port open dashboard (memory/debug_graph degrade gracefully)"
```

---

### Task 5: Remap stage_graph semantic badges to develop attrs

**Files:**
- Modify: `tools/ascend-debug/ascend_debug/stage_graph.py` (`_build_semantic_attrs`)
- Test: `tests/tools/ascend-debug/test_stage_graph_badges.py`

The generic structure layer (func/op/dataflow) needs no change. Only `_build_semantic_attrs` reads dev-nyh `ascend.*` attr names; remap to develop's per the spec table. The text-extractor helpers (`_extract_attr`, `_extract_string_list_attr`, `_extract_int_attr`, `_extract_tile_params_attr`) are attr-name-generic and stay as-is.

- [ ] **Step 1: Write the failing badge test**

Create `tests/tools/ascend-debug/test_stage_graph_badges.py`:

```python
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "ascend-debug"))

from ascend_debug import stage_graph


def test_kernel_id_from_auto_fuse_group_id():
    op_text = 'linalg.generic {auto_fuse.group_id = 3 : i64} ...'
    attrs = stage_graph._build_semantic_attrs("linalg.generic", op_text)
    assert attrs["kernel"]["id"] == "3"


def test_role_from_auto_fuse_kind():
    op_text = 'func.func @k(...) attributes {auto_fuse.kind = "vec"}'
    attrs = stage_graph._build_semantic_attrs("func.func", op_text)
    assert attrs["kernel"]["role"] == "vec"


def test_aclnn_op_surfaces_as_role_when_no_kind():
    op_text = 'func.call @aclnn ... {aclnn.op = "Matmul"}'
    attrs = stage_graph._build_semantic_attrs("func.call", op_text)
    assert attrs["kernel"]["role"] == "Matmul"


def test_no_ascend_namespace_attrs_referenced():
    src = (ROOT / "tools" / "ascend-debug" / "ascend_debug" / "stage_graph.py").read_text()
    # the badge builder must not key off dev-nyh ascend.* scheduler attrs
    assert "ascend.schedule" not in src
    assert "ascend.op_role" not in src
    assert "ascend.kernel" not in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/tools/ascend-debug/test_stage_graph_badges.py -v`
Expected: FAIL — `id` is `None` (still reads `ascend.kernel`) and the `ascend.*` strings are still present.

- [ ] **Step 3: Rewrite `_build_semantic_attrs`**

Replace the `_build_semantic_attrs` function body in `stage_graph.py` with develop-attr extraction. Mapping (spec §5):

| field | develop attr(s), in priority order |
|---|---|
| `kernel.id` | `auto_fuse.group_id` else `auto_fuse.topo_index` |
| `kernel.role` | `auto_fuse.kind` else `aclnn.op` else `aclnn.kind` |
| `kernel.template_families` | `auto_fuse.tiling_infos` (presence) / `afir.reduce_template` |
| `kernel.primary` | drop (no develop equivalent) |
| `schedule.family` | `afir.reduce_template` |
| `schedule.tile_params` | parse `auto_fuse.tiling_infos` (keep `_extract_tile_params_attr` retargeted to this attr name) |
| `schedule.default_tile_size` | `auto_fuse.default_tile_size` |
| `schedule.block_dim` | `afir.block_dim_expr` |
| `schedule.axis_extent` | `afir.axis_extent_expr` |

New body:

```python
def _build_semantic_attrs(op_name: str, op_text: str) -> dict[str, Any]:
    kernel = {
        key: value
        for key, value in {
            "id": _extract_attr(op_text, "auto_fuse.group_id")
            or _extract_attr(op_text, "auto_fuse.topo_index"),
            "role": _extract_attr(op_text, "auto_fuse.kind")
            or _extract_attr(op_text, "aclnn.op")
            or _extract_attr(op_text, "aclnn.kind"),
            "template_families": _extract_string_list_attr(
                op_text, "afir.reduce_template"
            ),
        }.items()
        if value not in (None, [], False)
    }
    schedule = {
        key: value
        for key, value in {
            "family": _extract_attr(op_text, "afir.reduce_template"),
            "default_tile_size": _extract_attr(op_text, "auto_fuse.default_tile_size"),
            "block_dim": _extract_attr(op_text, "afir.block_dim_expr"),
            "axis_extent": _extract_attr(op_text, "afir.axis_extent_expr"),
            "tile_params": _extract_tile_params_attr(op_text),
        }.items()
        if value not in (None, [], False)
    }
    phases = _extract_tail_phases(op_text)
    if op_name == "memref.copy" and "data_copy" not in phases:
        phases.append("data_copy")
    if op_name.startswith("ascendc.data_copy") and "data_copy" not in phases:
        phases.append("data_copy")
    movement = {"phases": phases} if phases else {}
    memory = _extract_tensor_buffer_attrs(op_text)
    return {
        key: value
        for key, value in {
            "kernel": kernel,
            "schedule": schedule,
            "movement": movement,
            "memory": memory,
        }.items()
        if value
    }
```

Then update `_extract_tile_params_attr` to read develop's attr name. Find:

```python
    body = _extract_balanced_attr_value(text, "ascend.schedule.tile_params", "[", "]")
```

Replace with:

```python
    body = _extract_balanced_attr_value(text, "auto_fuse.tiling_infos", "[", "]")
```

Finally, in `_build_node_badges`, the `tail_policies` / `runtime_top_k` references no longer populate (no develop equivalent); leave the code (it guards on presence and renders nothing when absent) — no edit required. Verify there are no remaining `ascend.schedule`/`ascend.op_role`/`ascend.kernel` literals:

```bash
grep -nE "ascend\.(schedule|op_role|op_roles|kernel|primary|kernelize)" tools/ascend-debug/ascend_debug/stage_graph.py
```
Expected: no output. Remove any stragglers.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/tools/ascend-debug/test_stage_graph_badges.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Run the full python unit suite**

Run: `python3 -m pytest tests/tools/ascend-debug/ -v`
Expected: PASS (all tests from Tasks 1, 3, 5).

- [ ] **Step 6: Commit**

```bash
git add tools/ascend-debug/ascend_debug/stage_graph.py tests/tools/ascend-debug/test_stage_graph_badges.py
git commit -m "feat(ascend-debug): remap stage_graph badges to develop auto_fuse/aclnn/afir attrs"
```

---

### Task 6: Build wiring + lit smoke test + end-to-end verification

**Files:**
- Create: `tools/ascend-debug/CMakeLists.txt`
- Modify: `tools/CMakeLists.txt`
- Modify: `test/CMakeLists.txt`
- Create: `test/tools/diagnostics/test_ascend_debug_cli.sh`
- Create: `test/tools/diagnostics/ascend-debug-cli.mlir`

- [ ] **Step 1: Write the failing lit smoke test**

Create `test/tools/diagnostics/ascend-debug-cli.mlir`:

```mlir
// RUN: bash %S/test_ascend_debug_cli.sh %s | FileCheck %s

func.func @elementwise(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = tensor.empty() : tensor<4x8xf32>
  %1 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%0 : tensor<4x8xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %s = arith.addf %a, %b : f32
    linalg.yield %s : f32
  } -> tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}

// CHECK: ascend-debug.collect.out=
// CHECK: STAGES_OK=6
// CHECK: OPEN_OK=1
```

Create `test/tools/diagnostics/test_ascend_debug_cli.sh`:

```bash
#!/usr/bin/env bash
# Smoke test: collect runs develop's 6-stage pipeline + open renders HTML.
set -euo pipefail

INPUT="$1"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# ascend-debug is installed to build/bin; FileCheck-friendly stdout.
ascend-debug collect "$INPUT" --out "$WORK/run"

# Count produced numbered stage artifacts (010..060 => 6).
STAGES_OK="$(find "$WORK/run/stages" -name '0[1-6]0-*.mlir' | wc -l | tr -d ' ')"
echo "STAGES_OK=$STAGES_OK"

ascend-debug open "$WORK/run" --no-browser >/dev/null
# open must produce at least one HTML file.
if find "$WORK/run" -name '*.html' | grep -q .; then
  echo "OPEN_OK=1"
else
  echo "OPEN_OK=0"
fi
```

- [ ] **Step 2: Wire the tool into CMake**

Create `tools/ascend-debug/CMakeLists.txt` (port dev-nyh's and adjust install name if needed):

```bash
git show upstream/dev-nyh:tools/ascend-debug/CMakeLists.txt > tools/ascend-debug/CMakeLists.txt
```

Read it; confirm it (a) installs a `build/bin/ascend-debug` wrapper pointing at `ascend-debug.py` + the `ascend_debug/` package, and (b) defines an `ascend-debug` target. If it references diff/locate/serve/run modules we did not port, trim those references. Then add to `tools/CMakeLists.txt`:

```cmake
add_subdirectory(ascend-debug)
```

And add `ascend-debug` to the test dependency list in `test/CMakeLists.txt` (find the existing `AFIR_TEST_DEPENDS`-style list and append `ascend-debug`).

- [ ] **Step 3: Build the tool + afir-opt**

Run (use the project's configured build dir):
```bash
cmake --build build --target afir-opt ascend-debug
```
Expected: both targets build; `build/bin/ascend-debug` exists and is executable.

- [ ] **Step 4: Run the lit test to verify it fails first, then passes**

If the build above already produced the tool, run lit:
```bash
cmake --build build --target check-afir 2>&1 | tail -40
# or run the single test:
build/bin/llvm-lit -v test/tools/diagnostics/ascend-debug-cli.mlir
```
Expected: PASS. If `STAGES_OK` < 6, a stage failed `afir-opt` — inspect `"$WORK/run/stages/<key>.report.txt"` (the per-stage stderr report) to see which pass errored, and reconcile that stage's flags in `collect.PASS_STEPS` (spec §4 risk: merge a state-dependent stage into its neighbor if it cannot run standalone).

- [ ] **Step 5: End-to-end check on a real develop example**

```bash
ascend-debug collect examples/add-mul-relu-e2e/add_mul_relu.mlir --out /tmp/amr-dbg
ascend-debug open /tmp/amr-dbg --no-browser
```
Expected: 6 numbered stage artifacts under `/tmp/amr-dbg/stages/`, a `manifest.json`, and HTML output. Open the HTML and confirm the per-stage structural graph renders and badges populate from `auto_fuse.*` attrs in the kernelize/schedule stages. (If running headless, grep the generated stage-graph JSON for non-empty `kernel`/`schedule` badges.)

- [ ] **Step 6: Commit**

```bash
git add tools/ascend-debug/CMakeLists.txt tools/CMakeLists.txt test/CMakeLists.txt test/tools/diagnostics/ascend-debug-cli.mlir test/tools/diagnostics/test_ascend_debug_cli.sh
git commit -m "test(ascend-debug): build wiring + collect/open lit smoke test"
```

---

## Self-Review

**Spec coverage:**
- §3 collect (6-stage afir-opt) → Task 3. ✓
- §3 open (manifest-driven dashboard) → Task 4. ✓
- §4 stage table → Task 3 `PASS_STEPS`. ✓
- §5 structural graph generic layer → ported as-is Task 4; semantic badges remapped → Task 5. ✓
- §5 graceful degradation (memory/debug_graph) → Task 4 Step 4. ✓
- §6 module list → Tasks 1-5. ✓
- §6 build wiring → Task 6 Steps 2-3. ✓
- §7 lit smoke + e2e check → Task 6 Steps 1,4,5. ✓
- Non-goals (diff/locate/C++/unified workspace/memory timeline) → not implemented; diff/locate/serve/run/kernel_dag/failure modules intentionally not ported (Task 1 trims CLI; Task 4 Step 2 trims imports; Task 6 Step 2 trims CMake). ✓

**Placeholder scan:** No TBD/TODO. Adaptation notes (Task 3 Step 3, Task 4 Steps 2/4, Task 6 Step 2) point at concrete ported files to conform to, with the exact grep/inspection to run — not vague "handle later."

**Type consistency:** `PASS_STEPS` tuple shape `(name, in_key, out_key, flags)` used identically in `collect.py` and both tests. Stage output keys (`010-normalize-out` … `060-finalize-out`) chain consistently (Task 3 `test_stage_io_chains` enforces it). `_build_semantic_attrs` return shape (`kernel`/`schedule`/`movement`/`memory`) unchanged from dev-nyh, so `_build_node_badges` consumers still match.

**Known conformance risk:** `layout.write_manifest` / `layout.StageArtifact` signatures were verified against `upstream/dev-nyh:layout.py` and pinned in Task 3 Step 3 (the "VERIFIED API" note). `collect_run` itself is exercised only by the Task 6 lit test, not the Task 3 unit test (which checks `PASS_STEPS` statically).
