# Ascend Debug CLI Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first usable `ascend-debug` CLI slice: command surface, `collect --preset quick`, numbered stage artifacts, manifest/provenance skeleton, and a minimal `open` dashboard.

**Architecture:** Implement `ascend-debug` as a Python CLI wrapper installed into `build/bin` by CMake. Keep this slice compile/artifact focused: it orchestrates `ascend-mlir-opt`, writes stable artifacts, and generates read-only HTML/JSON; it does not run NPU/sim, compare tensors, or implement checkpoints yet. Later plans will add `diff`, `locate`, runtime checkpointing, and enhanced visualizers on top of this artifact contract.

**Tech Stack:** Python 3 standard library, CMake custom target, `ascend-mlir-opt`, LLVM lit/FileCheck, JSON, static HTML.

---

## Scope Split

The approved debug design covers several subsystems: CLI, artifact collection, provenance, dashboard, tensor diff, first-bad-kernel location, runtime checkpoints, and visualization. This plan intentionally implements only the first independently testable slice:

- CLI command surface: `collect`, `open`, `diff`, `locate`
- working `collect --preset quick`
- working `open`
- stable artifact numbering and `manifest.json`
- minimal `provenance.json` skeleton
- deterministic errors for `diff` and `locate` until their own plans land

Follow-up plans should be separate:

- `ascend-debug-diff-runtime-plan`: final-output `.npy` diff and runtime evidence ingestion
- `ascend-debug-checkpoint-locate-plan`: per-kernel checkpoint, semantic boundaries, first-bad-kernel
- `ascend-debug-dashboard-viz-plan`: enhanced `ascend_kernel_dag_viz`, IR graph, memory timeline

## File Structure

- Create `tools/ascend-debug/CMakeLists.txt`
  - Installs a Python executable wrapper to `build/bin/ascend-debug`.
- Create `tools/ascend-debug/ascend-debug.py`
  - CLI entrypoint and subcommand dispatch.
- Create `tools/ascend-debug/ascend_debug/__init__.py`
  - Package marker and version string.
- Create `tools/ascend-debug/ascend_debug/layout.py`
  - Stage numbering, artifact paths, manifest/provenance writers.
- Create `tools/ascend-debug/ascend_debug/runner.py`
  - External command execution helpers with deterministic errors.
- Create `tools/ascend-debug/ascend_debug/collect.py`
  - `collect --preset quick` implementation.
- Create `tools/ascend-debug/ascend_debug/open_view.py`
  - Minimal dashboard generation and path printing.
- Modify `tools/CMakeLists.txt`
  - Add `add_subdirectory(ascend-debug)`.
- Modify `test/CMakeLists.txt`
  - Add `ascend-debug` to `AFIR_TEST_DEPENDS`.
- Create `test/tools/diagnostics/test_ascend_debug_cli.sh`
  - Focused shell test for help, quick collect, manifest ordering, and open.
- Create `test/tools/diagnostics/ascend-debug-cli.mlir`
  - Lit entrypoint that runs the shell test.

### Task 1: Wire `ascend-debug` Into the Build

**Files:**
- Create: `tools/ascend-debug/CMakeLists.txt`
- Create: `tools/ascend-debug/ascend-debug.py`
- Create: `tools/ascend-debug/ascend_debug/__init__.py`
- Modify: `tools/CMakeLists.txt`
- Modify: `test/CMakeLists.txt`

- [ ] **Step 1: Write the failing lit smoke test**

Create `test/tools/diagnostics/ascend-debug-cli.mlir`:

```mlir
// RUN: bash %S/test_ascend_debug_cli.sh %s | FileCheck %s

func.func @elementwise(%arg0: tensor<4x8xf16>, %arg1: tensor<4x8xf16>) -> tensor<4x8xf16> {
  %empty = tensor.empty() : tensor<4x8xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x8xf16>, tensor<4x8xf16>)
    outs(%empty : tensor<4x8xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<4x8xf16>
  return %out : tensor<4x8xf16>
}

// CHECK: ascend_debug.help=ok
// CHECK: ascend_debug.collect=ok
// CHECK: ascend_debug.open=ok
// CHECK: ascend_debug.manifest.stage_count=5
// CHECK: ascend_debug.stage.0=000-source.mlir
// CHECK: ascend_debug.stage.4=029-kernelize-out.mlir
// CHECK: ALL ASCEND DEBUG CLI TESTS PASSED
```

- [ ] **Step 2: Add the shell test driver**

Create `test/tools/diagnostics/test_ascend_debug_cli.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

INPUT_MLIR="$1"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ascend-debug-cli.XXXXXX")"
trap 'rm -rf "${TMP_DIR}"' EXIT

ascend-debug --help >/tmp/ascend-debug-help.txt 2>&1
grep -q 'collect' /tmp/ascend-debug-help.txt
grep -q 'open' /tmp/ascend-debug-help.txt
grep -q 'diff' /tmp/ascend-debug-help.txt
grep -q 'locate' /tmp/ascend-debug-help.txt
echo "ascend_debug.help=ok"

ascend-debug collect "${INPUT_MLIR}" \
  --out "${TMP_DIR}/debug-run" \
  --pipeline normalize-kernelize
test -f "${TMP_DIR}/debug-run/stages/000-source.mlir"
test -f "${TMP_DIR}/debug-run/stages/010-normalize-in.mlir"
test -f "${TMP_DIR}/debug-run/stages/019-normalize-out.mlir"
test -f "${TMP_DIR}/debug-run/stages/020-kernelize-in.mlir"
test -f "${TMP_DIR}/debug-run/stages/029-kernelize-out.mlir"
test -f "${TMP_DIR}/debug-run/manifest.json"
test -f "${TMP_DIR}/debug-run/provenance.json"
echo "ascend_debug.collect=ok"

ascend-debug open "${TMP_DIR}/debug-run" --no-browser >/tmp/ascend-debug-open.txt
grep -q 'index.html' /tmp/ascend-debug-open.txt
test -f "${TMP_DIR}/debug-run/index.html"
echo "ascend_debug.open=ok"

python3 - "${TMP_DIR}/debug-run/manifest.json" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
stages = manifest["stages"]
print(f"ascend_debug.manifest.stage_count={len(stages)}")
for i, stage in enumerate(stages):
    print(f"ascend_debug.stage.{i}={pathlib.Path(stage['path']).name}")
assert [stage["order"] for stage in stages] == [0, 10, 19, 20, 29]
assert manifest["tool"] == "ascend-debug"
assert manifest["preset"] == "quick"
assert manifest["device_scope"] == "single_run_single_device"
PY

echo "ALL ASCEND DEBUG CLI TESTS PASSED"
```

- [ ] **Step 3: Run the lit test and verify it fails before implementation**

Run:

```bash
llvm-lit -v build/test/tools/diagnostics/ascend-debug-cli.mlir
```

Expected result before implementation:

```text
FAIL: ... tools/diagnostics/ascend-debug-cli.mlir
ascend-debug: command not found
```

- [ ] **Step 4: Add CMake wiring**

Modify `tools/CMakeLists.txt`:

```cmake
add_subdirectory(afir-opt)
add_subdirectory(ascend-mlir-opt)
add_subdirectory(ascend-debug)
add_subdirectory(afir-translate)
add_subdirectory(autotuner)
add_subdirectory(mix-compiler)
add_subdirectory(mix-tiling-helper)
add_subdirectory(runtime-session)
```

Create `tools/ascend-debug/CMakeLists.txt`:

```cmake
set(ASCEND_DEBUG_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/ascend-debug.py")
set(ASCEND_DEBUG_OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ascend-debug")

add_custom_command(
  OUTPUT "${ASCEND_DEBUG_OUTPUT}"
  COMMAND "${CMAKE_COMMAND}" -E copy "${ASCEND_DEBUG_SOURCE}" "${ASCEND_DEBUG_OUTPUT}"
  COMMAND chmod +x "${ASCEND_DEBUG_OUTPUT}"
  DEPENDS "${ASCEND_DEBUG_SOURCE}"
  COMMENT "Installing ascend-debug Python CLI"
  VERBATIM
)

add_custom_target(ascend-debug ALL
  DEPENDS "${ASCEND_DEBUG_OUTPUT}"
)
```

Modify `test/CMakeLists.txt` so `AFIR_TEST_DEPENDS` includes `ascend-debug`:

```cmake
set(AFIR_TEST_DEPENDS
  afir-opt
  ascend-mlir-opt
  ascend-debug
  afir-translate
  FileCheck
  count
  not
)
```

- [ ] **Step 5: Add a minimal executable script**

Create `tools/ascend-debug/ascend_debug/__init__.py`:

```python
__version__ = "0.1"
```

Create `tools/ascend-debug/ascend-debug.py`:

```python
#!/usr/bin/env python3
import argparse
import pathlib
import sys

from ascend_debug import __version__


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
    collect.add_argument("--preset", choices=["quick"], default="quick")
    collect.add_argument("--pipeline", choices=["normalize-kernelize"], default="normalize-kernelize")
    collect.set_defaults(handler=lambda args: 0)

    open_cmd = subparsers.add_parser("open", help="Generate or open the debug dashboard")
    open_cmd.add_argument("run_dir", type=pathlib.Path)
    open_cmd.add_argument("--no-browser", action="store_true")
    open_cmd.set_defaults(handler=lambda args: 0)

    diff = subparsers.add_parser("diff", help="Compare collected tensors")
    diff.add_argument("run_dir", type=pathlib.Path)
    diff.set_defaults(handler=lambda args: parser.exit(2, "ascend-debug diff is implemented in a later slice\n"))

    locate = subparsers.add_parser("locate", help="Locate first bad kernel")
    locate.add_argument("run_dir", type=pathlib.Path)
    locate.set_defaults(handler=lambda args: parser.exit(2, "ascend-debug locate is implemented in a later slice\n"))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Configure and build the CLI target**

Run:

```bash
ninja -C build ascend-debug
```

Expected result:

```text
[...]/bin/ascend-debug
```

Then run:

```bash
build/bin/ascend-debug --help
```

Expected output contains:

```text
collect
open
diff
locate
```

- [ ] **Step 7: Commit this task**

Run:

```bash
git add tools/CMakeLists.txt test/CMakeLists.txt tools/ascend-debug test/tools/diagnostics/ascend-debug-cli.mlir test/tools/diagnostics/test_ascend_debug_cli.sh
git commit -m "feat: add ascend-debug cli shell"
```

### Task 2: Implement Artifact Layout and Manifest Skeleton

**Files:**
- Create: `tools/ascend-debug/ascend_debug/layout.py`
- Modify: `tools/ascend-debug/ascend-debug.py`
- Test: `test/tools/diagnostics/test_ascend_debug_cli.sh`

- [ ] **Step 1: Extend the shell test with manifest content checks**

Update the Python block in `test/tools/diagnostics/test_ascend_debug_cli.sh`:

```python
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
stages = manifest["stages"]
print(f"ascend_debug.manifest.stage_count={len(stages)}")
for i, stage in enumerate(stages):
    print(f"ascend_debug.stage.{i}={pathlib.Path(stage['path']).name}")
assert manifest["schema_version"] == 1
assert manifest["tool"] == "ascend-debug"
assert manifest["preset"] == "quick"
assert manifest["backend"] == "compile"
assert manifest["device_id"] is None
assert manifest["device_scope"] == "single_run_single_device"
assert [stage["order"] for stage in stages] == [0, 10, 19, 20, 29]
assert [stage["name"] for stage in stages] == [
    "source",
    "normalize-in",
    "normalize-out",
    "kernelize-in",
    "kernelize-out",
]
```

- [ ] **Step 2: Run the test and verify it fails on missing manifest fields**

Run:

```bash
llvm-lit -v build/test/tools/diagnostics/ascend-debug-cli.mlir
```

Expected failure after Task 1:

```text
KeyError: 'schema_version'
```

- [ ] **Step 3: Implement `layout.py`**

Create `tools/ascend-debug/ascend_debug/layout.py`:

```python
from __future__ import annotations

import json
import pathlib
import shutil
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StageArtifact:
    order: int
    name: str
    path: str


QUICK_NORMALIZE_KERNELIZE_STAGES: tuple[StageArtifact, ...] = (
    StageArtifact(0, "source", "stages/000-source.mlir"),
    StageArtifact(10, "normalize-in", "stages/010-normalize-in.mlir"),
    StageArtifact(19, "normalize-out", "stages/019-normalize-out.mlir"),
    StageArtifact(20, "kernelize-in", "stages/020-kernelize-in.mlir"),
    StageArtifact(29, "kernelize-out", "stages/029-kernelize-out.mlir"),
)


def prepare_run_dir(run_dir: pathlib.Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ["stages", "reports", "graphs", "tensors/final", "tensors/checkpoints", "profiles", "summaries"]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)


def write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_stage(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_manifest(
    run_dir: pathlib.Path,
    *,
    input_path: pathlib.Path,
    preset: str,
    pipeline: str,
    stages: tuple[StageArtifact, ...],
) -> None:
    write_json(
        run_dir / "manifest.json",
        {
            "schema_version": 1,
            "tool": "ascend-debug",
            "input": str(input_path),
            "preset": preset,
            "pipeline": pipeline,
            "backend": "compile",
            "device_id": None,
            "device_scope": "single_run_single_device",
            "stages": [
                {"order": stage.order, "name": stage.name, "path": stage.path}
                for stage in stages
            ],
        },
    )


def write_provenance_skeleton(run_dir: pathlib.Path) -> None:
    write_json(
        run_dir / "provenance.json",
        {
            "schema_version": 1,
            "boundaries": [],
            "kernels": [],
            "runtime_tasks": [],
        },
    )
```

- [ ] **Step 4: Import layout in the CLI**

Modify `tools/ascend-debug/ascend-debug.py` so imports include:

```python
from ascend_debug import layout
```

Keep the command handlers unchanged in this step.

- [ ] **Step 5: Run syntax checks**

Run:

```bash
python3 -m py_compile tools/ascend-debug/ascend-debug.py tools/ascend-debug/ascend_debug/layout.py
```

Expected result: command exits 0 with no output.

- [ ] **Step 6: Commit this task**

Run:

```bash
git add tools/ascend-debug/ascend_debug/layout.py tools/ascend-debug/ascend-debug.py test/tools/diagnostics/test_ascend_debug_cli.sh
git commit -m "feat: add ascend-debug artifact layout"
```

### Task 3: Implement `collect --preset quick`

**Files:**
- Create: `tools/ascend-debug/ascend_debug/runner.py`
- Create: `tools/ascend-debug/ascend_debug/collect.py`
- Modify: `tools/ascend-debug/ascend-debug.py`
- Test: `test/tools/diagnostics/test_ascend_debug_cli.sh`

- [ ] **Step 1: Run the test and verify collect does not create artifacts**

Run:

```bash
llvm-lit -v build/test/tools/diagnostics/ascend-debug-cli.mlir
```

Expected failure after Task 2:

```text
test: .../debug-run/stages/000-source.mlir: No such file or directory
```

- [ ] **Step 2: Implement external command runner**

Create `tools/ascend-debug/ascend_debug/runner.py`:

```python
from __future__ import annotations

import pathlib
import shutil
import subprocess


class CommandError(RuntimeError):
    pass


def find_tool(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    raise CommandError(f"required tool not found in PATH: {name}")


def run_command(argv: list[str], *, stdout_path: pathlib.Path | None = None) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True) if stdout_path else None
    stdout_file = stdout_path.open("w", encoding="utf-8") if stdout_path else subprocess.DEVNULL
    try:
        completed = subprocess.run(
            argv,
            stdout=stdout_file,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    finally:
        if stdout_path:
            stdout_file.close()
    if completed.returncode != 0:
        command = " ".join(argv)
        raise CommandError(
            f"command failed with exit code {completed.returncode}: {command}\n{completed.stderr}"
        )
```

- [ ] **Step 3: Implement quick collect**

Create `tools/ascend-debug/ascend_debug/collect.py`:

```python
from __future__ import annotations

import argparse

from ascend_debug import layout
from ascend_debug.runner import CommandError, find_tool, run_command


def collect_quick(args: argparse.Namespace) -> int:
    input_path = args.input.resolve()
    run_dir = args.out.resolve()
    if not input_path.exists():
        raise CommandError(f"input MLIR does not exist: {input_path}")

    stages = layout.QUICK_NORMALIZE_KERNELIZE_STAGES
    layout.prepare_run_dir(run_dir)

    source = run_dir / stages[0].path
    normalize_in = run_dir / stages[1].path
    normalize_out = run_dir / stages[2].path
    kernelize_in = run_dir / stages[3].path
    kernelize_out = run_dir / stages[4].path

    layout.copy_stage(input_path, source)
    layout.copy_stage(source, normalize_in)

    opt = find_tool("ascend-mlir-opt")
    run_command([opt, str(normalize_in), "--ascend-normalize"], stdout_path=normalize_out)
    layout.copy_stage(normalize_out, kernelize_in)
    run_command([opt, str(kernelize_in), "--ascend-kernelize"], stdout_path=kernelize_out)

    layout.write_manifest(
        run_dir,
        input_path=input_path,
        preset=args.preset,
        pipeline=args.pipeline,
        stages=stages,
    )
    layout.write_provenance_skeleton(run_dir)
    print(f"ascend-debug.collect.out={run_dir}")
    return 0
```

- [ ] **Step 4: Wire collect handler into CLI**

Modify `tools/ascend-debug/ascend-debug.py` imports:

```python
from ascend_debug.collect import collect_quick
from ascend_debug.runner import CommandError
```

Modify the collect handler:

```python
collect.set_defaults(handler=collect_quick)
```

Wrap `main` errors:

```python
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except CommandError as exc:
        print(f"ascend-debug: error: {exc}", file=sys.stderr)
        return 1
```

- [ ] **Step 5: Run syntax checks**

Run:

```bash
python3 -m py_compile \
  tools/ascend-debug/ascend-debug.py \
  tools/ascend-debug/ascend_debug/layout.py \
  tools/ascend-debug/ascend_debug/runner.py \
  tools/ascend-debug/ascend_debug/collect.py
```

Expected result: command exits 0 with no output.

- [ ] **Step 6: Rebuild the copied CLI**

Run:

```bash
ninja -C build ascend-debug
```

Expected result: command exits 0 and updates `build/bin/ascend-debug`.

- [ ] **Step 7: Run the focused lit test**

Run:

```bash
llvm-lit -v build/test/tools/diagnostics/ascend-debug-cli.mlir
```

Expected failure after this task: `open` has not generated `index.html`.

- [ ] **Step 8: Commit this task**

Run:

```bash
git add tools/ascend-debug/ascend_debug/runner.py tools/ascend-debug/ascend_debug/collect.py tools/ascend-debug/ascend-debug.py
git commit -m "feat: collect quick ascend debug artifacts"
```

### Task 4: Implement Minimal `open`

**Files:**
- Create: `tools/ascend-debug/ascend_debug/open_view.py`
- Modify: `tools/ascend-debug/ascend-debug.py`
- Test: `test/tools/diagnostics/test_ascend_debug_cli.sh`

- [ ] **Step 1: Verify the current focused test fails at `open`**

Run:

```bash
llvm-lit -v build/test/tools/diagnostics/ascend-debug-cli.mlir
```

Expected failure:

```text
test: .../debug-run/index.html: No such file or directory
```

- [ ] **Step 2: Implement dashboard generation**

Create `tools/ascend-debug/ascend_debug/open_view.py`:

```python
from __future__ import annotations

import argparse
import html
import json
import pathlib
import webbrowser

from ascend_debug.runner import CommandError


def load_manifest(run_dir: pathlib.Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise CommandError(f"manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def render_index(run_dir: pathlib.Path, manifest: dict) -> pathlib.Path:
    rows = []
    for stage in manifest.get("stages", []):
        path = html.escape(stage["path"])
        name = html.escape(stage["name"])
        order = html.escape(str(stage["order"]))
        rows.append(f"<tr><td>{order}</td><td>{name}</td><td><code>{path}</code></td></tr>")
    body = "\n".join(rows)
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>ascend-debug</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 32px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d8dee7; padding: 8px 10px; text-align: left; }}
    th {{ background: #f3f5f8; }}
    code {{ background: #eef2f7; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>ascend-debug run</h1>
  <p>input: <code>{html.escape(manifest.get("input", ""))}</code></p>
  <p>preset: <code>{html.escape(manifest.get("preset", ""))}</code></p>
  <table>
    <thead><tr><th>order</th><th>stage</th><th>path</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
</body>
</html>
"""
    index = run_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


def open_run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    index = render_index(run_dir, manifest)
    print(f"ascend-debug.open.index={index}")
    if not args.no_browser:
        webbrowser.open(index.as_uri())
    return 0
```

- [ ] **Step 3: Wire open handler into CLI**

Modify `tools/ascend-debug/ascend-debug.py` imports:

```python
from ascend_debug.open_view import open_run
```

Modify the open handler:

```python
open_cmd.set_defaults(handler=open_run)
```

- [ ] **Step 4: Run syntax checks**

Run:

```bash
python3 -m py_compile \
  tools/ascend-debug/ascend-debug.py \
  tools/ascend-debug/ascend_debug/open_view.py
```

Expected result: command exits 0 with no output.

- [ ] **Step 5: Rebuild and run the focused lit test**

Run:

```bash
ninja -C build ascend-debug
llvm-lit -v build/test/tools/diagnostics/ascend-debug-cli.mlir
```

Expected result:

```text
PASS: ... tools/diagnostics/ascend-debug-cli.mlir
```

- [ ] **Step 6: Commit this task**

Run:

```bash
git add tools/ascend-debug/ascend_debug/open_view.py tools/ascend-debug/ascend-debug.py
git commit -m "feat: open ascend debug dashboard"
```

### Task 5: Verify the Foundation Against Existing Gates

**Files:**
- No source changes unless verification exposes a failure.

- [ ] **Step 1: Run the focused diagnostics suite**

Run:

```bash
llvm-lit -v build/test/tools/diagnostics
```

Expected result:

```text
PASS: ... tools/diagnostics/ascend-debug-cli.mlir
PASS: ... tools/diagnostics/ascend-kernel-dag-viz.mlir
```

- [ ] **Step 2: Run the broader tools suite**

Run:

```bash
llvm-lit -v build/test/tools
```

Expected result: all enabled `test/tools` lit tests pass. Longrun tests remain gated by their existing environment variables and should not run unless explicitly enabled.

- [ ] **Step 3: Run the repository smoke gate that includes tools**

Run:

```bash
ninja -C build check-ascend-tools
```

Expected result: the target completes with 0 failing tests.

- [ ] **Step 4: Inspect changed files**

Run:

```bash
git status --short
git diff --stat
```

Expected changed files for this plan:

```text
tools/CMakeLists.txt
tools/ascend-debug/CMakeLists.txt
tools/ascend-debug/ascend-debug.py
tools/ascend-debug/ascend_debug/__init__.py
tools/ascend-debug/ascend_debug/layout.py
tools/ascend-debug/ascend_debug/runner.py
tools/ascend-debug/ascend_debug/collect.py
tools/ascend-debug/ascend_debug/open_view.py
test/CMakeLists.txt
test/tools/diagnostics/ascend-debug-cli.mlir
test/tools/diagnostics/test_ascend_debug_cli.sh
```

- [ ] **Step 5: Commit verification adjustments**

If Step 1 through Step 3 required fixes, commit them:

```bash
git add tools test
git commit -m "test: cover ascend-debug cli foundation"
```

If no fixes were needed after Task 4, do not create an empty commit.

## Self-Review

Spec coverage for this first slice:

- Covered: CLI-first `ascend-debug` command surface.
- Covered: `collect --preset quick`.
- Covered: numbered stage artifacts and manifest stage order.
- Covered: `provenance.json` skeleton with stable schema version.
- Covered: minimal `open` dashboard.
- Deferred by design: final-output diff, per-kernel checkpoints, first-bad-kernel locate, semantic boundary extraction, enhanced DAG/memory visualization, NPU `--device-id` runtime execution.

Placeholder scan:

- The plan contains no unresolved placeholder markers or open-ended implementation steps.
- The only deferred items are explicitly named follow-up plans outside this first slice.

Type and command consistency:

- CLI script name is consistently `ascend-debug`.
- Python package name is consistently `ascend_debug`.
- The first supported pipeline is consistently `normalize-kernelize`.
- The manifest stage order is consistently `[0, 10, 19, 20, 29]`.
