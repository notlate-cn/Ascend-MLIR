# Ascend Debug Contract Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move ascend-debug toward versioned Debug Contract JSON as the primary semantic interface while keeping the current stage dump/report/artifact parsing as legacy fallback.

**Architecture:** Add a schema-first Python consumption layer under `tools/ascend-debug/ascend_debug/contracts.py`. `collect.py` and `open_view.py` should prefer `debug_contract/*.json` files when present and only invoke existing MLIR/report/artifact parsers when contracts are absent. Compiler-side contract emission will be introduced incrementally after the consumer path is stable; current field-level parsing remains isolated as fallback and should not grow.

**Tech Stack:** Python ascend-debug CLI, existing JSON run manifest/view generation, MLIR lit diagnostics, C++/LLVM JSON emitters in later phases.

---

### Task 1: Document v1 Debug Contract Shape

**Files:**
- Create: `docs/schemas/ascend-debug/v1/README.md`
- Test: `test/tools/diagnostics/test_ascend_debug_cli.sh`

- [ ] **Step 1: Create the schema README**

Add `docs/schemas/ascend-debug/v1/README.md` with the following content:

```markdown
# Ascend Debug Contract v1

Ascend Debug Contract files are versioned JSON files produced by the compiler
or by a compatibility adapter. They are the primary semantic interface for
`ascend-debug`. Stage MLIR dumps and pass reports are source browsing and
legacy fallback inputs only.

Every contract file is a JSON object with:

```json
{
  "schema": "ascend.debug.<name>",
  "schema_version": 1,
  "producer": {
    "tool": "ascend-mlir-opt",
    "pass": "ascend-schedule"
  },
  "data": {}
}
```

Required v1 contracts:

- `stage_manifest.json`: ordered stage records, step catalog entries, report
  paths, graph paths, and contract paths.
- `schedule_decisions.json`: schedule decisions by kernel with guards,
  fallback markers, tile summaries, and raw extension payloads.
- `kernel_dag.json`: kernel DAG nodes and edges with stable display facts and
  optional raw runtime-artifact references.
- `memory_plan.json`: realize memory placement, movement, workspace slots, and
  failure reasons.

Unknown fields are preserved under `raw` or `extensions` and must not make the
debug UI fail.
```

- [ ] **Step 2: Add a CLI smoke assertion**

In `test/tools/diagnostics/test_ascend_debug_cli.sh`, add a filesystem check near the other documentation/fixture checks:

```bash
test -f "${REPO_ROOT}/docs/schemas/ascend-debug/v1/README.md"
grep -Fq 'schema_version' "${REPO_ROOT}/docs/schemas/ascend-debug/v1/README.md"
```

- [ ] **Step 3: Run the smoke test and verify it passes**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/tools/diagnostics/ascend-debug-cli.mlir'
```

Expected: `PASS: Ascend :: tools/diagnostics/ascend-debug-cli.mlir`.

### Task 2: Add Contract Loader Infrastructure

**Files:**
- Create: `tools/ascend-debug/ascend_debug/contracts.py`
- Modify: `tools/ascend-debug/ascend_debug/open_view.py`
- Test: `test/tools/diagnostics/test_ascend_debug_cli.sh`

- [ ] **Step 1: Add failing Python coverage**

Add this Python block to the existing embedded Python diagnostics section in
`test/tools/diagnostics/test_ascend_debug_cli.sh`:

```python
from ascend_debug import contracts

contract_root = pathlib.Path(tempfile.mkdtemp(prefix="ascend-debug-contracts."))
(contract_root / "stage_manifest.json").write_text(json.dumps({
    "schema": "ascend.debug.stage_manifest",
    "schema_version": 1,
    "producer": {"tool": "fixture"},
    "data": {"stages": []}
}), encoding="utf-8")
bundle = contracts.load_contract_bundle(contract_root)
assert bundle.has("ascend.debug.stage_manifest"), bundle.available_schemas()
assert bundle.get("ascend.debug.stage_manifest")["data"]["stages"] == []
```

Expected before implementation: import fails because `contracts.py` does not exist.

- [ ] **Step 2: Implement `contracts.py`**

Create `tools/ascend-debug/ascend_debug/contracts.py`:

```python
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class ContractBundle:
    root: pathlib.Path
    contracts: dict[str, dict[str, Any]]

    def has(self, schema: str) -> bool:
        return schema in self.contracts

    def get(self, schema: str) -> dict[str, Any]:
        return self.contracts[schema]

    def available_schemas(self) -> list[str]:
        return sorted(self.contracts)


def _load_contract(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid debug contract {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"debug contract must be an object: {path}")
    schema = value.get("schema")
    version = value.get("schema_version")
    data = value.get("data")
    if not isinstance(schema, str) or not schema.startswith("ascend.debug."):
        raise ContractError(f"debug contract missing valid schema: {path}")
    if version != 1:
        raise ContractError(f"unsupported debug contract version in {path}: {version}")
    if not isinstance(data, dict):
        raise ContractError(f"debug contract data must be an object: {path}")
    return value


def load_contract_bundle(root: pathlib.Path) -> ContractBundle:
    resolved = root.resolve()
    contracts: dict[str, dict[str, Any]] = {}
    for path in sorted(resolved.glob("*.json")):
        contract = _load_contract(path)
        contracts[contract["schema"]] = contract
    return ContractBundle(root=resolved, contracts=contracts)
```

- [ ] **Step 3: Run the diagnostics lit**

Run the same xvm lit command from Task 1. Expected: PASS.

### Task 3: Collect Contract Files Without Parsing Their Internals

**Files:**
- Modify: `tools/ascend-debug/ascend_debug/collect.py`
- Modify: `tools/ascend-debug/ascend-debug.py`
- Test: `test/tools/diagnostics/test_ascend_debug_cli.sh`

- [ ] **Step 1: Add failing CLI fixture**

In `test/tools/diagnostics/test_ascend_debug_cli.sh`, create a temporary
contract directory and assert collect preserves it:

```bash
mkdir -p "${TMP_DIR}/debug-contract"
cat >"${TMP_DIR}/debug-contract/stage_manifest.json" <<'JSON'
{
  "schema": "ascend.debug.stage_manifest",
  "schema_version": 1,
  "producer": {"tool": "fixture"},
  "data": {
    "stages": [
      {"order": 0, "name": "source", "path": "stages/000-source.mlir"}
    ]
  }
}
JSON
ascend-debug collect "${INPUT_MLIR}" \
  --out "${TMP_DIR}/debug-run-contract" \
  --mode quick \
  --debug-contract-dir "${TMP_DIR}/debug-contract"
test -f "${TMP_DIR}/debug-run-contract/debug_contract/stage_manifest.json"
grep -Fq 'ascend.debug.stage_manifest' "${TMP_DIR}/debug-run-contract/debug_contract/stage_manifest.json"
```

Expected before implementation: `--debug-contract-dir` is rejected.

- [ ] **Step 2: Add the CLI option**

In `tools/ascend-debug/ascend-debug.py`, add:

```python
collect_parser.add_argument(
    "--debug-contract-dir",
    type=pathlib.Path,
    help="Directory containing versioned Ascend Debug Contract JSON files.",
)
```

- [ ] **Step 3: Copy contract files during collect**

In `tools/ascend-debug/ascend_debug/collect.py`, import `contracts` and copy all
contract JSON files to `run_dir / "debug_contract"` after `layout.prepare_run_dir`.
Record each copied file in manifest `graphs` as:

```python
{"kind": "debug-contract", "schema": schema, "path": f"debug_contract/{path.name}"}
```

Use `contracts.load_contract_bundle(args.debug_contract_dir)` for validation.

- [ ] **Step 4: Run diagnostics lit**

Expected: PASS.

### Task 4: Prefer Contract Kernel DAG When Present

**Files:**
- Modify: `tools/ascend-debug/ascend_debug/kernel_dag.py`
- Modify: `tools/ascend-debug/ascend_debug/collect.py`
- Test: `test/tools/diagnostics/test_ascend_debug_cli.sh`

- [ ] **Step 1: Add a contract DAG fixture**

Extend the `debug-run-contract` fixture with `kernel_dag.json`:

```json
{
  "schema": "ascend.debug.kernel_dag",
  "schema_version": 1,
  "producer": {"tool": "fixture"},
  "data": {
    "kernel_count": 1,
    "task_count": 0,
    "graph_edges": 0,
    "kind_counts": {"vec": 1, "cube": 0, "mix": 0},
    "root_tasks": 1,
    "root_task_ids": ["kernel_0"],
    "leaf_tasks": 1,
    "leaf_task_ids": ["kernel_0"],
    "runtime_input_roots": 0,
    "runtime_input_root_ids": [],
    "prepack_candidate_roots": 1,
    "prepack_candidate_root_ids": ["kernel_0"],
    "critical_path_depth": 1,
    "critical_path": ["kernel_0"],
    "simple_fusion_edges": [],
    "edges": [],
    "nodes": {
      "kernel_0": {
        "kind": "vec",
        "depth": 1,
        "input_degree": 0,
        "output_degree": 0,
        "output_shape": "4x8",
        "output_dtype": "f16",
        "workspace_size": 0,
        "semantic_source": "debug_contract",
        "raw": {"unknown_field": "kept"}
      }
    }
  }
}
```

Assert `graphs/kernel_dag.summary.json` contains `semantic_source` and `raw`.

- [ ] **Step 2: Implement contract summary path**

Add `kernel_dag.summary_from_contract(contract: dict[str, Any]) -> dict[str, Any]`
that validates required top-level defaults, fills missing counters with zero,
and preserves `nodes[*].raw`.

- [ ] **Step 3: Make collect prefer the contract**

When `ascend.debug.kernel_dag` is in the bundle, call
`summary_from_contract()` and render/write summary/report from that data.
Only call `analyze_paths()` when the contract is absent.

- [ ] **Step 4: Run diagnostics lit**

Expected: PASS.

### Task 5: Mark Legacy Parsers Explicitly

**Files:**
- Modify: `tools/ascend-debug/ascend_debug/kernel_dag.py`
- Modify: `tools/ascend-debug/ascend_debug/stage_graph.py`
- Modify: `tools/ascend-debug/ascend_debug/debug_graph.py`
- Test: `test/tools/diagnostics/test_ascend_debug_cli.sh`

- [ ] **Step 1: Add assertions for legacy source markers**

For existing artifact/report-derived summaries, assert:

```python
assert graph["kernel_dag"]["semantic_source"] == "legacy_adapter"
```

- [ ] **Step 2: Add source markers**

Set `summary["semantic_source"] = "legacy_adapter"` in legacy `kernel_dag.analyze()`.
Set contract-derived summaries to `semantic_source = "debug_contract"`.
In `stage_graph.parse_stage_mlir`, set graph-level
`semantic_source = "legacy_mlir_regex"`.

- [ ] **Step 3: Show source in UI without blocking unknown fields**

Display `semantic_source` in the debug graph summary cards and kernel detail
facts. Unknown `raw` fields remain JSON text.

- [ ] **Step 4: Run diagnostics lit**

Expected: PASS.

### Task 6: Verification and Commit

**Files:**
- All files changed above.

- [ ] **Step 1: Run local Python compile check**

```bash
python3 -m py_compile \
  tools/ascend-debug/ascend_debug/contracts.py \
  tools/ascend-debug/ascend_debug/collect.py \
  tools/ascend-debug/ascend_debug/kernel_dag.py \
  tools/ascend-debug/ascend_debug/open_view.py \
  tools/ascend-debug/ascend_debug/debug_graph.py \
  tools/ascend-debug/ascend_debug/stage_graph.py
```

Expected: exit 0.

- [ ] **Step 2: Run xvm focused diagnostics**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/tools/diagnostics/ascend-debug-cli.mlir'
```

Expected: PASS.

- [ ] **Step 3: Run final hygiene**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended files modified.

- [ ] **Step 4: Commit**

```bash
git add docs/schemas/ascend-debug/v1/README.md \
  docs/superpowers/plans/2026-06-06-ascend-debug-contract-migration.md \
  test/tools/diagnostics/test_ascend_debug_cli.sh \
  tools/ascend-debug/ascend-debug.py \
  tools/ascend-debug/ascend_debug/contracts.py \
  tools/ascend-debug/ascend_debug/collect.py \
  tools/ascend-debug/ascend_debug/kernel_dag.py \
  tools/ascend-debug/ascend_debug/debug_graph.py \
  tools/ascend-debug/ascend_debug/stage_graph.py
git commit -m "Introduce ascend-debug contract consumer"
```

Expected: commit succeeds.

---

## Self-Review

- Spec coverage: The plan covers contract schema documentation, Python consumer
  infrastructure, collect integration, kernel DAG contract preference, legacy
  source markers, tests, verification, and commit.
- Placeholder scan: No TBD/TODO/later placeholders remain.
- Type consistency: Contract schema names use `ascend.debug.<name>` throughout;
  loader returns `ContractBundle`; consumer tasks use the same API names.
