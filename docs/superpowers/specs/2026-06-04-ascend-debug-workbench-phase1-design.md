# ascend-debug Workbench Phase 1 — Kernel-DAG from network.json

**Date:** 2026-06-04
**Status:** Approved (design phase)
**Follow-up to:** the collect+open MVP + func-level badges. Goal of the larger effort: a full-featured debug workbench. This Phase 1 lights up the **Kernel-DAG** panel from develop's existing network artifacts; later phases add Tensor Diff / Locate (need runs) and Memory.

---

## 1. What's already true (so we build the right thing)

The workbench (`debug_graph.html`) panels bind to:
- **Kernel-DAG** ← `workspace.kernel_dag = kernel_summary` (`graphs/kernel_dag.summary.json`). Needs `{nodes:{<kid>:{…}}, edges:[…], kernel_count, graph_edges, critical_path_depth}`. **Empty today** — this is Phase 1's target.
- **Provenance** ← `renderDataflowProvenance`: the selected stage-graph node's incoming/outgoing **edges**. Already works from our structural graph (per-node, on click). NOT fed by `network.provenance.json`. **No work needed.**
- **Tensor Diff / Locate / Memory** ← `overlay_details.*` — need runtime data. Phase 2/3, out of scope here.

So Phase 1 = produce a correct `kernel_dag.summary.json` and a run dir `open` can render. Provenance already works; the other panels stay empty (later phases).

## 2. Data source (no run, no recompile)

develop's network_runner outline phase already emits, in a work dir:
- `groups/network.json` — `{function, inputs:[{name,shape,dtype}], kernels:[{id, kind, op?, file, args:[{from:"input"|"kernel", name?|kernel, result?}], results:[{name,shape,dtype}]}], outputs}`
- `groups/network.provenance.json` — `{kernels:[{kernel_id, kind, fused_ops_summary, source_ops:[{id,loc}], boundary_source_ops, output_checkpoint_hint}]}`

Real samples exist at `examples/two-elewise-e2e/build_e2e/groups/` (used for dev/test). GPT-2 comes from a network_runner run's work dir (the user supplies a path / re-runs).

## 3. UX: a new `ingest` subcommand

```
ascend-debug ingest <network_workdir> --out <run_dir>
ascend-debug open <run_dir>
```
`ingest` reads `<network_workdir>/groups/network.json` (+ `network.provenance.json` if present), writes into `<run_dir>`:
- `graphs/kernel_dag.summary.json` (the adapted DAG — §4)
- `network.provenance.json` (copied through, for later phases / reference)
- `manifest.json` (via `layout.write_manifest`, `mode="network-ingest"`, with one stage entry pointing at the outlined network IR if present — see §5 — so `open` has something to render besides the DAG)

Rationale for a separate `ingest` (not folding into `collect`): `collect` runs the single-function `--auto-fuse-codegen` path that crashes on whole networks (e.g. GPT-2); `ingest` consumes already-produced network artifacts — decoupled, no recompile, reuses existing e2e outputs. Phase 2 (diff/locate) writes `summaries/*.json` into the same run dir.

## 4. Adapter: network.json (+provenance) → kernel_dag.summary.json

Pure Python, in a new module `ascend_debug/network_dag.py` with a function `build_kernel_dag_summary(network: dict, provenance: dict | None) -> dict`.

Mapping:
- `prov_by_id = {k["kernel_id"]: k for k in (provenance or {}).get("kernels", [])}`
- For each `k` in `network["kernels"]` (kid = `k["id"]`):
  - `nodes[kid] = {`
    - `"id": kid`,
    - `"kind": k.get("kind", "ascendc")`  (drives CSS class + display),
    - `"ops": [{"label": prov.get("fused_ops_summary") or k.get("op") or k.get("kind","kernel"), "op": k.get("op")}]`  (the render reads `node.ops[0].label`),
    - `"depth": <computed>`  (longest-path depth from roots over the kernel→kernel edges),
    - `"workspace_size": 0`  (unknown without runtime; placeholder, render tolerates),
    - `"output_shape": k["results"][0]["shape"] if results else None`,
    - `"output_dtype": k["results"][0]["dtype"] if results else None`,
    - `"source_ops": prov.get("source_ops", [])`,
    - `"aclnn_op": k.get("op")`  (when present) `}`
- Edges: for each kernel `k`, for each `arg` with `arg.get("from") == "kernel"`: `edges.append({"from": arg["kernel"], "to": kid})`. Dedupe.
- `kernel_count = len(nodes)`; `graph_edges = len(edges)`; `critical_path_depth = max(depths, default=0)`.
- Depth: topological — roots (no incoming kernel edge) have depth 0; `depth(n) = 1 + max(depth(pred))`. Guard against cycles (network DAG is acyclic; if a cycle is detected, cap depth and `log` a warning).

This mirrors what develop's `python/tools/ascend_kernel_dag_viz/gen_dag_json.py:build_dag` already does from the same inputs — we reuse its field conventions but emit the dev-nyh workbench summary shape (nodes keyed by id + edges + counts) rather than gen_dag_json's `nodes:[...]` list shape.

## 5. Manifest / IR view for the ingest run

`open` validates `manifest["stages"]` is a list and renders per-stage IR + structural graph. For an ingest run we have the outlined network IR available in the work dir (network_runner writes `_outlined_combined.mlir` and/or `groups/network.mlir` / per-kernel `kernel_group*.mlir`). Phase 1:
- If `<workdir>/_outlined_combined.mlir` (or `groups/network.mlir`) exists, copy it to `<run_dir>/stages/000-network.mlir` and register one `StageArtifact(order=0, name="network", path="stages/000-network.mlir", step="source")`. This gives a real IR view + structural graph of the outlined network alongside the Kernel-DAG.
- If no such file, write a stages list with zero entries is invalid for `open`'s validator — so register at least the source if available; otherwise `ingest` errors with a clear message naming what it looked for.

(Per-kernel 6-stage `collect` on each `kernel_group*.mlir` — i.e. full lowering IR per kernel — is a natural Phase-1b extension, not required to light up the Kernel-DAG panel. Out of scope for this spec.)

## 6. Files

- New: `tools/ascend-debug/ascend_debug/network_dag.py` — `build_kernel_dag_summary` (+ small depth helper).
- New: `tools/ascend-debug/ascend_debug/ingest.py` — `ingest_run(args)`: read network.json/provenance, call adapter, write `graphs/kernel_dag.summary.json` + `network.provenance.json` + `manifest.json`, copy network IR stage.
- Modify: `tools/ascend-debug/ascend-debug.py` — add the `ingest` subparser (positional `network_workdir`, required `--out`).
- Test: `tests/tools/ascend-debug/test_network_dag.py` (adapter unit tests) + extend the CLI parser test for `ingest`.

No changes to `open_view.py` / `debug_graph.py` (the Kernel-DAG render already consumes `kernel_dag.summary.json`).

## 7. Testing

Unit (pure, no afir-opt/run):
- Feed a synthetic 3-kernel network.json (k0,k1 → k2 via kernel args) + matching provenance → assert `summary["nodes"]` is a dict keyed by kid, `nodes["k2"]["depth"] == 1`, `nodes["k0"]["depth"] == 0`, `edges` contains `{from:"k0",to:"k2"}` and `{from:"k1",to:"k2"}`, `kernel_count == 3`, `nodes["k2"]["ops"][0]["label"]` comes from provenance `fused_ops_summary`.
- Cycle guard: a pathological self/loop edge doesn't infinite-loop.

Integration (real artifacts, no run):
- `ascend-debug ingest examples/two-elewise-e2e/build_e2e --out /tmp/te-wb` → `graphs/kernel_dag.summary.json` exists with ≥1 node; `ascend-debug open /tmp/te-wb` produces `index.html` + `views/debug_graph.html`; grep the workspace JSON shows `kernel_dag.nodes` non-empty (Kernel-DAG panel will render).
- Then GPT-2: point `ingest` at a network_runner GPT-2 work dir → Kernel-DAG shows the multi-kernel graph.

## 8. Risks / notes

- **two-elewise has very few kernels** — fine for wiring/schema validation; GPT-2 is the real visual test (dozens of kernels). The spec's "two-elewise first" is deliberate (cheap correctness), GPT-2 second (visual scale).
- **network.json `kind` values** (`ascendc`/`aclnn`/…) must map to CSS classes the render knows (`vec`/`cube`/`ascendc`). The render falls back to a default class for unknown kinds — acceptable; refine kind→class mapping if the DAG looks miscolored.
- **No depth/workspace runtime truth** — `workspace_size` is a placeholder (0); real values are a Phase-3 (memory/profiles) concern.
