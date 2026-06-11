# ascend-debug Func-Level Badges — Design

**Date:** 2026-06-03
**Status:** Approved (design phase)
**Follow-up to:** `2026-06-03-ascend-debug-retarget-develop-design.md` (the collect+open MVP). That MVP shipped the per-stage structural graph, but its semantic badges read attributes that, on the single-function `--auto-fuse-codegen` path, live at **func level** (`afir.block_dim_expr`, `afir.axis_extent_expr`, `ascendc.kernel_kind`, `afir.reduce_template`) or at **func-arg level** (`auto_fuse.default_tile_size`, `afir.symbolic_shape`). The MVP parser only feeds per-*op* text to badge extraction, so these never populate. This change surfaces them.

---

## 1. Problem

`stage_graph.parse_stage_mlir` builds graph nodes per op and per func-arg. `_build_semantic_attrs(op_name, op_text)` extracts badges from a node's op text. But:
- `_collect_func_header` stops at the args' closing `)`, so the trailing `attributes { afir.block_dim_expr=…, ascendc.kernel_kind=… }` block is never seen → func-level badges are empty.
- Per-arg attrs (`%arg4: index {auto_fuse.default_tile_size = 128 : i64}`) are captured inside each arg's `type` string but never parsed into the arg node's badges.

Net: on codegen-path IR the structural graph renders but is badge-sparse.

## 2. Goal / Non-Goals

**Goal:** populate the structural graph with func-level and per-arg metadata so each stage's graph shows kernel-kind / tiling / shape info that develop's codegen IR actually carries.

**Non-goals:**
- Module-level `auto_fuse.tiling_infos` per-op tile_params (still requires module-level parse + kernel_id association — separate follow-up).
- The optional C++ `ascend-stage-graph` tool path (off by default).
- Any change to `collect`, the manifest contract, or `open_view` rendering — node badges already render; we only produce more of them.

## 3. Design (Option A: function node + enriched arg nodes)

### 3a. Capture the func attribute block
Add a helper `_collect_func_attr_block(lines, body_open_index_hint)` (or extend `_parse_func_decl` to also return the attribute text) that, starting after the args' `)`, captures the text between `)` and the func body's opening `{` — i.e. the `attributes { … }` block when present. **Do not change `_collect_func_header`'s existing return index** (it is also used for function-boundary detection in the op-scan pass; changing it risks skipping body lines). The capture is used only in the function-node-building pass.

### 3b. Emit one `func.func` node per function
In `parse_stage_mlir`'s first pass (which already creates `func.arg` nodes), before/alongside the arg nodes, emit a node:
```
{
  "id": "n<i>", "line": <func decl line>, "function": <name>,
  "op_name": "func.func", "label": <name>,
  "input_values": [], "result_values": [], "result_type": None,
  "kernel_id": None, "op_role": None, "schedule_decision_id": None,
  "workspace_size_bytes": None,
  "semantic_attrs": _build_semantic_attrs("func.func", <signature + attr block>),
  "badges": _build_node_badges(<that semantic_attrs>),
}
```
The node is **isolated** (no edges) for v1 — isolated nodes render through the existing layout. `kernel_id=None` so `_finalize_graph`'s kernel aggregation ignores it.

### 3c. Enrich `func.arg` nodes
For each arg node, run badge extraction over the arg's `type` text (which carries `{auto_fuse.default_tile_size = …}` / `{afir.symbolic_shape = …}`) and set its `semantic_attrs`/`badges` instead of leaving them empty.

### 3d. Badge sources (extend `_build_semantic_attrs` minimally)
Add `ascendc.kernel_kind` as a role fallback (it is the codegen-path kernel-kind attr): role = first of `auto_fuse.kind` → `aclnn.op` → `aclnn.kind` → `ascendc.kernel_kind`. Add a `symbolic_shape` field reading `afir.symbolic_shape` (string) so arg shape shows. Existing schedule reads (`afir.block_dim_expr`, `afir.axis_extent_expr`, `auto_fuse.default_tile_size`, `afir.reduce_template`) already produce badges once the attr text reaches the extractor — 3a/3b/3c make that happen. `_build_node_badges` gains a small branch to emit a `shape …` badge from the new `symbolic_shape` field and a `block_dim`/`axis` badge from the existing schedule fields if not already surfaced.

Forward-compatibility: the role fallback chain still works on outlined network IR (where `auto_fuse.kind`/`aclnn.op` exist), so nothing regresses for that path.

## 4. Files

- `tools/ascend-debug/ascend_debug/stage_graph.py` — the only production file touched: `_collect_func_attr_block` (new), `parse_stage_mlir` (emit func node + enrich arg nodes), `_build_semantic_attrs` (role fallback + symbolic_shape), `_build_node_badges` (shape/block_dim/axis badges).
- `tests/tools/ascend-debug/test_stage_graph_badges.py` — extend with func-node + arg-node badge cases.

## 5. Testing

Unit (pytest), parsing real IR snippets:
- A func decl with `attributes {ascendc.kernel_kind = "vec", afir.block_dim_expr = "ceil(.../XBLOCK)", afir.axis_extent_expr = "..."}` → the emitted `func.func` node has role `"vec"` and `block_dim`/`axis` badges.
- A func arg `%a: index {auto_fuse.default_tile_size = 128 : i64}` → that arg node carries a tile-size badge; `%x: tensor<?x?xf32> {afir.symbolic_shape = "s0,s1"}` → a shape badge.
- Regression: the existing 4 badge tests + the `no ascend.* literals` guard still pass.

Integration (manual / existing lit): re-run `collect` + `open` on `add-mul-relu-e2e`; confirm the 020-schedule-out (and later) stage graphs now show a `func.func` node with kernel-kind/block-dim badges and arg nodes with tile-size/shape badges, and `open` still produces HTML without error.

## 6. Risks

- **`_collect_func_header` reuse:** must not alter its boundary-detection return; the attr capture is additive via a separate helper (3a). Verified-by: existing op-scan still groups ops under the right function (node `function` fields unchanged in tests).
- **Layout of isolated func node:** if the existing layout mishandles a node with no edges, fall back to giving the func node a `control`/`contains` edge to its first arg node. Decide during implementation based on the rendered graph.
