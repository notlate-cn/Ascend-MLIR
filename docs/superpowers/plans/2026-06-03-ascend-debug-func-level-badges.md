# ascend-debug Func-Level Badges Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Populate the stage structural graph with func-level and per-arg metadata from develop's codegen IR (kernel-kind, block-dim, axis-extent, tile-size, symbolic-shape).

**Architecture:** Extend `tools/ascend-debug/ascend_debug/stage_graph.py` only: capture each func's `attributes {…}` block, emit a per-function `func.func` graph node carrying func-level badges, enrich existing `func.arg` nodes with their per-arg attr badges, and add a few attr sources to `_build_semantic_attrs`/`_build_node_badges`. Node badges already render in the dashboard, so no `open_view`/`collect`/manifest changes.

**Tech Stack:** Python 3 stdlib; pytest; existing `afir-opt` at `build/bin`.

**Source of truth:** spec `docs/superpowers/specs/2026-06-03-ascend-debug-func-level-badges-design.md`. The implementer must READ the current `tools/ascend-debug/ascend_debug/stage_graph.py` to find exact insertion points; key functions: `_collect_func_header`, `_parse_func_decl`, `parse_stage_mlir` (first pass builds `func.arg` nodes), `_build_semantic_attrs`, `_build_node_badges`, `_extract_attr`, `_extract_attr_as_str`, `_extract_int_attr`.

---

### Task 1: Func-level + per-arg badges in stage_graph

**Files:**
- Modify: `tools/ascend-debug/ascend_debug/stage_graph.py`
- Test: `tests/tools/ascend-debug/test_stage_graph_badges.py` (extend)

- [ ] **Step 1: Write failing tests**

Append to `tests/tools/ascend-debug/test_stage_graph_badges.py`:

```python
def test_func_node_emitted_with_kind_and_tiling_badges():
    text = (
        'module {\n'
        '  func.func @k(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> '
        'attributes {ascendc.kernel_kind = "vec", '
        'afir.block_dim_expr = "ceil((d0*d1)/XBLOCK)", '
        'afir.axis_extent_expr = "(d0*d1)"} {\n'
        '    return %arg0 : tensor<?x?xf32>\n'
        '  }\n'
        '}\n'
    )
    graph = stage_graph.parse_stage_mlir({"order": 1, "name": "schedule", "path": "p"}, text)
    func_nodes = [n for n in graph["nodes"] if n["op_name"] == "func.func"]
    assert len(func_nodes) == 1
    fn = func_nodes[0]
    assert fn["function"] == "@k"
    assert fn["semantic_attrs"]["kernel"]["role"] == "vec"
    assert fn["semantic_attrs"]["schedule"]["block_dim"] == "ceil((d0*d1)/XBLOCK)"
    assert fn["semantic_attrs"]["schedule"]["axis_extent"] == "(d0*d1)"
    # at least one badge surfaced
    assert fn["badges"]


def test_arg_node_carries_tile_and_shape_badges():
    text = (
        'module {\n'
        '  func.func @k(%arg0: tensor<?x?xf32> {afir.symbolic_shape = "s0,s1"}, '
        '%arg1: index {auto_fuse.default_tile_size = 128 : i64}) -> tensor<?x?xf32> {\n'
        '    return %arg0 : tensor<?x?xf32>\n'
        '  }\n'
        '}\n'
    )
    graph = stage_graph.parse_stage_mlir({"order": 1, "name": "schedule", "path": "p"}, text)
    arg_nodes = {n["label"]: n for n in graph["nodes"] if n["op_name"] == "func.arg"}
    assert arg_nodes["%arg0"]["semantic_attrs"]["kernel"].get("symbolic_shape") == "s0,s1" \
        or arg_nodes["%arg0"]["semantic_attrs"].get("memory", {}).get("symbolic_shape") == "s0,s1" \
        or any("s0,s1" in b for b in arg_nodes["%arg0"]["badges"])
    assert arg_nodes["%arg1"]["semantic_attrs"]["schedule"].get("default_tile_size") == "128" \
        or any("128" in b for b in arg_nodes["%arg1"]["badges"])
```

> The arg-node test uses tolerant `or` assertions because the exact home of `symbolic_shape`/`default_tile_size` in `semantic_attrs` depends on how you extend `_build_semantic_attrs` (Step 3). Make at least one branch of each `or` true.

- [ ] **Step 2: Run tests → see them fail**

Run: `python3 -m pytest tests/tools/ascend-debug/test_stage_graph_badges.py -v`
Expected: the two new tests FAIL (no `func.func` node today; arg nodes have empty badges).

- [ ] **Step 3: Implement in `stage_graph.py`**

Read the file first. Then:

**(a) Capture the func attribute block.** Add a helper that, given the lines and the index of the func decl, returns the text of the `attributes { … }` block that sits between the args' closing `)` and the func body's opening `{` (empty string if none). Do **NOT** modify `_collect_func_header` (it is reused for boundary detection in the op-scan pass). Example:

```python
def _collect_func_attr_block(lines: list[str], start_index: int) -> str:
    """Return the `attributes {...}` text between the func args ')' and the
    body '{', or '' if absent. Scans from the func decl line forward."""
    depth = 0
    seen_args = False
    buf: list[str] = []
    cursor = start_index
    while cursor < len(lines):
        line = lines[cursor]
        # find end of arg list
        for ch in line:
            if ch == "(":
                seen_args = True
                depth += 1
            elif ch == ")" and seen_args:
                depth -= 1
        if seen_args and depth <= 0:
            # capture from after the ')' on this line up to the body '{'
            tail = line.split(")", 1)[1] if ")" in line else line
            collect = [tail]
            scan = cursor + 1
            joined = tail
            while "{" not in joined.split("attributes", 1)[-1] and scan < len(lines):
                collect.append(lines[scan])
                joined += " " + lines[scan]
                scan += 1
            blob = " ".join(collect)
            m = re.search(r"attributes\s*(\{.*)$", blob, re.S)
            if not m:
                return ""
            # balance braces to the matching close
            s = m.group(1)
            d = 0
            for i, ch in enumerate(s):
                if ch == "{":
                    d += 1
                elif ch == "}":
                    d -= 1
                    if d == 0:
                        return s[: i + 1]
            return s
        cursor += 1
    return ""
```
(If your reading of the file suggests a simpler capture given existing helpers like `_extract_balanced_attr_value`, you may use that instead — the requirement is: get the func-level attribute text so `_build_semantic_attrs` can read `ascendc.kernel_kind`/`afir.block_dim_expr`/`afir.axis_extent_expr`/`afir.reduce_template` from it.)

**(b) Emit a `func.func` node per function.** In `parse_stage_mlir`'s first pass (where it iterates func decls and creates `func.arg` nodes), for each function also append one node BEFORE its arg nodes:

```python
            attr_blob = _collect_func_attr_block(lines, header_start_index)
            func_sem = _build_semantic_attrs("func.func", func_header + " " + attr_blob)
            fnode_id = f"n{len(nodes)}"
            nodes.append({
                "id": fnode_id, "line": line_number, "function": function_name,
                "op_name": "func.func", "label": function_name,
                "input_values": [], "result_values": [], "result_type": None,
                "kernel_id": None, "op_role": None, "schedule_decision_id": None,
                "workspace_size_bytes": None,
                "semantic_attrs": func_sem,
                "badges": _build_node_badges(func_sem),
            })
            function_node_ids[function_name].append(fnode_id)
```
(`header_start_index` = the index of the func decl line in the first-pass loop; `func_header` = the value already produced by `_collect_func_header`. Use the names that exist in the current loop.)

**(c) Enrich `func.arg` nodes.** Where each arg node is built, populate its badges from the arg's `type` text (which carries `{auto_fuse.default_tile_size = …}` / `{afir.symbolic_shape = …}`):

```python
            arg_sem = _build_semantic_attrs("func.arg", arg["type"])
            ... node["semantic_attrs"] = arg_sem
            ... node["badges"] = _build_node_badges(arg_sem)
```

**(d) Extend `_build_semantic_attrs`.** Add `ascendc.kernel_kind` as the last role fallback, and capture `symbolic_shape`:
```python
            "role": _extract_attr(op_text, "auto_fuse.kind")
            or _extract_attr(op_text, "aclnn.op")
            or _extract_attr(op_text, "aclnn.kind")
            or _extract_attr(op_text, "ascendc.kernel_kind"),
            "symbolic_shape": _extract_attr(op_text, "afir.symbolic_shape"),
```
(Keep `symbolic_shape` inside the `kernel` dict, alongside `id`/`role`.)

**(e) Extend `_build_node_badges`** to surface the new fields. After the existing badge logic, add:
```python
    if kernel.get("role"):
        ...  # (existing role badge — keep)
    if kernel.get("symbolic_shape"):
        badges.append(f"shape {kernel['symbolic_shape']}")
    if schedule.get("block_dim"):
        badges.append("block_dim")
    if schedule.get("axis_extent"):
        badges.append("axis")
    if schedule.get("default_tile_size"):
        badges.append(f"tile {schedule['default_tile_size']}")
```
(Only add branches that don't already exist; read the current `_build_node_badges` and avoid duplicate badges.)

- [ ] **Step 4: Run tests → pass**

Run: `python3 -m pytest tests/tools/ascend-debug/ -v`
Expected: all pass (the 2 new + the prior 11). If the func-node test sees 0 nodes, your first-pass insertion point or `_collect_func_attr_block` is wrong — debug against the test's inline IR.

- [ ] **Step 5: Confirm no `ascend.*` regression + grep guard**

Run: `grep -nE "ascend\.(schedule|op_role|op_roles|kernel|primary|kernelize)" tools/ascend-debug/ascend_debug/stage_graph.py`
Expected: empty (the `test_no_ascend_namespace_attrs_referenced` test also enforces this).

- [ ] **Step 6: Commit**

```bash
git add tools/ascend-debug/ascend_debug/stage_graph.py tests/tools/ascend-debug/test_stage_graph_badges.py
git commit -m "feat(ascend-debug): surface func-level + per-arg badges in stage graph"
```
(trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; develop; don't stage externals/scripts.)

---

### Task 2: End-to-end verification on real IR

**Files:** none (verification only). Uses built `afir-opt`.

- [ ] **Step 1: Run collect + open on a real example**
```bash
cd /home/gser/code/Ascend-MLIR
export PATH="$PWD/build/bin:$PATH"
rm -rf /tmp/badge-check
PYTHONPATH=tools/ascend-debug python3 tools/ascend-debug/ascend-debug.py collect examples/add-mul-relu-e2e/add_mul_relu.mlir --out /tmp/badge-check
PYTHONPATH=tools/ascend-debug python3 tools/ascend-debug/ascend-debug.py open /tmp/badge-check --no-browser
```
Expected: both exit 0.

- [ ] **Step 2: Confirm func/arg badges now populate**
```bash
python3 - <<'PY'
import json, glob
hits = 0
for f in sorted(glob.glob("/tmp/badge-check/graphs/stages/*.graph.json")):
    g = json.load(open(f))
    funcs = [n for n in g["nodes"] if n["op_name"] == "func.func"]
    badged = [n for n in g["nodes"] if n.get("badges")]
    print(f.split("/")[-1], "func_nodes=", len(funcs),
          "func_badges=", [n["badges"] for n in funcs][:1],
          "total_badged_nodes=", len(badged))
    hits += sum(1 for n in funcs if n.get("badges"))
print("FUNC_BADGE_HITS=", hits)
PY
```
Expected: `FUNC_BADGE_HITS` > 0 — at least the schedule/realize/finalize stage func nodes show kernel-kind / block_dim / axis badges, and arg nodes show tile/shape badges. (`add_mul_relu` func has `ascendc.kernel_kind`, `afir.block_dim_expr`, `afir.axis_extent_expr` from the schedule stage onward; args carry `afir.symbolic_shape` and `auto_fuse.default_tile_size`.)

- [ ] **Step 3: Re-run the lit smoke test (no regression)**
```bash
LLVM_BUILD_DIR=externals/llvm-project/build /home/gser/anaconda3/envs/torch-mlir/bin/lit -v test/tools/diagnostics/ascend-debug-cli.mlir 2>&1 | tail -4
```
Expected: PASS.

(No commit — verification only. If Step 2 shows `FUNC_BADGE_HITS=0`, the func-node/attr capture is broken; return to Task 1.)

---

## Self-Review

- Spec §3a (capture attr block) → Task 1 Step 3(a). ✓
- Spec §3b (func node) → 3(b). ✓
- Spec §3c (enrich arg nodes) → 3(c). ✓
- Spec §3d (role fallback + symbolic_shape) → 3(d). ✓ §3e badges → 3(e). ✓
- Spec §5 testing (unit + integration) → Task 1 Steps 1/4, Task 2. ✓
- Spec §6 risk (don't change `_collect_func_header`; isolated-node layout fallback) → 3(a) note + Task 2 renders the graph so a layout problem would surface. ✓
- No placeholders; code shown for each edit; tolerant arg-test assertions documented. Function/attr names (`_build_semantic_attrs`, `_build_node_badges`, `_collect_func_attr_block`, fields `kernel.role/symbolic_shape`, `schedule.block_dim/axis_extent/default_tile_size`) are consistent across tasks.
