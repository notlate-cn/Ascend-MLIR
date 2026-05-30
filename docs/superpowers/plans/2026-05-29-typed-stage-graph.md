# Typed Stage Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace regex-only stage graph extraction with an MLIR-backed typed graph source and add AscendC resource/effect edges so lowered stage 60 graphs are connected by execution semantics.

**Architecture:** Add a small C++ diagnostic tool that reads one MLIR file and emits stage graph JSON from MLIR operations, operands, regions, and selected semantic classifiers. `ascend-debug open` prefers this tool when present and falls back to the current Python parser. The UI keeps rendering the existing schema while preserving edge `kind`, `effect`, and `label`.

**Tech Stack:** MLIR C++ APIs, PyAsc AscendC/EmitAsc dialects, Python `ascend-debug`, xvm build and diagnostic tests.

---

### Task 1: Add MLIR-backed graph dumper executable

**Files:**
- Create: `tools/ascend-stage-graph/CMakeLists.txt`
- Create: `tools/ascend-stage-graph/ascend-stage-graph.cpp`
- Modify: `tools/CMakeLists.txt`
- Test: `test/tools/diagnostics/test_ascend_debug_cli.sh`

- [ ] **Step 1: Write a failing CLI smoke test**

Add a test block to `test/tools/diagnostics/test_ascend_debug_cli.sh` that writes a small MLIR file with `memref.copy`, runs `build/bin/ascend-stage-graph`, and asserts JSON has `nodes` and `edges`.

- [ ] **Step 2: Run test to verify it fails**

Run on xvm:

```bash
cd /home/niu/code/Ascend-MLIR
source examples/env.sh
cd build
ninja ascend-debug
cd ..
bash test/tools/diagnostics/test_ascend_debug_cli.sh test/tools/diagnostics/ascend-debug-cli.mlir
```

Expected: failure because `build/bin/ascend-stage-graph` does not exist.

- [ ] **Step 3: Implement executable skeleton**

Create `ascend-stage-graph` that accepts:

```text
ascend-stage-graph <input.mlir> --stage-order <int> --stage-name <name> --stage-path <relpath> --output <graph.json>
```

It should parse MLIR with registered builtin, func, arith, affine, scf, memref, tensor, linalg, AscendC, and EmitAsc dialects, then emit JSON with schema-compatible `nodes`, `edges`, `stage`, `function`, `kernel_ids`, and `layout`.

- [ ] **Step 4: Verify executable builds**

Run:

```bash
cd /home/niu/code/Ascend-MLIR
source examples/env.sh
cd build
ninja ascend-stage-graph
```

Expected: build succeeds and `build/bin/ascend-stage-graph` exists.

### Task 2: Emit typed MLIR graph edges

**Files:**
- Modify: `tools/ascend-stage-graph/ascend-stage-graph.cpp`
- Test: `test/tools/diagnostics/test_ascend_debug_cli.sh`

- [ ] **Step 1: Add graph assertions**

Extend the CLI smoke test to assert:

```python
assert any(edge["kind"] == "value" for edge in graph["edges"])
assert any(edge["kind"] == "region" for edge in graph["edges"])
```

- [ ] **Step 2: Implement node and value edge extraction**

For every operation, emit stable node ids, op name, source location line, result values, input values, kernel attrs, and source excerpt where available. For every operand whose defining op or block argument is known, emit a `kind=value` edge.

- [ ] **Step 3: Implement region edges**

For each parent op with nested regions, emit `kind=region` edges from parent to direct child ops. Include block argument nodes for `scf.for` induction variables and region arguments.

- [ ] **Step 4: Verify typed graph smoke**

Run:

```bash
cd /home/niu/code/Ascend-MLIR
source examples/env.sh
cd build
ninja ascend-stage-graph
cd ..
bash test/tools/diagnostics/test_ascend_debug_cli.sh test/tools/diagnostics/ascend-debug-cli.mlir
```

Expected: test passes and graph JSON contains typed `value` and `region` edges.

### Task 3: Add AscendC semantic resource/effect edges

**Files:**
- Modify: `tools/ascend-stage-graph/ascend-stage-graph.cpp`
- Test: `test/tools/diagnostics/test_ascend_debug_cli.sh`

- [ ] **Step 1: Add stage 60 style test MLIR**

Add a small AscendC MLIR test with:

```mlir
%pipe = ascendc.pipe
%q = ascendc.queue : <vecin, 1>
ascendc.pipe.init_queue %pipe, %q, %c1_i32, %c64 : !ascendc.queue<vecin, 1>, i32, index
%t = ascendc.que_bind.alloc_tensor %q : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
ascendc.que_bind.enque_tensor %q, %t : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
%u = ascendc.que_bind.deque_tensor %q : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
ascendc.pipe_barrier pipe_all
ascendc.que_bind.free_tensor %q, %u : !ascendc.queue<vecin, 1>, !ascendc.local_tensor<*xf16>
```

Assert `resource_effect` edges connect queue initialization, enqueue/dequeue/free, and barrier ordering.

- [ ] **Step 2: Implement AscendC resource classifier**

Classify resource writes and reads by op name and operand position:

```text
pipe.init_queue: write queue operand
pipe.init_buffer: write tbuf operand
que_bind.alloc_tensor: read/write queue, result local tensor
que_bind.enque_tensor: write queue, read tensor
que_bind.deque_tensor: read/write queue, result local tensor
que_bind.free_tensor: write queue, read tensor
global_tensor.set_global_buffer: write global tensor, read memref
broadcast_l2/add_l2/mul_l2: write first local tensor, read remaining tensor operands
emitasc.verbatim: conservative write first tensor operand, read remaining tensor operands
pipe_barrier: order prior resource/effect ops before following resource/effect ops
```

- [ ] **Step 3: Emit resource labels**

Use labels such as `write %q`, `read %t`, `barrier pipe_all`, and keep `kind=resource_effect` or `kind=control`.

- [ ] **Step 4: Verify stage 60 no longer has isolated barrier**

After regenerating concat deep mode, inspect `graphs/stages/060-compute-lower-out.graph.json`; expected: `ascendc.pipe_barrier` nodes have incoming and outgoing typed edges.

### Task 4: Wire C++ dumper into ascend-debug open

**Files:**
- Modify: `tools/ascend-debug/ascend_debug/stage_graph.py`
- Modify: `tools/ascend-debug/ascend_debug/open_view.py`
- Modify: `tools/ascend-debug/ascend_debug/debug_graph.py`
- Test: `test/tools/diagnostics/test_ascend_debug_cli.sh`

- [ ] **Step 1: Add tool discovery fallback**

In `stage_graph.render_stage_graphs`, find `ascend-stage-graph` next to `ascend-debug` or on `PATH`. If found, generate each stage graph through C++; otherwise use the current Python parser.

- [ ] **Step 2: Keep Python fallback**

Do not remove the existing parser; keep it for source-tree local use and failure fallback. Add report metadata indicating `graph_source` is `mlir-tool` or `python-parser`.

- [ ] **Step 3: Render typed edge classes**

Map edge kinds to CSS classes in static and dynamic SVG:

```text
value: current solid gray
memory_effect: dashed teal
resource_effect: dashed purple
control: dotted orange
region: faint blue
symbol: dashed slate
```

- [ ] **Step 4: Verify workbench renders typed edges**

Run `ascend-debug open` on a generated run and check `views/debug_graph.html` contains CSS for typed edge classes and `graph_source`.

### Task 5: End-to-end verification on concat deep mode

**Files:**
- Test-only generated outputs under `/tmp/ascend-debug-add-broadcast-concat-semantic-view`

- [ ] **Step 1: Build and run xvm diagnostics**

Run:

```bash
cd /home/niu/code/Ascend-MLIR
source examples/env.sh
cd build
ninja ascend-debug ascend-stage-graph
cd ..
bash test/tools/diagnostics/test_ascend_debug_cli.sh test/tools/diagnostics/ascend-debug-cli.mlir
```

Expected: `ALL ASCEND DEBUG CLI TESTS PASSED`.

- [ ] **Step 2: Regenerate concat deep mode**

Run:

```bash
cd /home/niu/code/Ascend-MLIR
source examples/env.sh
rm -rf /tmp/ascend-debug-add-broadcast-concat-semantic-view
ascend-debug collect examples/add-broadcast-concat/step0_input.mlir --out /tmp/ascend-debug-add-broadcast-concat-semantic-view --mode deep
ascend-debug open /tmp/ascend-debug-add-broadcast-concat-semantic-view --no-browser
```

- [ ] **Step 3: Inspect stage 60 coverage**

Run:

```bash
cd /tmp/ascend-debug-add-broadcast-concat-semantic-view
python3 - <<'PY'
import json
from collections import defaultdict
from pathlib import Path
g=json.loads(Path("graphs/stages/060-compute-lower-out.graph.json").read_text())
incoming=defaultdict(list); outgoing=defaultdict(list)
for e in g["edges"]:
    incoming[e["to"]].append(e); outgoing[e["from"]].append(e)
for n in g["nodes"]:
    if n["op_name"] == "ascendc.pipe_barrier":
        assert incoming[n["id"]], n
        assert outgoing[n["id"]], n
print("stage60_barrier_edges=ok")
PY
```

Expected: `stage60_barrier_edges=ok`.

- [ ] **Step 4: Restart local server**

Run:

```bash
cd /home/niu/code/Ascend-MLIR
source examples/env.sh
ascend-debug serve /tmp/ascend-debug-add-broadcast-concat-semantic-view --host 127.0.0.1 --port 18765 --no-browser
```

Expected: `http://127.0.0.1:18765/` serves refreshed workbench data.

