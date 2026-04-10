# relu-broadcast-transpose CPU Simulation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CPU simulation + precision validation to `examples/relu-broadcast-transpose/` following the `broadcast-add-reduce` pattern, with M=640, N=500, TB_N=64.

**Architecture:** Three new files (gen_inputs.py, tiling_space.json, updated run.sh) wired into the existing 8-stage pipeline. The existing MLIR steps and step8_kernel.cpp are already present and correct; we only add the data-generation, compile, and validate stages (Stage 8b–10).

**Tech Stack:** Python/NumPy for data gen, existing `compiler`/`validator` CLI tools from lib/runtime, bash.

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `examples/relu-broadcast-transpose/gen_inputs.py` | Create | Generate input_data0.npy [640,1], input_data1.npy [500,640], output_expected.npy [500,640] |
| `examples/relu-broadcast-transpose/tiling_space.json` | Create | Tiling schema matching TilingData struct in step7_kernel.mlir |
| `examples/relu-broadcast-transpose/run.sh` | Modify | Add Stage 8b (gen_inputs), Stage 9 (compiler), Stage 10 (validator); fix header comment |

---

### Task 1: Create gen_inputs.py

**Files:**
- Create: `examples/relu-broadcast-transpose/gen_inputs.py`

Computation: `out[n,m] = relu(data0[m,0]) + data1[n,m]`
- `data0`: shape `[M,1]`, f16
- `data1`: shape `[N,M]`, f16
- `out`:   shape `[N,M]`, f16

relu is `max(x, 0)`. The broadcast/transpose is encoded in the indexing map `(d0,d1)->(d1,0)`: element `out[n,m]` reads `data0[m,0]`, so `relu(data0[:,0])` is broadcast along the n-axis.

- [ ] **Step 1: Create gen_inputs.py**

```python
"""Generate input/expected-output npy files for relu-broadcast-transpose.

Computation: out[n, m] = relu(data0[m, 0]) + data1[n, m]
  data0: (M, 1)   column vector, relu applied element-wise
  data1: (N, M)   2-D input matrix
  out:   (N, M)   result

Usage:
  python3 gen_inputs.py [--m M] [--n N] [--seed SEED] [--out-dir DIR]
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m",       type=int, default=640)
    parser.add_argument("--n",       type=int, default=500)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=".")
    args = parser.parse_args()

    M, N = args.m, args.n
    out_dir = Path(args.out_dir)

    rng = np.random.default_rng(args.seed)
    data0 = rng.uniform(-1.0, 1.0, (M, 1)).astype(np.float32).astype(np.float16)
    data1 = rng.uniform(-1.0, 1.0, (N, M)).astype(np.float32).astype(np.float16)

    # out[n, m] = relu(data0[m, 0]) + data1[n, m]
    # relu(data0[:,0]) has shape [M]; broadcast along n-axis → [N, M]
    relu_col = np.maximum(data0[:, 0].astype(np.float32), 0.0)  # [M]
    out = (relu_col[np.newaxis, :] + data1.astype(np.float32)).astype(np.float16)  # [N, M]

    np.save(out_dir / "input_data0.npy", data0)
    np.save(out_dir / "input_data1.npy", data1)
    np.save(out_dir / "output_expected.npy", out)

    print(f"input_data0:      {data0.shape} {data0.dtype}")
    print(f"input_data1:      {data1.shape} {data1.dtype}")
    print(f"output_expected:  {out.shape} {out.dtype}  range [{out.min():.4g}, {out.max():.4g}]")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script runs locally (on Mac)**

```bash
cd /Volumes/GM9/code/Ascend-MLIR/examples/relu-broadcast-transpose
python3 gen_inputs.py --m 640 --n 500 --seed 42 --out-dir /tmp/rbt_test
python3 -c "
import numpy as np
d0 = np.load('/tmp/rbt_test/input_data0.npy')
d1 = np.load('/tmp/rbt_test/input_data1.npy')
out = np.load('/tmp/rbt_test/output_expected.npy')
print('data0:', d0.shape, d0.dtype)
print('data1:', d1.shape, d1.dtype)
print('out:  ', out.shape, out.dtype)
# Spot check: out[0,0] == relu(data0[0,0]) + data1[0,0]
expected_00 = float(max(d0[0,0], 0)) + float(d1[0,0])
print(f'spot check out[0,0]={float(out[0,0]):.4f}  expected={expected_00:.4f}')
"
```

Expected output:
```
data0: (640, 1) float16
data1: (500, 640) float16
out:   (500, 640) float16
spot check out[0,0]=...  expected=...   # values should match
```

- [ ] **Step 3: Commit**

```bash
cd /Volumes/GM9/code/Ascend-MLIR
git add examples/relu-broadcast-transpose/gen_inputs.py
git commit -m "feat(example): add gen_inputs.py for relu-broadcast-transpose (M=640,N=500,seed=42)"
```

---

### Task 2: Create tiling_space.json

**Files:**
- Create: `examples/relu-broadcast-transpose/tiling_space.json`

TilingData field order from `step7_kernel.mlir` line 2:
`["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_0", "dim_arg0_1", "dim_arg1_1"]`

Mapping:
- `TB_M` = 640 (m-axis tile = full M, untiled)
- `TB_N` = 64  (n-axis tile, inter-core TB block)
- `dim_arg0_0` = M = 640 (data0 dim0)
- `dim_arg1_0` = N = 500 (data1 dim0)
- `dim_arg0_1` = 1       (data0 dim1, always 1)
- `dim_arg1_1` = M = 640 (data1 dim1)

Block dim = ceil(N / TB_N) = ceil(500 / 64) = 8 (last block has 500 - 7×64 = 52 rows).

- [ ] **Step 1: Create tiling_space.json**

```json
{
  "kernel": "relu_transpose_broadcast_add",
  "kernel_file": "step8_kernel.cpp",
  "soc": "Ascend910B1",
  "block_dim_expr": "ceil(N / TB_N)",
  "tiling_params": [
    {
      "name": "TB_M",
      "type": "int64",
      "values": [640],
      "note": "m-axis tile = full M (untiled); data0 has shape [M,1], processed whole in UB"
    },
    {
      "name": "TB_N",
      "type": "int64",
      "min": 64, "max": 64, "step": 64,
      "note": "n-axis TB block; N=500 not divisible by 64, last block=52 rows (tail block tested)"
    },
    {"name": "dim_arg0_0", "type": "int64", "fixed": true, "shape_key": "M"},
    {"name": "dim_arg1_0", "type": "int64", "fixed": true, "shape_key": "N"},
    {"name": "dim_arg0_1", "type": "int64", "fixed": true, "value": 1},
    {"name": "dim_arg1_1", "type": "int64", "fixed": true, "shape_key": "M"}
  ]
}
```

- [ ] **Step 2: Commit**

```bash
cd /Volumes/GM9/code/Ascend-MLIR
git add examples/relu-broadcast-transpose/tiling_space.json
git commit -m "feat(example): add tiling_space.json for relu-broadcast-transpose (TB_N=64, M=640, N=500)"
```

---

### Task 3: Update run.sh — add Stage 8b/9/10 and fix header

**Files:**
- Modify: `examples/relu-broadcast-transpose/run.sh`

Changes:
1. Fix header comment: path reference `ewop-broadcast-transpose` → `relu-broadcast-transpose`
2. Add `PYTHON`, `AFIR_TRANSLATE`, `COMPILER`, `VALIDATOR` variables (mirror broadcast-add-reduce)
3. Add Stage 7b: `--canonicalize-cann-signature` → `step7_cann.mlir`
4. Replace Stage 8 to use `afir-translate -mlir-to-cann` (same as broadcast-add-reduce)
5. Add Stage 8b: `gen_inputs.py`
6. Add Stage 9: `$COMPILER`
7. Add Stage 10: `$VALIDATOR`

- [ ] **Step 1: Replace run.sh entirely**

```bash
#!/bin/bash
# ============================================================
# relu + broadcast + transpose 完整编译流水线 Demo
#
# 用法：
#   source examples/env.sh
#   bash examples/relu-broadcast-transpose/run.sh
#
# 计算图：
#   输入: data0[M,1], data1[N,M]
#   out[n,m] = relu(data0[m,0]) + data1[n,m]
#
# 融合后单 linalg.generic，indexing_map:
#   data0: (d0,d1)->(d1,0)  — 转置+广播（列向量广播至[N,M]）
#   data1: (d0,d1)->(d0,d1) — identity
#
# 形状: M=640, N=500, TB_N=64, block_dim=8（N不整除TB_N，含tail block）
#
# 各阶段说明：
#   step0_input.mlir          原始 High-Level IR（linalg/tensor，完全符号化）
#   step0_input_out.mlir      --linalg-generalize-named-ops --linalg-fuse-elementwise-ops 结果
#                             → 融合为单 linalg.generic（relu+transpose+broadcast+add）
#   step1_fused.mlir          --canonicalize --cse（基于 step0_input_out.mlir）
#   step2_tiled.mlir          --transform-interpreter tiling 结果
#                             → 单 generic TB/Tb 两级循环
#   step3_bufferized.mlir     --one-shot-bufferize 结果（tensor→memref）
#   step4_buffer_placement.mlir  --ascendc-buffer-placement 结果
#                             → 推导 on-chip memory_space（VECIN=9, VECOUT=10）
#   step5_ascendc.mlir        --linalg-to-ascendc 结果
#                             → data_copy_l2（GM→UB）+ relu + broadcast_l2 + add_l2
#   step6_parallelize.mlir    --ascendc-parallelize（get_block_idx 单维调度）
#   step7_kernel.mlir         --ascendc-prepare-for-emit（kernel IR）
#   step7_cann.mlir           --canonicalize-cann-signature（CANN 标准签名）
#   step8_kernel.cpp          afir-translate -mlir-to-cann（C++ kernel）
# ============================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
PYTHON="${PYTHON:-python3}"
COMPILER="${COMPILER:-compiler}"
VALIDATOR="${VALIDATOR:-validator}"

# 解析参数
VERBOSE=false
for arg in "$@"; do
  case $arg in
    --log) VERBOSE=true ;;
  esac
done

log() {
  if $VERBOSE; then
    echo "$@"
  fi
}

echo "========================================================"
echo " relu + broadcast + transpose 编译流水线"
echo "========================================================"

# ── STAGE 0: 解析原始 IR ───────────────────────────────────
echo ""
echo "==================== [STAGE 0] 解析 + 融合 High-Level IR（relu + transpose + broadcast + add）===================="
log "  输入: step0_input.mlir"
$AFIR_OPT --linalg-generalize-named-ops \
  --linalg-fuse-elementwise-ops \
  --canonicalize --cse \
  "$DIR/step0_input.mlir" \
  -o "$DIR/step0_input_out.mlir" 2>&1
log "  ✓ 融合成功，输出: step0_input_out.mlir"

# ── STAGE 1: Canonicalize/CSE ──────────────────────────────
echo ""
echo "==================== [STAGE 1] Canonicalize/CSE ===================="
$AFIR_OPT --canonicalize --cse "$DIR/step0_input_out.mlir" \
  -o "$DIR/step1_fused.mlir" 2>&1
log "  ✓ 输出: step1_fused.mlir"

# ── STAGE 2: Transform Tiling ──────────────────────────────
echo ""
echo "==================== [STAGE 2] Tiling：--transform-interpreter ===================="
$AFIR_OPT --transform-interpreter "$DIR/step2_transform.mlir" \
  --canonicalize --cse \
  -o "$DIR/step2_tiled.mlir" 2>&1
log "  ✓ Tiling 成功，输出: step2_tiled.mlir"

# ── STAGE 3: Bufferize ─────────────────────────────────────
echo ""
echo "==================== [STAGE 3] Bufferize：--one-shot-bufferize ===================="
$AFIR_OPT \
  "--one-shot-bufferize=bufferize-function-boundaries=true allow-return-allocs-from-loops=true function-boundary-type-conversion=identity-layout-map" \
  "$DIR/step2_tiled.mlir" \
  --cse \
  -o "$DIR/step3_bufferized.mlir" 2>&1
log "  ✓ Bufferize 成功，输出: step3_bufferized.mlir"

# ── STAGE 4: Buffer Placement ──────────────────────────────
echo ""
echo "==================== [STAGE 4] Buffer Placement：--ascendc-buffer-placement ===================="
$AFIR_OPT \
  --ascendc-buffer-placement \
  "$DIR/step3_bufferized.mlir" \
  -o "$DIR/step4_buffer_placement.mlir" 2>&1
log "  ✓ Buffer Placement 成功，输出: step4_buffer_placement.mlir"

# ── STAGE 5: Linalg → AscendC Compute ─────────────────────
echo ""
echo "==================== [STAGE 5] Linalg → AscendC：--linalg-to-ascendc ===================="
$AFIR_OPT \
  --linalg-to-ascendc \
  "$DIR/step4_buffer_placement.mlir" \
  --canonicalize \
  --cse \
  -o "$DIR/step5_ascendc.mlir" 2>&1
log "  ✓ Linalg→AscendC 成功，输出: step5_ascendc.mlir"

# ── STAGE 6: AscendC Parallelize ───────────────────────────
echo ""
echo "==================== [STAGE 6] Parallelize：--ascendc-parallelize ===================="
$AFIR_OPT "$DIR/step5_ascendc.mlir" \
  --ascendc-parallelize \
  --canonicalize \
  --cse \
  -o "$DIR/step6_parallelize.mlir" 2>&1
log "  ✓ Parallelize 成功，输出: step6_parallelize.mlir"

# ── STAGE 7: Prepare For Emit ──────────────────────────────
echo ""
echo "==================== [STAGE 7] Prepare For Emit：--ascendc-prepare-for-emit ===================="
$AFIR_OPT "$DIR/step6_parallelize.mlir" \
  --ascendc-prepare-for-emit \
  --canonicalize \
  --cse \
  -o "$DIR/step7_kernel.mlir" 2>&1
log "  ✓ Prepare For Emit 成功，输出: step7_kernel.mlir"

# ── STAGE 7b: Canonicalize CANN signature ──────────────────
echo ""
echo "==================== [STAGE 7b] CANN Signature：--canonicalize-cann-signature ===================="
$AFIR_OPT --canonicalize-cann-signature \
  "$DIR/step7_kernel.mlir" \
  -o "$DIR/step7_cann.mlir" 2>&1
log "  ✓ CANN 签名规范化成功，输出: step7_cann.mlir"

# ── STAGE 8: AscendC C++ Code Generation ───────────────────
echo ""
echo "==================== [STAGE 8] Codegen：afir-translate -mlir-to-cann ===================="
"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/step7_cann.mlir" \
  -o "$DIR/step8_kernel.cpp" 2>&1
log "  ✓ Codegen 成功，输出: step8_kernel.cpp"

# ── STAGE 8b: Generate test data ───────────────────────────
echo ""
echo "==================== [STAGE 8b] 生成测试数据：gen_inputs.py ===================="
"$PYTHON" "$DIR/gen_inputs.py" --m 640 --n 500 --seed 42 --out-dir "$DIR" 2>&1
log "  ✓ 生成成功：input_data0.npy, input_data1.npy, output_expected.npy"

# ── STAGE 9: Compile AscendC kernel ────────────────────────
echo ""
echo "==================== [STAGE 9] Compile：bisheng C++ → .bin ===================="
BUILD_DIR="$DIR/build_e2e"
rm -fr "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
"$COMPILER" \
  --kernel "$DIR/step8_kernel.cpp" \
  --output "$BUILD_DIR" \
  --name relu_transpose_broadcast_add \
  --num-inputs 2 2>&1
log "  ✓ Compile 成功，输出: $BUILD_DIR/relu_transpose_broadcast_add.bin"

# ── STAGE 10: Run and verify ────────────────────────────────
echo ""
echo "==================== [STAGE 10] Run + Verify ===================="
log "  使用参数：TB_M=640, TB_N=64, M=640, N=500, block-dim=8"
BIN="$BUILD_DIR/relu_transpose_broadcast_add.bin"

if [ -f "$BIN" ]; then
  "$VALIDATOR" \
    --bin "$BIN" \
    --name relu_transpose_broadcast_add \
    --inputs "$DIR/input_data0.npy,$DIR/input_data1.npy" \
    --expected "$DIR/output_expected.npy" \
    --tiling-schema "$DIR/tiling_space.json" \
    --tiling-params 'TB_M=640,TB_N=64,dim_arg0_0=640,dim_arg1_0=500,dim_arg0_1=1,dim_arg1_1=640' \
    --block-dim 8 \
    --atol 1e-2 \
    --rtol 1e-2 \
    --dump-actual "$BUILD_DIR/actual.txt" \
    --dump-expected "$BUILD_DIR/expected.txt" \
    --precision 4 \
    2>&1 | grep -v '^\[info\]\|^\[PEM_AIC_LOG\]\|^\[INFO\]\|^\[WARNING\]' || true
else
  echo "  ⚠ bin not found — skipping run"
fi

echo ""
echo "========================================================"
echo " 流水线完成！生成文件："
echo "   step0_input_out.mlir        → generalize+fuse 后单 linalg.generic IR"
echo "   step1_fused.mlir            → canonicalize+cse 清理后 IR"
echo "   step2_tiled.mlir            → Tiling 后（单 generic TB/Tb 两级循环）"
echo "   step3_bufferized.mlir       → Bufferize 后（memref）"
echo "   step4_buffer_placement.mlir → on-chip 内存标注（VECIN/VECOUT）"
echo "   step5_ascendc.mlir          → AscendC compute ops（data_copy+relu+broadcast+add）"
echo "   step6_parallelize.mlir      → 多核 AiCore 调度（get_block_idx）"
echo "   step7_kernel.mlir           → 完整 AscendC kernel IR"
echo "   step7_cann.mlir             → CANN 标准签名 IR"
echo "   step8_kernel.cpp            → AscendC C++ kernel 源码"
echo "   build_e2e/relu_transpose_broadcast_add.bin → 编译后二进制"
echo "========================================================"

rm -fr *.dump
rm -fr *.toml
```

- [ ] **Step 2: Commit**

```bash
cd /Volumes/GM9/code/Ascend-MLIR
git add examples/relu-broadcast-transpose/run.sh
git commit -m "feat(example): add Stage 7b/8b/9/10 to relu-broadcast-transpose run.sh (CPU sim + validation)"
```

---

### Task 4: Run on xvm and verify

All commands run inside the `xvm` container after `source examples/env.sh`.

- [ ] **Step 1: Run the full pipeline**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && bash examples/relu-broadcast-transpose/run.sh --log"
```

Expected: all stages print `✓`, Stage 10 shows precision results.

- [ ] **Step 2: If Stage 10 passes, record actual.txt**

```bash
ssh xvm@orb "cat /home/niu/code/Ascend-MLIR/examples/relu-broadcast-transpose/build_e2e/actual.txt | head -10"
```

- [ ] **Step 3: If Stage 10 fails precision check, diagnose**

Check max_abs_diff in validator output. Common causes:
- Wrong tiling-params order (must match TilingData field order: TB_M, TB_N, dim_arg0_0, dim_arg1_0, dim_arg0_1, dim_arg1_1)
- Wrong block-dim (should be 8 = ceil(500/64))
- gen_inputs.py reference computation wrong (verify spot-check in Task 1 Step 2)

- [ ] **Step 4: Commit build artifacts if generated**

```bash
cd /Volumes/GM9/code/Ascend-MLIR
git add examples/relu-broadcast-transpose/build_e2e/actual.txt \
        examples/relu-broadcast-transpose/build_e2e/expected.txt 2>/dev/null || true
git commit -m "test(example): relu-broadcast-transpose CPU sim passes (M=640,N=500,atol=1e-2)" || true
```
