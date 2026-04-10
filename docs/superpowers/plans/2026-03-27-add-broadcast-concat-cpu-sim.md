# add-broadcast-concat CPU 仿真验证 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `examples/add-broadcast-concat` 添加 CPU 仿真验证（stage 9 编译 + stage 10 精度验证），并升级编译路径（stage 7b + stage 8）对齐 `broadcast-add-reduce` 的工具链。

**Architecture:** 在 VM（xvm）上运行所有编译和仿真步骤。先生成测试数据，再升级 run.sh 的编译路径（引入 `--canonicalize-cann-signature` + `afir-translate -mlir-to-cann`），最后补充 stage 9（`compiler`）和 stage 10（`validator`）并端到端验证精度。

**Tech Stack:** bash, afir-opt, afir-translate, compiler, validator, Python/numpy（数据生成）

---

## 文件变更清单

| 操作 | 文件 |
|------|------|
| 新增 | `examples/add-broadcast-concat/input_a.npy` |
| 新增 | `examples/add-broadcast-concat/input_b.npy` |
| 新增 | `examples/add-broadcast-concat/input_c.npy` |
| 新增 | `examples/add-broadcast-concat/input_d.npy` |
| 新增 | `examples/add-broadcast-concat/output.npy` |
| 新增 | `examples/add-broadcast-concat/tiling_space.json` |
| 修改 | `examples/add-broadcast-concat/run.sh` |
| 新增（由 run.sh 生成） | `examples/add-broadcast-concat/step7_cann.mlir` |
| 更新（由 run.sh 生成） | `examples/add-broadcast-concat/step8_kernel.cpp` |

---

## Task 1：生成测试数据 (.npy)

**Files:**
- Create: `examples/add-broadcast-concat/gen_inputs.py`（临时脚本，运行后可删除）
- Create: `examples/add-broadcast-concat/input_a.npy`
- Create: `examples/add-broadcast-concat/input_b.npy`
- Create: `examples/add-broadcast-concat/input_c.npy`
- Create: `examples/add-broadcast-concat/input_d.npy`
- Create: `examples/add-broadcast-concat/output.npy`

- [ ] **Step 1: 在 xvm 上运行数据生成脚本**

SSH 进 xvm，进入项目目录：
```bash
ssh xvm@orb
cd /home/niu/code/Ascend-MLIR
```

运行以下 Python 命令（inline，无需创建文件）：
```bash
python3 - <<'EOF'
import numpy as np, os
DIR = "examples/add-broadcast-concat"
M, N = 50, 20
rng = np.random.default_rng(42)
input_a = rng.standard_normal((M,)).astype(np.float16)
input_b = rng.standard_normal((M, N)).astype(np.float16)
input_c = rng.standard_normal((M,)).astype(np.float16)
input_d = rng.standard_normal((M, N)).astype(np.float16)
C = input_a[:, None] + input_b
D = input_c[:, None] * input_d
output = np.concatenate([C, D], axis=0)
np.save(f"{DIR}/input_a.npy", input_a)
np.save(f"{DIR}/input_b.npy", input_b)
np.save(f"{DIR}/input_c.npy", input_c)
np.save(f"{DIR}/input_d.npy", input_d)
np.save(f"{DIR}/output.npy", output)
print(f"input_a: {input_a.shape}, input_b: {input_b.shape}")
print(f"input_c: {input_c.shape}, input_d: {input_d.shape}")
print(f"output:  {output.shape}")
print("Done.")
EOF
```

Expected output:
```
input_a: (50,), input_b: (50, 20)
input_c: (50,), input_d: (50, 20)
output:  (100, 20)
Done.
```

- [ ] **Step 2: 验证文件已生成**

```bash
ls -lh examples/add-broadcast-concat/*.npy
```

Expected: 5 个 .npy 文件，大小合理（input_a ~200B，input_b ~2KB，output ~4KB）。

- [ ] **Step 3: 快速验证数据内容**

```bash
python3 -c "
import numpy as np
a = np.load('examples/add-broadcast-concat/input_a.npy')
b = np.load('examples/add-broadcast-concat/input_b.npy')
o = np.load('examples/add-broadcast-concat/output.npy')
print('input_a dtype:', a.dtype, 'shape:', a.shape)
print('input_b dtype:', b.dtype, 'shape:', b.shape)
print('output  dtype:', o.dtype, 'shape:', o.shape)
print('output[:3,:3] =', o[:3,:3])
"
```

Expected: dtype 均为 float16，shape 正确，output 前几个值非零。

- [ ] **Step 4: 提交测试数据**

```bash
cd /home/niu/code/Ascend-MLIR
git add examples/add-broadcast-concat/input_a.npy \
        examples/add-broadcast-concat/input_b.npy \
        examples/add-broadcast-concat/input_c.npy \
        examples/add-broadcast-concat/input_d.npy \
        examples/add-broadcast-concat/output.npy
git commit -m "feat(example): add test data for add-broadcast-concat (M=50, N=20, non-32B aligned)"
```

---

## Task 2：升级 run.sh stage 7b + stage 8

**Files:**
- Modify: `examples/add-broadcast-concat/run.sh`

目标：
- 新增 stage 7b（`--canonicalize-cann-signature`）生成 `step7_cann.mlir`
- stage 8 改用 `afir-translate -mlir-to-cann`，去掉临时 Python 脚本和 `step8_no_transform.mlir`

- [ ] **Step 1: 在 run.sh 中替换 stage 8 并插入 stage 7b**

找到 run.sh 中的 STAGE 8 代码块（约第 246-269 行），将其替换为以下内容：

```bash
# ── STAGE 7b: Canonicalize CANN signature ──────────────────
echo ""
echo "==================== [STAGE 7b] CANN Signature：--canonicalize-cann-signature ===================="
log "  输入: step7_kernel.mlir"
log "  输出: step7_cann.mlir（CANN 标准签名，去除 transform ops）"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
$AFIR_OPT --canonicalize-cann-signature \
  "$DIR/step7_kernel.mlir" \
  -o "$DIR/step7_cann.mlir" 2>&1
log "  ✓ CANN 签名规范化成功，输出: step7_cann.mlir"
log "$(grep -E 'func.func|cann.num_inputs|py_struct|memref<ui8>' "$DIR/step7_cann.mlir" | head -3)"


# ── STAGE 8: AscendC C++ Code Generation (CANN standard) ───
echo ""
echo "==================== [STAGE 8] Codegen：afir-translate -mlir-to-cann ===================="
log "  输入: step7_cann.mlir"
log "  输出: step8_kernel.cpp（CANN 标准 C++ kernel）"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/step7_cann.mlir" \
  -o "$DIR/step8_kernel.cpp" \
  --tiling-space-out "$DIR/tiling_space.json" 2>&1
log "  ✓ Codegen 成功，输出: step8_kernel.cpp"
log "  ✓ Tiling space 骨架: tiling_space.json"
log ""
log "  [生成的 C++ kernel 头部]"
log "$(head -5 "$DIR/step8_kernel.cpp")"
```

- [ ] **Step 2: 更新 run.sh 末尾的汇总信息**

在 run.sh 末尾的 `流水线完成！` 汇总块中，将原来的文件列表更新为：

```bash
echo "========================================================"
echo " 流水线完成！生成文件："
echo "   step0_input_out.mlir        → 解析后 IR"
echo "   step1_fused.mlir            → 融合后（本场景基本不变）"
echo "   step2_tiled.mlir            → Tiling 后（Op1/Op2 各自 TB/Tb 两级循环）"
echo "   step3_bufferized.mlir       → Bufferize 后（memref，insert_slice→subview）"
echo "   step4_buffer_placement.mlir → on-chip 内存标注（VECIN/VECOUT）"
echo "   step5_ascendc.mlir          → AscendC compute ops（broadcast_l2/add_l2/mul_l2）"
echo "   step6_parallelize.mlir      → 多核 AiCore 调度（get_block_idx）"
echo "   step7_kernel.mlir           → 完整 AscendC kernel IR"
echo "   step7_cann.mlir             → CANN 标准签名 IR（去除 transform ops）"
echo "   step8_kernel.cpp            → AscendC C++ kernel 源码"
echo "   tiling_space.json           → tiling 参数空间（JSON）"
echo "========================================================"
```

- [ ] **Step 3: 在 xvm 上运行 stage 0-8，验证新路径通过**

```bash
ssh xvm@orb
cd /home/niu/code/Ascend-MLIR
source examples/env.sh
bash examples/add-broadcast-concat/run.sh --log 2>&1 | tail -30
```

Expected：所有阶段输出 `✓`，末尾显示生成文件列表，无报错。

- [ ] **Step 4: 验证 step7_cann.mlir 包含正确的 cann.num_inputs**

```bash
grep 'cann.num_inputs' examples/add-broadcast-concat/step7_cann.mlir
```

Expected：
```
cann.num_inputs = 4 : i32
```

- [ ] **Step 5: 验证 tiling_space.json 已生成（骨架）**

```bash
cat examples/add-broadcast-concat/tiling_space.json
```

Expected：包含 `TB_M`, `TB_N`, `dim_arg0_0` 等字段的 JSON。

- [ ] **Step 6: 提交**

```bash
git add examples/add-broadcast-concat/run.sh \
        examples/add-broadcast-concat/step7_cann.mlir \
        examples/add-broadcast-concat/step8_kernel.cpp \
        examples/add-broadcast-concat/tiling_space.json
git commit -m "feat(example): upgrade add-broadcast-concat to afir-translate codegen path (stage 7b+8)"
```

---

## Task 3：完善 tiling_space.json

**Files:**
- Modify: `examples/add-broadcast-concat/tiling_space.json`

`afir-translate` 自动生成的骨架只有参数名和类型，需手动补充约束信息（搜索范围、fixed 标记、shape_key、block_dim_expr）。

- [ ] **Step 1: 用完整内容覆写 tiling_space.json**

将 `examples/add-broadcast-concat/tiling_space.json` 内容替换为：

```json
{
  "kernel": "ewop_broadcast_concat",
  "kernel_file": "step8_kernel.cpp",
  "soc": "Ascend910B1",
  "block_dim_expr": "ceil(M / TB_M)",
  "tiling_params": [
    {
      "name": "TB_M",
      "type": "int64",
      "min": 16, "max": 32, "step": 16,
      "note": "Tiling rows per block; tail block may be smaller"
    },
    {
      "name": "TB_N",
      "type": "int64",
      "fixed": true, "shape_key": "N",
      "note": "N axis not tiled; TB_N == N"
    },
    {"name": "dim_arg0_0", "type": "int64", "fixed": true, "shape_key": "M"},
    {"name": "dim_arg1_1", "type": "int64", "fixed": true, "shape_key": "N"},
    {"name": "dim_arg2_0", "type": "int64", "fixed": true, "shape_key": "M"},
    {"name": "dim_arg3_1", "type": "int64", "fixed": true, "shape_key": "N"},
    {"name": "dim_arg0_1", "type": "int64", "fixed": true, "shape_key": "N"},
    {"name": "dim_arg1_0", "type": "int64", "fixed": true, "shape_key": "M"},
    {"name": "dim_arg2_1", "type": "int64", "fixed": true, "shape_key": "N"},
    {"name": "dim_arg3_0", "type": "int64", "fixed": true, "shape_key": "M"}
  ]
}
```

注意：`dim_argX_Y` 字段名称必须与 `step8_kernel.cpp` 中 `TilingData` 结构体成员名完全一致。用以下命令确认：

```bash
grep 'v7\.' examples/add-broadcast-concat/step8_kernel.cpp | head -15
```

Expected：出现 `v7.TB_M`, `v7.TB_N`, `v7.dim_arg0_0`, ..., `v7.dim_arg3_0`，与 JSON 中的名称一一对应。

- [ ] **Step 2: 提交**

```bash
git add examples/add-broadcast-concat/tiling_space.json
git commit -m "feat(example): add tiling_space.json for add-broadcast-concat"
```

---

## Task 4：新增 stage 9（compiler）

**Files:**
- Modify: `examples/add-broadcast-concat/run.sh`

- [ ] **Step 1: 在 run.sh 末尾汇总块之前插入 stage 9**

在 run.sh 的 `流水线完成！` 汇总 echo 块之前插入：

```bash
# ── STAGE 9: Compile AscendC kernel ──────────────────────────────────
echo ""
echo "==================== [STAGE 9] Compile：bisheng C++ → .bin ===================="
log "  输入: step8_kernel.cpp"
log "  输出: build_e2e/ewop_broadcast_concat.bin"
COMPILER="${COMPILER:-compiler}"
BUILD_DIR="$DIR/build_e2e"
rm -fr "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
"$COMPILER" \
  --kernel "$DIR/step8_kernel.cpp" \
  --output "$BUILD_DIR" \
  --name ewop_broadcast_concat \
  --num-inputs 4 2>&1
log "  ✓ Compile 成功，输出: $BUILD_DIR/ewop_broadcast_concat.bin"
```

- [ ] **Step 2: 在 xvm 上运行验证 stage 9**

```bash
ssh xvm@orb
cd /home/niu/code/Ascend-MLIR
source examples/env.sh
bash examples/add-broadcast-concat/run.sh 2>&1 | grep -E 'STAGE 9|✓|ERROR|error' | head -20
```

Expected：出现 `[STAGE 9]` 行，无报错，`build_e2e/ewop_broadcast_concat.bin` 存在。

```bash
ls -lh examples/add-broadcast-concat/build_e2e/ewop_broadcast_concat.bin
```

Expected：文件存在，大小非零。

- [ ] **Step 3: 提交**

```bash
git add examples/add-broadcast-concat/run.sh
git commit -m "feat(example): add stage 9 compiler step to add-broadcast-concat"
```

---

## Task 5：新增 stage 10（validator）并端到端验证

**Files:**
- Modify: `examples/add-broadcast-concat/run.sh`

- [ ] **Step 1: 在 stage 9 之后插入 stage 10**

在 stage 9 代码块之后、汇总 echo 块之前插入：

```bash
# ── STAGE 10: Run and verify ──────────────────────────────────────────
echo ""
echo "==================== [STAGE 10] Run + Verify ===================="
log "  使用参数：TB_M=16, TB_N=20, M=50, N=20, block-dim=4"
VALIDATOR="${VALIDATOR:-validator}"
BIN="$BUILD_DIR/ewop_broadcast_concat.bin"

if [ -f "$BIN" ]; then
  "$VALIDATOR" \
    --bin "$BIN" \
    --name ewop_broadcast_concat \
    --inputs "$DIR/input_a.npy,$DIR/input_b.npy,$DIR/input_c.npy,$DIR/input_d.npy" \
    --expected "$DIR/output.npy" \
    --tiling-schema "$DIR/tiling_space.json" \
    --tiling-params 'TB_M=16,TB_N=20,dim_arg0_0=50,dim_arg1_1=20,dim_arg2_0=50,dim_arg3_1=20,dim_arg0_1=20,dim_arg1_0=50,dim_arg2_1=20,dim_arg3_0=50' \
    --block-dim 4 \
    --atol 1e-2 \
    --rtol 1e-2 \
    --dump-actual "$BUILD_DIR/actual.txt" \
    --dump-expected "$BUILD_DIR/expected.txt" \
    --precision 4 \
    2>&1 | grep -v '^\[info\]\|^\[PEM_AIC_LOG\]\|^\[INFO\]\|^\[WARNING\]' || true
else
  echo "  ⚠ bin not found — skipping run"
fi
```

同时更新末尾汇总 echo 块，追加 stage 9/10 的输出文件：

```bash
echo "========================================================"
echo " 流水线完成！生成文件："
echo "   step0_input_out.mlir        → 解析后 IR"
echo "   step1_fused.mlir            → 融合后（本场景基本不变）"
echo "   step2_tiled.mlir            → Tiling 后（Op1/Op2 各自 TB/Tb 两级循环）"
echo "   step3_bufferized.mlir       → Bufferize 后（memref，insert_slice→subview）"
echo "   step4_buffer_placement.mlir → on-chip 内存标注（VECIN/VECOUT）"
echo "   step5_ascendc.mlir          → AscendC compute ops（broadcast_l2/add_l2/mul_l2）"
echo "   step6_parallelize.mlir      → 多核 AiCore 调度（get_block_idx）"
echo "   step7_kernel.mlir           → 完整 AscendC kernel IR"
echo "   step7_cann.mlir             → CANN 标准签名 IR（去除 transform ops）"
echo "   step8_kernel.cpp            → AscendC C++ kernel 源码"
echo "   tiling_space.json           → tiling 参数空间（JSON）"
echo "   build_e2e/ewop_broadcast_concat.bin → 编译后二进制"
echo "========================================================"
```

- [ ] **Step 2: 端到端运行完整流水线**

```bash
ssh xvm@orb
cd /home/niu/code/Ascend-MLIR
source examples/env.sh
bash examples/add-broadcast-concat/run.sh 2>&1 | grep -v '^\[info\]\|^\[PEM_AIC_LOG\]\|^\[INFO\]\|^\[WARNING\]'
```

Expected 关键输出：
```
==================== [STAGE 10] Run + Verify ====================
passed: true
max_abs_diff: ...
mean_abs_diff: ...
cycle_count: ...
```

- [ ] **Step 3: 如果 validator 报告 passed: false，诊断**

检查 actual vs expected 的差异：
```bash
diff <(head -20 examples/add-broadcast-concat/build_e2e/actual.txt) \
     <(head -20 examples/add-broadcast-concat/build_e2e/expected.txt)
```

常见原因：
- `dim_arg` 字段名与 TilingData 不匹配 → 检查 `step8_kernel.cpp` 中的成员名
- tiling_params 中 M/N 的 `shape_key` 映射错误 → 核对 `block_dim_expr` 中的 `M` 对应哪个 `dim_arg`
- DataCopy 非对齐问题 → 查看 simulator log 中是否有对齐报错

- [ ] **Step 4: 验证通过后提交**

```bash
git add examples/add-broadcast-concat/run.sh
git commit -m "feat(example): add stage 10 validator + end-to-end CPU sim for add-broadcast-concat (M=50,N=20 non-aligned)"
```

---

## Task 6：提交 step7_cann.mlir 和 step8_kernel.cpp 中间产物

**Files:**
- Create: `examples/add-broadcast-concat/step7_cann.mlir`（由 run.sh 生成，提交到仓库作为参考）
- Modify: `examples/add-broadcast-concat/step8_kernel.cpp`（由新 codegen 路径重新生成）

- [ ] **Step 1: 确认文件已由 run.sh 生成且内容正确**

```bash
# 确认 step7_cann.mlir 有 cann.num_inputs
grep 'cann.num_inputs' examples/add-broadcast-concat/step7_cann.mlir

# 确认 step8_kernel.cpp 函数签名为 CANN 标准格式（inputs, outputs, workspace, tiling）
head -5 examples/add-broadcast-concat/step8_kernel.cpp
```

- [ ] **Step 2: 删除旧的临时文件**

```bash
rm -f examples/add-broadcast-concat/step8_no_transform.mlir
```

- [ ] **Step 3: 提交所有中间产物**

```bash
git add examples/add-broadcast-concat/step7_cann.mlir \
        examples/add-broadcast-concat/step8_kernel.cpp
git rm --cached examples/add-broadcast-concat/step8_no_transform.mlir 2>/dev/null || true
git commit -m "chore(example): update add-broadcast-concat intermediate IR artifacts (CANN codegen path)"
```

---

## Self-Review 检查

**Spec coverage：**
- [x] 测试数据生成（M=50, N=20）→ Task 1
- [x] stage 7b（`--canonicalize-cann-signature`）→ Task 2
- [x] stage 8（`afir-translate -mlir-to-cann`）→ Task 2
- [x] tiling_space.json 完善 → Task 3
- [x] stage 9（compiler）→ Task 4
- [x] stage 10（validator）→ Task 5
- [x] 端到端验证通过 → Task 5 Step 2
- [x] 中间产物更新 → Task 6

**Placeholder 检查：** 无 TBD/TODO，所有命令有 expected output。

**类型一致性：** kernel 名 `ewop_broadcast_concat` 在 Task 2-5 中一致；`tiling_params` 字段名在 Task 3 和 Task 5 的 `--tiling-params` 中一致；`--num-inputs 4` 与 cann.num_inputs=4 一致。
