# add-broadcast-concat CPU 仿真验证设计文档

**日期：** 2026-03-27
**目标：** 参照 `examples/broadcast-add-reduce`，为 `examples/add-broadcast-concat` 添加 CPU 仿真验证（stage 9 编译 + stage 10 验证），并升级编译路径对齐新工具链。

---

## 1. 背景

### 计算图

```
input_a[M]   + input_b[M,N] → C[M,N]   （Op1：广播加法）
input_c[M]   * input_d[M,N] → D[M,N]   （Op2：广播乘法）
concat(C, D, axis=0)        → output[2M,N]
```

### 现状缺陷

| 项目 | broadcast-add-reduce | add-broadcast-concat |
|------|---------------------|---------------------|
| stage 7b（CANN签名）| ✓ `afir-translate` | ✗ 临时 Python 脚本去 transform |
| stage 8（codegen）| ✓ `afir-translate -mlir-to-cann` | ✗ `ascir-translate -mlir-to-ascendc`（旧路径）|
| stage 9（编译 .bin）| ✓ | ✗ 缺失 |
| stage 10（CPU仿真）| ✓ | ✗ 缺失 |
| `tiling_space.json` | ✓ | ✗ 缺失 |
| `.npy` 测试数据 | ✓ | ✗ 缺失 |

---

## 2. 测试数据设计

### 尺寸选择：M=50, N=20

目的是测试 DataCopy 非对齐场景的通用处理能力：

- **M=50**：`50 = 16×3 + 2`，不是16的倍数。TB_M=16 时最后一个 block 只处理2行（尾块），触发行方向非对齐处理。
- **N=20**：`20 × 2B(half) = 40B`，不是32B的倍数，触发列方向 DataCopy 非对齐。
- **block-dim=4**：ceil(50/16) = 4，4个 AiCore block 并行。
- **TB_M=16, TB_N=20**（TB_N = N，不在 N 轴切分）。

### 数据生成（Python）

```python
import numpy as np

M, N = 50, 20
rng = np.random.default_rng(42)

input_a = rng.standard_normal((M,)).astype(np.float16)      # [M]
input_b = rng.standard_normal((M, N)).astype(np.float16)    # [M,N]
input_c = rng.standard_normal((M,)).astype(np.float16)      # [M]
input_d = rng.standard_normal((M, N)).astype(np.float16)    # [M,N]

C = input_a[:, None] + input_b   # [M,N] 广播加法
D = input_c[:, None] * input_d   # [M,N] 广播乘法
output = np.concatenate([C, D], axis=0)  # [2M,N] = [100,20]

np.save("input_a.npy", input_a)
np.save("input_b.npy", input_b)
np.save("input_c.npy", input_c)
np.save("input_d.npy", input_d)
np.save("output.npy", output)
```

预生成后提交到仓库，run.sh 直接引用，不在运行时动态生成。

---

## 3. 编译路径升级（stage 7b + stage 8）

### Stage 7b：新增 `--canonicalize-cann-signature`

```bash
afir-opt --canonicalize-cann-signature \
  step7_kernel.mlir \
  -o step7_cann.mlir
```

**效果：**
- 函数签名从 `(inputs..., memref<?x!emitasc.py_struct<...>, 22:i32>, outputs...)` 转为 `(inputs..., outputs..., memref<ui8>, !emitasc.py_struct<...>)`
- 自动计算 `cann.num_inputs = 4`（tilingIdx 之前有4个输入参数）
- 删除 module 级别的 transform 序列（取代临时 Python 脚本）

### Stage 8：改用 `afir-translate -mlir-to-cann`

```bash
afir-translate -mlir-to-cann step7_cann.mlir \
  -o step8_kernel.cpp \
  --tiling-space-out tiling_space.json
```

**效果：**
- 输出符合 CANN 标准签名的 C++ kernel
- 自动生成 `tiling_space.json` 骨架（包含所有10个 TilingData 字段）
- 删除 `step8_no_transform.mlir` 临时文件

---

## 4. tiling_space.json

`afir-translate` 自动生成骨架后，手动补充约束信息：

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

---

## 5. Stage 9：编译

```bash
compiler \
  --kernel step8_kernel.cpp \
  --output build_e2e \
  --name ewop_broadcast_concat \
  --num-inputs 4
```

输出：`build_e2e/ewop_broadcast_concat.bin`

---

## 6. Stage 10：CPU 仿真验证

```bash
validator \
  --bin build_e2e/ewop_broadcast_concat.bin \
  --name ewop_broadcast_concat \
  --inputs "input_a.npy,input_b.npy,input_c.npy,input_d.npy" \
  --expected output.npy \
  --tiling-schema tiling_space.json \
  --tiling-params 'TB_M=16,TB_N=20,dim_arg0_0=50,dim_arg1_1=20,dim_arg2_0=50,dim_arg3_1=20,dim_arg0_1=20,dim_arg1_0=50,dim_arg2_1=20,dim_arg3_0=50' \
  --block-dim 4 \
  --atol 1e-2 \
  --rtol 1e-2 \
  --dump-actual build_e2e/actual.txt \
  --dump-expected build_e2e/expected.txt \
  --precision 4
```

**精度容忍：** atol=1e-2, rtol=1e-2（float16 计算误差，略宽于 broadcast-add-reduce 的 1e-3）。

---

## 7. 文件变更清单

### 新增文件
- `examples/add-broadcast-concat/input_a.npy`（M=50）
- `examples/add-broadcast-concat/input_b.npy`（M×N=50×20）
- `examples/add-broadcast-concat/input_c.npy`（M=50）
- `examples/add-broadcast-concat/input_d.npy`（M×N=50×20）
- `examples/add-broadcast-concat/output.npy`（2M×N=100×20）
- `examples/add-broadcast-concat/tiling_space.json`

### 修改文件
- `examples/add-broadcast-concat/run.sh`：
  - stage 7b：新增 `--canonicalize-cann-signature` → `step7_cann.mlir`
  - stage 8：改用 `afir-translate -mlir-to-cann`，删除临时 Python 脚本和 `step8_no_transform.mlir`
  - stage 9：新增 `compiler` 调用
  - stage 10：新增 `validator` 调用
  - 汇总输出信息更新

### 可能更新文件（视 afir-translate 输出而定）
- `examples/add-broadcast-concat/step7_cann.mlir`（新生成）
- `examples/add-broadcast-concat/step8_kernel.cpp`（可能因新 codegen 路径有差异）

---

## 8. 非对齐处理验证目标

测试通过意味着：

1. **行尾块处理正确**：block 3 处理第 48-49 行（仅2行），DataCopy 使用正确的 count 参数而非固定 TB_M
2. **N 轴非对齐 DataCopy 正确**：N=20（40B）非32B对齐时，DataCopy API 选择和参数填写正确
3. **concat 输出正确**：output[0:50, :] = C，output[50:100, :] = D，strided subview offset 计算正确
4. **精度验证通过**：float16 计算结果在 atol=1e-2, rtol=1e-2 内

---

## 9. 实现顺序

1. 生成 `.npy` 测试数据（Python 脚本在 VM 上运行）
2. 升级 run.sh：stage 7b + stage 8 新路径
3. 运行 stage 7b/8，获得新的 `step7_cann.mlir`、`step8_kernel.cpp`、`tiling_space.json`
4. 完善 `tiling_space.json`（补充约束信息）
5. 新增 stage 9（compiler）
6. 新增 stage 10（validator）
7. 端到端运行验证精度通过
