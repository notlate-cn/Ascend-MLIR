# aclnn 直调方案设计

**Date:** 2026-04-22
**Status:** Approved for implementation
**Supersedes:** `2026-04-20-aclnn-fallback-design.md`（本文档为最终采纳方案）

---

## 1. 目标

将 `tm_tensor.attention` 等非 linalg op 转为对 CANN aclnn 算子库的直接调用，打通从 torch-mlir 输出到 Ascend NPU 上 FlashAttention 执行的完整链路。

**设计约束：**
- 保留 4D BNSD 布局（pre-collapse 4D 源）
- 面向动态 shape，运行时传 shape
- 独立 AclnnBackend，不修改现有 MixDirectBackend
- 两阶段产物：先输出 `network.mlir`，再单独 emit C++

---

## 2. 完整 Pipeline

```
torch-opt
  ConvertTmTensorAttentionPass          [已有，修改 attr]
  → func.func private @__aclnn_flash_attention
        attributes {aclnn.kind = "flash_attention"}
  → coordinator: func.call @__aclnn_flash_attention(q4d, k4d, v4d, mask4d, init4d)

afir-opt
  --vector-plan-group-analysis          [不变]
  --vector-plan-group-outline           [不变]
  --aclnn-finalize-decl                 [新增]
  → func.func private @__aclnn_flash_attention
        attributes {aclnn.op = "FlashAttentionScore", aclnn.layout = "BNSD"}
  → aclnn.kind 被移除

输出 network.mlir                       [新增产物]

AclnnBackend                            [新增，独立]
  读 network.mlir → network_host.cpp
```

---

## 3. 两个 Pass 的接口契约

### 3.1 职责划分

| Pass | 阶段 | 职责 |
|------|------|------|
| `ConvertTmTensorAttentionPass` | torch-opt | 结构变换：4D bypass，创建 `func.call`，声明 `func.func private` |
| `AclnnFinalizeDeclPass` | afir-opt | 语义填充：按注册表将 `aclnn.kind` 映射为最终 `aclnn.op` + `aclnn.layout` |

### 3.2 握手属性

**`aclnn.kind`** 是两个 pass 之间唯一的契约属性：

- `ConvertTmTensorAttentionPass` 负责设置
- `AclnnFinalizeDeclPass` 负责消费并替换
- `aclnn.kind` 在 `AclnnFinalizeDeclPass` 执行后不再存在

```
torch-opt 输出                      afir-opt 输出（AclnnBackend 的输入）
─────────────────────────────       ─────────────────────────────────────
aclnn.kind = "flash_attention"  →   aclnn.op     = "
"
                                    aclnn.layout  = "BNSD"
```

**不使用 `ascendc.unit`**：该属性语义上属于 AscendC 设备 unit（Cube/Vector），不应用于标记 host 侧 aclnn 调用。

### 3.3 IR 变化示意

**ConvertTmTensorAttentionPass 输出（torch-opt 之后）：**

```mlir
// module 顶层声明
func.func private @__aclnn_flash_attention(
    tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>,
    tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>
) -> tensor<?x?x?x?xf16>
    attributes {aclnn.kind = "flash_attention"}

// coordinator 中的调用点
%q_dyn  = tensor.cast %q4d  : tensor<1x2x8x4xf16>  to tensor<?x?x?x?xf16>
%k_dyn  = tensor.cast %k4d  : tensor<1x2x8x4xf16>  to tensor<?x?x?x?xf16>
%v_dyn  = tensor.cast %v4d  : tensor<1x2x8x4xf16>  to tensor<?x?x?x?xf16>
%m_dyn  = tensor.cast %m4d  : tensor<1x2x8x8xf16>  to tensor<?x?x?x?xf16>
%i_dyn  = tensor.cast %i4d  : tensor<1x2x8x4xf16>  to tensor<?x?x?x?xf16>
%out = func.call @__aclnn_flash_attention(%q_dyn, %k_dyn, %v_dyn, %m_dyn, %i_dyn)
    : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>,
       tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16>
```

**AclnnFinalizeDeclPass 输出（network.mlir 中的声明）：**

```mlir
func.func private @__aclnn_flash_attention(
    tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>,
    tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>
) -> tensor<?x?x?x?xf16>
    attributes {aclnn.op = "FlashAttentionScore", aclnn.layout = "BNSD"}
```

---

## 4. AclnnFinalizeDeclPass

**Pass 级别：** `ModuleOp`

**注册表：** 静态 `StringMap`，key = `aclnn.kind` 值，value = 最终元数据

```cpp
struct AclnnOpMeta {
  StringRef op;      // aclnn API 名，AclnnBackend 按此查 wrapper
  StringRef layout;  // 输入 tensor layout
};

static const StringMap<AclnnOpMeta> kRegistry = {
  {"flash_attention", {"FlashAttentionScore", "BNSD"}},
  // 扩展新 op：追加一行，实现对应的 C++ wrapper
};
```

**Pass 逻辑：**

```
对 module 中每个 func.func private：
  读 aclnn.kind attr
  若不存在 → 跳过
  若存在但不在注册表 → emitError + signalPassFailure（fail-fast）
  若在注册表：
    setAttr aclnn.op = meta.op
    setAttr aclnn.layout = meta.layout
    removeAttr aclnn.kind
```

**与现有 vector-plan passes 的关系：** 在 `--vector-plan-group-outline` 之后运行。`func.call @__aclnn_*` 在 group analysis 中天然不进入任何 Group（非 LinalgOp），在 outline pass 中留在 coordinator func，两个 pass 均无需修改。

---

## 5. network.mlir 格式

`afir-opt` 全部 passes 执行后的完整 module，是 AclnnBackend 的输入，也是调试检查点。

```mlir
module {
  // ① aclnn 声明（已完成属性替换）
  func.func private @__aclnn_flash_attention(...) -> tensor<?x?x?x?xf16>
      attributes {aclnn.op = "FlashAttentionScore", aclnn.layout = "BNSD"}

  // ② coordinator（host 侧主函数）
  func.func @model(%arg0: tensor<?x?xf16>, ...) -> tensor<?x?xf16> {
    ...
    %out = func.call @__aclnn_flash_attention(%q, %k, %v, %mask, %init) : ...
    %r   = func.call @kernel_group_0(%out, ...) : ...
    return %r
  }

  // ③ AscendC kernel 签名（函数体已由 outline 提取到独立文件）
  func.func private @kernel_group_0(...) -> ...
  func.func private @kernel_group_1(...) -> ...
}
```

coordinator 中的 `func.call` 调用两类函数：
- `@__aclnn_*`：带 `aclnn.op` attr → AclnnBackend 生成 aclnn C++ dispatch
- `@kernel_group_X`：无特殊 attr → AclnnBackend 生成 `ACLRT_LAUNCH_KERNEL` 调用

---

## 6. AclnnBackend

### 6.1 定位

独立于 `MixDirectBackend`。`MixDirectBackend` 是单 kernel 编译器（一个 AscendC 核一次）；`AclnnBackend` 是 network 级别的 orchestrator（读 coordinator func，生成多 op 调度序列）。

### 6.2 TensorInfo ABI

生成的 C++ 全程使用统一结构传递 tensor，不区分静态/动态 shape：

```cpp
struct TensorInfo {
  void    *data;
  int64_t  shape[8];
  int64_t  strides[8];   // row-major，运行时从 shape 计算
  int      rank;
  int      dtype;        // aclDataType 值
};
```

### 6.3 network_host.cpp 结构

```
① Helper：TensorInfo 工具函数
    makeAclTensor(TensorInfo) → aclTensor*
    allocTensorLike(TensorInfo src, TensorInfo *dst)
    rowMajorStrides(shape, rank, strides[])

② aclnn wrapper 函数（每种 aclnn.op 一个）
    run_FlashAttentionScore(q, k, v, mask, out*, stream)

③ AscendC kernel launch wrapper（每个 kernel group 一个）
    launch_kernel_group_0(inputs..., stream)

④ network() 主函数（由 coordinator MLIR 翻译）
    void network(TensorInfo inputs[], TensorInfo outputs[], aclrtStream stream)
```

### 6.4 run_FlashAttentionScore

numHeads、scale、numKvHeads 全部从运行时 shape 推导，不在 IR 中静态存储：

```cpp
static void run_FlashAttentionScore(
    TensorInfo q, TensorInfo k, TensorInfo v,
    TensorInfo mask, TensorInfo *out, aclrtStream stream) {

  // 从 BNSD 运行时 shape 推导 aclnn 参数
  int64_t numHeads   = q.shape[1];                      // dim N
  int64_t headDim    = q.shape[3];                      // dim D
  float   scale      = 1.0f / sqrtf((float)headDim);
  int64_t numKvHeads = k.shape[1];

  rowMajorStrides(q.shape, q.rank, q.strides);
  // ... k, v, mask 同理 ...

  aclTensor *qT = makeAclTensor(q), *kT = makeAclTensor(k);
  aclTensor *vT = makeAclTensor(v), *mT = makeAclTensor(mask);

  // 运行时分配输出（BNSD，shape 同 q）
  allocTensorLike(q, out);
  aclTensor *outT = makeAclTensor(*out);

  uint64_t wsSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnFlashAttentionScoreGetWorkspaceSize(
      qT, kT, vT,
      /*attn_bias=*/nullptr, mT,
      scale, /*keep_prob=*/1.0f,
      /*pre_tokens=*/65536, /*next_tokens=*/0,
      numHeads, "BNSD", numKvHeads,
      /*sparse_mode=*/0, /*inner_precise=*/0,
      outT, &wsSize, &executor);

  void *ws = nullptr;
  if (wsSize > 0) aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_NORMAL_ONLY);
  aclnnFlashAttentionScore(ws, wsSize, executor, stream);
  if (ws) aclrtFree(ws);

  aclDestroyTensor(qT); aclDestroyTensor(kT);
  aclDestroyTensor(vT); aclDestroyTensor(mT);
  aclDestroyTensor(outT);
}
```

### 6.5 coordinator 翻译规则

AclnnBackend 遍历 coordinator func 中的 op，按类型生成对应 C++：

| MLIR op | 生成 C++ |
|---------|---------|
| `func.call @__aclnn_*`（有 `aclnn.op` attr）| 查注册表调对应 `run_*` wrapper |
| `func.call @kernel_group_X` | `launch_kernel_group_X(...)` |
| `tensor.cast` | SSA 别名，无代码 |
| `tensor.empty(%d0, %d1, ...)` | `allocTensor(rank, dtype, {d0, d1, ...})` |
| `linalg.fill ins(%c)` | `fillTensor(buf, c)` |
| `tensor.dim %t, %c` | `int64_t vN = getShape(v_t)[c]` |
| `arith.constant` | C++ 字面量 |
| `tensor.collapse_shape` / `expand_shape` | 仅调整 strides，不 copy |

---

## 7. 文件地图

| 文件 | 动作 | 说明 |
|------|------|------|
| `lib/Conversion/TorchFrontend/ConvertTmTensorAttentionPass.cpp` | 修改 | `ascendc.unit` → `aclnn.kind = "flash_attention"` |
| `test/Conversion/TorchFrontend/convert-attention.mlir` | 修改 | 同步更新 CHECK 中的 attr |
| `include/Conversion/Passes.td` | 修改 | 新增 `AclnnFinalizeDecl` pass 声明 |
| `include/Conversion/LowerNonLinalgOps/LowerNonLinalgOpsPass.h` | 新建 | `createAclnnFinalizeDeclPass()` 声明 |
| `lib/Conversion/LowerNonLinalgOps/AclnnFinalizeDeclPass.cpp` | 新建 | Pass 实现 + 注册表 |
| `lib/Conversion/LowerNonLinalgOps/CMakeLists.txt` | 新建 | |
| `lib/Conversion/CMakeLists.txt` | 修改 | `add_subdirectory(LowerNonLinalgOps)` |
| `test/Conversion/aclnn-finalize-decl.mlir` | 新建 | FileCheck 测试（成功路径 + fail-fast） |
| `include/Runtime/AclnnBackend.h` | 新建 | |
| `lib/Runtime/AclnnBackend.cpp` | 新建 | emit `network_host.cpp` |
| `lib/Runtime/AclnnOps.cpp` | 新建 | `run_FlashAttentionScore` 等 wrapper |
| `lib/Runtime/CMakeLists.txt` | 修改 | 加入 `AclnnBackend.cpp`、`AclnnOps.cpp` |

---

## 8. 不在范围内（本阶段）

- aclnn workspace 复用优化
- aclnn op 与 mix kernel 之间的数据格式转换
- softmax / layernorm → aclnn pattern matching（见 `project_aclnn_pattern_matching.md`）
- Q3（AclnnBackend 触发时机）：待确认是 `afir-translate` 新模式还是独立工具