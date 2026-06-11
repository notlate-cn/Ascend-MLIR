# aclnn Fallback Path — 设计方案

**Date:** 2026-04-20
**Status:** Draft for review

---

## 1. 背景与目标

当前 Ascend-MLIR 的 main 编译路径（linalg-on-tensors → vector-plan passes → LinalgToAscendC → BishengCompiler）只处理 linalg op。对于来自 torch-mlir 的非 linalg op（如 `tm_tensor.attention`），pipeline 目前无法处理，会直接卡住或报错。

**目标**：设计一套通用的 aclnn fallback 机制，让无法走 mix 编译的 op 能够调用 CANN 预编译的 aclnn 算子库，作为高性能的融合算子调用入口。第一个落地 case 是 `tm_tensor.attention` → `aclnnPromptFlashAttention`。

---

## 2. 核心设计原则

### 2.1 aclnn op 不参与融合

非 linalg op 在现有 vector-plan Group Analysis（§3.1 of vector-plan-unified-design）里天然不进入任何 Group，是融合边界。aclnn fallback op 沿用这一语义——它是不可融合的黑盒算子。

### 2.2 不引入新方言

用 `func.call` + function-level attribute 表达 aclnn 调用，零基础设施成本，扩展新 op 只需注册白名单条目。

### 2.3 三层决策逻辑

遇到非 linalg op 时，按顺序执行：

```
非 linalg op
    ↓
1. 白名单命中？ → 是 → 转 func.call @__aclnn_xxx__（高性能 aclnn 路径）
    ↓ 否
2. 有 decomposition pattern？ → 是 → expand 成 linalg → 走 mix 路径（保底正确）
    ↓ 否
3. Fail-fast（编译报错，明确说明不支持）
```

---

## 3. IR 表示

### 3.1 函数声明

Pre-Pass 在 module 顶层插入外部函数声明，携带 aclnn 元数据 attribute：

```mlir
func.func private @__aclnn_PromptFlashAttentionV3__(
    %q:    tensor<12x8x64xf32>,
    %k:    tensor<12x8x64xf32>,
    %v:    tensor<12x8x64xf32>,
    %mask: tensor<12x8x8xf32>
) -> tensor<12x8x64xf32>
    attributes {
        aclnn.op          = "PromptFlashAttentionV3",
        aclnn.num_heads   = 12 : i64,
        aclnn.scale       = 0.125 : f64,
        aclnn.layout      = "BNSD"
    }
```

`numHeads` 在 Pass 中从 ins[0] shape[0] 读取（折叠形式下 batch=1，故 numHeads = dim[0]）；`scale = 1/sqrt(head_dim)`，head_dim = dim[2]。

### 3.2 调用点

原 `tm_tensor.attention` 被替换为对该函数的 `func.call`：

```mlir
%attn = func.call @__aclnn_PromptFlashAttentionV3__(%q, %k, %v, %mask)
    : (tensor<...>, tensor<...>, tensor<...>, tensor<...>) -> tensor<...>
```

### 3.3 在 network.mlir 中的位置

`func.call @__aclnn_xxx__` 出现在 coordinator func 里，与 `call @kernel_groupN` 并列：

```mlir
func.func @network(%input: ...) -> ... {
    %r0 = call @kernel_group0(%input)           // mix kernel
    %attn = call @__aclnn_PromptFlashAttention__(%q, %k, %v, %mask)  // aclnn
    %r1 = call @kernel_group1(%attn)            // mix kernel
    return %r1
}
```

---

## 4. Pipeline 集成

```
linalg-on-tensors（含 tm_tensor.attention 等非 linalg op）
    ↓
[新 Pass] DecompLoweringPass                 ← 必须在 Pass 1 之前
    有 decomp pattern → expand 成 linalg op（需参与后续融合分组）
    否则              → 保留原 op 不动
    ↓
Pass 1: vector-plan-group-analysis           ← 不变
    linalg op 参与融合分组
    非 linalg op（tm_tensor.attention 等）天然不进任何 Group，留在原 func
    ↓
Outline Pass: vector-plan-group-outline      ← 不变
    linalg group → kernel_groupN.mlir（mix 路径）
    非 linalg op 随 coordinator func 保留
    ↓
[新 Pass] AclnnLoweringPass                  ← 在 Outline Pass 之后
    白名单命中 → func.call @__aclnn_xxx__ + 函数声明
    否则        → 编译报错（fail-fast）
    ↓
network.mlir（host 侧）
    ↓
[扩展] network.mlir lowering pass
    call @kernel_groupN  → aclrtlaunch_xxx(blockDim, stream, ...)  // 已有
    call @__aclnn_xxx__  → aclnnPromptFlashAttention(q, k, v, ...) // 新增 pattern
```

**顺序说明**：
- `DecompLoweringPass` 必须在 Pass 1 之前——expand 出的 linalg op 需要参与 Group Analysis
- `AclnnLoweringPass` 在 Outline 之后——白名单转换与 vector-plan 无耦合关系，非 linalg op 在 vector-plan 各 pass 中被自然忽略
- vector-plan 三个 pass（group-analysis、outline、tile-fuse）无需任何修改

---

## 5. 白名单注册

白名单以 **C++ 函数指针表**形式维护。每条记录包含一个 op-specific 的转换函数，而非统一参数结构——这是因为不同 aclnn API 的参数签名差异极大（`aclTensor*` vs `aclTensorList*`、参数数量从 14 到 30+），无法用统一字段覆盖。

```cpp
using AclnnConversionFn = LogicalResult (*)(
    Operation *op, ModuleOp module, IRRewriter &rewriter);

struct WhitelistEntry {
    AclnnConversionFn convert;
};

// 扩展新 op：实现一个 convertXxx 函数，追加一行到此表
static const llvm::StringMap<WhitelistEntry> &getAclnnWhitelist() {
    static llvm::StringMap<WhitelistEntry> table = {
        {"tm_tensor.attention", {convertAttentionToPromptFlashV3}},
    };
    return table;
}
```

`tm_tensor.attention` → `aclnnPromptFlashAttentionV3` 的转换函数负责：

- 从 ins[0] shape[0] 读取 `numHeads`（`N`，如 12）
- 计算 `scaleValue = 1/sqrt(shape[2])`（如 0.125）
- 在 `func.func private` 声明上写入 `aclnn.op`、`aclnn.num_heads`、`aclnn.scale`、`aclnn.layout = "BNSD"` 四个 attributes
- 将 ins[0..2]（q/k/v）和 ins[3]（mask）映射到对应参数位置

---

## 6. host 侧 codegen

### 6.1 组织原则：统一任务接口

network 函数本体**只包含调度序列**，不含任何资源管理代码。每个 kernel——无论 mix 还是 aclnn——生成一个独立的 wrapper 函数，网络函数只调用这些 wrapper：

```cpp
// network 函数：只有线性调度，无资源管理噪音
void network(void *inputs[], void *outputs[], aclrtStream stream) {
    launch_kernel_group0(inputs[0], buf0, stream);              // mix wrapper（已有）
    launch_pfa_v3(buf0, buf1, buf2, buf3, buf4, stream);        // aclnn wrapper（新增）
    launch_kernel_group1(buf4, outputs[0], stream);             // mix wrapper（已有）
}
```

### 6.2 aclnn wrapper 的生成模板

network.mlir lowering pass 在识别到 `aclnn.op = "PromptFlashAttentionV3"` attribute 后，为该调用点生成一个 `launch_pfa_v3` wrapper 函数。`numHeads`、`scaleValue`、`numKeyValueHeads` 在编译期从 `aclnn.*` attributes 读出并硬编码：

```cpp
// 生成：每个 aclnn call site 一个 wrapper（同 op 多次调用共享同一个 wrapper）
static void launch_pfa_v3(void *q, void *k, void *v, void *mask,
                           void *out, aclrtStream stream) {
    // 1. 从裸指针构造 aclTensor（shape/dtype 编译期已知，硬编码）
    int64_t qShape[] = {12, 8, 64};
    aclTensor *qT   = aclCreateTensor(qShape, 3, ACL_FLOAT, ..., q);
    aclTensor *kT   = aclCreateTensor(qShape, 3, ACL_FLOAT, ..., k);
    aclTensor *vT   = aclCreateTensor(qShape, 3, ACL_FLOAT, ..., v);
    int64_t mShape[] = {12, 8, 8};
    aclTensor *maskT = aclCreateTensor(mShape, 3, ACL_FLOAT, ..., mask);
    aclTensor *outT  = aclCreateTensor(qShape, 3, ACL_FLOAT, ..., out);

    // 2. 查询 workspace 大小
    uint64_t wsSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnPromptFlashAttentionV3GetWorkspaceSize(
        qT, kT, vT,
        /*pseShift=*/nullptr, maskT,
        /*actualSeqLengths=*/nullptr, /*actualSeqLengthsKv=*/nullptr,
        /*deqScale1..quantOffset2=*/nullptr, nullptr, nullptr, nullptr, nullptr,
        /*numHeads=*/12, /*scaleValue=*/0.125,
        /*preTokens=*/65535, /*nextTokens=*/0,
        /*inputLayout=*/"NSD",
        /*numKeyValueHeads=*/12, /*sparseMode=*/0, /*innerPrecise=*/0,
        outT, &wsSize, &executor);

    // 3. 分配 workspace、执行、释放
    void *ws = nullptr;
    if (wsSize > 0) aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclnnPromptFlashAttentionV3(ws, wsSize, executor, stream);
    if (ws) aclrtFree(ws);

    // 4. 销毁 aclTensor
    aclDestroyTensor(qT); aclDestroyTensor(kT); aclDestroyTensor(vT);
    aclDestroyTensor(maskT); aclDestroyTensor(outT);
}
```

`inputLayout` 用 **`"NSD"`**：tensor 形状 `[N, S, D]` 与 NSD layout 直接对应（N=batch×numHeads，batch=1 已折叠），无需 reshape。BNSD 要求 4D tensor，不适用。

---

## 7. 不在范围内

- aclnn workspace 复用优化（一期不做）
- 动态 shape 下 aclnn 参数推导（一期只支持静态 shape）
- aclnn op 与 mix kernel 之间的数据格式转换（假设格式兼容）

---

## 8. 接口边界

| 组件 | 修改内容 |
|------|---------|
| `AclnnLoweringPass`（新增） | 扫描非 linalg op，按三层逻辑处理 |
| `AclnnRegistry`（新增） | 白名单注册表，静态 C++ 数据结构 |
| `vector-plan-group-analysis` | 不变，`func.call` 天然不进 Group |
| `vector-plan-group-outline` | 不变，`func.call` 留在 coordinator func |
| `network.mlir lowering pass`（扩展） | 新增 `aclnn.op` pattern → aclnn C++ codegen |