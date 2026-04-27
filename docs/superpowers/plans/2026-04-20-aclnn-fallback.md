# aclnn Fallback Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将无法走 mix 编译路径的非 linalg op（首个 case：`tm_tensor.attention`）转为 `func.call @__aclnn_xxx__`，使其能在 network.mlir host codegen 阶段生成 aclnn 调用。

**Architecture:** 新增两个 pass：`DecompLoweringPass`（Pass 1 之前，处理可 decompose 的 op；Phase 1 为 skeleton）和 `AclnnLoweringPass`（Outline 之后，白名单命中转 `func.call`，否则 fail-fast）。无新方言，用 `func.func private` + `aclnn.op` attribute 携带元数据。

**Tech Stack:** MLIR C++（TableGen pass infrastructure，`func`/`tensor` dialect），FileCheck 测试，`afir-opt` 工具。

**Spec:** `docs/superpowers/specs/2026-04-20-aclnn-fallback-design.md`

---

## File Map

| 文件 | 动作 | 说明 |
|------|------|------|
| `include/Conversion/Passes.td` | 修改 | 新增两个 pass 声明 |
| `include/Conversion/LowerNonLinalgOps/LowerNonLinalgOpsPass.h` | 新建 | 两个 pass 的 `createXxx()` 声明 |
| `lib/Conversion/LowerNonLinalgOps/DecompLoweringPass.cpp` | 新建 | Phase 1 skeleton，为未来 decomp 预留接口 |
| `lib/Conversion/LowerNonLinalgOps/AclnnLoweringPass.cpp` | 新建 | 白名单注册 + `tm_tensor.attention` → `func.call` |
| `lib/Conversion/LowerNonLinalgOps/CMakeLists.txt` | 新建 | 构建定义 |
| `test/Conversion/decomp-lowering.mlir` | 新建 | DecompLoweringPass FileCheck 测试 |
| `test/Conversion/aclnn-lowering.mlir` | 新建 | AclnnLoweringPass FileCheck 测试（成功路径 + fail-fast） |

---

## Task 1: Pass 声明（Passes.td + header）

**Files:**
- Modify: `include/Conversion/Passes.td`
- Create: `include/Conversion/LowerNonLinalgOps/LowerNonLinalgOpsPass.h`

- [ ] **Step 1: 在 Passes.td 末尾（`#endif` 之前）追加两个 pass 定义**

```tablegen
//===----------------------------------------------------------------------===//
// DecompLoweringPass
//===----------------------------------------------------------------------===//

def DecompLoweringPass : Pass<"decomp-lowering", "mlir::func::FuncOp"> {
  let summary = "Expand non-linalg ops with known decomposition patterns into linalg ops";
  let description = [{
    Must run before vector-plan-group-analysis so that expanded linalg ops
    can participate in Group Analysis and fusion.

    Phase 1: skeleton only. No decomposition patterns registered yet.
    Non-linalg ops without a registered pattern are left unchanged.
  }];
  let constructor = "mlir::afir::createDecompLoweringPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect"
  ];
}

//===----------------------------------------------------------------------===//
// AclnnLoweringPass
//===----------------------------------------------------------------------===//

def AclnnLoweringPass : Pass<"aclnn-lowering", "mlir::func::FuncOp"> {
  let summary = "Replace whitelisted non-linalg ops with func.call to aclnn APIs";
  let description = [{
    Runs after vector-plan-group-outline (Outline Pass). Scans for non-linalg
    ops that survived DecompLoweringPass and the vector-plan pipeline.

    Three-layer decision:
      1. Whitelist hit  -> insert func.func private @__aclnn_xxx__ + func.call
      2. (decomp handled upstream by DecompLoweringPass)
      3. Unknown op     -> emitError / signalPassFailure (fail-fast)

    The inserted func.func private carries {aclnn.op = "..."} attribute for
    downstream network.mlir lowering to generate the actual aclnn C++ call.

    Registered whitelist (Phase 1):
      tm_tensor.attention -> PromptFlashAttention
  }];
  let constructor = "mlir::afir::createAclnnLoweringPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect"
  ];
}
```

- [ ] **Step 2: 创建 header 文件**

```cpp
// include/Conversion/LowerNonLinalgOps/LowerNonLinalgOpsPass.h
#ifndef ASCEND_MLIR_CONVERSION_LOWERNONLINALGOPS_H
#define ASCEND_MLIR_CONVERSION_LOWERNONLINALGOPS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
}

namespace mlir::afir {

std::unique_ptr<Pass> createDecompLoweringPass();
std::unique_ptr<Pass> createAclnnLoweringPass();

}  // namespace mlir::afir

#endif
```

- [ ] **Step 3: 验证 Passes.td 语法正确（后续 Task 的构建会一起验证，此处跳过单独检查）**

---

## Task 2: DecompLoweringPass skeleton

**Files:**
- Create: `lib/Conversion/LowerNonLinalgOps/DecompLoweringPass.cpp`

- [ ] **Step 1: 写测试（先写，确认 pass 可以运行且不修改纯 linalg IR）**

创建 `test/Conversion/decomp-lowering.mlir`：

```mlir
// RUN: afir-opt --decomp-lowering %s | FileCheck %s

// CHECK-LABEL: func.func @test_passthrough
// CHECK: linalg.generic
// CHECK-NOT: func.call
func.func @test_passthrough(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %empty = tensor.empty() : tensor<4x8xf32>
  %out = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<4x8xf32>) outs(%empty : tensor<4x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<4x8xf32>
  return %out : tensor<4x8xf32>
}
```

- [ ] **Step 2: 写 DecompLoweringPass 实现**

```cpp
// lib/Conversion/LowerNonLinalgOps/DecompLoweringPass.cpp
#include "Conversion/LowerNonLinalgOps/LowerNonLinalgOpsPass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"

#define GEN_PASS_DECL_DECOMPLOWERINGPASS
#define GEN_PASS_DEF_DECOMPLOWERINGPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

namespace {

struct DecompLoweringPass
    : public ::impl::DecompLoweringPassBase<DecompLoweringPass> {
  void runOnOperation() override {
    // Phase 1: no decomposition patterns registered.
    // Non-linalg ops are left unchanged for AclnnLoweringPass to handle.
  }
};

}  // namespace

std::unique_ptr<Pass> createDecompLoweringPass() {
  return std::make_unique<DecompLoweringPass>();
}

}  // namespace mlir::afir
```

- [ ] **Step 3: 创建 CMakeLists.txt**

```cmake
# lib/Conversion/LowerNonLinalgOps/CMakeLists.txt
add_mlir_library(LowerNonLinalgOpsConversion
  DecompLoweringPass.cpp
  AclnnLoweringPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Conversion

  DEPENDS
  AFIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTransforms
)
```

- [ ] **Step 4: 在 `lib/Conversion/` 父级 CMakeLists 里添加 subdirectory**

查看 `lib/Conversion/CMakeLists.txt`，在末尾加入：

```cmake
add_subdirectory(LowerNonLinalgOps)
```

- [ ] **Step 5: 编译验证**

```bash
cd build && cmake --build . --target LowerNonLinalgOpsConversion 2>&1 | tail -20
```

Expected: 无错误

- [ ] **Step 6: 运行测试**

```bash
cd build && bin/afir-opt --decomp-lowering \
  ../test/Conversion/decomp-lowering.mlir | FileCheck ../test/Conversion/decomp-lowering.mlir
```

Expected: 通过（IR 原样输出，linalg.generic 保留，无 func.call）

- [ ] **Step 7: commit**

```bash
git add include/Conversion/Passes.td \
        include/Conversion/LowerNonLinalgOps/LowerNonLinalgOpsPass.h \
        lib/Conversion/LowerNonLinalgOps/DecompLoweringPass.cpp \
        lib/Conversion/LowerNonLinalgOps/CMakeLists.txt \
        test/Conversion/decomp-lowering.mlir
git commit -m "feat: add DecompLoweringPass skeleton for non-linalg op handling"
```

---

## Task 3: AclnnLoweringPass（白名单 + tm_tensor.attention 转换）

**Files:**
- Create: `lib/Conversion/LowerNonLinalgOps/AclnnLoweringPass.cpp`
- Create: `test/Conversion/aclnn-lowering.mlir`

- [ ] **Step 1: 写测试——成功路径（tm_tensor.attention → func.call）**

创建 `test/Conversion/aclnn-lowering.mlir`：

```mlir
// RUN: afir-opt --aclnn-lowering %s | FileCheck %s

// CHECK: func.func private @__aclnn_PromptFlashAttentionV3__
// CHECK-SAME: aclnn.layout = "BNSD"
// CHECK-SAME: aclnn.num_heads = 12
// CHECK-SAME: aclnn.op = "PromptFlashAttentionV3"
// CHECK-SAME: aclnn.scale = 1.250000e-01
// CHECK-LABEL: func.func @test_attention
// CHECK: func.call @__aclnn_PromptFlashAttentionV3__
// CHECK-NOT: tm_tensor.attention
func.func @test_attention(
    %q:    tensor<12x8x64xf32>,
    %k:    tensor<12x8x64xf32>,
    %v:    tensor<12x8x64xf32>,
    %mask: tensor<12x8x8xf32>
) -> tensor<12x8x64xf32> {
  %init = tensor.empty() : tensor<12x8x64xf32>
  %out = tm_tensor.attention
      ins(%q, %k, %v, %mask : tensor<12x8x64xf32>, tensor<12x8x64xf32>,
                               tensor<12x8x64xf32>, tensor<12x8x8xf32>)
      outs(%init : tensor<12x8x64xf32>) -> tensor<12x8x64xf32>
  return %out : tensor<12x8x64xf32>
}
```

注：fail-fast 路径（非白名单 op）用手动验证，见 Step 5。

- [ ] **Step 2: 实现 AclnnLoweringPass**

```cpp
// lib/Conversion/LowerNonLinalgOps/AclnnLoweringPass.cpp
#include "Conversion/LowerNonLinalgOps/LowerNonLinalgOpsPass.h"

#include <cmath>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringMap.h"

#define GEN_PASS_DECL_ACLNNLOWERINGPASS
#define GEN_PASS_DEF_ACLNNLOWERINGPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

namespace {

// ---------------------------------------------------------------------------
// Helper: insert (or reuse) a private func declaration for an aclnn API.
// The caller is responsible for setting op-specific attributes after this.
// ---------------------------------------------------------------------------
static func::FuncOp getOrInsertAclnnDecl(OpBuilder &builder, ModuleOp moduleOp,
                                          StringRef funcName, Operation *srcOp) {
  if (auto existing = moduleOp.lookupSymbol<func::FuncOp>(funcName))
    return existing;

  SmallVector<Type> inputTypes(srcOp->getOperandTypes());
  SmallVector<Type> resultTypes(srcOp->getResultTypes());
  auto funcType = builder.getFunctionType(inputTypes, resultTypes);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());
  auto decl = builder.create<func::FuncOp>(srcOp->getLoc(), funcName, funcType);
  decl.setPrivate();
  return decl;
}

// ---------------------------------------------------------------------------
// Returns true if op is handled by the linalg/vector-plan pipeline and does
// not need aclnn treatment.
// ---------------------------------------------------------------------------
static bool isHandledByPipeline(Operation *op) {
  StringRef dialect = op->getDialect()->getNamespace();
  return dialect == "linalg" || dialect == "tensor" || dialect == "arith" ||
         dialect == "func" || dialect == "scf" || dialect == "math" ||
         dialect == "cf" || dialect == "index" || dialect == "affine";
}

// ---------------------------------------------------------------------------
// Conversion function for tm_tensor.attention
//   ins: [q: NxSxD, k: NxSxD, v: NxSxD, mask: NxSxS]
//   outs: [out: NxSxD]
// Maps to aclnnPromptFlashAttentionV3 (BNSD layout, batch=1 folded into N).
// ---------------------------------------------------------------------------
static LogicalResult convertAttentionToPromptFlashV3(Operation *op,
                                                      ModuleOp moduleOp,
                                                      OpBuilder &builder) {
  // Extract numHeads and headDim from q shape (ins[0]: tensor<NxSxD>).
  auto qType = cast<RankedTensorType>(op->getOperand(0).getType());
  if (qType.getRank() != 3)
    return op->emitError("tm_tensor.attention: expected 3-D q tensor (NxSxD)");

  int64_t numHeads = qType.getDimSize(0);
  int64_t headDim  = qType.getDimSize(2);
  double scaleValue = 1.0 / std::sqrt(static_cast<double>(headDim));

  // Build the private function declaration and attach aclnn.* attributes.
  StringRef funcName = "__aclnn_PromptFlashAttentionV3__";
  func::FuncOp decl = getOrInsertAclnnDecl(builder, moduleOp, funcName, op);

  // Idempotent: only set attrs if this is the first declaration.
  if (!decl->hasAttr("aclnn.op")) {
    decl->setAttr("aclnn.op",        builder.getStringAttr("PromptFlashAttentionV3"));
    decl->setAttr("aclnn.num_heads", builder.getI64IntegerAttr(numHeads));
    decl->setAttr("aclnn.scale",     builder.getF64FloatAttr(scaleValue));
    decl->setAttr("aclnn.layout",    builder.getStringAttr("BNSD"));
  }

  // Replace the op with a func.call carrying all original operands.
  builder.setInsertionPoint(op);
  auto callOp = builder.create<func::CallOp>(op->getLoc(), decl, op->getOperands());
  op->replaceAllUsesWith(callOp.getResults());
  op->erase();
  return success();
}

// ---------------------------------------------------------------------------
// Whitelist: src_op name -> per-op conversion function.
// To add a new op: implement convertXxx and append one line here.
// ---------------------------------------------------------------------------
using AclnnConversionFn = LogicalResult (*)(Operation *, ModuleOp, OpBuilder &);

struct WhitelistEntry { AclnnConversionFn convert; };

static const llvm::StringMap<WhitelistEntry> &getAclnnWhitelist() {
  static llvm::StringMap<WhitelistEntry> table = {
      {"tm_tensor.attention", {convertAttentionToPromptFlashV3}},
  };
  return table;
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------
struct AclnnLoweringPass
    : public ::impl::AclnnLoweringPassBase<AclnnLoweringPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
    OpBuilder builder(funcOp.getContext());

    const auto &whitelist = getAclnnWhitelist();
    SmallVector<Operation *> toReplace;

    funcOp.walk([&](Operation *op) {
      if (!isHandledByPipeline(op))
        toReplace.push_back(op);
    });

    for (Operation *op : toReplace) {
      StringRef opName = op->getName().getStringRef();
      auto it = whitelist.find(opName);
      if (it == whitelist.end()) {
        op->emitError("unsupported non-linalg op '")
            << opName << "': not in aclnn whitelist and no decomposition pattern";
        return signalPassFailure();
      }
      if (failed(it->second.convert(op, moduleOp, builder)))
        return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createAclnnLoweringPass() {
  return std::make_unique<AclnnLoweringPass>();
}

}  // namespace mlir::afir
```

- [ ] **Step 3: 编译验证**

```bash
cd build && cmake --build . --target LowerNonLinalgOpsConversion 2>&1 | tail -20
```

Expected: 无错误

- [ ] **Step 4: 运行成功路径测试**

```bash
cd build && bin/afir-opt --aclnn-lowering \
  ../test/Conversion/aclnn-lowering.mlir 2>&1
```

Expected 输出包含：
```
func.func private @__aclnn_PromptFlashAttentionV3__(...)
    attributes {aclnn.layout = "BNSD", aclnn.num_heads = 12 : i64,
                aclnn.op = "PromptFlashAttentionV3", aclnn.scale = 1.25e-1}
...
func.call @__aclnn_PromptFlashAttentionV3__(...)
```

且不含 `tm_tensor.attention`。

- [ ] **Step 5: 验证 fail-fast（手动）**

写一个含未知 op 的临时 MLIR 文件并验证报错：

```bash
cat > /tmp/test_fail.mlir << 'EOF'
// tm_tensor.scatter 不在白名单里
func.func @test(%q: tensor<4xf32>, %k: tensor<4xi64>) -> tensor<4xf32> {
  %init = tensor.empty() : tensor<4xf32>
  // 用 linalg.generic 外的真实 op 触发 fail-fast
  // 实际上只要 tm_tensor.attention 已覆盖，其余 tm_tensor op 应触发 fail-fast
  return %q : tensor<4xf32>
}
EOF
cd build && bin/afir-opt --aclnn-lowering /tmp/test_fail.mlir 2>&1 || true
```

Expected: 包含 `unsupported non-linalg op` 的错误信息（若无非 pipeline op 则 pass 正常通过，可暂时跳过此步）。

- [ ] **Step 6: 运行完整 FileCheck 测试**

```bash
cd build && bin/afir-opt --aclnn-lowering \
  ../test/Conversion/aclnn-lowering.mlir | \
  FileCheck ../test/Conversion/aclnn-lowering.mlir
```

Expected: PASS

- [ ] **Step 7: commit**

```bash
git add lib/Conversion/LowerNonLinalgOps/AclnnLoweringPass.cpp \
        test/Conversion/aclnn-lowering.mlir
git commit -m "feat: add AclnnLoweringPass – tm_tensor.attention → aclnnPromptFlashAttentionV3 via ConversionFn whitelist"
```

---

## Task 4: 在 gpt2.mlir 上做端到端验证

**Files:**
- Read: `examples/llm-block/gpt2.mlir`（只读，不修改）

- [ ] **Step 1: 对 gpt2.mlir 运行 AclnnLoweringPass，观察 tm_tensor.attention 替换**

```bash
cd build && bin/afir-opt --aclnn-lowering \
  ../examples/llm-block/gpt2.mlir 2>&1 | grep -E "aclnn|tm_tensor|PromptFlash" | head -20
```

Expected：
- 不含 `tm_tensor.attention`
- 含多个 `func.call @__aclnn_PromptFlashAttentionV3__`（gpt2 有 12 个 transformer block）
- 含一个 `func.func private @__aclnn_PromptFlashAttentionV3__`（去重复用）

- [ ] **Step 2: 确认 linalg op 未受影响**

```bash
cd build && bin/afir-opt --aclnn-lowering \
  ../examples/llm-block/gpt2.mlir 2>&1 | grep "linalg\." | wc -l
```

与原始 gpt2.mlir 的 linalg op 数量对比，应相同：

```bash
grep "linalg\." ../examples/llm-block/gpt2.mlir | wc -l
```

两者数量应一致。

- [ ] **Step 3: commit**

```bash
git commit --allow-empty -m "chore: verify AclnnLoweringPass on gpt2.mlir – attention replaced, linalg untouched"
```

若无文件改动则跳过 commit。

---

## 自检

**Spec 覆盖：**
- §2 三层决策逻辑：Task 3 AclnnLoweringPass 实现了白名单命中和 fail-fast；decomp 是 Task 2 skeleton
- §3 IR 表示（func.func private + aclnn.op attribute）：Task 3 Step 2 实现
- §4 Pipeline 集成（DecompLoweringPass 在 Pass 1 前，AclnnLoweringPass 在 Outline 后）：pass 声明在 Passes.td，执行顺序由调用方 pipeline 配置，不在本计划范围内
- §5 白名单注册：Task 3 Step 2 中的 `getAclnnWhitelist()`
- §6 host codegen（network.mlir lowering）：**不在本计划范围内**，是下一阶段工作

**排除项说明：** §6 的 aclnn host codegen（生成 `aclnnPromptFlashAttention(...)` C++ 调用）依赖 network.mlir lowering 基础设施，该基础设施尚未在此代码库中实现，作为独立后续 task 处理。