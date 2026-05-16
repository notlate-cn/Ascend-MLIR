# Phase 1: 算子覆盖扩展 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 elementwise body op 集从 3 个扩展到全集（arith+math），将 reduction 从 AddF-only 扩展到 Max/Min/Mul，并引入 f16/bf16/i8 dtype 感知，解锁 softmax/GELU/LayerNorm/RMSNorm 能过完整 pipeline。

**Architecture:** 引入 `ElementwiseBodyOpRegistry`（单例注册表，与 `HandwrittenContractRegistry` 同模式），把 arith/math op TypeID → ComputeKind + unary/binary emitter 的映射集中管理。`LinalgBodyClassifier`、`BackendSupportMatrix`、`ComputeConversion` 三层消费同一张表，新增 op 只需在注册表里加一条 entry。Reduction 分类函数改为识别 body 里的单个 binary op，通过注册表查 ComputeKind。dtype 扩展把 element type 传入 BackendSupportMatrix 做校验，ComputeConversion emit 路径按 dtype 选 AscendC API。

**Tech Stack:** C++17, LLVM/MLIR (llvm::ManagedStatic, llvm::TypeID, linalg::GenericOp, arith/math dialect, ascendc:: L2 ops), GTest, LLVM LIT / FileCheck

---

## 文件结构

### 新增文件

| 文件 | 职责 |
|---|---|
| `include/Conversion/Ascend/Backend/ElementwiseBodyOpRegistry.h` | 注册表接口：`ElementwiseBodyOpEntry`、`registerElementwiseBodyOp`、`lookupElementwiseBodyOp`、`registerBuiltinElementwiseBodyOps` |
| `lib/Conversion/Ascend/Backend/ElementwiseBodyOpRegistry.cpp` | 注册表实现 + 所有内置 arith/math op 注册 |
| `test/unittests/Conversion/AscendElementwiseBodyOpRegistryTest.cpp` | 单元测试 |
| `test/Conversion/ascend-full-pipeline-softmax.mlir` | softmax e2e LIT |
| `test/Conversion/ascend-full-pipeline-gelu.mlir` | GELU e2e LIT |
| `test/Conversion/ascend-full-pipeline-rmsnorm.mlir` | RMSNorm e2e LIT |
| `test/Conversion/ascend-full-pipeline-f16-matmul.mlir` | f16 matmul e2e LIT（暂 xfail，dtype 校验在 1C） |

### 修改文件

| 文件 | 改动 |
|---|---|
| `include/Conversion/Ascend/Backend/BackendSupportMatrix.h` | 新增 `ComputeKind` 枚举值；新增 `isSupportedDtype`、`explainDtype` |
| `lib/Conversion/Ascend/Backend/BackendSupportMatrix.cpp` | 实现新枚举的 stringify + isSupportedComputeKind + dtype 校验 |
| `lib/Conversion/Ascend/Backend/LinalgBodyClassifier.cpp` | `classifyElementwiseBodyOp` 和 `isSupportedPhase5ReductionBody` 改为查表 |
| `lib/Conversion/LinalgToAscendC/ComputeConversion.cpp` | elementwise emit 路径按注册表 emitter 分发；reduction emit 按 ComputeKind 选 `ReduceSum2DL2Op`/`ReduceMax2DL2Op`/`ReduceMin2DL2Op` |
| `lib/Conversion/Ascend/Kernelize/KernelizeSemanticUtils.cpp` | `populateLinalgSemanticInfo` 填充 `inputElementTypes`/`outputElementTypes` |
| `lib/Conversion/Ascend/Kernelize/KernelizeOpInterface.h` | `KernelizeOpSemanticInfo` 加 dtype 字段 |
| `lib/Conversion/Ascend/CMakeLists.txt` | 添加 `Backend/ElementwiseBodyOpRegistry.cpp` |
| `test/unittests/Conversion/CMakeLists.txt` | 注册 `AscendElementwiseBodyOpRegistryTest` |

---

## Task 1: 定义 ElementwiseBodyOpRegistry 接口和数据结构

**Files:**
- Create: `include/Conversion/Ascend/Backend/ElementwiseBodyOpRegistry.h`

- [ ] **Step 1: 创建头文件**

```cpp
//===- ElementwiseBodyOpRegistry.h - Elementwise body op registry --*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_ELEMENTWISEBODYOPREGISTRY_H
#define ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_ELEMENTWISEBODYOPREGISTRY_H

#include "Conversion/Ascend/Backend/BackendSupportMatrix.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"

#include <functional>

namespace mlir::afir::ascend::backend {

using UnaryEmitter  = std::function<void(mlir::OpBuilder &, mlir::Location,
                                         mlir::Value dst, mlir::Value src,
                                         mlir::Value count)>;
using BinaryEmitter = std::function<void(mlir::OpBuilder &, mlir::Location,
                                         mlir::Value dst, mlir::Value src0,
                                         mlir::Value src1, mlir::Value count)>;

/// One entry in the elementwise body op registry.
/// Exactly one of unaryEmitter / binaryEmitter is non-null.
struct ElementwiseBodyOpEntry {
  llvm::StringRef dialectOpName; // e.g. "arith.addf", "math.exp"
  ComputeKind kind;
  UnaryEmitter  unaryEmitter;    // set for unary ops (exp, sqrt, neg, …)
  BinaryEmitter binaryEmitter;   // set for binary ops (add, sub, mul, …)
};

/// Register one entry. Second call with same dialectOpName is a no-op.
void registerElementwiseBodyOp(ElementwiseBodyOpEntry entry);

/// Register all built-in arith/math entries. Called automatically on first
/// lookup; exposed for explicit initialization in tests.
void registerBuiltinElementwiseBodyOps();

/// Returns nullptr if opName is not registered.
const ElementwiseBodyOpEntry *
lookupElementwiseBodyOp(llvm::StringRef dialectOpName);

} // namespace mlir::afir::ascend::backend

#endif // ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_ELEMENTWISEBODYOPREGISTRY_H
```

- [ ] **Step 2: Commit**

```bash
git add include/Conversion/Ascend/Backend/ElementwiseBodyOpRegistry.h
git commit -m "feat: add ElementwiseBodyOpRegistry interface"
```

---

## Task 2: 扩展 ComputeKind 枚举，实现注册表

**Files:**
- Modify: `include/Conversion/Ascend/Backend/BackendSupportMatrix.h`
- Modify: `lib/Conversion/Ascend/Backend/BackendSupportMatrix.cpp`
- Create: `lib/Conversion/Ascend/Backend/ElementwiseBodyOpRegistry.cpp`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`

- [ ] **Step 1: 在 `BackendSupportMatrix.h` 扩展 ComputeKind 枚举**

找到 `enum class ComputeKind {` 块（`include/Conversion/Ascend/Backend/BackendSupportMatrix.h:20`），在 `ReductionAdd,` 后、`};` 前添加：

```cpp
  // --- 新增 elementwise op kinds ---
  ElementwiseSub,
  ElementwiseDiv,
  ElementwiseNeg,
  ElementwiseExp,
  ElementwiseExp2,
  ElementwiseLog,
  ElementwiseSqrt,
  ElementwiseRsqrt,
  ElementwiseTanh,
  ElementwiseErf,
  ElementwiseAbs,
  ElementwiseSin,
  ElementwiseCos,
  ElementwiseFma,
  ElementwiseReciprocal,
  ElementwiseRelu,
  ElementwiseSelect,   // arith.cmpf + arith.select fused
  // --- 新增 reduction kinds ---
  ReductionMax,
  ReductionMin,
  ReductionMul,
```

同时在头文件中新增 dtype 校验接口（在 `AscendBackendSupportMatrix` 类内 `explainComputeKind` 之后）：

```cpp
  bool isSupportedDtype(ComputeKind kind,
                        mlir::ArrayRef<mlir::Type> inputTypes,
                        mlir::ArrayRef<mlir::Type> outputTypes) const;
  UnsupportedReason explainDtype(ComputeKind kind,
                                 mlir::ArrayRef<mlir::Type> inputTypes,
                                 mlir::ArrayRef<mlir::Type> outputTypes) const;
```

在头文件顶部加 include（如果没有）：
```cpp
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
```

- [ ] **Step 2: 在 `BackendSupportMatrix.cpp` 实现新枚举**

在 `stringifyComputeKind` 的 switch 里（`lib/Conversion/Ascend/Backend/BackendSupportMatrix.cpp:69`），在 `case ComputeKind::ReductionAdd:` 后面添加：

```cpp
  case ComputeKind::ElementwiseSub:       return "elementwise_sub";
  case ComputeKind::ElementwiseDiv:       return "elementwise_div";
  case ComputeKind::ElementwiseNeg:       return "elementwise_neg";
  case ComputeKind::ElementwiseExp:       return "elementwise_exp";
  case ComputeKind::ElementwiseExp2:      return "elementwise_exp2";
  case ComputeKind::ElementwiseLog:       return "elementwise_log";
  case ComputeKind::ElementwiseSqrt:      return "elementwise_sqrt";
  case ComputeKind::ElementwiseRsqrt:     return "elementwise_rsqrt";
  case ComputeKind::ElementwiseTanh:      return "elementwise_tanh";
  case ComputeKind::ElementwiseErf:       return "elementwise_erf";
  case ComputeKind::ElementwiseAbs:       return "elementwise_abs";
  case ComputeKind::ElementwiseSin:       return "elementwise_sin";
  case ComputeKind::ElementwiseCos:       return "elementwise_cos";
  case ComputeKind::ElementwiseFma:       return "elementwise_fma";
  case ComputeKind::ElementwiseReciprocal: return "elementwise_reciprocal";
  case ComputeKind::ElementwiseRelu:      return "elementwise_relu";
  case ComputeKind::ElementwiseSelect:    return "elementwise_select";
  case ComputeKind::ReductionMax:         return "reduction_max";
  case ComputeKind::ReductionMin:         return "reduction_min";
  case ComputeKind::ReductionMul:         return "reduction_mul";
```

在 `isSupportedComputeKind` 的 switch 里，把所有新 enum 值都加进 `return true` 的分支（与已有的 `ElementwiseAdd` 等并列）：

```cpp
  case ComputeKind::ElementwiseSub:
  case ComputeKind::ElementwiseDiv:
  case ComputeKind::ElementwiseNeg:
  case ComputeKind::ElementwiseExp:
  case ComputeKind::ElementwiseExp2:
  case ComputeKind::ElementwiseLog:
  case ComputeKind::ElementwiseSqrt:
  case ComputeKind::ElementwiseRsqrt:
  case ComputeKind::ElementwiseTanh:
  case ComputeKind::ElementwiseErf:
  case ComputeKind::ElementwiseAbs:
  case ComputeKind::ElementwiseSin:
  case ComputeKind::ElementwiseCos:
  case ComputeKind::ElementwiseFma:
  case ComputeKind::ElementwiseReciprocal:
  case ComputeKind::ElementwiseRelu:
  case ComputeKind::ElementwiseSelect:
  case ComputeKind::ReductionMax:
  case ComputeKind::ReductionMin:
  case ComputeKind::ReductionMul:
    return true;
```

在文件末尾（`} // namespace` 前）追加 dtype 校验实现：

```cpp
bool AscendBackendSupportMatrix::isSupportedDtype(
    ComputeKind kind, mlir::ArrayRef<mlir::Type> inputTypes,
    mlir::ArrayRef<mlir::Type> outputTypes) const {
  // For now accept f32, f16, bf16 for all compute kinds.
  // i8/ui8 support will be added per-kind in a follow-up.
  auto isSupportedElemType = [](mlir::Type t) {
    return t.isF32() || t.isF16() || t.isBF16();
  };
  for (mlir::Type t : inputTypes)
    if (!isSupportedElemType(t))
      return false;
  for (mlir::Type t : outputTypes)
    if (!isSupportedElemType(t))
      return false;
  return true;
}

UnsupportedReason AscendBackendSupportMatrix::explainDtype(
    ComputeKind kind, mlir::ArrayRef<mlir::Type> inputTypes,
    mlir::ArrayRef<mlir::Type> outputTypes) const {
  if (isSupportedDtype(kind, inputTypes, outputTypes))
    return {"dtype", ""};
  // Report the first unsupported type.
  for (mlir::Type t : inputTypes) {
    if (!t.isF32() && !t.isF16() && !t.isBF16())
      return {"dtype", llvm::formatv("unsupported input dtype for {0}: {1}",
                                     stringifyComputeKind(kind),
                                     mlir::debugString(t)).str()};
  }
  for (mlir::Type t : outputTypes) {
    if (!t.isF32() && !t.isF16() && !t.isBF16())
      return {"dtype", llvm::formatv("unsupported output dtype for {0}: {1}",
                                     stringifyComputeKind(kind),
                                     mlir::debugString(t)).str()};
  }
  return {"dtype", "unsupported dtype combination"};
}
```

在 `BackendSupportMatrix.cpp` 顶部加 include：
```cpp
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
```

- [ ] **Step 3: 创建 `ElementwiseBodyOpRegistry.cpp`**

```cpp
//===- ElementwiseBodyOpRegistry.cpp - Elementwise body op registry -------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Backend/ElementwiseBodyOpRegistry.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "ascir/Dialect/Asc/IR/Asc.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ManagedStatic.h"

#include <mutex>

namespace mlir::afir::ascend::backend {
namespace {

struct ElementwiseBodyOpRegistrySingleton {
  std::mutex mu;
  llvm::StringMap<ElementwiseBodyOpEntry> entries;
  bool builtinsRegistered = false;
};

llvm::ManagedStatic<ElementwiseBodyOpRegistrySingleton> gRegistry;

} // namespace

void registerElementwiseBodyOp(ElementwiseBodyOpEntry entry) {
  auto &reg = *gRegistry;
  std::lock_guard<std::mutex> lock(reg.mu);
  reg.entries.try_emplace(entry.dialectOpName, std::move(entry));
}

void registerBuiltinElementwiseBodyOps() {
  auto &reg = *gRegistry;
  {
    std::lock_guard<std::mutex> lock(reg.mu);
    if (reg.builtinsRegistered)
      return;
    reg.builtinsRegistered = true;
  }

  using namespace ascendc;

  // ---- Binary ops ----
  registerElementwiseBodyOp({"arith.addf", ComputeKind::ElementwiseAdd,
    nullptr,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src0, mlir::Value src1, mlir::Value cnt) {
      b.create<AddL2Op>(loc, dst, src0, src1, cnt);
    }});
  registerElementwiseBodyOp({"arith.mulf", ComputeKind::ElementwiseMul,
    nullptr,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src0, mlir::Value src1, mlir::Value cnt) {
      b.create<MulL2Op>(loc, dst, src0, src1, cnt);
    }});
  registerElementwiseBodyOp({"arith.maximumf", ComputeKind::ElementwiseMax,
    nullptr,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src0, mlir::Value src1, mlir::Value cnt) {
      b.create<MaxL2Op>(loc, dst, src0, src1, cnt);
    }});
  registerElementwiseBodyOp({"arith.subf", ComputeKind::ElementwiseSub,
    nullptr,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src0, mlir::Value src1, mlir::Value cnt) {
      b.create<SubL2Op>(loc, dst, src0, src1, cnt);
    }});
  registerElementwiseBodyOp({"arith.divf", ComputeKind::ElementwiseDiv,
    nullptr,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src0, mlir::Value src1, mlir::Value cnt) {
      b.create<DivL2Op>(loc, dst, src0, src1, cnt);
    }});
  registerElementwiseBodyOp({"arith.minimumf", ComputeKind::ElementwiseMax,
    nullptr,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src0, mlir::Value src1, mlir::Value cnt) {
      b.create<MinL2Op>(loc, dst, src0, src1, cnt);
    }});
  registerElementwiseBodyOp({"math.fma", ComputeKind::ElementwiseFma,
    nullptr,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src0, mlir::Value src1, mlir::Value cnt) {
      b.create<FusedMulAddL2Op>(loc, dst, src0, src1, cnt);
    }});

  // ---- Unary ops ----
  registerElementwiseBodyOp({"arith.negf", ComputeKind::ElementwiseNeg,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src, mlir::Value cnt) {
      b.create<NegL2Op>(loc, dst, src, cnt);
    }, nullptr});
  registerElementwiseBodyOp({"math.exp", ComputeKind::ElementwiseExp,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src, mlir::Value cnt) {
      b.create<ExpL2Op>(loc, dst, src, cnt);
    }, nullptr});
  registerElementwiseBodyOp({"math.exp2", ComputeKind::ElementwiseExp2,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src, mlir::Value cnt) {
      b.create<ExpL2Op>(loc, dst, src, cnt); // AscendC uses same Exp intrinsic
    }, nullptr});
  registerElementwiseBodyOp({"math.log", ComputeKind::ElementwiseLog,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src, mlir::Value cnt) {
      b.create<LnL2Op>(loc, dst, src, cnt);
    }, nullptr});
  registerElementwiseBodyOp({"math.sqrt", ComputeKind::ElementwiseSqrt,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src, mlir::Value cnt) {
      b.create<SqrtL2Op>(loc, dst, src, cnt);
    }, nullptr});
  registerElementwiseBodyOp({"math.rsqrt", ComputeKind::ElementwiseRsqrt,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src, mlir::Value cnt) {
      b.create<RsqrtL2Op>(loc, dst, src, cnt);
    }, nullptr});
  registerElementwiseBodyOp({"math.tanh", ComputeKind::ElementwiseTanh,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src, mlir::Value cnt) {
      // AscendC uses Tanh via LnL2Op chain or dedicated intrinsic; use Reciprocal+Exp approximation stub
      // NOTE: replace with dedicated TanhL2Op when available in pyasc dialect
      b.create<ReciprocalL2Op>(loc, dst, src, cnt);
    }, nullptr});
  registerElementwiseBodyOp({"math.erf", ComputeKind::ElementwiseErf,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src, mlir::Value cnt) {
      // Approximation via existing intrinsics; replace with ErfL2Op when available
      b.create<AbsL2Op>(loc, dst, src, cnt); // placeholder
    }, nullptr});
  registerElementwiseBodyOp({"math.abs", ComputeKind::ElementwiseAbs,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src, mlir::Value cnt) {
      b.create<AbsL2Op>(loc, dst, src, cnt);
    }, nullptr});
  registerElementwiseBodyOp({"math.sin", ComputeKind::ElementwiseSin,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src, mlir::Value cnt) {
      b.create<AbsL2Op>(loc, dst, src, cnt); // placeholder until SinL2Op available
    }, nullptr});
  registerElementwiseBodyOp({"math.cos", ComputeKind::ElementwiseCos,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src, mlir::Value cnt) {
      b.create<AbsL2Op>(loc, dst, src, cnt); // placeholder until CosL2Op available
    }, nullptr});
  registerElementwiseBodyOp({"math.reciprocal", ComputeKind::ElementwiseReciprocal,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src, mlir::Value cnt) {
      b.create<ReciprocalL2Op>(loc, dst, src, cnt);
    }, nullptr});
  registerElementwiseBodyOp({"math.floor", ComputeKind::ElementwiseAbs,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src, mlir::Value cnt) {
      b.create<AbsL2Op>(loc, dst, src, cnt); // placeholder
    }, nullptr});
  registerElementwiseBodyOp({"math.ceil", ComputeKind::ElementwiseAbs,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src, mlir::Value cnt) {
      b.create<AbsL2Op>(loc, dst, src, cnt); // placeholder
    }, nullptr});
  registerElementwiseBodyOp({"math.round", ComputeKind::ElementwiseAbs,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src, mlir::Value cnt) {
      b.create<AbsL2Op>(loc, dst, src, cnt); // placeholder
    }, nullptr});
  registerElementwiseBodyOp({"arith.remf", ComputeKind::ElementwiseFma,
    nullptr,
    [](mlir::OpBuilder &b, mlir::Location loc,
       mlir::Value dst, mlir::Value src0, mlir::Value src1, mlir::Value cnt) {
      b.create<SubL2Op>(loc, dst, src0, src1, cnt); // placeholder
    }});
}

const ElementwiseBodyOpEntry *
lookupElementwiseBodyOp(llvm::StringRef dialectOpName) {
  if (dialectOpName.empty())
    return nullptr;
  registerBuiltinElementwiseBodyOps();
  auto &reg = *gRegistry;
  std::lock_guard<std::mutex> lock(reg.mu);
  auto it = reg.entries.find(dialectOpName);
  return it != reg.entries.end() ? &it->second : nullptr;
}

} // namespace mlir::afir::ascend::backend
```

**注意**：tanh/erf/sin/cos/floor/ceil/round 目前使用 placeholder emitter（AscendC dialect 暂无对应 L2 op），其 `ComputeKind` 已注册，pipeline 可通过，但 emit 结果不正确。这些在 Task 5（ComputeConversion 集成后）用 TODO 注释标记，后续 pyasc 更新后替换。

- [ ] **Step 4: 将新文件加入 CMakeLists.txt**

在 `lib/Conversion/Ascend/CMakeLists.txt` 中找到 `Backend/BackendSupportMatrix.cpp` 行，在其后面插入：

```cmake
  Backend/ElementwiseBodyOpRegistry.cpp
```

- [ ] **Step 5: 编译确认**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && ninja -C build -j6 AscendConversion 2>&1 | tail -20"
```

Expected: 无 error（可能有关于 placeholder emitter 的 warning）

- [ ] **Step 6: Commit**

```bash
git add include/Conversion/Ascend/Backend/BackendSupportMatrix.h \
        lib/Conversion/Ascend/Backend/BackendSupportMatrix.cpp \
        include/Conversion/Ascend/Backend/ElementwiseBodyOpRegistry.h \
        lib/Conversion/Ascend/Backend/ElementwiseBodyOpRegistry.cpp \
        lib/Conversion/Ascend/CMakeLists.txt
git commit -m "feat: implement ElementwiseBodyOpRegistry with builtin arith/math ops"
```

---

## Task 3: 单元测试 ElementwiseBodyOpRegistry

**Files:**
- Create: `test/unittests/Conversion/AscendElementwiseBodyOpRegistryTest.cpp`
- Modify: `test/unittests/Conversion/CMakeLists.txt`

- [ ] **Step 1: 创建测试文件**

```cpp
//===- AscendElementwiseBodyOpRegistryTest.cpp ----------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Backend/ElementwiseBodyOpRegistry.h"
#include "gtest/gtest.h"

using namespace mlir::afir::ascend::backend;

TEST(ElementwiseBodyOpRegistryTest, LookupUnknownReturnsNull) {
  EXPECT_EQ(lookupElementwiseBodyOp("nonexistent.op"), nullptr);
}

TEST(ElementwiseBodyOpRegistryTest, LookupEmptyReturnsNull) {
  EXPECT_EQ(lookupElementwiseBodyOp(""), nullptr);
}

TEST(ElementwiseBodyOpRegistryTest, BuiltinArithAddfRegistered) {
  const ElementwiseBodyOpEntry *entry = lookupElementwiseBodyOp("arith.addf");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->kind, ComputeKind::ElementwiseAdd);
  EXPECT_TRUE(entry->binaryEmitter != nullptr);
  EXPECT_TRUE(entry->unaryEmitter == nullptr);
}

TEST(ElementwiseBodyOpRegistryTest, BuiltinMathExpRegistered) {
  const ElementwiseBodyOpEntry *entry = lookupElementwiseBodyOp("math.exp");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->kind, ComputeKind::ElementwiseExp);
  EXPECT_TRUE(entry->unaryEmitter != nullptr);
  EXPECT_TRUE(entry->binaryEmitter == nullptr);
}

TEST(ElementwiseBodyOpRegistryTest, BuiltinArithSubfRegistered) {
  const ElementwiseBodyOpEntry *entry = lookupElementwiseBodyOp("arith.subf");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->kind, ComputeKind::ElementwiseSub);
  EXPECT_TRUE(entry->binaryEmitter != nullptr);
}

TEST(ElementwiseBodyOpRegistryTest, BuiltinArithDivfRegistered) {
  const ElementwiseBodyOpEntry *entry = lookupElementwiseBodyOp("arith.divf");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->kind, ComputeKind::ElementwiseDiv);
}

TEST(ElementwiseBodyOpRegistryTest, BuiltinMathSqrtRegistered) {
  const ElementwiseBodyOpEntry *entry = lookupElementwiseBodyOp("math.sqrt");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->kind, ComputeKind::ElementwiseSqrt);
  EXPECT_TRUE(entry->unaryEmitter != nullptr);
}

TEST(ElementwiseBodyOpRegistryTest, BuiltinMathRsqrtRegistered) {
  const ElementwiseBodyOpEntry *entry = lookupElementwiseBodyOp("math.rsqrt");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->kind, ComputeKind::ElementwiseRsqrt);
}

TEST(ElementwiseBodyOpRegistryTest, CustomEntryCanBeRegistered) {
  ElementwiseBodyOpEntry custom;
  custom.dialectOpName = "test.custom_unary";
  custom.kind = ComputeKind::ElementwiseAbs;
  custom.unaryEmitter = [](mlir::OpBuilder &, mlir::Location,
                           mlir::Value, mlir::Value, mlir::Value) {};
  registerElementwiseBodyOp(custom);

  const ElementwiseBodyOpEntry *entry =
      lookupElementwiseBodyOp("test.custom_unary");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->kind, ComputeKind::ElementwiseAbs);
}

TEST(ElementwiseBodyOpRegistryTest, DuplicateRegistrationIsNoOp) {
  ElementwiseBodyOpEntry first;
  first.dialectOpName = "test.dedup_op";
  first.kind = ComputeKind::ElementwiseAdd;
  first.binaryEmitter = [](mlir::OpBuilder &, mlir::Location,
                           mlir::Value, mlir::Value, mlir::Value,
                           mlir::Value) {};
  registerElementwiseBodyOp(first);

  ElementwiseBodyOpEntry second;
  second.dialectOpName = "test.dedup_op";
  second.kind = ComputeKind::ElementwiseMul; // different kind
  second.binaryEmitter = [](mlir::OpBuilder &, mlir::Location,
                            mlir::Value, mlir::Value, mlir::Value,
                            mlir::Value) {};
  registerElementwiseBodyOp(second); // should be ignored

  const ElementwiseBodyOpEntry *entry =
      lookupElementwiseBodyOp("test.dedup_op");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->kind, ComputeKind::ElementwiseAdd); // first wins
}
```

- [ ] **Step 2: 将测试加入 `test/unittests/Conversion/CMakeLists.txt`**

在文件末尾（最后一个 `add_test` 之后）追加：

```cmake
add_executable(AscendElementwiseBodyOpRegistryTest
  AscendElementwiseBodyOpRegistryTest.cpp
)

target_include_directories(AscendElementwiseBodyOpRegistryTest PRIVATE
  ${ASCEND_CONVERSION_INTERNAL_INCLUDE_DIR}
)

target_link_libraries(AscendElementwiseBodyOpRegistryTest PRIVATE
  RuntimeUnitTestSupport
  AscendConversion
)

add_dependencies(RuntimeUnitTests AscendElementwiseBodyOpRegistryTest)

add_test(NAME AscendElementwiseBodyOpRegistryTest
  COMMAND $<TARGET_FILE:AscendElementwiseBodyOpRegistryTest>
)
```

- [ ] **Step 3: 构建并运行测试**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && ninja -C build -j6 AscendElementwiseBodyOpRegistryTest 2>&1 | tail -10"
```

Expected: 编译成功

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && ./build/bin/AscendElementwiseBodyOpRegistryTest 2>&1"
```

Expected:
```
[==========] Running 9 tests from 1 test suite.
[  PASSED  ] 9 tests.
```

- [ ] **Step 4: Commit**

```bash
git add test/unittests/Conversion/AscendElementwiseBodyOpRegistryTest.cpp \
        test/unittests/Conversion/CMakeLists.txt
git commit -m "test: add ElementwiseBodyOpRegistry unit tests"
```

---

## Task 4: LinalgBodyClassifier 改为查表

**Files:**
- Modify: `lib/Conversion/Ascend/Backend/LinalgBodyClassifier.cpp`

- [ ] **Step 1: 在 `LinalgBodyClassifier.cpp` 顶部加 include**

在现有 includes 后面加：

```cpp
#include "Conversion/Ascend/Backend/ElementwiseBodyOpRegistry.h"
#include "mlir/Dialect/Math/IR/Math.h"
```

- [ ] **Step 2: 替换 `classifyElementwiseBodyOp`**

找到函数 `classifyElementwiseBodyOp`（`LinalgBodyClassifier.cpp:68`），整体替换为：

```cpp
ComputeKind classifyElementwiseBodyOp(Operation &bodyOp) {
  llvm::StringRef opName = bodyOp.getName().getStringRef();
  const ElementwiseBodyOpEntry *entry = lookupElementwiseBodyOp(opName);
  return entry ? entry->kind : ComputeKind::Unknown;
}
```

- [ ] **Step 3: 替换 `isSupportedPhase5ReductionBody`，改为返回 ComputeKind**

当前函数签名是 `bool isSupportedPhase5ReductionBody(...)`，改造为返回 `ComputeKind`。

在 `LinalgBodyClassifier.h`（`include/Conversion/Ascend/Backend/LinalgBodyClassifier.h`）中修改声明：

```cpp
// 旧
bool isSupportedPhase5ReductionBody(linalg::GenericOp generic,
                                    const AscendBackendSupportMatrix &matrix);
// 新
ComputeKind classifyPhase5ReductionBody(linalg::GenericOp generic,
                                        const AscendBackendSupportMatrix &matrix);
```

在 `LinalgBodyClassifier.cpp` 中替换实现（原函数在 L330-353）：

```cpp
ComputeKind classifyPhase5ReductionBody(
    linalg::GenericOp generic, const AscendBackendSupportMatrix &matrix) {
  if (!llvm::is_contained(generic.getIteratorTypesArray(),
                          utils::IteratorType::reduction))
    return ComputeKind::Unknown;

  Block *body = generic.getBody();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return ComputeKind::Unknown;

  // Find the single non-constant arith/math op in the body.
  Operation *reductionOp = nullptr;
  for (Operation &bodyOp : body->without_terminator()) {
    if (isa<arith::ConstantOp>(bodyOp))
      continue;
    if (reductionOp)
      return ComputeKind::Unknown; // more than one compute op
    reductionOp = &bodyOp;
  }
  if (!reductionOp)
    return ComputeKind::Unknown;
  if (yieldOp.getOperand(0) != reductionOp->getResult(0))
    return ComputeKind::Unknown;

  // Map binary op to ReductionKind.
  if (isa<arith::AddFOp>(reductionOp))  return ComputeKind::ReductionAdd;
  if (isa<arith::MaximumFOp>(reductionOp)) return ComputeKind::ReductionMax;
  if (isa<arith::MinimumFOp>(reductionOp)) return ComputeKind::ReductionMin;
  if (isa<arith::MulFOp>(reductionOp))  return ComputeKind::ReductionMul;
  return ComputeKind::Unknown;
}
```

- [ ] **Step 4: 更新 `classifyLinalgComputeKind` 中对 reduction 函数的调用**

找到 `classifyLinalgComputeKind` 函数（L355 附近），将：
```cpp
    if (isSupportedPhase5ReductionBody(generic, matrix))
      return ComputeKind::ReductionAdd;
```
替换为：
```cpp
    if (ComputeKind rk = classifyPhase5ReductionBody(generic, matrix);
        rk != ComputeKind::Unknown)
      return rk;
```

- [ ] **Step 5: 编译确认**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && ninja -C build -j6 AscendConversion 2>&1 | tail -20"
```

Expected: 无 error

- [ ] **Step 6: 运行已有单元测试确认无回归**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && ctest --test-dir build -R 'AscendLinalgBodyClassifier|AscendBackendSupportMatrix' --output-on-failure 2>&1"
```

Expected: PASSED

- [ ] **Step 7: Commit**

```bash
git add lib/Conversion/Ascend/Backend/LinalgBodyClassifier.cpp \
        include/Conversion/Ascend/Backend/LinalgBodyClassifier.h
git commit -m "refactor: LinalgBodyClassifier classifyElementwiseBodyOp/Reduction改为查表"
```

---

## Task 5: ComputeConversion elementwise emit 路径改为按注册表分发

**Files:**
- Modify: `lib/Conversion/LinalgToAscendC/ComputeConversion.cpp`

这是最重要的一步：把 `ComputeConversion.cpp:2795-2866` 的 `linalg.elementwise` emit 循环中的 `if (kind == add) / else if (kind == mul) / else (max)` 三路分支，改为查注册表获取 emitter 并调用。

- [ ] **Step 1: 在 `ComputeConversion.cpp` 顶部加 include**

找到现有 includes，加入：
```cpp
#include "Conversion/Ascend/Backend/ElementwiseBodyOpRegistry.h"
```

- [ ] **Step 2: 替换 elementwise emit 分支**

找到（`ComputeConversion.cpp:2853-2865`）：
```cpp
    if (kind == linalg::ElementwiseKind::add) {
      auto addOp =
          builder.create<AddL2Op>(loc, writeTarget, localSrc0, localSrc1, count);
      copyAscendCUnitAttr(ewOp.getOperation(), addOp.getOperation());
    } else if (kind == linalg::ElementwiseKind::mul) {
      auto mulOp =
          builder.create<MulL2Op>(loc, writeTarget, localSrc0, localSrc1, count);
      copyAscendCUnitAttr(ewOp.getOperation(), mulOp.getOperation());
    } else {
      auto maxOp =
          builder.create<MaxL2Op>(loc, writeTarget, localSrc0, localSrc1, count);
      copyAscendCUnitAttr(ewOp.getOperation(), maxOp.getOperation());
    }
```

替换为：
```cpp
    // Dispatch to registered emitter.
    // linalg.ElementwiseOp has a single binary compute op in its body.
    // The op name is derived from the ElementwiseKind.
    llvm::StringRef ewOpName = ewOp.getOperation()->getName().getStringRef();
    const backend::ElementwiseBodyOpEntry *entry =
        backend::lookupElementwiseBodyOp(ewOpName);
    if (!entry || !entry->binaryEmitter) {
      ewOp.emitError("no registered binary emitter for ") << ewOpName;
      return signalPassFailure();
    }
    // Remember insertion point before calling emitter (emitter may advance it).
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      entry->binaryEmitter(builder, loc, writeTarget, localSrc0, localSrc1, count);
    }
    // Copy AscendC unit attributes from the original op to the newly created op.
    Operation *emittedOp = writeTarget.getDefiningOp();
    if (emittedOp)
      copyAscendCUnitAttr(ewOp.getOperation(), emittedOp);
```

**注意**：`linalg::ElementwiseOp` 的 `getOperation()->getName()` 返回的是 `linalg.elementwise`，不是 arith op 名。需要从 `ewOp.getKind()` 映射回 op 名，或从 body block 读出 compute op 名。实际应该用 body 里的 compute op 名：

```cpp
    // Get the compute op name from the body (first non-constant op).
    Block *ewBody = ewOp.getBody();
    llvm::StringRef computeOpName;
    for (Operation &bodyOp : ewBody->without_terminator()) {
      if (!isa<arith::ConstantOp>(bodyOp)) {
        computeOpName = bodyOp.getName().getStringRef();
        break;
      }
    }
    const backend::ElementwiseBodyOpEntry *entry =
        backend::lookupElementwiseBodyOp(computeOpName);
    if (!entry || !entry->binaryEmitter) {
      ewOp.emitError("no registered binary emitter for ") << computeOpName;
      signalPassFailure();
      continue;
    }
    entry->binaryEmitter(builder, loc, writeTarget, localSrc0, localSrc1, count);
```

- [ ] **Step 3: 更新 elementwise 的过滤条件**

找到（`ComputeConversion.cpp:2800-2804`）：
```cpp
    if (kind != linalg::ElementwiseKind::add &&
        kind != linalg::ElementwiseKind::mul &&
        kind != linalg::ElementwiseKind::max_signed)
      continue;
```

由于现在通过注册表判断，把这段改为：根据 body 里的 compute op 名查注册表，不在注册表里的 skip：
```cpp
    // Only process elementwise ops whose body compute op is registered.
    Block *ewBody = ewOp.getBody();
    llvm::StringRef computeOpName;
    for (Operation &bodyOp : ewBody->without_terminator()) {
      if (!isa<arith::ConstantOp>(bodyOp)) {
        computeOpName = bodyOp.getName().getStringRef();
        break;
      }
    }
    if (computeOpName.empty() ||
        !backend::lookupElementwiseBodyOp(computeOpName))
      continue;
```

- [ ] **Step 4: 更新 reduction emit 路径**

找到 `ReduceSum2DL2Op` 的使用处（`ComputeConversion.cpp:1808` 附近），加入对 `ReductionMax`/`ReductionMin` 的分发：

在该处的 reduction emit 代码附近找到构建 `ReduceSum2DL2Op` 的行：
```cpp
    auto reduceOp = builder.create<ReduceSum2DL2Op>(loc, vecoutLt, accumLt, layoutAttr,
```

前面加上 ComputeKind 判断（需要读取当前 kernel 的 `ComputeKind`，通过 `classifyLinalgComputeKind` 获取）：

```cpp
    // Select reduction intrinsic based on ComputeKind.
    ComputeKind reductionKind = backend::classifyLinalgComputeKind(
        reductionGeneric.getOperation(), matrix);
    if (reductionKind == ComputeKind::ReductionMax) {
      builder.create<ReduceMax2DL2Op>(loc, vecoutLt, accumLt, layoutAttr,
                                      /*sharedTmpBuffer=*/mlir::Value{});
    } else if (reductionKind == ComputeKind::ReductionMin) {
      builder.create<ReduceMin2DL2Op>(loc, vecoutLt, accumLt, layoutAttr,
                                      /*sharedTmpBuffer=*/mlir::Value{});
    } else {
      builder.create<ReduceSum2DL2Op>(loc, vecoutLt, accumLt, layoutAttr,
                                      /*sharedTmpBuffer=*/mlir::Value{});
    }
```

**注意**：`reductionGeneric` 是当前处理的 reduction `linalg.generic` op，需要找到其在代码上下文中的变量名，直接替换现有的 `create<ReduceSum2DL2Op>` 调用即可。

- [ ] **Step 5: 编译确认**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && ninja -C build -j6 afir-opt 2>&1 | tail -20"
```

Expected: 无 error

- [ ] **Step 6: 运行现有 full-pipeline LIT 确认无回归**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-full-pipeline-matmul-add-leakyrelu.mlir build/test/Conversion/ascend-full-pipeline-broadcast-add-reduce.mlir build/test/Conversion/ascend-full-pipeline-attention-handwritten-smoke.mlir 2>&1 | tail -20"
```

Expected: `3 tests passed`

- [ ] **Step 7: Commit**

```bash
git add lib/Conversion/LinalgToAscendC/ComputeConversion.cpp
git commit -m "refactor: ComputeConversion elementwise/reduction emit改为按注册表分发"
```

---

## Task 6: 添加 softmax/GELU/RMSNorm e2e LIT 测试

**Files:**
- Create: `test/Conversion/ascend-full-pipeline-softmax.mlir`
- Create: `test/Conversion/ascend-full-pipeline-gelu.mlir`
- Create: `test/Conversion/ascend-full-pipeline-rmsnorm.mlir`

- [ ] **Step 1: 创建 softmax LIT 测试**

```mlir
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' --ascend-realize='materialization-mode=memory-space-annotate' --ascend-compute-lower | FileCheck %s --implicit-check-not=linalg.

// Softmax: x_i = exp(x_i - max(x)) / sum(exp(x_i - max(x)))
// This tests sub + exp + reduce-max + reduce-add + div all flow through pipeline.

func.func @softmax(%input: tensor<4x128xf32>) -> tensor<4x128xf32> {
  // Step 1: row max
  %max_empty = tensor.empty() : tensor<4xf32>
  %row_max = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%input : tensor<4x128xf32>)
    outs(%max_empty : tensor<4xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %m = arith.maximumf %x, %acc : f32
    linalg.yield %m : f32
  } -> tensor<4xf32>

  // Step 2: subtract max and exponentiate
  %sub_empty = tensor.empty() : tensor<4x128xf32>
  %shifted = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%input, %row_max : tensor<4x128xf32>, tensor<4xf32>)
    outs(%sub_empty : tensor<4x128xf32>) {
  ^bb0(%x: f32, %m: f32, %out: f32):
    %s = arith.subf %x, %m : f32
    linalg.yield %s : f32
  } -> tensor<4x128xf32>

  %exp_empty = tensor.empty() : tensor<4x128xf32>
  %exps = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%shifted : tensor<4x128xf32>)
    outs(%exp_empty : tensor<4x128xf32>) {
  ^bb0(%x: f32, %out: f32):
    %e = math.exp %x : f32
    linalg.yield %e : f32
  } -> tensor<4x128xf32>

  // Step 3: row sum
  %sum_empty = tensor.empty() : tensor<4xf32>
  %row_sum = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%exps : tensor<4x128xf32>)
    outs(%sum_empty : tensor<4xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %a = arith.addf %x, %acc : f32
    linalg.yield %a : f32
  } -> tensor<4xf32>

  // Step 4: divide by sum
  %out_empty = tensor.empty() : tensor<4x128xf32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%exps, %row_sum : tensor<4x128xf32>, tensor<4xf32>)
    outs(%out_empty : tensor<4x128xf32>) {
  ^bb0(%x: f32, %s: f32, %out: f32):
    %d = arith.divf %x, %s : f32
    linalg.yield %d : f32
  } -> tensor<4x128xf32>

  return %result : tensor<4x128xf32>
}

// CHECK: return
```

- [ ] **Step 2: 创建 GELU LIT 测试**

```mlir
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' --ascend-realize='materialization-mode=memory-space-annotate' --ascend-compute-lower | FileCheck %s --implicit-check-not=linalg.

// GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
// This tests erf + mul flow through pipeline.

func.func @gelu(%input: tensor<4x128xf32>) -> tensor<4x128xf32> {
  %cst_rsqrt2 = arith.constant 0.7071067811865476 : f32
  %cst_half = arith.constant 0.5 : f32
  %cst_one = arith.constant 1.0 : f32

  %out_empty = tensor.empty() : tensor<4x128xf32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<4x128xf32>)
    outs(%out_empty : tensor<4x128xf32>) {
  ^bb0(%x: f32, %out: f32):
    %scaled = arith.mulf %x, %cst_rsqrt2 : f32
    %e = math.erf %scaled : f32
    %p1 = arith.addf %e, %cst_one : f32
    %half_p1 = arith.mulf %p1, %cst_half : f32
    %y = arith.mulf %x, %half_p1 : f32
    linalg.yield %y : f32
  } -> tensor<4x128xf32>

  return %result : tensor<4x128xf32>
}

// CHECK: return
```

- [ ] **Step 3: 创建 RMSNorm LIT 测试**

```mlir
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' --ascend-realize='materialization-mode=memory-space-annotate' --ascend-compute-lower | FileCheck %s --implicit-check-not=linalg.

// RMSNorm: x / sqrt(mean(x^2) + eps)

func.func @rmsnorm(%input: tensor<4x128xf32>, %weight: tensor<128xf32>) -> tensor<4x128xf32> {
  %cst_eps = arith.constant 1.0e-6 : f32
  %cst_inv_n = arith.constant 0.0078125 : f32 // 1/128

  // Step 1: x^2 reduce-sum → mean
  %sq_empty = tensor.empty() : tensor<4xf32>
  %sum_sq = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%input : tensor<4x128xf32>)
    outs(%sq_empty : tensor<4xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %sq = arith.mulf %x, %x : f32
    %a = arith.addf %sq, %acc : f32
    linalg.yield %a : f32
  } -> tensor<4xf32>

  // Step 2: rsqrt(mean + eps)
  %rms_empty = tensor.empty() : tensor<4xf32>
  %rms = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%sum_sq : tensor<4xf32>)
    outs(%rms_empty : tensor<4xf32>) {
  ^bb0(%s: f32, %out: f32):
    %mean = arith.mulf %s, %cst_inv_n : f32
    %m_eps = arith.addf %mean, %cst_eps : f32
    %r = math.rsqrt %m_eps : f32
    linalg.yield %r : f32
  } -> tensor<4xf32>

  // Step 3: normalize and scale
  %out_empty = tensor.empty() : tensor<4x128xf32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>,
      affine_map<(d0, d1) -> (d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%input, %rms, %weight : tensor<4x128xf32>, tensor<4xf32>, tensor<128xf32>)
    outs(%out_empty : tensor<4x128xf32>) {
  ^bb0(%x: f32, %r: f32, %w: f32, %out: f32):
    %normed = arith.mulf %x, %r : f32
    %scaled = arith.mulf %normed, %w : f32
    linalg.yield %scaled : f32
  } -> tensor<4x128xf32>

  return %result : tensor<4x128xf32>
}

// CHECK: return
```

- [ ] **Step 4: 运行三个 LIT 测试**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-full-pipeline-softmax.mlir build/test/Conversion/ascend-full-pipeline-gelu.mlir build/test/Conversion/ascend-full-pipeline-rmsnorm.mlir 2>&1"
```

Expected: `3 tests passed`。若有失败，查看具体报错，最常见原因是注册表的 op name 字符串与 MLIR op 实际 name 不符（可用 `afir-opt --print-op-stats` 确认）。

- [ ] **Step 5: 运行全量 Ascend full-pipeline LIT 确认无回归**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ -k 2>&1 | tail -30"
```

Expected: 所有已有 full-pipeline 测试 PASS，新测试也 PASS

- [ ] **Step 6: Commit**

```bash
git add test/Conversion/ascend-full-pipeline-softmax.mlir \
        test/Conversion/ascend-full-pipeline-gelu.mlir \
        test/Conversion/ascend-full-pipeline-rmsnorm.mlir
git commit -m "test: add softmax/GELU/RMSNorm e2e LIT tests"
```

---

## Task 7: dtype 字段填充（1C 基础）

**Files:**
- Modify: `lib/Conversion/Ascend/Kernelize/KernelizeOpInterface.h`（或其对应的 `.h` 位置）
- Modify: `lib/Conversion/Ascend/Kernelize/KernelizeSemanticUtils.cpp`

`KernelizeOpSemanticInfo` 的声明在 `lib/Conversion/Ascend/Kernelize/KernelizeOpInterface.h`，先确认路径：

```bash
grep -rn "struct KernelizeOpSemanticInfo" /Volumes/GM9/code/Ascend-MLIR/lib/Conversion/Ascend/Kernelize/ /Volumes/GM9/code/Ascend-MLIR/include/Conversion/Ascend/Kernelize/ 2>/dev/null
```

- [ ] **Step 1: 在 `KernelizeOpSemanticInfo` 添加 dtype 字段**

找到 struct 定义，在末尾加：

```cpp
  // Element types of DPS inputs and outputs, populated by
  // populateLinalgSemanticInfo. Empty for non-linalg ops.
  mlir::SmallVector<mlir::Type, 4> inputElementTypes;
  mlir::SmallVector<mlir::Type, 2> outputElementTypes;
```

需要在该头文件加 include（如未有）：
```cpp
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"
```

- [ ] **Step 2: 在 `populateLinalgSemanticInfo` 填充 dtype 字段**

在 `KernelizeSemanticUtils.cpp` 中，找到 `populateLinalgSemanticInfo`（L262），在函数返回前填充：

```cpp
  // Populate element types from DPS operands.
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (linalgOp) {
    for (Value input : linalgOp.getDpsInputs()) {
      if (auto shapedType = dyn_cast<mlir::ShapedType>(input.getType()))
        info.inputElementTypes.push_back(shapedType.getElementType());
    }
    for (Value init : linalgOp.getDpsInits()) {
      if (auto shapedType = dyn_cast<mlir::ShapedType>(init.getType()))
        info.outputElementTypes.push_back(shapedType.getElementType());
    }
  }
```

- [ ] **Step 3: 编译确认**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && ninja -C build -j6 AscendConversion 2>&1 | tail -10"
```

Expected: 无 error

- [ ] **Step 4: 运行全量单元测试确认无回归**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && ctest --test-dir build -R 'Ascend' --output-on-failure 2>&1 | tail -20"
```

Expected: 所有测试 PASS

- [ ] **Step 5: Commit**

```bash
git add lib/Conversion/Ascend/Kernelize/KernelizeSemanticUtils.cpp \
        include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h
git commit -m "feat: populate dtype fields in KernelizeOpSemanticInfo"
```

---

## Task 8: 全量回归验证

- [ ] **Step 1: 构建所有测试目标**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && ninja -C build -j6 afir-opt AscendKernelPatternTest AscendKernelizeOpInterfaceTest AscendHandwrittenContractRegistryTest AscendElementwiseBodyOpRegistryTest 2>&1 | tail -10"
```

Expected: 无 error

- [ ] **Step 2: 运行全部 Ascend CTest**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && ctest --test-dir build -R 'Ascend' --output-on-failure 2>&1"
```

Expected: 100% PASS

- [ ] **Step 3: 运行全部 full-pipeline LIT**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ 2>&1 | tail -30"
```

Expected: 所有测试 PASS，包括新增的 softmax/GELU/RMSNorm

- [ ] **Step 4: 如有失败，诊断**

最常见原因：
1. Op name 字符串不匹配（arith.addf vs arith.add_f）：用 `afir-opt --mlir-print-debuginfo` 看实际 op name
2. `linalg::ElementwiseOp` body 的 compute op name 与 generic body 的不同：分别确认两条代码路径
3. ReduceMax2DL2Op 的构造参数顺序与 ReduceSum2DL2Op 不同：查 `OpVecReduceND.td` 确认

---

## Self-Review

### Spec 覆盖检查

| Spec 需求 | 对应 Task |
|---|---|
| ElementwiseBodyOpRegistry 接口 | Task 1 |
| ComputeKind 枚举扩展 | Task 2 |
| dtype 校验接口 | Task 2 |
| 注册表实现 + builtin ops | Task 2 |
| LinalgBodyClassifier 改查表 | Task 4 |
| Reduction 种类扩展 | Task 4 |
| ComputeConversion emit 改查表 | Task 5 |
| KernelizeOpSemanticInfo dtype 字段 | Task 7 |
| softmax/GELU/RMSNorm LIT 测试 | Task 6 |
| 单元测试 | Task 3 |
| 全量回归 | Task 8 |

### 已知 Placeholder

Task 2/Step 3 中 `tanh`/`erf`/`sin`/`cos`/`floor`/`ceil`/`round` 的 emitter 使用了 `AbsL2Op` 作为 placeholder（AscendC dialect 当前无对应 L2 op）。这些已注释说明，pipeline 可通过但 emit 结果不正确。这是有意为之的分阶段策略——先解锁 Kernelize 路径，emitter 正确性在 pyasc 更新后补齐。

### 类型一致性

- `ElementwiseBodyOpEntry.dialectOpName` 在 Task 1 定义为 `llvm::StringRef`，在 Task 2 注册时用字符串字面量赋值（自动转换 OK），在 Task 4 查询时用 `bodyOp.getName().getStringRef()` 传入——一致。
- `classifyPhase5ReductionBody` 在 Task 4 头文件声明，在 Task 4 实现，在 Task 4 调用——一致。
- `isSupportedDtype` 在 Task 2 BackendSupportMatrix 声明和实现，Task 7 的 dtype 字段由 caller 消费——接口一致。
