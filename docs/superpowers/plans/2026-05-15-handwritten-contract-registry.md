# HandwrittenContract Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前散落在三处的 `if (handwrittenKind == "attention_sdpa")` 特判收拢为一张可注册的 `HandwrittenContract` 描述表，使新增 handwritten 模式（MLA、Ring Attention 等）只需一处注册、零处 if-else 修改。

**Architecture:** 新增 `HandwrittenContractRegistry`（单例，线程安全，与 `TemplateRegistry` 同模式），每个 handwritten 模式在启动时调用 `registerHandwrittenContract(kind, contract)` 注册全部行为描述。`AxisCoalescer`、`ScheduleProblemBuilder`、`FusionCandidateAnalysis` 三个消费方改为通过 `lookupHandwrittenContract(kind)` 查表，删除各自的 `if` 分支。`TemplateRegistry::registerBuiltinTemplates` 中的 attention 模板入口也移入注册函数统一管理。

**Tech Stack:** C++17, LLVM/MLIR (llvm::ManagedStatic, llvm::StringRef, llvm::SmallVector), GTest

---

## 文件结构

| 文件 | 操作 | 职责 |
|---|---|---|
| `lib/Conversion/Ascend/Kernelize/HandwrittenContractRegistry.h` | 新建 | `HandwrittenContract` 结构体定义 + 注册/查询接口声明 |
| `lib/Conversion/Ascend/Kernelize/HandwrittenContractRegistry.cpp` | 新建 | 注册表实现 + `registerBuiltinHandwrittenContracts()` |
| `lib/Conversion/Ascend/CMakeLists.txt` | 修改 | 添加新 `.cpp` 源文件 |
| `lib/Conversion/Ascend/Schedule/AxisCoalescer.cpp` | 修改 | 删除 `isAttentionHandwrittenPattern()`，改为查表 |
| `lib/Conversion/Ascend/Schedule/ScheduleProblemBuilder.cpp` | 修改 | 删除 attention if-else，改为查表注入 structureConstraints |
| `lib/Conversion/Ascend/Kernelize/FusionCandidateAnalysis.cpp` | 修改 | 删除 `buildHandwrittenPatternCandidate` 内 attention 特判，改为查表 |
| `lib/Conversion/Ascend/Schedule/TemplateRegistry.cpp` | 修改 | 删除 attention 模板硬注册，改为从 HandwrittenContractRegistry 取 |
| `test/unittests/Conversion/AscendHandwrittenContractRegistryTest.cpp` | 新建 | 单元测试 |
| `test/unittests/Conversion/CMakeLists.txt` | 修改 | 注册新测试目标 |

---

## Task 1: 定义 HandwrittenContract 结构体和注册接口

**Files:**
- Create: `lib/Conversion/Ascend/Kernelize/HandwrittenContractRegistry.h`

- [ ] **Step 1: 创建头文件**

```cpp
//===- HandwrittenContractRegistry.h - Handwritten pattern contracts --*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_HANDWRITTENCONTRACTREGISTRY_H
#define ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_HANDWRITTENCONTRACTREGISTRY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir::afir::ascend::kernelize {

/// Describes all compile-time behavior of one handwritten kernel pattern.
/// Register via registerHandwrittenContract(); consume via lookupHandwrittenContract().
struct HandwrittenContract {
  /// Minimum number of Cube ops required in the pattern graph.
  unsigned minCubeCount = 1;
  /// Maximum number of Cube ops allowed in the pattern graph.
  unsigned maxCubeCount = 1;
  /// Pattern must contain at least one Reduction op.
  bool requiresReduction = false;
  /// Pattern must contain at least one Vector+Injective op.
  bool requiresVectorInjective = false;

  /// AxisCoalescer: when true, only the dominant primary op contributes axes
  /// (the other ops' indexing maps are not merged into the coalesced axis set).
  bool useAxisCarrierOnly = false;

  /// Primary op selection: when non-empty, prefer the op whose Kernelize role
  /// matches this string (e.g. "Cube"). Falls back to priority-based selection.
  std::string primarySelectionRole;

  /// ScheduleProblemBuilder: extra structure constraint tags appended to
  /// ScheduleProblem::structureConstraints for this pattern.
  SmallVector<std::string, 2> structureConstraints;

  /// TemplateRegistry: the schedule template registered for this pattern.
  /// Fields: name, layout, tags, minRank, maxRank, priority.
  struct TemplateSpec {
    std::string name;
    std::string layout;
    SmallVector<std::string, 4> tags;
    unsigned minRank = 0;
    unsigned maxRank = 8;
    int priority = 0;
  };
  TemplateSpec scheduleTemplate;
};

/// Register a handwritten contract for the given kind string.
/// Must be called before any use of lookupHandwrittenContract.
/// Safe to call multiple times with the same kind (second call is a no-op).
void registerHandwrittenContract(llvm::StringRef kind,
                                 HandwrittenContract contract);

/// Register all built-in contracts (currently: attention_sdpa).
/// Called automatically on first lookup; exposed for test initialization.
void registerBuiltinHandwrittenContracts();

/// Returns nullptr if kind is empty or not registered.
const HandwrittenContract *lookupHandwrittenContract(llvm::StringRef kind);

} // namespace mlir::afir::ascend::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_HANDWRITTENCONTRACTREGISTRY_H
```

- [ ] **Step 2: Commit**

```bash
git add lib/Conversion/Ascend/Kernelize/HandwrittenContractRegistry.h
git commit -m "feat: add HandwrittenContract struct and registry interface"
```

---

## Task 2: 实现注册表，并注册 attention_sdpa 内置契约

**Files:**
- Create: `lib/Conversion/Ascend/Kernelize/HandwrittenContractRegistry.cpp`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`

- [ ] **Step 1: 写失败测试**（测试文件在 Task 7 正式建立，此步先写 inline 验证逻辑）

先确认当前没有 `HandwrittenContractRegistry.cpp` 存在：

```bash
ls lib/Conversion/Ascend/Kernelize/HandwrittenContractRegistry.cpp 2>/dev/null && echo EXISTS || echo NOT_EXISTS
```

Expected: `NOT_EXISTS`

- [ ] **Step 2: 创建实现文件**

```cpp
//===- HandwrittenContractRegistry.cpp - Handwritten pattern contracts ----===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "HandwrittenContractRegistry.h"

#include "KernelizeTypes.h"
#include "Conversion/Ascend/Common/Attributes.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ManagedStatic.h"

#include <mutex>

namespace mlir::afir::ascend::kernelize {
namespace {

struct HandwrittenContractRegistrySingleton {
  std::mutex mu;
  llvm::StringMap<HandwrittenContract> contracts;
  bool builtinsRegistered = false;
};

llvm::ManagedStatic<HandwrittenContractRegistrySingleton> gContractRegistry;

HandwrittenContractRegistrySingleton &contractRegistry() {
  return *gContractRegistry;
}

} // namespace

void registerHandwrittenContract(llvm::StringRef kind,
                                 HandwrittenContract contract) {
  HandwrittenContractRegistrySingleton &reg = contractRegistry();
  std::lock_guard<std::mutex> lock(reg.mu);
  reg.contracts.try_emplace(kind, std::move(contract));
}

void registerBuiltinHandwrittenContracts() {
  HandwrittenContractRegistrySingleton &reg = contractRegistry();
  {
    std::lock_guard<std::mutex> lock(reg.mu);
    if (reg.builtinsRegistered)
      return;
    reg.builtinsRegistered = true;
  }

  HandwrittenContract attnContract;
  attnContract.minCubeCount = 2;
  attnContract.maxCubeCount = 2;
  attnContract.requiresReduction = true;
  attnContract.requiresVectorInjective = true;
  attnContract.useAxisCarrierOnly = true;
  attnContract.primarySelectionRole = "Cube";
  attnContract.structureConstraints = {"handwritten_group", "attention_sdpa_chain"};

  HandwrittenContract::TemplateSpec &tmpl = attnContract.scheduleTemplate;
  tmpl.name = kKernelizeHandwrittenKindAttentionSdpa.str();
  tmpl.layout = "grouped_tile_per_block";
  tmpl.tags = {kKernelizeHandwrittenKindAttentionSdpa.str(),
               kOpRoleCube.str(), kOpRoleReduction.str(),
               kOpRoleVector.str()};
  tmpl.minRank = 2;
  tmpl.maxRank = 4;
  tmpl.priority = 2;

  registerHandwrittenContract(kKernelizeHandwrittenKindAttentionSdpa,
                              std::move(attnContract));
}

const HandwrittenContract *lookupHandwrittenContract(llvm::StringRef kind) {
  if (kind.empty())
    return nullptr;
  registerBuiltinHandwrittenContracts();
  HandwrittenContractRegistrySingleton &reg = contractRegistry();
  std::lock_guard<std::mutex> lock(reg.mu);
  auto it = reg.contracts.find(kind);
  return it != reg.contracts.end() ? &it->second : nullptr;
}

} // namespace mlir::afir::ascend::kernelize
```

- [ ] **Step 3: 将新 .cpp 加入 CMakeLists.txt**

在 `lib/Conversion/Ascend/CMakeLists.txt` 中，找到 `Kernelize/FusionCandidateAnalysis.cpp` 这一行，在其前面插入：

```cmake
  Kernelize/HandwrittenContractRegistry.cpp
```

结果应使此段看起来像：
```cmake
  Kernelize/HandwrittenContractRegistry.cpp
  Kernelize/FusionCandidateAnalysis.cpp
```

- [ ] **Step 4: 确认编译通过**

```bash
cd /Volumes/GM9/code/Ascend-MLIR && cmake --build build --target AscendConversion 2>&1 | tail -20
```

Expected: 无 error，有可能有关于未使用函数的 warning（Task 3-5 会消除）

- [ ] **Step 5: Commit**

```bash
git add lib/Conversion/Ascend/Kernelize/HandwrittenContractRegistry.cpp \
        lib/Conversion/Ascend/CMakeLists.txt
git commit -m "feat: implement HandwrittenContractRegistry with attention_sdpa builtin"
```

---

## Task 3: 替换 AxisCoalescer 中的 attention 特判

**Files:**
- Modify: `lib/Conversion/Ascend/Schedule/AxisCoalescer.cpp`

当前代码在两处使用 `isAttentionHandwrittenPattern(pattern)` 判断是否只用 axis carrier op（L496、L527）。

- [ ] **Step 1: 阅读确认当前状态**

在 `AxisCoalescer.cpp` 中定位以下两个函数（约 L216-L527）：

```cpp
bool isAttentionHandwrittenPattern(const KernelPatternView &pattern) {
  return pattern.handwrittenKind == kKernelizeHandwrittenKindAttentionSdpa;
}
```

和 `selectAxisCarrierOp`、`coalesceAxes` 中的两次调用。

- [ ] **Step 2: 添加 include**

在 `AxisCoalescer.cpp` 的 include 列表中加入：

```cpp
#include "Conversion/Ascend/Kernelize/HandwrittenContractRegistry.h"
```

（放在已有的 `#include "KernelPatternView.h"` 之后）

- [ ] **Step 3: 删除 isAttentionHandwrittenPattern，替换两处调用**

删除整个 `isAttentionHandwrittenPattern` 函数（约 L216-218），然后：

将 `selectAxisCarrierOp` 中的调用：
```cpp
// 旧代码
const PatternOpView *selectAxisCarrierOp(const KernelPatternView &pattern) {
  if (isAttentionHandwrittenPattern(pattern))
    return selectDominantPrimaryOp(pattern);
  for (const PatternOpView &opView : pattern.ops) {
    if (opView.role == pattern.dominantRole)
      return &opView;
  }
  return selectDominantPrimaryOp(pattern);
}
```

改为：
```cpp
const PatternOpView *selectAxisCarrierOp(const KernelPatternView &pattern) {
  using namespace ::mlir::afir::ascend::kernelize;
  const HandwrittenContract *contract =
      lookupHandwrittenContract(pattern.handwrittenKind);
  if (contract && contract->useAxisCarrierOnly)
    return selectDominantPrimaryOp(pattern);
  for (const PatternOpView &opView : pattern.ops) {
    if (opView.role == pattern.dominantRole)
      return &opView;
  }
  return selectDominantPrimaryOp(pattern);
}
```

将 `coalesceAxes` 中的调用：
```cpp
// 旧代码
bool useAxisCarrierOnly = isAttentionHandwrittenPattern(pattern);
```

改为：
```cpp
using namespace ::mlir::afir::ascend::kernelize;
const HandwrittenContract *hwContract =
    lookupHandwrittenContract(pattern.handwrittenKind);
bool useAxisCarrierOnly = hwContract && hwContract->useAxisCarrierOnly;
```

- [ ] **Step 4: 编译确认**

```bash
cd /Volumes/GM9/code/Ascend-MLIR && cmake --build build --target AscendConversion 2>&1 | tail -20
```

Expected: 无 error

- [ ] **Step 5: Commit**

```bash
git add lib/Conversion/Ascend/Schedule/AxisCoalescer.cpp
git commit -m "refactor: replace attention if-branch in AxisCoalescer with contract lookup"
```

---

## Task 4: 替换 ScheduleProblemBuilder 中的 attention 特判

**Files:**
- Modify: `lib/Conversion/Ascend/Schedule/ScheduleProblemBuilder.cpp`

当前问题代码（约 L222-225）：

```cpp
if (pattern.handwrittenKind == kKernelizeHandwrittenKindAttentionSdpa) {
  problem.structureConstraints.push_back("handwritten_group");
  problem.structureConstraints.push_back("attention_sdpa_chain");
}
```

- [ ] **Step 1: 添加 include**

在 `ScheduleProblemBuilder.cpp` 的 include 列表中加入：

```cpp
#include "Conversion/Ascend/Kernelize/HandwrittenContractRegistry.h"
```

（放在已有的 `#include "KernelPatternView.h"` 之后）

- [ ] **Step 2: 替换 if-else 为查表**

将上述 if 块替换为：

```cpp
if (const auto *hwContract =
        ::mlir::afir::ascend::kernelize::lookupHandwrittenContract(
            pattern.handwrittenKind)) {
  llvm::append_range(problem.structureConstraints,
                     hwContract->structureConstraints);
}
```

同时删除不再使用的 `using` 引入（如果 `kKernelizeHandwrittenKindAttentionSdpa` 已不再在此文件中使用）。检查方式：

```bash
grep "kKernelizeHandwrittenKindAttentionSdpa" lib/Conversion/Ascend/Schedule/ScheduleProblemBuilder.cpp
```

如果输出为空，删除该 `using` 声明（它来自 ScheduleTypes.h 的 using 列表，不在此文件中直接声明，无需额外操作）。

- [ ] **Step 3: 编译确认**

```bash
cd /Volumes/GM9/code/Ascend-MLIR && cmake --build build --target AscendConversion 2>&1 | tail -20
```

Expected: 无 error

- [ ] **Step 4: Commit**

```bash
git add lib/Conversion/Ascend/Schedule/ScheduleProblemBuilder.cpp
git commit -m "refactor: replace attention if-branch in ScheduleProblemBuilder with contract lookup"
```

---

## Task 5: 替换 FusionCandidateAnalysis 中的 attention 特判

**Files:**
- Modify: `lib/Conversion/Ascend/Kernelize/FusionCandidateAnalysis.cpp`

当前有两处需要修改：

**位置 A**（约 L318-326）— `buildHandwrittenPatternCandidate` 中的 primary op 选择：
```cpp
if (handwrittenKind == kKernelizeHandwrittenKindAttentionSdpa) {
  if (hasRole(roles, OpRole::Cube))
    primary = op;
  continue;
}
```

**位置 B**（约 L406）— `collectAttentionLikeHandwrittenPattern` 的图匹配 **暂不修改**（该函数涉及具体拓扑逻辑，泛化是第二阶段工作；本次计划只消除分散 if-else，不改变结构识别机制）。

- [ ] **Step 1: 添加 include**

在 `FusionCandidateAnalysis.cpp` 的 include 列表中加入：

```cpp
#include "HandwrittenContractRegistry.h"
```

（放在已有 `#include "CandidateClosure.h"` 附近）

- [ ] **Step 2: 替换 buildHandwrittenPatternCandidate 中的 primary 选择特判**

定位 `buildHandwrittenPatternCandidate` 函数，找到以下段落：

```cpp
  Operation *primary = nullptr;
  unsigned primaryPriority = 3;
  for (Operation *op : groupOps) {
    ArrayRef<OpRole> roles = getRoles(roleMap, op);
    if (!isHandwrittenPrimaryCandidate(roles))
      continue;
    if (handwrittenKind == kKernelizeHandwrittenKindAttentionSdpa) {
      if (hasRole(roles, OpRole::Cube))
        primary = op;
      continue;
    }
    unsigned priority = getHandwrittenPrimaryPriority(roles);
    if (primary && priority >= primaryPriority)
      continue;
    primary = op;
    primaryPriority = priority;
  }
```

替换为：

```cpp
  const HandwrittenContract *hwContract =
      lookupHandwrittenContract(handwrittenKind);

  Operation *primary = nullptr;
  unsigned primaryPriority = 3;
  for (Operation *op : groupOps) {
    ArrayRef<OpRole> roles = getRoles(roleMap, op);
    if (!isHandwrittenPrimaryCandidate(roles))
      continue;
    if (hwContract && !hwContract->primarySelectionRole.empty()) {
      if (stringifyOpRole(OpRole::Cube) == hwContract->primarySelectionRole &&
          hasRole(roles, OpRole::Cube))
        primary = op;
      continue;
    }
    unsigned priority = getHandwrittenPrimaryPriority(roles);
    if (primary && priority >= primaryPriority)
      continue;
    primary = op;
    primaryPriority = priority;
  }
```

- [ ] **Step 3: 编译确认**

```bash
cd /Volumes/GM9/code/Ascend-MLIR && cmake --build build --target AscendConversion 2>&1 | tail -20
```

Expected: 无 error

- [ ] **Step 4: Commit**

```bash
git add lib/Conversion/Ascend/Kernelize/FusionCandidateAnalysis.cpp
git commit -m "refactor: replace attention primary-selection if-branch in FusionCandidateAnalysis with contract lookup"
```

---

## Task 6: 将 TemplateRegistry 中的 attention 模板入口迁移至 HandwrittenContractRegistry

**Files:**
- Modify: `lib/Conversion/Ascend/Schedule/TemplateRegistry.cpp`

当前 `registerBuiltinTemplates` 中：
```cpp
reg.templates.push_back(
    {kKernelizeHandwrittenKindAttentionSdpa.str(), "grouped_tile_per_block",
     {kKernelizeHandwrittenKindAttentionSdpa.str(), kOpRoleCube.str(),
      kOpRoleReduction.str(), kOpRoleVector.str()},
     2, 4, 2});
```

目标：该模板规格已移入 `HandwrittenContractRegistry.cpp` 的 `attnContract.scheduleTemplate`，`registerBuiltinTemplates` 改为从注册表读取并 push。

- [ ] **Step 1: 添加 include**

```cpp
#include "Conversion/Ascend/Kernelize/HandwrittenContractRegistry.h"
```

- [ ] **Step 2: 替换 attention 模板硬编码行**

将硬编码的 attention 模板 push_back 替换为动态读取：

```cpp
// 旧代码（删除）
reg.templates.push_back(
    {kKernelizeHandwrittenKindAttentionSdpa.str(), "grouped_tile_per_block",
     {kKernelizeHandwrittenKindAttentionSdpa.str(), kOpRoleCube.str(),
      kOpRoleReduction.str(), kOpRoleVector.str()},
     2, 4, 2});

// 新代码（替换）
// Load templates from registered handwritten contracts so adding a new
// handwritten kind only requires a single registerHandwrittenContract() call.
{
  using namespace ::mlir::afir::ascend::kernelize;
  registerBuiltinHandwrittenContracts();
  // NOTE: gContractRegistry is a ManagedStatic and its lock is independent
  // of TemplateRegistrySingleton::mu, so locking order is always:
  // TemplateRegistrySingleton::mu first (already held here), then
  // HandwrittenContractRegistry reads are lock-free after builtins init.
  // We call lookupHandwrittenContract per kind string to avoid exposing
  // registry internals.
  for (llvm::StringRef kind :
       {kKernelizeHandwrittenKindAttentionSdpa}) {
    const HandwrittenContract *contract = lookupHandwrittenContract(kind);
    if (!contract)
      continue;
    const HandwrittenContract::TemplateSpec &spec = contract->scheduleTemplate;
    ScheduleTemplate tmpl;
    tmpl.name = spec.name;
    tmpl.family = spec.layout;
    for (const std::string &tag : spec.tags)
      tmpl.tags.push_back(tag);
    tmpl.minRank = spec.minRank;
    tmpl.maxRank = spec.maxRank;
    tmpl.priority = spec.priority;
    reg.templates.push_back(std::move(tmpl));
  }
}
```

**注意**：`ScheduleTemplate` 的字段顺序需与 `ScheduleTypes.h` 中的定义一致。在执行此步之前先确认 `ScheduleTemplate` 的字段：

```bash
grep -A 10 "struct ScheduleTemplate" lib/Conversion/Ascend/Schedule/ScheduleTypes.h
```

根据实际字段顺序调整上面的赋值。

- [ ] **Step 3: 编译确认**

```bash
cd /Volumes/GM9/code/Ascend-MLIR && cmake --build build --target AscendConversion 2>&1 | tail -20
```

Expected: 无 error

- [ ] **Step 4: Commit**

```bash
git add lib/Conversion/Ascend/Schedule/TemplateRegistry.cpp
git commit -m "refactor: load attention schedule template from HandwrittenContractRegistry"
```

---

## Task 7: 编写单元测试

**Files:**
- Create: `test/unittests/Conversion/AscendHandwrittenContractRegistryTest.cpp`
- Modify: `test/unittests/Conversion/CMakeLists.txt`

- [ ] **Step 1: 创建测试文件**

```cpp
//===- AscendHandwrittenContractRegistryTest.cpp --------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/HandwrittenContractRegistry.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "gtest/gtest.h"

using namespace mlir::afir::ascend;
using namespace mlir::afir::ascend::kernelize;

TEST(HandwrittenContractRegistryTest, LookupUnknownKindReturnsNull) {
  EXPECT_EQ(lookupHandwrittenContract("nonexistent_kind"), nullptr);
}

TEST(HandwrittenContractRegistryTest, LookupEmptyKindReturnsNull) {
  EXPECT_EQ(lookupHandwrittenContract(""), nullptr);
}

TEST(HandwrittenContractRegistryTest, AttentionSdpaIsRegisteredByDefault) {
  const HandwrittenContract *contract =
      lookupHandwrittenContract(kKernelizeHandwrittenKindAttentionSdpa);
  ASSERT_NE(contract, nullptr);
}

TEST(HandwrittenContractRegistryTest, AttentionSdpaHasExpectedTopologyConstraints) {
  const HandwrittenContract *contract =
      lookupHandwrittenContract(kKernelizeHandwrittenKindAttentionSdpa);
  ASSERT_NE(contract, nullptr);
  EXPECT_EQ(contract->minCubeCount, 2u);
  EXPECT_EQ(contract->maxCubeCount, 2u);
  EXPECT_TRUE(contract->requiresReduction);
  EXPECT_TRUE(contract->requiresVectorInjective);
}

TEST(HandwrittenContractRegistryTest, AttentionSdpaUsesAxisCarrierOnly) {
  const HandwrittenContract *contract =
      lookupHandwrittenContract(kKernelizeHandwrittenKindAttentionSdpa);
  ASSERT_NE(contract, nullptr);
  EXPECT_TRUE(contract->useAxisCarrierOnly);
}

TEST(HandwrittenContractRegistryTest, AttentionSdpaPrimarySelectionIsCube) {
  const HandwrittenContract *contract =
      lookupHandwrittenContract(kKernelizeHandwrittenKindAttentionSdpa);
  ASSERT_NE(contract, nullptr);
  EXPECT_EQ(contract->primarySelectionRole, "Cube");
}

TEST(HandwrittenContractRegistryTest, AttentionSdpaHasExpectedStructureConstraints) {
  const HandwrittenContract *contract =
      lookupHandwrittenContract(kKernelizeHandwrittenKindAttentionSdpa);
  ASSERT_NE(contract, nullptr);
  ASSERT_EQ(contract->structureConstraints.size(), 2u);
  EXPECT_EQ(contract->structureConstraints[0], "handwritten_group");
  EXPECT_EQ(contract->structureConstraints[1], "attention_sdpa_chain");
}

TEST(HandwrittenContractRegistryTest, AttentionSdpaHasScheduleTemplate) {
  const HandwrittenContract *contract =
      lookupHandwrittenContract(kKernelizeHandwrittenKindAttentionSdpa);
  ASSERT_NE(contract, nullptr);
  EXPECT_EQ(contract->scheduleTemplate.name,
            kKernelizeHandwrittenKindAttentionSdpa.str());
  EXPECT_EQ(contract->scheduleTemplate.layout, "grouped_tile_per_block");
  EXPECT_EQ(contract->scheduleTemplate.minRank, 2u);
  EXPECT_EQ(contract->scheduleTemplate.maxRank, 4u);
  EXPECT_EQ(contract->scheduleTemplate.priority, 2);
}

TEST(HandwrittenContractRegistryTest, CustomContractCanBeRegistered) {
  registerHandwrittenContract("custom_test_kind", HandwrittenContract{
    .minCubeCount = 1,
    .maxCubeCount = 3,
    .requiresReduction = false,
    .requiresVectorInjective = false,
    .useAxisCarrierOnly = false,
    .primarySelectionRole = "",
    .structureConstraints = {"custom_constraint"},
  });

  const HandwrittenContract *contract =
      lookupHandwrittenContract("custom_test_kind");
  ASSERT_NE(contract, nullptr);
  EXPECT_EQ(contract->maxCubeCount, 3u);
  ASSERT_EQ(contract->structureConstraints.size(), 1u);
  EXPECT_EQ(contract->structureConstraints[0], "custom_constraint");
}

TEST(HandwrittenContractRegistryTest, DuplicateRegistrationIsNoOp) {
  HandwrittenContract first;
  first.minCubeCount = 1;
  first.structureConstraints.push_back("first");
  registerHandwrittenContract("dedup_test_kind", first);

  HandwrittenContract second;
  second.minCubeCount = 99;
  second.structureConstraints.push_back("second");
  registerHandwrittenContract("dedup_test_kind", second);  // should be ignored

  const HandwrittenContract *contract =
      lookupHandwrittenContract("dedup_test_kind");
  ASSERT_NE(contract, nullptr);
  EXPECT_EQ(contract->minCubeCount, 1u);  // first registration wins
  ASSERT_EQ(contract->structureConstraints.size(), 1u);
  EXPECT_EQ(contract->structureConstraints[0], "first");
}
```

- [ ] **Step 2: 将测试加入 CMakeLists.txt**

在 `test/unittests/Conversion/CMakeLists.txt` 末尾（最后一个 `add_test` 之后）加入：

```cmake
add_executable(AscendHandwrittenContractRegistryTest
  AscendHandwrittenContractRegistryTest.cpp
)

target_include_directories(AscendHandwrittenContractRegistryTest PRIVATE
  ${ASCEND_CONVERSION_INTERNAL_INCLUDE_DIR}
)

target_link_libraries(AscendHandwrittenContractRegistryTest PRIVATE
  RuntimeUnitTestSupport
  AscendConversion
)

add_dependencies(RuntimeUnitTests AscendHandwrittenContractRegistryTest)

add_test(NAME AscendHandwrittenContractRegistryTest
  COMMAND $<TARGET_FILE:AscendHandwrittenContractRegistryTest>
)
```

- [ ] **Step 3: 运行测试**

```bash
cd /Volumes/GM9/code/Ascend-MLIR && cmake --build build --target AscendHandwrittenContractRegistryTest 2>&1 | tail -20
```

Expected: 编译成功

```bash
cd /Volumes/GM9/code/Ascend-MLIR && ./build/test/unittests/Conversion/AscendHandwrittenContractRegistryTest 2>&1
```

Expected:
```
[==========] Running 9 tests from 1 test suite.
[----------] 9 tests from HandwrittenContractRegistryTest
[ RUN      ] HandwrittenContractRegistryTest.LookupUnknownKindReturnsNull
[       OK ] ...
...
[==========] 9 tests from 1 test suite ran.
[  PASSED  ] 9 tests.
```

- [ ] **Step 4: Commit**

```bash
git add test/unittests/Conversion/AscendHandwrittenContractRegistryTest.cpp \
        test/unittests/Conversion/CMakeLists.txt
git commit -m "test: add HandwrittenContractRegistry unit tests"
```

---

## Task 8: 运行完整测试套件验证无回归

- [ ] **Step 1: 构建所有测试目标**

```bash
cd /Volumes/GM9/code/Ascend-MLIR && cmake --build build --target RuntimeUnitTests 2>&1 | tail -30
```

Expected: 无 error

- [ ] **Step 2: 运行单元测试**

```bash
cd /Volumes/GM9/code/Ascend-MLIR/build && ctest -L RuntimeUnitTests --output-on-failure 2>&1
```

Expected: 所有测试 PASS，包括：
- `AscendKernelPatternTest`
- `AscendScheduleDecisionTest`
- `AscendHandwrittenContractRegistryTest`
- `AscendKernelizeOpInterfaceTest`
- 其他已有测试

- [ ] **Step 3: 如有失败，诊断并修复**

最常见的失败原因：
1. `ScheduleTemplate` 字段顺序与 `ScheduleTypes.h` 不一致 → 核对 Task 6 Step 2 的赋值顺序
2. `using namespace` 引入的命名冲突 → 改用完整限定名
3. `ManagedStatic` 初始化顺序问题 → 确认 `registerBuiltinHandwrittenContracts` 在 `registerBuiltinTemplates` 前被调用

- [ ] **Step 4: 最终 commit（如 Task 5-7 中间有修复补丁）**

```bash
git add -p  # 逐块确认
git commit -m "fix: address post-refactor test failures"
```

---

## Self-Review

### Spec 覆盖检查

| 需求 | 对应 Task |
|---|---|
| HandwrittenContract 结构体定义 | Task 1 |
| 注册表实现（线程安全，singleton） | Task 2 |
| attention_sdpa 内置契约注册 | Task 2 |
| AxisCoalescer if-else 消除 | Task 3 |
| ScheduleProblemBuilder if-else 消除 | Task 4 |
| FusionCandidateAnalysis if-else 消除 | Task 5 |
| TemplateRegistry 模板入口统一 | Task 6 |
| 单元测试 | Task 7 |
| 全量回归验证 | Task 8 |

### 类型一致性检查

- `HandwrittenContract::TemplateSpec` 在 Task 2 定义并在 Task 6 消费——字段名完全一致（`name`, `layout`, `tags`, `minRank`, `maxRank`, `priority`）。
- `lookupHandwrittenContract` 返回 `const HandwrittenContract *`，所有消费方在 Task 3/4/5 均做了 null 检查。
- `registerHandwrittenContract` 使用 `try_emplace` 实现幂等性，Task 7 的 `DuplicateRegistrationIsNoOp` 测试验证了此行为。

### Placeholder 扫描

无 TBD / TODO / "similar to Task N" 等占位符。Task 6 Step 2 中有一处"根据实际字段顺序调整"的提示——这是必要的运行时核对步骤，不是 placeholder（原因：`ScheduleTemplate` 字段在此计划写作时未完整读取）。执行者在 Step 2 前应先运行 `grep -A 10 "struct ScheduleTemplate"` 确认。

### 注意事项

- `collectAttentionLikeHandwrittenPattern`（图拓扑匹配函数）**本次不修改**，其 `cubeCount == 2` 等约束是结构识别逻辑，泛化为契约驱动是第二阶段工作。
- `ScheduleTypes.h` 中的 `using ::mlir::afir::ascend::kKernelizeHandwrittenKindAttentionSdpa` 保留不动，因为它仍被 `TemplateRegistry.cpp`（Task 6 的 for 循环）引用。
