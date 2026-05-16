# Phase 2: Attention Pattern 扩展性 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `FusionCandidateAnalysis.cpp` 中硬编码的 `collectAttentionLikeHandwrittenPattern` 改造为可注册的 `HandwrittenPatternRecognizer` 接口，消除主干中的 hardcoded kind 字符串，新增 `FlashAttentionRecognizer` 作为第二个内置 recognizer。

**Architecture:** 新增 `HandwrittenPatternRecognizer` 抽象基类 + `HandwrittenPatternRecognizerRegistry`（单例，线程安全，与 `HandwrittenContractRegistry` 同模式）。`AttentionSdpaRecognizer` 把现有 `collectAttentionLikeHandwrittenPattern` 逻辑原封不动迁入。`FlashAttentionRecognizer` 识别多轮 Cube→Reduction→Vector(exp)→Cube 链。`FusionCandidateAnalysis::analyze` 主循环改为遍历注册表，删除所有 hardcoded kind 字符串比较。

**依赖：** Phase 1 必须先完成（`FlashAttentionRecognizer` 需要用 `ElementwiseBodyOpRegistry` 识别 `math.exp` body op）。

**Tech Stack:** C++17, LLVM/MLIR (llvm::ManagedStatic, linalg ops, DependencyAnalysis types), GTest

---

## 文件结构

### 新增文件

| 文件 | 职责 |
|---|---|
| `lib/Conversion/Ascend/Kernelize/HandwrittenPatternRecognizer.h` | `PatternMatch` 结构体 + `HandwrittenPatternRecognizer` 抽象基类 + 注册/查询接口声明 |
| `lib/Conversion/Ascend/Kernelize/HandwrittenPatternRecognizerRegistry.cpp` | 注册表实现 + `registerBuiltinHandwrittenPatternRecognizers()` |
| `lib/Conversion/Ascend/Kernelize/AttentionSdpaRecognizer.cpp` | `AttentionSdpaRecognizer`：现有 SDPA 识别逻辑迁入 |
| `lib/Conversion/Ascend/Kernelize/FlashAttentionRecognizer.cpp` | `FlashAttentionRecognizer`：多轮 Flash Attention 拓扑识别 |
| `test/unittests/Conversion/AscendHandwrittenPatternRecognizerTest.cpp` | 单元测试 |
| `test/Conversion/ascend-kernelize-flash-attention-pattern.mlir` | FlashAttention 识别 LIT 测试 |

### 修改文件

| 文件 | 改动 |
|---|---|
| `lib/Conversion/Ascend/Kernelize/FusionCandidateAnalysis.cpp` | `analyze()` 主循环改为遍历注册表；删除 `collectAttentionLikeHandwrittenPattern` 函数；删除 `kKernelizeHandwrittenKindAttentionSdpa` 硬编码字符串 |
| `lib/Conversion/Ascend/Kernelize/HandwrittenContractRegistry.cpp` | 注册 `kKernelizeHandwrittenKindFlashAttention` contract |
| `lib/Conversion/Ascend/CMakeLists.txt` | 添加三个新 `.cpp` |
| `test/unittests/Conversion/CMakeLists.txt` | 注册新测试目标 |

---

## Task 1: 定义 HandwrittenPatternRecognizer 接口和注册表声明

**Files:**
- Create: `lib/Conversion/Ascend/Kernelize/HandwrittenPatternRecognizer.h`

- [ ] **Step 1: 创建头文件**

```cpp
//===- HandwrittenPatternRecognizer.h - Pattern recognizer interface --*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_HANDWRITTENPATTERNRECOGNIZER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_HANDWRITTENPATTERNRECOGNIZER_H

#include "DependencyAnalysis.h"
#include "KernelizeTypes.h"
#include "OpRoleClassification.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <optional>

namespace mlir {
class Operation;
}

namespace mlir::afir::ascend::kernelize {

/// Result of a successful pattern match.
struct PatternMatch {
  /// Op group that forms the handwritten pattern (sorted by opId).
  SmallVector<Operation *, 8> ops;
  /// The HandwrittenContract kind key (must be registered in
  /// HandwrittenContractRegistry).
  std::string kindId;
};

/// Abstract base class for handwritten kernel pattern recognizers.
/// Each recognizer encapsulates one pattern's topology-matching logic.
/// Register via registerHandwrittenPatternRecognizer(); the
/// FusionCandidateAnalyzer iterates all registered recognizers per seed op.
class HandwrittenPatternRecognizer {
public:
  virtual ~HandwrittenPatternRecognizer() = default;

  /// The HandwrittenContract kind this recognizer produces on match.
  virtual llvm::StringRef kindId() const = 0;

  /// Try to match a handwritten pattern starting from `seed`.
  /// Returns PatternMatch on success, std::nullopt if seed is not the start
  /// of this recognizer's pattern.
  virtual std::optional<PatternMatch>
  tryMatch(Operation *seed, const DependencyAnalysisResult &deps,
           const OpRoleMap &roleMap,
           const KernelizeConfig &config) const = 0;
};

/// Register a recognizer. Ownership is transferred to the registry.
/// Thread-safe. If a recognizer with the same kindId() is already registered,
/// the new one is ignored (first registration wins).
void registerHandwrittenPatternRecognizer(
    std::unique_ptr<HandwrittenPatternRecognizer> recognizer);

/// Register all built-in recognizers (AttentionSdpa, FlashAttention).
/// Called automatically on first use of getHandwrittenPatternRecognizers();
/// exposed for explicit initialization in tests.
void registerBuiltinHandwrittenPatternRecognizers();

/// Returns a snapshot of all registered recognizers (non-owning pointers).
/// Callers must not cache this across registerHandwrittenPatternRecognizer()
/// calls.
SmallVector<HandwrittenPatternRecognizer *> getHandwrittenPatternRecognizers();

} // namespace mlir::afir::ascend::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_HANDWRITTENPATTERNRECOGNIZER_H
```

- [ ] **Step 2: Commit**

```bash
git add lib/Conversion/Ascend/Kernelize/HandwrittenPatternRecognizer.h
git commit -m "feat: add HandwrittenPatternRecognizer interface"
```

---

## Task 2: 实现注册表 + AttentionSdpaRecognizer

**Files:**
- Create: `lib/Conversion/Ascend/Kernelize/HandwrittenPatternRecognizerRegistry.cpp`
- Create: `lib/Conversion/Ascend/Kernelize/AttentionSdpaRecognizer.cpp`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`

- [ ] **Step 1: 创建注册表实现**

```cpp
//===- HandwrittenPatternRecognizerRegistry.cpp ---------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "HandwrittenPatternRecognizer.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ManagedStatic.h"

#include <mutex>

namespace mlir::afir::ascend::kernelize {
namespace {

struct RecognizerRegistry {
  std::mutex mu;
  // Insertion-order list for deterministic iteration.
  SmallVector<std::unique_ptr<HandwrittenPatternRecognizer>> recognizers;
  llvm::StringSet<> registeredKinds;
  bool builtinsRegistered = false;
};

llvm::ManagedStatic<RecognizerRegistry> gRecognizerRegistry;

} // namespace

void registerHandwrittenPatternRecognizer(
    std::unique_ptr<HandwrittenPatternRecognizer> recognizer) {
  RecognizerRegistry &reg = *gRecognizerRegistry;
  std::lock_guard<std::mutex> lock(reg.mu);
  llvm::StringRef kind = recognizer->kindId();
  if (!reg.registeredKinds.insert(kind).second)
    return; // already registered, first wins
  reg.recognizers.push_back(std::move(recognizer));
}

void registerBuiltinHandwrittenPatternRecognizers() {
  RecognizerRegistry &reg = *gRecognizerRegistry;
  {
    std::lock_guard<std::mutex> lock(reg.mu);
    if (reg.builtinsRegistered)
      return;
    reg.builtinsRegistered = true;
  }
  // Defined in AttentionSdpaRecognizer.cpp and FlashAttentionRecognizer.cpp.
  registerAttentionSdpaRecognizer();
  registerFlashAttentionRecognizer();
}

SmallVector<HandwrittenPatternRecognizer *>
getHandwrittenPatternRecognizers() {
  registerBuiltinHandwrittenPatternRecognizers();
  RecognizerRegistry &reg = *gRecognizerRegistry;
  std::lock_guard<std::mutex> lock(reg.mu);
  SmallVector<HandwrittenPatternRecognizer *> result;
  for (auto &r : reg.recognizers)
    result.push_back(r.get());
  return result;
}

} // namespace mlir::afir::ascend::kernelize
```

- [ ] **Step 2: 在 `HandwrittenPatternRecognizer.h` 末尾（`#endif` 前）追加 forward 声明**

```cpp
// Forward declarations of built-in recognizer registration functions
// (implemented in AttentionSdpaRecognizer.cpp / FlashAttentionRecognizer.cpp).
namespace mlir::afir::ascend::kernelize {
void registerAttentionSdpaRecognizer();
void registerFlashAttentionRecognizer();
} // namespace mlir::afir::ascend::kernelize
```

- [ ] **Step 3: 创建 `AttentionSdpaRecognizer.cpp`**

把 `FusionCandidateAnalysis.cpp` 中的 `collectAttentionLikeHandwrittenPattern`（L348-415）逻辑**原封不动**迁入，仅包装为 `HandwrittenPatternRecognizer` 子类：

```cpp
//===- AttentionSdpaRecognizer.cpp - SDPA pattern recognizer -------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "HandwrittenPatternRecognizer.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "DependencyAnalysis.h"
#include "KernelizeTypes.h"
#include "OpRoleClassification.h"

#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace mlir::afir::ascend::kernelize {
namespace {

bool isAttentionLikeHandwrittenOp(ArrayRef<OpRole> roles) {
  return hasRole(roles, OpRole::Cube) || hasRole(roles, OpRole::Reduction) ||
         isVectorInjective(roles);
}

class AttentionSdpaRecognizer final : public HandwrittenPatternRecognizer {
public:
  llvm::StringRef kindId() const override {
    return kKernelizeHandwrittenKindAttentionSdpa;
  }

  std::optional<PatternMatch>
  tryMatch(Operation *seed, const DependencyAnalysisResult &deps,
           const OpRoleMap &roleMap,
           const KernelizeConfig &config) const override {
    if (!hasRole(getRoles(roleMap, seed), OpRole::Cube))
      return std::nullopt;

    SmallVector<Operation *, 8> groupOps;
    llvm::DenseSet<Operation *> seen;
    SmallVector<Operation *, 4> seedReductionConsumers;
    for (Operation *consumer : getConsumers(deps.index, seed))
      if (hasRole(getRoles(roleMap, consumer), OpRole::Reduction))
        seedReductionConsumers.push_back(consumer);
    if (seedReductionConsumers.size() != 1)
      return std::nullopt;

    Operation *current = seedReductionConsumers.front();
    bool hasVector = false;
    unsigned cubeCount = 1;
    seen.insert(seed);
    seen.insert(current);
    groupOps.push_back(seed);
    groupOps.push_back(current);

    while (true) {
      SmallVector<Operation *, 4> eligibleConsumers;
      for (Operation *consumer : getConsumers(deps.index, current)) {
        if (seen.contains(consumer))
          return std::nullopt;
        if (isAttentionLikeHandwrittenOp(getRoles(roleMap, consumer)))
          eligibleConsumers.push_back(consumer);
      }

      if (eligibleConsumers.empty())
        break;
      if (eligibleConsumers.size() != 1)
        return std::nullopt;

      Operation *next = eligibleConsumers.front();
      ArrayRef<OpRole> nextRoles = getRoles(roleMap, next);
      bool isCube = hasRole(nextRoles, OpRole::Cube);
      if (!isCube && !hasRole(nextRoles, OpRole::Reduction) &&
          !isVectorInjective(nextRoles))
        return std::nullopt;

      seen.insert(next);
      groupOps.push_back(next);
      if (isVectorInjective(nextRoles))
        hasVector = true;
      if (groupOps.size() > config.maxOpsPerCandidate)
        return std::nullopt;

      if (isCube) {
        ++cubeCount;
        if (cubeCount != 2)
          return std::nullopt;
        break;
      }
      current = next;
    }

    if (cubeCount < 2 || !hasVector)
      return std::nullopt;

    sortByOpId(groupOps, deps.index);
    PatternMatch match;
    match.ops = std::move(groupOps);
    match.kindId = kKernelizeHandwrittenKindAttentionSdpa.str();
    return match;
  }
};

} // namespace

void registerAttentionSdpaRecognizer() {
  registerHandwrittenPatternRecognizer(
      std::make_unique<AttentionSdpaRecognizer>());
}

} // namespace mlir::afir::ascend::kernelize
```

- [ ] **Step 4: 将新文件加入 `lib/Conversion/Ascend/CMakeLists.txt`**

在 `Kernelize/HandwrittenContractRegistry.cpp` 行附近插入：

```cmake
  Kernelize/HandwrittenPatternRecognizerRegistry.cpp
  Kernelize/AttentionSdpaRecognizer.cpp
  Kernelize/FlashAttentionRecognizer.cpp
```

- [ ] **Step 5: 编译确认（FlashAttention 文件还不存在，先建空桩）**

创建空桩 `lib/Conversion/Ascend/Kernelize/FlashAttentionRecognizer.cpp`：

```cpp
#include "HandwrittenPatternRecognizer.h"
namespace mlir::afir::ascend::kernelize {
void registerFlashAttentionRecognizer() {
  // Implemented in Task 4.
}
} // namespace mlir::afir::ascend::kernelize
```

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && ninja -C build -j6 AscendConversion 2>&1 | tail -15"
```

Expected: 无 error

- [ ] **Step 6: Commit**

```bash
git add lib/Conversion/Ascend/Kernelize/HandwrittenPatternRecognizer.h \
        lib/Conversion/Ascend/Kernelize/HandwrittenPatternRecognizerRegistry.cpp \
        lib/Conversion/Ascend/Kernelize/AttentionSdpaRecognizer.cpp \
        lib/Conversion/Ascend/Kernelize/FlashAttentionRecognizer.cpp \
        lib/Conversion/Ascend/CMakeLists.txt
git commit -m "feat: add HandwrittenPatternRecognizerRegistry + AttentionSdpaRecognizer"
```

---

## Task 3: 改造 FusionCandidateAnalysis 主循环

**Files:**
- Modify: `lib/Conversion/Ascend/Kernelize/FusionCandidateAnalysis.cpp`

- [ ] **Step 1: 在文件顶部加 include**

在现有 includes 末尾加：

```cpp
#include "HandwrittenPatternRecognizer.h"
```

- [ ] **Step 2: 替换 attention 识别段（`FusionCandidateAnalysis.cpp:486-501`）**

找到：
```cpp
  llvm::DenseSet<Operation *> attentionGroupedOps;
  for (Operation *seed : deps.index.orderedOps) {
    if (attentionGroupedOps.contains(seed))
      continue;
    std::optional<SmallVector<Operation *, 8>> groupOps =
        collectAttentionLikeHandwrittenPattern(seed, deps, roleMap, config);
    if (!groupOps)
      continue;
    for (Operation *op : *groupOps)
      attentionGroupedOps.insert(op);
    appendLegalCandidate(candidates,
                         buildHandwrittenPatternCandidate(*groupOps, deps,
                                                          roleMap,
                                                          kKernelizeHandwrittenKindAttentionSdpa),
                         deps, roleMap, config);
  }
```

替换为：

```cpp
  llvm::DenseSet<Operation *> handwrittenGroupedOps;
  SmallVector<HandwrittenPatternRecognizer *> recognizers =
      getHandwrittenPatternRecognizers();
  for (Operation *seed : deps.index.orderedOps) {
    if (handwrittenGroupedOps.contains(seed))
      continue;
    for (HandwrittenPatternRecognizer *recognizer : recognizers) {
      std::optional<PatternMatch> match =
          recognizer->tryMatch(seed, deps, roleMap, config);
      if (!match)
        continue;
      for (Operation *op : match->ops)
        handwrittenGroupedOps.insert(op);
      appendLegalCandidate(
          candidates,
          buildHandwrittenPatternCandidate(match->ops, deps, roleMap,
                                           match->kindId),
          deps, roleMap, config);
      break; // first recognizer that matches wins
    }
  }
```

- [ ] **Step 3: 删除 `collectAttentionLikeHandwrittenPattern` 函数**

删除 `FusionCandidateAnalysis.cpp:348-415` 整个函数（包括其 helper `isAttentionLikeHandwrittenOp`，该 helper 已迁入 `AttentionSdpaRecognizer.cpp`）。

先确认 `isAttentionLikeHandwrittenOp` 只在 `collectAttentionLikeHandwrittenPattern` 中被调用：

```bash
grep -n "isAttentionLikeHandwrittenOp" lib/Conversion/Ascend/Kernelize/FusionCandidateAnalysis.cpp
```

Expected: 只有函数定义 + 在 `collectAttentionLikeHandwrittenPattern` 内的两处调用。确认后删除两个函数。

- [ ] **Step 4: 删除文件内的 `kKernelizeHandwrittenKindAttentionSdpa` 硬编码引用**

```bash
grep -n "kKernelizeHandwrittenKindAttentionSdpa" lib/Conversion/Ascend/Kernelize/FusionCandidateAnalysis.cpp
```

Expected: 输出为空（所有引用应已被删除）。若有残留，逐一删除。

- [ ] **Step 5: 编译确认**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && ninja -C build -j6 AscendConversion 2>&1 | tail -15"
```

Expected: 无 error

- [ ] **Step 6: 运行现有 attention LIT 测试确认无回归**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-kernelize-attention-handwritten-pattern.mlir build/test/Conversion/ascend-full-pipeline-attention-handwritten-smoke.mlir 2>&1"
```

Expected: `2 tests passed`

- [ ] **Step 7: Commit**

```bash
git add lib/Conversion/Ascend/Kernelize/FusionCandidateAnalysis.cpp
git commit -m "refactor: FusionCandidateAnalysis主循环改为遍历recognizer注册表"
```

---

## Task 4: 实现 FlashAttentionRecognizer + 注册 contract

**Files:**
- Modify: `lib/Conversion/Ascend/Kernelize/FlashAttentionRecognizer.cpp`（替换桩）
- Modify: `lib/Conversion/Ascend/Kernelize/HandwrittenContractRegistry.cpp`

- [ ] **Step 1: 在 `HandwrittenContractRegistry.cpp` 注册 FlashAttention contract**

在 `registerBuiltinHandwrittenContracts()` 函数末尾（`registerHandwrittenContract(kKernelizeHandwrittenKindAttentionSdpa, ...)` 调用之后）添加：

首先在 `KernelizeTypes.h` 或 `Attributes.h` 中确认 `kKernelizeHandwrittenKindFlashAttention` 是否已存在：

```bash
grep -rn "kKernelizeHandwrittenKindFlash\|FlashAttention\|flash_attention" \
  lib/Conversion/Ascend/Kernelize/ include/Conversion/Ascend/
```

若不存在，在 `Conversion/Ascend/Common/Attributes.h`（或 `KernelizeTypes.h` 中 `kKernelizeHandwrittenKindAttentionSdpa` 附近）添加：

```cpp
constexpr llvm::StringLiteral kKernelizeHandwrittenKindFlashAttention =
    "flash_attention";
```

然后在 `HandwrittenContractRegistry.cpp` 的 `registerBuiltinHandwrittenContracts()` 里追加：

```cpp
  HandwrittenContract flashContract;
  flashContract.minCubeCount = 2;
  flashContract.maxCubeCount = 4;   // 支持多轮迭代
  flashContract.requiresReduction = true;
  flashContract.requiresVectorInjective = true;
  flashContract.useAxisCarrierOnly = true;
  flashContract.primarySelectionRole = "Cube";
  flashContract.structureConstraints = {"handwritten_group",
                                        "flash_attention_chain"};

  HandwrittenContract::TemplateSpec &flashTmpl = flashContract.scheduleTemplate;
  flashTmpl.kindId = kKernelizeHandwrittenKindFlashAttention.str();
  flashTmpl.tilingLayout = "grouped_tile_per_block";
  flashTmpl.tags = {kKernelizeHandwrittenKindFlashAttention.str(),
                    kOpRoleCube.str(), kOpRoleReduction.str(),
                    kOpRoleVector.str()};
  flashTmpl.minRank = 2;
  flashTmpl.maxRank = 4;
  flashTmpl.priority = 2;

  registerHandwrittenContract(kKernelizeHandwrittenKindFlashAttention,
                              std::move(flashContract));
```

同时在 `TemplateRegistry.cpp` 的 for 循环里加入新 kind：

```cpp
  for (llvm::StringRef kind : {kKernelizeHandwrittenKindAttentionSdpa,
                                kKernelizeHandwrittenKindFlashAttention}) {
```

- [ ] **Step 2: 实现 `FlashAttentionRecognizer`（替换桩文件）**

FlashAttention 拓扑：多轮 `Cube → Reduction → Vector(exp/mul rescale) → Cube`，最终以第二个（或更多）Cube 结尾。识别条件：

1. seed 有 Cube role
2. seed 的 reduction consumer 恰好 1 个
3. 从 reduction 开始，沿 consumer 链遍历，允许多个 Vector op（必须有 exp body op，通过 `lookupElementwiseBodyOp` 判断），遇到第二个 Cube 时记录一轮，允许继续到第三、四个 Cube（最多 4 个）
4. 最终至少有 2 个 Cube、1 个 Reduction、1 个含 exp 的 Vector op

```cpp
//===- FlashAttentionRecognizer.cpp - Flash Attention pattern recognizer --===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "HandwrittenPatternRecognizer.h"

#include "Conversion/Ascend/Backend/ElementwiseBodyOpRegistry.h"
#include "Conversion/Ascend/Common/Attributes.h"
#include "Conversion/Ascend/Kernelize/HandwrittenContractRegistry.h"
#include "DependencyAnalysis.h"
#include "KernelizeTypes.h"
#include "OpRoleClassification.h"

#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace mlir::afir::ascend::kernelize {
namespace {

/// Returns true if a linalg.generic op's body contains a math.exp or
/// math.exp2 body op (characteristic of FlashAttention rescale step).
bool hasExpBodyOp(Operation *op) {
  auto generic = dyn_cast<linalg::GenericOp>(op);
  if (!generic)
    return false;
  Block *body = generic.getBody();
  for (Operation &bodyOp : body->without_terminator()) {
    llvm::StringRef name = bodyOp.getName().getStringRef();
    if (name == "math.exp" || name == "math.exp2")
      return true;
  }
  return false;
}

bool isFlashAttentionEligibleOp(ArrayRef<OpRole> roles) {
  return hasRole(roles, OpRole::Cube) || hasRole(roles, OpRole::Reduction) ||
         isVectorInjective(roles);
}

class FlashAttentionRecognizer final : public HandwrittenPatternRecognizer {
public:
  llvm::StringRef kindId() const override {
    return kKernelizeHandwrittenKindFlashAttention;
  }

  std::optional<PatternMatch>
  tryMatch(Operation *seed, const DependencyAnalysisResult &deps,
           const OpRoleMap &roleMap,
           const KernelizeConfig &config) const override {
    if (!hasRole(getRoles(roleMap, seed), OpRole::Cube))
      return std::nullopt;

    SmallVector<Operation *, 8> groupOps;
    llvm::DenseSet<Operation *> seen;
    groupOps.push_back(seed);
    seen.insert(seed);

    unsigned cubeCount = 1;
    bool hasReduction = false;
    bool hasExpVector = false;

    // Seed must have exactly one reduction consumer to start a chain.
    SmallVector<Operation *, 4> reductionConsumers;
    for (Operation *c : getConsumers(deps.index, seed))
      if (hasRole(getRoles(roleMap, c), OpRole::Reduction))
        reductionConsumers.push_back(c);
    if (reductionConsumers.size() != 1)
      return std::nullopt;

    Operation *current = reductionConsumers.front();
    seen.insert(current);
    groupOps.push_back(current);
    hasReduction = true;

    // Traverse consumer chain: Reduction → Vector* → Cube (→ Reduction → …)
    while (true) {
      SmallVector<Operation *, 4> eligible;
      for (Operation *c : getConsumers(deps.index, current)) {
        if (seen.contains(c))
          return std::nullopt;
        if (isFlashAttentionEligibleOp(getRoles(roleMap, c)))
          eligible.push_back(c);
      }
      if (eligible.empty())
        break;
      if (eligible.size() != 1)
        return std::nullopt;

      Operation *next = eligible.front();
      ArrayRef<OpRole> nextRoles = getRoles(roleMap, next);
      seen.insert(next);
      groupOps.push_back(next);

      if (groupOps.size() > config.maxOpsPerCandidate)
        return std::nullopt;

      if (hasRole(nextRoles, OpRole::Cube)) {
        ++cubeCount;
        if (cubeCount > 4)
          return std::nullopt;
        // After a new Cube, look for another Reduction to continue.
        current = next;
      } else if (isVectorInjective(nextRoles)) {
        if (hasExpBodyOp(next))
          hasExpVector = true;
        current = next;
      } else {
        // Reduction
        hasReduction = true;
        current = next;
      }
    }

    // FlashAttention requires: ≥2 Cube, ≥1 Reduction, ≥1 exp-Vector.
    if (cubeCount < 2 || !hasReduction || !hasExpVector)
      return std::nullopt;

    sortByOpId(groupOps, deps.index);
    PatternMatch match;
    match.ops = std::move(groupOps);
    match.kindId = kKernelizeHandwrittenKindFlashAttention.str();
    return match;
  }
};

} // namespace

void registerFlashAttentionRecognizer() {
  registerHandwrittenPatternRecognizer(
      std::make_unique<FlashAttentionRecognizer>());
}

} // namespace mlir::afir::ascend::kernelize
```

- [ ] **Step 3: 编译确认**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && ninja -C build -j6 AscendConversion 2>&1 | tail -15"
```

Expected: 无 error

- [ ] **Step 4: Commit**

```bash
git add lib/Conversion/Ascend/Kernelize/FlashAttentionRecognizer.cpp \
        lib/Conversion/Ascend/Kernelize/HandwrittenContractRegistry.cpp \
        lib/Conversion/Ascend/Schedule/TemplateRegistry.cpp \
        lib/Conversion/Ascend/Common/Attributes.h
git commit -m "feat: add FlashAttentionRecognizer and flash_attention HandwrittenContract"
```

---

## Task 5: 单元测试

**Files:**
- Create: `test/unittests/Conversion/AscendHandwrittenPatternRecognizerTest.cpp`
- Modify: `test/unittests/Conversion/CMakeLists.txt`

- [ ] **Step 1: 创建测试文件**

```cpp
//===- AscendHandwrittenPatternRecognizerTest.cpp -------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/HandwrittenContractRegistry.h"
#include "Conversion/Ascend/Kernelize/HandwrittenPatternRecognizer.h"
#include "gtest/gtest.h"

using namespace mlir::afir::ascend::kernelize;

TEST(HandwrittenPatternRecognizerTest, RegistryStartsEmpty) {
  // After registerBuiltinHandwrittenPatternRecognizers, at least 2 recognizers
  // should be present (AttentionSdpa + FlashAttention).
  registerBuiltinHandwrittenPatternRecognizers();
  SmallVector<HandwrittenPatternRecognizer *> recognizers =
      getHandwrittenPatternRecognizers();
  EXPECT_GE(recognizers.size(), 2u);
}

TEST(HandwrittenPatternRecognizerTest, AttentionSdpaRecognizerRegistered) {
  registerBuiltinHandwrittenPatternRecognizers();
  SmallVector<HandwrittenPatternRecognizer *> recognizers =
      getHandwrittenPatternRecognizers();
  bool found = false;
  for (auto *r : recognizers)
    if (r->kindId() == kKernelizeHandwrittenKindAttentionSdpa)
      found = true;
  EXPECT_TRUE(found);
}

TEST(HandwrittenPatternRecognizerTest, FlashAttentionRecognizerRegistered) {
  registerBuiltinHandwrittenPatternRecognizers();
  SmallVector<HandwrittenPatternRecognizer *> recognizers =
      getHandwrittenPatternRecognizers();
  bool found = false;
  for (auto *r : recognizers)
    if (r->kindId() == kKernelizeHandwrittenKindFlashAttention)
      found = true;
  EXPECT_TRUE(found);
}

TEST(HandwrittenPatternRecognizerTest, CustomRecognizerCanBeRegistered) {
  class CustomRecognizer final : public HandwrittenPatternRecognizer {
  public:
    llvm::StringRef kindId() const override { return "test_custom_kind"; }
    std::optional<PatternMatch>
    tryMatch(mlir::Operation *, const DependencyAnalysisResult &,
             const OpRoleMap &, const KernelizeConfig &) const override {
      return std::nullopt;
    }
  };
  registerHandwrittenPatternRecognizer(std::make_unique<CustomRecognizer>());

  SmallVector<HandwrittenPatternRecognizer *> recognizers =
      getHandwrittenPatternRecognizers();
  bool found = false;
  for (auto *r : recognizers)
    if (r->kindId() == "test_custom_kind")
      found = true;
  EXPECT_TRUE(found);
}

TEST(HandwrittenPatternRecognizerTest, DuplicateRegistrationIsNoOp) {
  unsigned countBefore = 0;
  for (auto *r : getHandwrittenPatternRecognizers())
    if (r->kindId() == kKernelizeHandwrittenKindAttentionSdpa)
      ++countBefore;

  // Re-register AttentionSdpa — should be ignored.
  registerAttentionSdpaRecognizer();

  unsigned countAfter = 0;
  for (auto *r : getHandwrittenPatternRecognizers())
    if (r->kindId() == kKernelizeHandwrittenKindAttentionSdpa)
      ++countAfter;

  EXPECT_EQ(countBefore, countAfter);
}

TEST(HandwrittenPatternRecognizerTest, FlashAttentionContractIsRegistered) {
  const HandwrittenContract *contract =
      lookupHandwrittenContract(kKernelizeHandwrittenKindFlashAttention);
  ASSERT_NE(contract, nullptr);
  EXPECT_GE(contract->maxCubeCount, 2u);
  EXPECT_TRUE(contract->requiresReduction);
  EXPECT_TRUE(contract->requiresVectorInjective);
}
```

- [ ] **Step 2: 将测试加入 `test/unittests/Conversion/CMakeLists.txt`**

在文件末尾追加：

```cmake
add_executable(AscendHandwrittenPatternRecognizerTest
  AscendHandwrittenPatternRecognizerTest.cpp
)

target_include_directories(AscendHandwrittenPatternRecognizerTest PRIVATE
  ${ASCEND_CONVERSION_INTERNAL_INCLUDE_DIR}
)

target_link_libraries(AscendHandwrittenPatternRecognizerTest PRIVATE
  RuntimeUnitTestSupport
  AscendConversion
)

add_dependencies(RuntimeUnitTests AscendHandwrittenPatternRecognizerTest)

add_test(NAME AscendHandwrittenPatternRecognizerTest
  COMMAND $<TARGET_FILE:AscendHandwrittenPatternRecognizerTest>
)
```

- [ ] **Step 3: 构建并运行**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && ninja -C build -j6 AscendHandwrittenPatternRecognizerTest 2>&1 | tail -10 && ./build/bin/AscendHandwrittenPatternRecognizerTest 2>&1"
```

Expected:
```
[==========] Running 5 tests from 1 test suite.
[  PASSED  ] 5 tests.
```

- [ ] **Step 4: Commit**

```bash
git add test/unittests/Conversion/AscendHandwrittenPatternRecognizerTest.cpp \
        test/unittests/Conversion/CMakeLists.txt
git commit -m "test: add HandwrittenPatternRecognizer unit tests"
```

---

## Task 6: FlashAttention LIT 测试 + 全量回归

**Files:**
- Create: `test/Conversion/ascend-kernelize-flash-attention-pattern.mlir`

- [ ] **Step 1: 创建 FlashAttention LIT 测试**

```mlir
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize | FileCheck %s

// Tests that a 2-round FlashAttention-like topology
// (Matmul1 -> RowMax -> Vector(exp) -> Matmul2 -> RowMax2 -> Vector(exp2) -> Matmul3)
// is recognized as flash_attention pattern by FlashAttentionRecognizer.

func.func @flash_attention_like(
    %q: tensor<2x4x8xf32>,
    %k: tensor<2x8x4xf32>,
    %v: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {

  // Round 1: Matmul1
  %s1_empty = tensor.empty() : tensor<2x4x4xf32>
  %s1 = linalg.batch_matmul
      ins(%q, %k : tensor<2x4x8xf32>, tensor<2x8x4xf32>)
      outs(%s1_empty : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>

  // Round 1: RowMax (Reduction)
  %m1_empty = tensor.empty() : tensor<2x4xf32>
  %m1 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%s1 : tensor<2x4x4xf32>)
    outs(%m1_empty : tensor<2x4xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %r = arith.maximumf %x, %acc : f32
    linalg.yield %r : f32
  } -> tensor<2x4xf32>

  // Round 1: exp(s - max) rescale (Vector with exp)
  %p1_empty = tensor.empty() : tensor<2x4x4xf32>
  %p1 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    ],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%s1, %m1 : tensor<2x4x4xf32>, tensor<2x4xf32>)
    outs(%p1_empty : tensor<2x4x4xf32>) {
  ^bb0(%x: f32, %m: f32, %out: f32):
    %shifted = arith.subf %x, %m : f32
    %e = math.exp %shifted : f32
    linalg.yield %e : f32
  } -> tensor<2x4x4xf32>

  // Round 2: Matmul2 (second Cube)
  %out_empty = tensor.empty() : tensor<2x4x8xf32>
  %out = linalg.batch_matmul
      ins(%p1, %v : tensor<2x4x4xf32>, tensor<2x4x8xf32>)
      outs(%out_empty : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>

  return %out : tensor<2x4x8xf32>
}

// CHECK: ascend.kernelize.handwritten_kind = "flash_attention"
```

- [ ] **Step 2: 运行 LIT 测试**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-kernelize-flash-attention-pattern.mlir 2>&1"
```

Expected: `PASSED`。若 CHECK 未匹配，用 `--ascend-kernelize --mlir-print-ir-after-all` 查看中间 IR 确认 `handwritten_kind` attribute 名称。

- [ ] **Step 3: 全量回归**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && ctest --test-dir build -R 'Ascend' --output-on-failure 2>&1 | tail -15"
```

Expected: 100% PASS

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ 2>&1 | tail -20"
```

Expected: 所有测试 PASS

- [ ] **Step 4: Commit**

```bash
git add test/Conversion/ascend-kernelize-flash-attention-pattern.mlir
git commit -m "test: add FlashAttention recognizer LIT test + full regression"
```

---

## Self-Review

### Spec 覆盖检查

| Spec 需求 | 对应 Task |
|---|---|
| HandwrittenPatternRecognizer 接口 | Task 1 |
| 注册表实现（线程安全，幂等） | Task 2 |
| AttentionSdpaRecognizer 迁移 | Task 2 |
| FusionCandidateAnalysis 主循环改为注册表 | Task 3 |
| 主干无 hardcoded kind 字符串 | Task 3 Step 4 |
| FlashAttentionRecognizer 实现 | Task 4 |
| FlashAttention HandwrittenContract 注册 | Task 4 Step 1 |
| 单元测试：注册→识别→contract lookup | Task 5 |
| FlashAttention LIT 测试 | Task 6 |
| 全量回归 | Task 6 |

### Placeholder 扫描

Task 2 Step 5 中的空桩 `FlashAttentionRecognizer.cpp` 是有意的临时状态，在 Task 4 被完整实现替换。无其他 placeholder。

### 类型一致性

- `PatternMatch.ops` 类型 `SmallVector<Operation *, 8>`：在 Task 1 头文件定义，在 Task 2 `AttentionSdpaRecognizer` 中赋值，在 Task 3 主循环中消费（`match->ops`）——一致。
- `PatternMatch.kindId` 类型 `std::string`：在 Task 1 定义，在 Task 2/4 中用 `.str()` 赋值，在 Task 3 中传给 `buildHandwrittenPatternCandidate(match->ops, deps, roleMap, match->kindId)`——`buildHandwrittenPatternCandidate` 的第四参数是 `StringRef`，`std::string` 隐式转换 OK。
- `registerAttentionSdpaRecognizer()` 在 Task 2 头文件 forward declare，在 Task 2 `AttentionSdpaRecognizer.cpp` 实现，在 `HandwrittenPatternRecognizerRegistry.cpp` 调用——一致。
- `kKernelizeHandwrittenKindFlashAttention` 在 Task 4 Step 1 添加到 `Attributes.h`，在 `HandwrittenContractRegistry.cpp`、`TemplateRegistry.cpp`、`FlashAttentionRecognizer.cpp` 中使用——一致。
