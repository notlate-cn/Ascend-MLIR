# Kernelize Op Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the private Kernelize op registry into a stable semantic contract and resolver so new op models can be registered without editing `DependencyAnalysis.cpp`.

**Architecture:** Add a thin public Kernelize semantic contract under `include/Conversion/Ascend/Kernelize`, then keep the resolver, default linalg/tensor-view models, and adapter to `OpSemanticSummary` inside `lib/Conversion/Ascend/Kernelize`. `DependencyAnalysis` will consume resolved participation (`Analyze`, `Transparent`, `Ignore`, `Unsupported`) instead of matching target ops and view chains directly.

**Tech Stack:** MLIR/LLVM C++17, existing AscendConversion library, GoogleTest runtime unit tests, LIT/FileCheck, xvm/docker verification via `examples/dev-env.md`.

---

## File Structure

- Create: `include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h`
- Create: `lib/Conversion/Ascend/Kernelize/KernelizeOpInterface.cpp`
- Modify: `lib/Conversion/Ascend/Kernelize/KernelizeTypes.h`
- Modify: `lib/Conversion/Ascend/Kernelize/KernelizeOpRegistry.h`
- Modify: `lib/Conversion/Ascend/Kernelize/KernelizeOpRegistry.cpp`
- Modify: `lib/Conversion/Ascend/Kernelize/DependencyAnalysis.h`
- Modify: `lib/Conversion/Ascend/Kernelize/DependencyAnalysis.cpp`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`
- Modify: `test/tools/check_ascend_public_headers.sh`
- Create: `test/unittests/Conversion/AscendKernelizeOpInterfaceTest.cpp`
- Modify: `test/unittests/Conversion/CMakeLists.txt`
- Create: `test/Conversion/ascend-kernelize-op-interface-linalg.mlir`
- Create: `test/Conversion/ascend-kernelize-op-interface-tensor-view.mlir`
- Create: `test/Conversion/ascend-kernelize-op-interface-unsupported.mlir`
- Modify: `test/Conversion/ascend-kernelize-linalg-interface.mlir`
- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

Do not stage or modify `docs/Ascend-MLIR-V2-Problem-Formulation.zh.md`; it is a user dirty file in the current worktree.

## Task 1: Public Semantic Contract And Header Boundary

**Files:**
- Create: `include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h`
- Create: `lib/Conversion/Ascend/Kernelize/KernelizeOpInterface.cpp`
- Modify: `lib/Conversion/Ascend/Kernelize/KernelizeTypes.h`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`
- Modify: `test/tools/check_ascend_public_headers.sh`
- Create: `test/unittests/Conversion/AscendKernelizeOpInterfaceTest.cpp`
- Modify: `test/unittests/Conversion/CMakeLists.txt`

- [ ] **Step 1: Write the failing public contract unit test**

Create `test/unittests/Conversion/AscendKernelizeOpInterfaceTest.cpp`:

```cpp
//===- AscendKernelizeOpInterfaceTest.cpp - Kernelize model tests --------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/KernelizeOpInterface.h"

#include "gtest/gtest.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"

using namespace mlir;
using namespace mlir::afir::ascend::kernelize;

namespace {

class TestOp {
public:
  TestOp(MLIRContext &context, StringRef name) {
    OperationState state(UnknownLoc::get(&context), name);
    op = Operation::create(state);
  }

  ~TestOp() {
    if (op)
      op->destroy();
  }

  operator Operation *() const { return op; }

private:
  Operation *op = nullptr;
};

bool matchTestAnalyze(Operation *op) {
  return op->getName().getStringRef() == "test.analyze";
}

LogicalResult populateTestAnalyze(Operation *, KernelizeOpSemanticInfo &info) {
  info.participation = KernelizeParticipationKind::Analyze;
  info.accessPattern = AccessPatternKind::Elementwise;
  info.iteratorKinds.push_back(IteratorKind::Parallel);
  info.resultRanks.push_back(1);
  info.traits.push_back(KernelizeSemanticTrait::Structured);
  info.modelName = "test_model";
  return success();
}

} // namespace

TEST(AscendKernelizeOpInterfaceTest, RegistryResolvesRegisteredModel) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  TestOp op(context, "test.analyze");

  KernelizeOpModelRegistry registry;
  registry.registerModel({"test_model", matchTestAnalyze, populateTestAnalyze});

  FailureOr<KernelizeOpSemanticInfo> info =
      registry.resolve(static_cast<Operation *>(op));

  ASSERT_TRUE(succeeded(info));
  EXPECT_EQ(info->participation, KernelizeParticipationKind::Analyze);
  EXPECT_EQ(info->accessPattern, AccessPatternKind::Elementwise);
  ASSERT_EQ(info->iteratorKinds.size(), 1u);
  EXPECT_EQ(info->iteratorKinds.front(), IteratorKind::Parallel);
  ASSERT_EQ(info->resultRanks.size(), 1u);
  EXPECT_EQ(info->resultRanks.front(), 1u);
  ASSERT_EQ(info->traits.size(), 1u);
  EXPECT_EQ(info->traits.front(), KernelizeSemanticTrait::Structured);
  EXPECT_EQ(info->modelName, "test_model");
}

TEST(AscendKernelizeOpInterfaceTest, RegistryReportsUnsupportedByDefault) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  TestOp op(context, "test.unknown");

  KernelizeOpModelRegistry registry;
  FailureOr<KernelizeOpSemanticInfo> info =
      registry.resolve(static_cast<Operation *>(op));

  ASSERT_TRUE(succeeded(info));
  EXPECT_EQ(info->participation, KernelizeParticipationKind::Unsupported);
  EXPECT_EQ(info->accessPattern, AccessPatternKind::Unknown);
  EXPECT_EQ(info->modelName, "unknown");
}

TEST(AscendKernelizeOpInterfaceTest, StringifiesPublicEnums) {
  EXPECT_EQ(stringifyKernelizeParticipation(
                KernelizeParticipationKind::Analyze),
            "analyze");
  EXPECT_EQ(stringifyKernelizeParticipation(
                KernelizeParticipationKind::Transparent),
            "transparent");
  EXPECT_EQ(stringifyKernelizeSemanticTrait(
                KernelizeSemanticTrait::TensorView),
            "tensor_view");
  EXPECT_EQ(stringifyAccessPattern(AccessPatternKind::Contraction),
            "Contraction");
  EXPECT_EQ(stringifyIteratorKind(IteratorKind::Reduction), "reduction");
}
```

Add the test to `test/unittests/Conversion/CMakeLists.txt`:

```cmake
add_executable(AscendKernelizeOpInterfaceTest
  AscendKernelizeOpInterfaceTest.cpp
)

target_link_libraries(AscendKernelizeOpInterfaceTest PRIVATE
  RuntimeUnitTestSupport
  AscendConversion
)

add_dependencies(RuntimeUnitTests AscendKernelizeOpInterfaceTest)

add_test(NAME AscendKernelizeOpInterfaceTest
  COMMAND $<TARGET_FILE:AscendKernelizeOpInterfaceTest>
)
```

- [ ] **Step 2: Run RED verification**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ninja -C build AscendKernelizeOpInterfaceTest'
```

Expected: build fails because `Conversion/Ascend/Kernelize/KernelizeOpInterface.h` does not exist.

- [ ] **Step 3: Add the public contract header**

Create `include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h`:

```cpp
//===- KernelizeOpInterface.h - Ascend kernelize op interface -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_KERNELIZEOPINTERFACE_H
#define ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_KERNELIZEOPINTERFACE_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir::afir::ascend::kernelize {

enum class AccessPatternKind {
  NotApplicable,
  Elementwise,
  Broadcast,
  Reduction,
  Contraction,
  Gather,
  Scatter,
  LayoutTransform,
  Unknown
};

enum class IteratorKind {
  Parallel,
  Reduction,
  Unknown
};

enum class KernelizeParticipationKind {
  Ignore,
  Analyze,
  Transparent,
  Unsupported,
};

enum class KernelizeSemanticTrait {
  Unknown,
  Structured,
  TensorView,
  LayoutTransform,
  HandwrittenGroup,
};

struct KernelizeOpSemanticInfo {
  KernelizeParticipationKind participation =
      KernelizeParticipationKind::Unsupported;
  AccessPatternKind accessPattern = AccessPatternKind::Unknown;
  SmallVector<IteratorKind, 4> iteratorKinds;
  SmallVector<AffineMap, 4> indexingMaps;
  SmallVector<unsigned, 2> resultRanks;
  SmallVector<KernelizeSemanticTrait, 4> traits;
  std::string modelName = "unknown";
  std::string unsupportedReason;
};

struct KernelizeOpModel {
  using MatchFn = bool (*)(Operation *op);
  using PopulateFn =
      LogicalResult (*)(Operation *op, KernelizeOpSemanticInfo &info);

  llvm::StringRef name = "unknown";
  MatchFn match = nullptr;
  PopulateFn populate = nullptr;
};

class KernelizeOpModelRegistry {
public:
  void registerModel(KernelizeOpModel model);
  FailureOr<KernelizeOpSemanticInfo> resolve(Operation *op) const;

private:
  SmallVector<KernelizeOpModel, 8> models;
};

llvm::StringRef stringifyAccessPattern(AccessPatternKind kind);
llvm::StringRef stringifyIteratorKind(IteratorKind kind);
llvm::StringRef stringifyKernelizeParticipation(
    KernelizeParticipationKind kind);
llvm::StringRef stringifyKernelizeSemanticTrait(KernelizeSemanticTrait trait);

} // namespace mlir::afir::ascend::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_KERNELIZEOPINTERFACE_H
```

Move `AccessPatternKind`, `IteratorKind`, `stringifyAccessPattern`, and `stringifyIteratorKind` out of `lib/Conversion/Ascend/Kernelize/KernelizeTypes.h` by including the new public header:

```cpp
#include "Conversion/Ascend/Kernelize/KernelizeOpInterface.h"
```

Keep `OpRole`, `CandidateKind`, `KernelizePrimitiveKind`, `KernelPatternEdgeKind`, `OperationId`, and `KernelizeConfig` private in `KernelizeTypes.h`.

- [ ] **Step 4: Add the public contract implementation**

Create `lib/Conversion/Ascend/Kernelize/KernelizeOpInterface.cpp`:

```cpp
//===- KernelizeOpInterface.cpp - Ascend kernelize op interface ----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/KernelizeOpInterface.h"

namespace mlir::afir::ascend::kernelize {

void KernelizeOpModelRegistry::registerModel(KernelizeOpModel model) {
  models.push_back(model);
}

FailureOr<KernelizeOpSemanticInfo>
KernelizeOpModelRegistry::resolve(Operation *op) const {
  for (const KernelizeOpModel &model : models) {
    if (!model.match || !model.match(op))
      continue;

    KernelizeOpSemanticInfo info;
    info.modelName = model.name.str();
    if (!model.populate)
      return info;
    if (failed(model.populate(op, info)))
      return failure();
    if (info.modelName == "unknown")
      info.modelName = model.name.str();
    return info;
  }

  KernelizeOpSemanticInfo info;
  info.participation = KernelizeParticipationKind::Unsupported;
  info.accessPattern = AccessPatternKind::Unknown;
  info.modelName = "unknown";
  info.unsupportedReason = "no kernelize semantic model for op " +
                           op->getName().getStringRef().str();
  return info;
}

llvm::StringRef stringifyAccessPattern(AccessPatternKind kind) {
  switch (kind) {
  case AccessPatternKind::NotApplicable:
    return "NotApplicable";
  case AccessPatternKind::Elementwise:
    return "Elementwise";
  case AccessPatternKind::Broadcast:
    return "Broadcast";
  case AccessPatternKind::Reduction:
    return "Reduction";
  case AccessPatternKind::Contraction:
    return "Contraction";
  case AccessPatternKind::Gather:
    return "Gather";
  case AccessPatternKind::Scatter:
    return "Scatter";
  case AccessPatternKind::LayoutTransform:
    return "LayoutTransform";
  case AccessPatternKind::Unknown:
    return "Unknown";
  }
  return "Unknown";
}

llvm::StringRef stringifyIteratorKind(IteratorKind kind) {
  switch (kind) {
  case IteratorKind::Parallel:
    return "parallel";
  case IteratorKind::Reduction:
    return "reduction";
  case IteratorKind::Unknown:
    return "unknown";
  }
  return "unknown";
}

llvm::StringRef stringifyKernelizeParticipation(
    KernelizeParticipationKind kind) {
  switch (kind) {
  case KernelizeParticipationKind::Ignore:
    return "ignore";
  case KernelizeParticipationKind::Analyze:
    return "analyze";
  case KernelizeParticipationKind::Transparent:
    return "transparent";
  case KernelizeParticipationKind::Unsupported:
    return "unsupported";
  }
  return "unsupported";
}

llvm::StringRef stringifyKernelizeSemanticTrait(
    KernelizeSemanticTrait trait) {
  switch (trait) {
  case KernelizeSemanticTrait::Unknown:
    return "unknown";
  case KernelizeSemanticTrait::Structured:
    return "structured";
  case KernelizeSemanticTrait::TensorView:
    return "tensor_view";
  case KernelizeSemanticTrait::LayoutTransform:
    return "layout_transform";
  case KernelizeSemanticTrait::HandwrittenGroup:
    return "handwritten_group";
  }
  return "unknown";
}

} // namespace mlir::afir::ascend::kernelize
```

Add `Kernelize/KernelizeOpInterface.cpp` to `lib/Conversion/Ascend/CMakeLists.txt`.

- [ ] **Step 5: Update public header boundary guard**

Modify `test/tools/check_ascend_public_headers.sh`:

```bash
check_phase_headers() {
  local phase="$1"
  shift
  local allowed
  local header

  while IFS= read -r header; do
    local ok=false
    for allowed in "$@"; do
      if [[ "$header" == "$allowed" ]]; then
        ok=true
        break
      fi
    done
    if [[ "$ok" != true ]]; then
      echo "unexpected public Ascend ${phase} internal header: ${header}" >&2
      return 1
    fi
  done < <(find "include/Conversion/Ascend/${phase}" -maxdepth 1 -type f -name '*.h' | sort)
}

check_phase_headers Kernelize \
  "include/Conversion/Ascend/Kernelize/KernelizePass.h" \
  "include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h"
check_phase_headers Schedule "include/Conversion/Ascend/Schedule/SchedulePass.h"
check_phase_headers Realize "include/Conversion/Ascend/Realize/RealizePass.h"
```

- [ ] **Step 6: Run GREEN verification and commit**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ninja -C build AscendKernelizeOpInterfaceTest afir-opt && ctest --test-dir build -R AscendKernelizeOpInterfaceTest --output-on-failure && bash test/tools/check_ascend_no_v2_code_naming.sh && llvm-lit -q test/Conversion/ascend-public-header-boundary.mlir'
git diff --check
```

Expected:

- `AscendKernelizeOpInterfaceTest` passes.
- public header boundary passes.
- code naming guard passes.
- `git diff --check` prints no output.

Commit:

```bash
git add include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h \
  lib/Conversion/Ascend/Kernelize/KernelizeOpInterface.cpp \
  lib/Conversion/Ascend/Kernelize/KernelizeTypes.h \
  lib/Conversion/Ascend/CMakeLists.txt \
  test/tools/check_ascend_public_headers.sh \
  test/unittests/Conversion/AscendKernelizeOpInterfaceTest.cpp \
  test/unittests/Conversion/CMakeLists.txt
git commit -m "feat: add Kernelize op semantic contract"
```

## Task 2: Default Resolver And Linalg External Model

**Files:**
- Modify: `lib/Conversion/Ascend/Kernelize/KernelizeOpRegistry.h`
- Modify: `lib/Conversion/Ascend/Kernelize/KernelizeOpRegistry.cpp`
- Modify: `lib/Conversion/Ascend/Kernelize/DependencyAnalysis.h`
- Modify: `lib/Conversion/Ascend/Kernelize/DependencyAnalysis.cpp`
- Create: `test/Conversion/ascend-kernelize-op-interface-linalg.mlir`
- Modify: `test/Conversion/ascend-kernelize-linalg-interface.mlir`

- [ ] **Step 1: Write the failing linalg resolver LIT**

Create `test/Conversion/ascend-kernelize-op-interface-linalg.mlir`:

```mlir
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s

func.func @generic_contraction(%arg0: tensor<4x8xf32>,
                               %arg1: tensor<8x16xf32>,
                               %arg2: tensor<4x16xf32>) -> tensor<4x16xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (k, n)>,
      affine_map<(m, n, k) -> (m, n)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<8x16xf32>)
    outs(%arg2 : tensor<4x16xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %out, %mul : f32
    linalg.yield %add : f32
  } -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK: DependencyAnalysis
// CHECK: op = "linalg.generic"
// CHECK-SAME: participation = "analyze"
// CHECK-SAME: model = "linalg"
// CHECK-SAME: traits = ["structured"]
// CHECK-SAME: access = "Contraction"
// CHECK-SAME: result_ranks = [2]
// CHECK-SAME: iterators = [parallel, parallel, reduction]
// CHECK: OpRoleClassification
// CHECK-SAME: roles = ["Primary", "Cube"]
```

- [ ] **Step 2: Run RED verification**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && llvm-lit -a test/Conversion/ascend-kernelize-op-interface-linalg.mlir'
```

Expected: FileCheck fails because report does not contain `participation`, `traits`, or `result_ranks`.

- [ ] **Step 3: Refactor private registry to use public semantic info**

Update `lib/Conversion/Ascend/Kernelize/KernelizeOpRegistry.h` so it no longer defines a parallel trait enum. It should expose only internal default registration:

```cpp
//===- KernelizeOpRegistry.h - Ascend kernelize op registry -*- C++ -*-===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_KERNELIZEOPREGISTRY_H
#define ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_KERNELIZEOPREGISTRY_H

#include "Conversion/Ascend/Kernelize/KernelizeOpInterface.h"

namespace mlir::afir::ascend::kernelize {

void registerDefaultKernelizeOpModels(KernelizeOpModelRegistry &registry);

} // namespace mlir::afir::ascend::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_KERNELIZEOPREGISTRY_H
```

Update `lib/Conversion/Ascend/Kernelize/KernelizeOpRegistry.cpp`:

- Keep the existing linalg semantic helper functions.
- Rename `populateLinalgSemanticSummary` to `populateLinalgSemanticInfo`.
- Fill `KernelizeOpSemanticInfo` instead of `OpSemanticSummary`.
- Record all ranked result ranks and all ranked DPS init ranks.
- On inconsistent non-empty result ranks, set:

```cpp
info.participation = KernelizeParticipationKind::Unsupported;
info.accessPattern = AccessPatternKind::Unknown;
info.unsupportedReason = "linalg op has inconsistent ranked result ranks";
return success();
```

The linalg populate function must set:

```cpp
info.participation = KernelizeParticipationKind::Analyze;
info.modelName = "linalg";
info.traits.push_back(KernelizeSemanticTrait::Structured);
```

Register it:

```cpp
void registerDefaultKernelizeOpModels(KernelizeOpModelRegistry &registry) {
  registry.registerModel({"linalg", matchLinalgOp, populateLinalgSemanticInfo});
}
```

- [ ] **Step 4: Adapt `OpSemanticSummary` to store public semantic fields**

Update `lib/Conversion/Ascend/Kernelize/DependencyAnalysis.h`:

```cpp
struct OpSemanticSummary {
  Operation *op = nullptr;
  OperationId opId;
  KernelizeParticipationKind participation =
      KernelizeParticipationKind::Unsupported;
  AccessPatternKind accessPattern = AccessPatternKind::Unknown;
  SmallVector<IteratorKind> iteratorTypes;
  SmallVector<AffineMap> indexingMaps;
  SmallVector<unsigned> resultRanks;
  SmallVector<KernelizeSemanticTrait> traits;
  std::string modelName = "unknown";
  std::string unsupportedReason;
  unsigned resultRank = 0;
  bool hasReductionIterator = false;
  bool hasOnlyParallelIterators = false;
};
```

Keep `resultRank` temporarily as compatibility for current role/schedule code. Set it to the first rank in `resultRanks`, or `0` when empty.

- [ ] **Step 5: Update `DependencyAnalysis.cpp` to build Analyze ops from resolver**

In `DependencyAnalyzer::analyze`:

```cpp
KernelizeOpModelRegistry registry;
registerDefaultKernelizeOpModels(registry);

DenseMap<Operation *, KernelizeOpSemanticInfo> resolved;
module.walk([&](Operation *op) {
  FailureOr<KernelizeOpSemanticInfo> info = registry.resolve(op);
  if (failed(info))
    return;
  resolved.try_emplace(op, std::move(*info));
});

module.walk([&](Operation *op) {
  auto it = resolved.find(op);
  if (it == resolved.end() ||
      it->second.participation != KernelizeParticipationKind::Analyze)
    return;

  OperationId opId{static_cast<unsigned>(result.index.orderedOps.size())};
  result.index.orderedOps.push_back(op);
  result.index.opIds.try_emplace(op, opId);
});
```

Add a helper:

```cpp
static OpSemanticSummary makeSummary(Operation *op, OperationId opId,
                                     const KernelizeOpSemanticInfo &info) {
  OpSemanticSummary summary;
  summary.op = op;
  summary.opId = opId;
  summary.participation = info.participation;
  summary.accessPattern = info.accessPattern;
  summary.iteratorTypes.append(info.iteratorKinds.begin(),
                               info.iteratorKinds.end());
  summary.indexingMaps.append(info.indexingMaps.begin(),
                              info.indexingMaps.end());
  summary.resultRanks.append(info.resultRanks.begin(), info.resultRanks.end());
  summary.traits.append(info.traits.begin(), info.traits.end());
  summary.modelName = info.modelName;
  summary.unsupportedReason = info.unsupportedReason;
  summary.resultRank = summary.resultRanks.empty() ? 0 : summary.resultRanks.front();
  summary.hasReductionIterator =
      llvm::is_contained(summary.iteratorTypes, IteratorKind::Reduction);
  summary.hasOnlyParallelIterators =
      !summary.iteratorTypes.empty() &&
      llvm::all_of(summary.iteratorTypes, [](IteratorKind kind) {
        return kind == IteratorKind::Parallel;
      });
  return summary;
}
```

Use `makeSummary` when populating `result.summaries`.

- [ ] **Step 6: Update dependency report fields**

In `emitDependencyAnalysisReport`, print:

```cpp
os << " participation = \""
   << stringifyKernelizeParticipation(summary.participation) << "\"";
os << " model = \"" << summary.modelName << "\" traits = [";
llvm::interleaveComma(summary.traits, os,
                      [&](KernelizeSemanticTrait trait) {
                        os << "\"" << stringifyKernelizeSemanticTrait(trait)
                           << "\"";
                      });
os << "] access = \"" << stringifyAccessPattern(summary.accessPattern) << "\"";
os << " result_ranks = [";
llvm::interleaveComma(summary.resultRanks, os,
                      [&](unsigned rank) { os << rank; });
os << "]";
```

If `unsupportedReason` is non-empty, print:

```cpp
os << " unsupported_reason = \"" << summary.unsupportedReason << "\"";
```

Update `test/Conversion/ascend-kernelize-linalg-interface.mlir` expectations from `trait = "structured_linalg"` to `traits = ["structured"]` and add `participation = "analyze"` / `result_ranks = [...]`.

- [ ] **Step 7: Run GREEN verification and commit**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ninja -C build afir-opt AscendKernelizeOpInterfaceTest AscendKernelPatternTest && ctest --test-dir build -R "Ascend(KernelizeOpInterface|KernelPattern)Test" --output-on-failure && source examples/env.sh >/tmp/ascend_env.log && llvm-lit -q test/Conversion/ascend-kernelize-linalg-interface.mlir test/Conversion/ascend-kernelize-op-interface-linalg.mlir'
git diff --check
```

Expected:

- Two unit targets pass.
- Two linalg interface LIT files pass.
- `git diff --check` prints no output.

Commit:

```bash
git add lib/Conversion/Ascend/Kernelize/KernelizeOpRegistry.h \
  lib/Conversion/Ascend/Kernelize/KernelizeOpRegistry.cpp \
  lib/Conversion/Ascend/Kernelize/DependencyAnalysis.h \
  lib/Conversion/Ascend/Kernelize/DependencyAnalysis.cpp \
  test/Conversion/ascend-kernelize-linalg-interface.mlir \
  test/Conversion/ascend-kernelize-op-interface-linalg.mlir
git commit -m "feat: route Kernelize linalg semantics through resolver"
```

## Task 3: Tensor View Transparent Model

**Files:**
- Modify: `lib/Conversion/Ascend/Kernelize/KernelizeOpRegistry.cpp`
- Modify: `lib/Conversion/Ascend/Kernelize/DependencyAnalysis.cpp`
- Create: `test/Conversion/ascend-kernelize-op-interface-tensor-view.mlir`

- [ ] **Step 1: Write the failing tensor view LIT**

Create `test/Conversion/ascend-kernelize-op-interface-tensor-view.mlir`:

```mlir
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s

func.func @view_chain(%arg0: tensor<4x16xf32>,
                      %arg1: tensor<4x16xf32>,
                      %arg2: tensor<64xf32>) -> tensor<64xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x16xf32>, tensor<4x16xf32>)
    outs(%arg0 : tensor<4x16xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %add = arith.addf %a, %b : f32
    linalg.yield %add : f32
  } -> tensor<4x16xf32>

  %1 = tensor.collapse_shape %0 [[0, 1]]
      : tensor<4x16xf32> into tensor<64xf32>
  %2 = tensor.cast %1 : tensor<64xf32> to tensor<64xf32>

  %3 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%2, %arg2 : tensor<64xf32>, tensor<64xf32>)
    outs(%arg2 : tensor<64xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %add = arith.addf %a, %b : f32
    linalg.yield %add : f32
  } -> tensor<64xf32>

  return %3 : tensor<64xf32>
}

// CHECK: DependencyAnalysis
// CHECK: op_id = 0
// CHECK-SAME: op = "linalg.generic"
// CHECK: op_id = 1
// CHECK-SAME: op = "linalg.generic"
// CHECK-SAME: producers = 1
// CHECK: KernelPartition
// CHECK: linalg.generic
// CHECK-SAME: ascend.kernel = "kernel_0"
// CHECK: linalg.generic
// CHECK-SAME: ascend.kernel = "kernel_0"
```

- [ ] **Step 2: Run RED verification**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && llvm-lit -a test/Conversion/ascend-kernelize-op-interface-tensor-view.mlir'
```

Expected: FileCheck fails because traversal is not yet model-driven or the report does not expose transparent view handling.

- [ ] **Step 3: Add tensor view model registration**

In `KernelizeOpRegistry.cpp`, include tensor dialect:

```cpp
#include "mlir/Dialect/Tensor/IR/Tensor.h"
```

Add match/populate helpers:

```cpp
bool matchTensorViewOp(Operation *op) {
  return isa<tensor::CastOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp,
             tensor::ExtractSliceOp>(op);
}

LogicalResult populateTensorViewSemanticInfo(
    Operation *, KernelizeOpSemanticInfo &info) {
  info.participation = KernelizeParticipationKind::Transparent;
  info.accessPattern = AccessPatternKind::LayoutTransform;
  info.traits.push_back(KernelizeSemanticTrait::TensorView);
  info.modelName = "tensor_view";
  return success();
}
```

Register after linalg:

```cpp
registry.registerModel({"tensor_view", matchTensorViewOp,
                        populateTensorViewSemanticInfo});
```

- [ ] **Step 4: Make producer traversal use participation**

Change `collectAnalyzedProducers` in `DependencyAnalysis.cpp` to accept resolved semantic info:

```cpp
void collectAnalyzedProducers(
    Value value, const ProducerConsumerIndex &index,
    const DenseMap<Operation *, KernelizeOpSemanticInfo> &resolved,
    SmallVectorImpl<Operation *> &producers,
    SmallVectorImpl<Operation *> &unsupportedProducers,
    DenseSet<Operation *> &visited) {
  Operation *producer = value.getDefiningOp();
  if (!producer)
    return;

  if (index.opIds.contains(producer)) {
    producers.push_back(producer);
    return;
  }

  if (!visited.insert(producer).second)
    return;

  auto resolvedIt = resolved.find(producer);
  KernelizeParticipationKind participation =
      resolvedIt == resolved.end()
          ? KernelizeParticipationKind::Ignore
          : resolvedIt->second.participation;

  if (participation == KernelizeParticipationKind::Transparent) {
    for (Value operand : producer->getOperands()) {
      if (!isa<TensorType>(operand.getType()))
        continue;
      collectAnalyzedProducers(operand, index, resolved, producers,
                               unsupportedProducers, visited);
    }
    return;
  }

  if (participation == KernelizeParticipationKind::Unsupported &&
      llvm::any_of(producer->getResultTypes(),
                   [](Type type) { return isa<TensorType>(type); })) {
    unsupportedProducers.push_back(producer);
  }
}
```

Use this helper for every operand of Analyze ops.

- [ ] **Step 5: Run GREEN verification and commit**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ninja -C build afir-opt && source examples/env.sh >/tmp/ascend_env.log && llvm-lit -q test/Conversion/ascend-kernelize-op-interface-tensor-view.mlir test/Conversion/ascend-kernelize-op-interface-linalg.mlir'
git diff --check
```

Expected: both LIT tests pass.

Commit:

```bash
git add lib/Conversion/Ascend/Kernelize/KernelizeOpRegistry.cpp \
  lib/Conversion/Ascend/Kernelize/DependencyAnalysis.cpp \
  test/Conversion/ascend-kernelize-op-interface-tensor-view.mlir
git commit -m "feat: model Kernelize tensor views as transparent"
```

## Task 4: Unsupported Tensor Producer Diagnostics

**Files:**
- Modify: `lib/Conversion/Ascend/Kernelize/DependencyAnalysis.h`
- Modify: `lib/Conversion/Ascend/Kernelize/DependencyAnalysis.cpp`
- Create: `test/Conversion/ascend-kernelize-op-interface-unsupported.mlir`

- [ ] **Step 1: Write the failing unsupported producer LIT**

Create `test/Conversion/ascend-kernelize-op-interface-unsupported.mlir`:

```mlir
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s

func.func @unsupported_tensor_producer(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %c4 = arith.constant 4 : index
  %generated = tensor.generate %c4 {
  ^bb0(%i: index):
    %v = arith.index_cast %i : index to i32
    %f = arith.sitofp %v : i32 to f32
    tensor.yield %f : f32
  } : tensor<4xf32>

  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%generated, %arg0 : tensor<4xf32>, tensor<4xf32>)
    outs(%arg0 : tensor<4xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %add = arith.addf %a, %b : f32
    linalg.yield %add : f32
  } -> tensor<4xf32>

  return %0 : tensor<4xf32>
}

// CHECK: error: unsupported Kernelize tensor producer "tensor.generate"
// CHECK-SAME: consumed by "linalg.generic"
```

- [ ] **Step 2: Run RED verification**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && llvm-lit -a test/Conversion/ascend-kernelize-op-interface-unsupported.mlir'
```

Expected: the command succeeds or fails without the requested diagnostic, proving unsupported tensor producers are still not reported correctly.

- [ ] **Step 3: Track unsupported producers in the analysis result**

Add to `DependencyAnalysis.h`:

```cpp
struct UnsupportedProducerDiagnostic {
  Operation *producer = nullptr;
  Operation *consumer = nullptr;
  std::string reason;
};
```

Add to `DependencyAnalysisResult`:

```cpp
SmallVector<UnsupportedProducerDiagnostic, 4> unsupportedProducers;
```

- [ ] **Step 4: Emit fail-closed diagnostics**

In `DependencyAnalyzer::analyze`, after collecting producers for an operand:

```cpp
for (Operation *unsupported : unsupportedProducers) {
  std::string reason = "no kernelize semantic model for op " +
                       unsupported->getName().getStringRef().str();
  result.unsupportedProducers.push_back({unsupported, op, reason});
}
```

After building all producer/consumer edges:

```cpp
if (!result.unsupportedProducers.empty()) {
  const UnsupportedProducerDiagnostic &diag =
      result.unsupportedProducers.front();
  diag.producer->emitError()
      << "unsupported Kernelize tensor producer \""
      << diag.producer->getName().getStringRef() << "\" consumed by \""
      << diag.consumer->getName().getStringRef() << "\"";
  return failure();
}
```

In `emitDependencyAnalysisReport`, add a section:

```cpp
if (!result.unsupportedProducers.empty()) {
  os << "UnsupportedProducers\n";
  for (const UnsupportedProducerDiagnostic &diag :
       result.unsupportedProducers) {
    os << "  producer = \"" << diag.producer->getName().getStringRef()
       << "\" consumer = \"" << diag.consumer->getName().getStringRef()
       << "\" unsupported_reason = \"" << diag.reason << "\"\n";
  }
}
```

- [ ] **Step 5: Run GREEN verification and commit**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ninja -C build afir-opt && source examples/env.sh >/tmp/ascend_env.log && llvm-lit -q test/Conversion/ascend-kernelize-op-interface-unsupported.mlir test/Conversion/ascend-kernelize-op-interface-tensor-view.mlir'
git diff --check
```

Expected: unsupported producer LIT fails closed with the expected diagnostic, tensor view LIT still passes.

Commit:

```bash
git add lib/Conversion/Ascend/Kernelize/DependencyAnalysis.h \
  lib/Conversion/Ascend/Kernelize/DependencyAnalysis.cpp \
  test/Conversion/ascend-kernelize-op-interface-unsupported.mlir
git commit -m "fix: diagnose unsupported Kernelize tensor producers"
```

## Task 5: Regression Sweep And Tracking

**Files:**
- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [ ] **Step 1: Update tracking doc**

In `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`, update the `KernelizeOpRegistry` row to state:

```markdown
| KernelizeOpInterface / trait model | `Done` | `KernelizeOpSemanticInfo` public contract added; linalg and tensor view default models resolve through `KernelizeOpModelRegistry`; `DependencyAnalysis` consumes participation (`Analyze` / `Transparent` / `Unsupported`) instead of private target-op matching; unsupported tensor producers now fail closed with producer/consumer diagnostic | `ascend-kernelize-op-interface-*.mlir`; `AscendKernelizeOpInterfaceTest` |
```

Update the gap board note from “后续把 registry 公开为正式 OpInterface / trait model” to “首轮 public semantic contract + external trait model registry 已完成，TableGen OpInterface 可后续用于 AFIR 自有 op”。

- [ ] **Step 2: Run focused regression**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ninja -C build afir-opt AscendKernelizeOpInterfaceTest AscendKernelPatternTest && ctest --test-dir build -R "Ascend(KernelizeOpInterface|KernelPattern)Test" --output-on-failure && source examples/env.sh >/tmp/ascend_env.log && llvm-lit -q test/Conversion/ascend-kernelize*.mlir test/Conversion/ascend-schedule*.mlir test/Conversion/ascend-realize*.mlir test/Conversion/ascend-full-pipeline-*.mlir'
```

Expected: all discovered tests pass.

- [ ] **Step 3: Run broader regression and examples**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && export PATH=/home/niu/code/Ascend-MLIR/build/bin:/home/niu/code/llvm-project/llvm/build/bin:$PATH && source examples/env.sh >/tmp/ascend_env.log && llvm-lit -q test/Conversion/ascend-*.mlir test/Target/cann-translate*.mlir'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ctest --test-dir build -R "Ascend.*Test" --output-on-failure'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && export PATH=/home/niu/code/Ascend-MLIR/build/bin:/home/niu/code/llvm-project/llvm/build/bin:$PATH && source examples/env.sh >/tmp/ascend_env.log && bash examples/transformer/run-mainline.sh && bash examples/relu-broadcast-transpose/run-mainline.sh'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && bash test/tools/check_ascend_no_v2_code_naming.sh'
git diff --check
```

Expected:

- Conversion/Target LIT pass.
- Ascend ctest pass.
- Transformer mainline reports `transformer_dynamic.full_codegen=pass`.
- `relu-broadcast-transpose` reports `session.validation=pass`.
- naming guard passes.
- `git diff --check` prints no output.

- [ ] **Step 4: Commit tracking update**

Commit:

```bash
git add docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md
git commit -m "docs: track Kernelize op interface closure"
```

## Final Verification Before Push

Run:

```bash
git status --short
git log --oneline -8
```

Expected:

- Only user-owned `docs/Ascend-MLIR-V2-Problem-Formulation.zh.md` may remain dirty.
- New commits are visible on top of the branch.

Then push only after the implementation commits are complete:

```bash
git push
```

## Review Checklist

- Public headers under `include/Conversion/Ascend/Kernelize` are exactly `KernelizePass.h` and `KernelizeOpInterface.h`.
- `DependencyAnalysis.cpp` no longer constructs target op participation from private linalg-only registry matching.
- Default linalg model still does not branch on concrete linalg op names.
- Tensor view traversal is driven by `KernelizeParticipationKind::Transparent`.
- Unsupported tensor producer diagnostic anchors the producer op and names the consumer op.
- Existing Kernelize/Schedule/Realize/full-pipeline tests and selected examples remain green.
