# Ascend Target Intrinsic Model MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue Phase 4 by adding a queryable `TargetIntrinsicModel` derived from CANN intrinsic entries.

**Architecture:** Keep `TargetProfile` as the raw CANN-derived fact carrier and build a local `TargetIntrinsicModel` from `TargetProfile::intrinsics`, matching the existing `TargetMemoryModelBuilder` pattern. The MVP classifies intrinsic names into unit, movement path kind, compute kind, and exact dtype-pattern token support; it does not perform profile closure validation or feed `TargetMemoryModel` yet.

**Scope boundary:** This task completes the query-only `TargetIntrinsicModel` MVP. Full V2-8 closure remains pending in `ProfileVerifier`, including intrinsic-backed path validation, path constraints, and `TargetMemoryModel` integration.

**Tech Stack:** MLIR/LLVM C++ libraries, LLVM ADT containers, GoogleTest unit tests, LIT/FileCheck, xvm/docker verification via `examples/dev-env.md`.

---

## File Structure

- Create `include/Target/Ascend/TargetIntrinsicModel.h`
- Create `lib/Target/Ascend/TargetIntrinsicModel.cpp`
- Modify `include/Target/Ascend/TargetProfile.h`
- Modify `include/Target/Ascend/TargetMemoryModel.h`
- Modify `lib/Target/Ascend/CannTargetProfileLoader.cpp`
- Modify `lib/Target/Ascend/CMakeLists.txt`
- Modify `test/unittests/Target/CMakeLists.txt`
- Create `test/unittests/Target/AscendTargetIntrinsicModelTest.cpp`
- Modify `test/Target/ascend-target-profile.mlir`
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

Do not modify or stage `AGENTS.md`; it contains unrelated local user edits.

## Task 1: Red Tests For TargetIntrinsicModel

**Files:**
- Create: `test/unittests/Target/AscendTargetIntrinsicModelTest.cpp`
- Modify: `test/unittests/Target/CMakeLists.txt`

- [ ] **Step 1: Add the test target**

Append to `test/unittests/Target/CMakeLists.txt`:

```cmake
add_executable(AscendTargetIntrinsicModelTest
  AscendTargetIntrinsicModelTest.cpp
)

target_link_libraries(AscendTargetIntrinsicModelTest PRIVATE
  RuntimeUnitTestSupport
  AscendTargetProfile
)

add_dependencies(RuntimeUnitTests AscendTargetIntrinsicModelTest)

add_test(NAME AscendTargetIntrinsicModelTest
  COMMAND $<TARGET_FILE:AscendTargetIntrinsicModelTest>
)
```

- [ ] **Step 2: Add unit tests for the desired query API**

Create `test/unittests/Target/AscendTargetIntrinsicModelTest.cpp`:

```cpp
#include "Target/Ascend/TargetIntrinsicModel.h"
#include "gtest/gtest.h"

using namespace mlir::ascend;
using llvm::FailureOr;
using llvm::succeeded;

static TargetProfile makeProfileWithIntrinsics() {
  TargetProfile profile;
  profile.identity.socVersion = "TestSoC";
  profile.hardware.aiCoreCount = 1;
  profile.hardware.l1SizeBytes = 512 * 1024;
  profile.hardware.ubSizeBytes = 192 * 1024;
  profile.intrinsics.push_back(
      {"Intrinsic_mmad", {"f16f16f16", "s32s8s8"},
       {ExecutionUnit::Cube}});
  profile.intrinsics.push_back({"Intrinsic_vadd", {"f16", "f32"}});
  profile.intrinsics.push_back({"Intrinsic_vexp", {"f16"}});
  profile.intrinsics.push_back({"Intrinsic_data_move_out2l1", {"f16"}});
  profile.intrinsics.push_back(
      {"Intrinsic_data_move_transpose_l12l0b", {"f16"}});
  profile.intrinsics.push_back({"Intrinsic_fix_pipe_l0c2out", {"f16"}});
  return profile;
}

TEST(AscendTargetIntrinsicModelTest, SupportsIntrinsicAndDtypes) {
  FailureOr<TargetIntrinsicModel> model =
      TargetIntrinsicModelBuilder().build(makeProfileWithIntrinsics());
  ASSERT_TRUE(succeeded(model));

  EXPECT_TRUE(model->supportsIntrinsic("Intrinsic_mmad"));
  EXPECT_FALSE(model->supportsIntrinsic("Intrinsic_missing"));
  EXPECT_TRUE(model->supportsDTypePattern("Intrinsic_mmad", "f16f16f16"));
  EXPECT_TRUE(model->supportsDTypePattern("Intrinsic_mmad", "s32s8s8"));
  EXPECT_FALSE(model->supportsDTypePattern("Intrinsic_mmad", "f16"));
}

TEST(AscendTargetIntrinsicModelTest, ClassifiesExecutionUnits) {
  FailureOr<TargetIntrinsicModel> model =
      TargetIntrinsicModelBuilder().build(makeProfileWithIntrinsics());
  ASSERT_TRUE(succeeded(model));

  EXPECT_EQ(model->getIntrinsicsForUnit(ExecutionUnit::Cube),
            (SmallVector<std::string>{"Intrinsic_mmad"}));
  EXPECT_EQ(model->getIntrinsicsForUnit(ExecutionUnit::Vector),
            (SmallVector<std::string>{"Intrinsic_vadd", "Intrinsic_vexp"}));
  EXPECT_EQ(model->getIntrinsicsForUnit(ExecutionUnit::DMA),
            (SmallVector<std::string>{"Intrinsic_data_move_out2l1",
                                      "Intrinsic_data_move_transpose_l12l0b",
                                      "Intrinsic_fix_pipe_l0c2out"}));
}

TEST(AscendTargetIntrinsicModelTest, ClassifiesMovementIntrinsicKinds) {
  FailureOr<TargetIntrinsicModel> model =
      TargetIntrinsicModelBuilder().build(makeProfileWithIntrinsics());
  ASSERT_TRUE(succeeded(model));

  EXPECT_EQ(model->getIntrinsicsForPathKind(PathKind::Load2D),
            (SmallVector<std::string>{"Intrinsic_data_move_out2l1"}));
  EXPECT_EQ(model->getIntrinsicsForPathKind(PathKind::Load2DTranspose),
            (SmallVector<std::string>{
                "Intrinsic_data_move_transpose_l12l0b"}));
  EXPECT_EQ(model->getIntrinsicsForPathKind(PathKind::FixPipe),
            (SmallVector<std::string>{"Intrinsic_fix_pipe_l0c2out"}));
  EXPECT_TRUE(model->getIntrinsicsForPathKind(PathKind::QueueTransfer).empty());
}

TEST(AscendTargetIntrinsicModelTest, ClassifiesComputeIntrinsicKinds) {
  FailureOr<TargetIntrinsicModel> model =
      TargetIntrinsicModelBuilder().build(makeProfileWithIntrinsics());
  ASSERT_TRUE(succeeded(model));

  EXPECT_EQ(model->getIntrinsicsForComputeKind(ComputeKind::Matmul),
            (SmallVector<std::string>{"Intrinsic_mmad"}));
  EXPECT_EQ(model->getIntrinsicsForComputeKind(ComputeKind::VectorAdd),
            (SmallVector<std::string>{"Intrinsic_vadd"}));
  EXPECT_EQ(model->getIntrinsicsForComputeKind(ComputeKind::VectorExp),
            (SmallVector<std::string>{"Intrinsic_vexp"}));
}

TEST(AscendTargetIntrinsicModelTest, MergesDuplicateCapabilities) {
  TargetProfile profile;
  profile.intrinsics.push_back(
      {"Intrinsic_mmad", {"f16f16f16"}, {ExecutionUnit::Cube}});
  profile.intrinsics.push_back(
      {"Intrinsic_mmad", {"s32s8s8"}, {ExecutionUnit::Cube}});

  FailureOr<TargetIntrinsicModel> model =
      TargetIntrinsicModelBuilder().build(profile);
  ASSERT_TRUE(succeeded(model));

  EXPECT_TRUE(model->supportsDTypePattern("Intrinsic_mmad", "f16f16f16"));
  EXPECT_TRUE(model->supportsDTypePattern("Intrinsic_mmad", "s32s8s8"));
  EXPECT_EQ(model->getIntrinsicsForUnit(ExecutionUnit::Cube),
            (SmallVector<std::string>{"Intrinsic_mmad"}));
}
```

The final test file must also cover:

- `PathKind::DirectCopy` via a non-Load2D data movement intrinsic such as `Intrinsic_data_move_l12l0a`
- `Intrinsic_fix_pipe_*` metadata, such as `Intrinsic_fix_pipe_unit_list`, remaining queryable without entering `ExecutionUnit::DMA` or `PathKind::FixPipe`
- all MVP compute kinds: `Matmul`, `VectorAdd`, `VectorExp`, `VectorTranspose`, `VectorGather`, and `VectorReduce`
- CANN loader section merge with a synthetic `platform_config` tree covering `AICoreintrinsicDtypeMap`, `CUBECoreintrinsicDtypeMap`, `VectorCoreintrinsicDtypeMap`, and `AICoreMemoryRates`

- [ ] **Step 3: Run RED verification in xvm/docker**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build AscendTargetIntrinsicModelTest'
```

Expected: FAIL because `Target/Ascend/TargetIntrinsicModel.h` is not implemented yet.

## Task 2: Implement TargetIntrinsicModel

**Files:**
- Create: `include/Target/Ascend/TargetIntrinsicModel.h`
- Create: `lib/Target/Ascend/TargetIntrinsicModel.cpp`
- Modify: `include/Target/Ascend/TargetProfile.h`
- Modify: `include/Target/Ascend/TargetMemoryModel.h`
- Modify: `lib/Target/Ascend/CMakeLists.txt`

- [ ] **Step 1: Move shared execution-unit enum**

Move `ExecutionUnit` from `TargetMemoryModel.h` to `TargetProfile.h` so both target memory and intrinsic models can use it:

```cpp
enum class ExecutionUnit { DMA, Cube, Vector };

struct TargetIntrinsicInfo {
  std::string name;
  SmallVector<std::string> dtypes;
  SmallVector<ExecutionUnit> units;
};
```

Remove the duplicate enum declaration from `TargetMemoryModel.h`.

- [ ] **Step 2: Add the target intrinsic model public API**

Create `include/Target/Ascend/TargetIntrinsicModel.h`:

```cpp
#ifndef ASCEND_MLIR_TARGET_ASCEND_TARGET_INTRINSIC_MODEL_H
#define ASCEND_MLIR_TARGET_ASCEND_TARGET_INTRINSIC_MODEL_H

#include "Target/Ascend/TargetMemoryModel.h"
#include "Target/Ascend/TargetProfile.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Support/LLVM.h"
#include <string>

namespace mlir::ascend {

enum class ComputeKind {
  Matmul,
  VectorAdd,
  VectorExp,
  VectorTranspose,
  VectorGather,
  VectorReduce,
};

struct IntrinsicCapability {
  std::string name;
  SmallVector<std::string> dtypes;
};

class TargetIntrinsicModel {
public:
  bool supportsIntrinsic(StringRef name) const;
  FailureOr<IntrinsicCapability> getIntrinsic(StringRef name) const;
  bool supportsDTypePattern(StringRef name, StringRef dtypePattern) const;
  SmallVector<std::string> getIntrinsicsForUnit(ExecutionUnit unit) const;
  SmallVector<std::string> getIntrinsicsForPathKind(PathKind kind) const;
  SmallVector<std::string> getIntrinsicsForComputeKind(ComputeKind kind) const;

private:
  friend class TargetIntrinsicModelBuilder;
  llvm::StringMap<IntrinsicCapability> intrinsicTable;
  SmallVector<std::string> dmaIntrinsics;
  SmallVector<std::string> cubeIntrinsics;
  SmallVector<std::string> vectorIntrinsics;
  SmallVector<std::string> directCopyIntrinsics;
  SmallVector<std::string> load2DIntrinsics;
  SmallVector<std::string> load2DTransposeIntrinsics;
  SmallVector<std::string> fixPipeIntrinsics;
  SmallVector<std::string> matmulIntrinsics;
  SmallVector<std::string> vectorAddIntrinsics;
  SmallVector<std::string> vectorExpIntrinsics;
  SmallVector<std::string> vectorTransposeIntrinsics;
  SmallVector<std::string> vectorGatherIntrinsics;
  SmallVector<std::string> vectorReduceIntrinsics;
};

class TargetIntrinsicModelBuilder {
public:
  FailureOr<TargetIntrinsicModel> build(const TargetProfile &profile) const;
};

} // namespace mlir::ascend

#endif
```

- [ ] **Step 3: Implement name-based classification**

In `lib/Target/Ascend/TargetIntrinsicModel.cpp`:

- Insert all `profile.intrinsics` into `intrinsicTable`.
- Sort and uniquify each returned vector for deterministic output.
- Unit classification:
  - Prefer `TargetIntrinsicInfo::units` recorded by the CANN loader.
  - If `units` is empty, use name fallback:
    - `Intrinsic_mmad` -> `ExecutionUnit::Cube`
    - names starting with `Intrinsic_v` -> `ExecutionUnit::Vector`
    - names starting with `Intrinsic_data_move_` or concrete path names starting with `Intrinsic_fix_pipe_l` -> `ExecutionUnit::DMA`
- Movement classification:
  - names containing `_transpose_` -> `PathKind::Load2DTranspose`
  - concrete path names starting with `Intrinsic_fix_pipe_l` -> `PathKind::FixPipe`
  - `Intrinsic_data_move_out2l1`, `Intrinsic_data_move_out2l0a`, `Intrinsic_data_move_out2l0b` -> `PathKind::Load2D`
  - other names starting with `Intrinsic_data_move_` -> `PathKind::DirectCopy`
- Compute classification:
  - `Intrinsic_mmad` -> `ComputeKind::Matmul`
  - `Intrinsic_vadd` -> `ComputeKind::VectorAdd`
  - `Intrinsic_vexp` -> `ComputeKind::VectorExp`
  - `Intrinsic_vtranspose` -> `ComputeKind::VectorTranspose`
  - `Intrinsic_vgather` -> `ComputeKind::VectorGather`
  - `Intrinsic_vreduce` -> `ComputeKind::VectorReduce`

- [ ] **Step 4: Run GREEN unit tests**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build AscendTargetIntrinsicModelTest && ./build/bin/AscendTargetIntrinsicModelTest'
```

Expected: unit tests pass.

## Task 3: Expand CANN Intrinsic Loading

**Files:**
- Modify: `lib/Target/Ascend/CannTargetProfileLoader.cpp`

- [ ] **Step 1: Parse all intrinsic dtype sections**

Update loader parsing so it reads these sections into `profile.intrinsics`:

- `AICoreintrinsicDtypeMap` -> record unit provenance by intrinsic name fallback
- `CUBECoreintrinsicDtypeMap` -> record unit provenance as `ExecutionUnit::Cube`
- `VectorCoreintrinsicDtypeMap` -> record unit provenance as `ExecutionUnit::Vector`

Continue scanning `AICoreMemoryRates` for intrinsic-like keys as a fallback for movement intrinsic names with no dtype list.

- [ ] **Step 2: Merge duplicate intrinsic names**

When the same intrinsic name appears in multiple sections, keep one `TargetIntrinsicInfo` entry and merge dtype strings and units without duplicates. Preserve final deterministic sorting by intrinsic name and dtype order.

Add a unit test with a synthetic CANN `platform_config` tree that proves duplicate dtype strings and unit provenance survive the merge across AICore/CUBECore/VectorCore sections.

- [ ] **Step 3: Add real CANN profile smoke checks**

Update `test/Target/ascend-target-profile.mlir` to keep the existing movement intrinsic check and add checks for at least:

```mlir
// CHECK: intrinsic = "Intrinsic_mmad"
// CHECK: intrinsic = "Intrinsic_vadd"
```

- [ ] **Step 4: Run target profile smoke**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Target/ascend-target-profile.mlir'
```

Expected: target profile lit still passes.

## Task 4: Tracking And Verification

**Files:**
- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [ ] **Step 1: Update Phase 4 table**

Change the `TargetIntrinsicModel` row status from `Planned` to `Done`:

```markdown
| `TargetIntrinsicModel MVP` | `Done` | query-only intrinsic table、unit map、movement map、compute map、dtype-pattern token lookup；ProfileVerifier / path constraints / memory-model integration 后续继续推进 | `AscendTargetIntrinsicModelTest` + target profile lit |
```

- [ ] **Step 2: Run final verification**

```bash
test/tools/check_ascend_no_v2_code_naming.sh
git diff --check -- . ':!AGENTS.md'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt AscendCommonAttributesTest AscendKernelPatternTest AscendRealizePlannerTest AscendTargetMemoryModelTest AscendTargetIntrinsicModelTest'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ctest --test-dir build -R "Ascend(CommonAttributes|KernelPattern|RealizePlanner|TargetMemoryModel|TargetIntrinsicModel)Test" --output-on-failure'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Target/ascend-target-profile.mlir'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion --filter="ascend-"'
```

Expected: all commands pass.

- [ ] **Step 3: Commit only related files**

```bash
git status --short
git add include/Target/Ascend/TargetProfile.h include/Target/Ascend/TargetMemoryModel.h include/Target/Ascend/TargetIntrinsicModel.h lib/Target/Ascend/TargetIntrinsicModel.cpp lib/Target/Ascend/CannTargetProfileLoader.cpp lib/Target/Ascend/CMakeLists.txt test/unittests/Target/CMakeLists.txt test/unittests/Target/AscendTargetIntrinsicModelTest.cpp test/Target/ascend-target-profile.mlir docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md docs/superpowers/plans/2026-05-10-ascend-target-intrinsic-model-mvp.md
git commit -m "feat: add Ascend target intrinsic model MVP"
git push
```

Do not stage `AGENTS.md`.
