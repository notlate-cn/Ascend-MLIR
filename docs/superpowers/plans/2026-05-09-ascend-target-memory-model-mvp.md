# Ascend Target Memory Model MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start Phase 4 by turning the existing CANN-derived `TargetProfile` into a queryable `TargetMemoryModel`.

**Architecture:** Keep `CannTargetProfileLoader` as the source of raw SoC facts, then derive a logical memory model from those facts. The MVP implements the V2-8 logical memory places, capacity rules, basic alignment and visibility rules, direct path graph queries, and positive-capacity validation; multi-hop routing, intrinsic-backed path validation, and cost model logic are deferred.

**Spec-review corrections applied locally:**
- `MemoryPlace` uses the explicit V2-8 `memory_space` numeric values: `GM=0`、`A1=1`、`A2=2`、`B1=3`、`B2=4`、`CO1=7`、`VECIN=9`、`VECOUT=10`、`VECCALC=11`、`GMFlat=22`.
- This MVP exposes direct-edge lookup as `findDirectPaths(src, dst)`; multi-hop route search is deferred to a later target routing task.
- Intrinsic-backed required-path validation is deferred to `TargetIntrinsicModel` / `ProfileVerifier`; this task must not expand intrinsic parsing or fail on missing intrinsics.
- `TargetProfile` does not store `TargetMemoryModel` by value. Consumers build the model from `TargetProfile` where needed to avoid include cycles.

**Tech Stack:** MLIR/LLVM C++ libraries, LLVM ADT containers, GoogleTest unit tests, LIT/FileCheck, xvm/docker verification via `examples/dev-env.md`.

---

## File Structure

- Modify `include/Target/Ascend/TargetProfile.h`
- Create `include/Target/Ascend/TargetMemoryModel.h`
- Create `lib/Target/Ascend/TargetMemoryModel.cpp`
- Modify `lib/Target/Ascend/CannTargetProfileLoader.cpp`
- Modify `lib/Target/Ascend/TargetProfile.cpp`
- Modify `lib/Target/Ascend/CMakeLists.txt`
- Create `test/unittests/Target/CMakeLists.txt`
- Create `test/unittests/Target/AscendTargetMemoryModelTest.cpp`
- Modify `test/unittests/CMakeLists.txt`
- Modify `test/Target/ascend-target-profile.mlir`
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

Do not modify or stage `AGENTS.md`; it contains unrelated local user edits.

## Task 1: Red Tests For Logical Memory Model

**Files:**
- Create: `test/unittests/Target/CMakeLists.txt`
- Create: `test/unittests/Target/AscendTargetMemoryModelTest.cpp`
- Modify: `test/unittests/CMakeLists.txt`

- [ ] **Step 1: Add the target unit test directory**

In `test/unittests/CMakeLists.txt`, add the target test subdirectory inside the `_runtime_unittest_support_ready` branch:

```cmake
  add_subdirectory(Target)
```

Create `test/unittests/Target/CMakeLists.txt`:

```cmake
add_executable(AscendTargetMemoryModelTest
  AscendTargetMemoryModelTest.cpp
)

target_link_libraries(AscendTargetMemoryModelTest PRIVATE
  RuntimeUnitTestSupport
  AscendTargetProfile
)

add_dependencies(RuntimeUnitTests AscendTargetMemoryModelTest)

add_test(NAME AscendTargetMemoryModelTest
  COMMAND $<TARGET_FILE:AscendTargetMemoryModelTest>
)
```

- [ ] **Step 2: Add unit tests that describe the desired API**

Create `test/unittests/Target/AscendTargetMemoryModelTest.cpp` with tests for:

```cpp
#include "Target/Ascend/TargetMemoryModel.h"
#include "gtest/gtest.h"

using namespace mlir::ascend;

static TargetProfile makeCompleteProfile() {
  TargetProfile profile;
  profile.identity.socVersion = "TestSoC";
  profile.hardware.aiCoreCount = 1;
  profile.hardware.l1SizeBytes = 512 * 1024;
  profile.hardware.ubSizeBytes = 192 * 1024;
  profile.hardware.supportFixpipe = true;
  profile.capacityBytes[MemoryPlace::GM] = 1024 * 1024 * 1024;
  profile.capacityBytes[MemoryPlace::A1] = 512 * 1024;
  profile.capacityBytes[MemoryPlace::B1] = 512 * 1024;
  profile.capacityBytes[MemoryPlace::A2] = 64 * 1024;
  profile.capacityBytes[MemoryPlace::B2] = 64 * 1024;
  profile.capacityBytes[MemoryPlace::CO1] = 128 * 1024;
  profile.capacityBytes[MemoryPlace::VECIN] = 192 * 1024;
  profile.capacityBytes[MemoryPlace::VECOUT] = 192 * 1024;
  profile.capacityBytes[MemoryPlace::VECCALC] = 192 * 1024;
  return profile;
}

TEST(AscendTargetMemoryModelTest, BuildsLogicalPlacesAndCapacities) {
  llvm::raw_null_ostream os;
  FailureOr<TargetMemoryModel> model =
      TargetMemoryModelBuilder().build(makeCompleteProfile(), os);
  ASSERT_TRUE(succeeded(model));
  EXPECT_TRUE(model->supportsMemoryPlace(MemoryPlace::A1));
  EXPECT_FALSE(model->supportsMemoryPlace(MemoryPlace::GMFlat));
  ASSERT_TRUE(succeeded(model->getCapacity(MemoryPlace::A2)));
  EXPECT_EQ(model->getCapacity(MemoryPlace::A2)->staticCapacityBytes,
            64 * 1024);
}

TEST(AscendTargetMemoryModelTest, UsesDocumentedMemorySpaceEncodings) {
  EXPECT_EQ(static_cast<int>(MemoryPlace::GM), 0);
  EXPECT_EQ(static_cast<int>(MemoryPlace::A1), 1);
  EXPECT_EQ(static_cast<int>(MemoryPlace::A2), 2);
  EXPECT_EQ(static_cast<int>(MemoryPlace::B1), 3);
  EXPECT_EQ(static_cast<int>(MemoryPlace::B2), 4);
  EXPECT_EQ(static_cast<int>(MemoryPlace::CO1), 7);
  EXPECT_EQ(static_cast<int>(MemoryPlace::VECIN), 9);
  EXPECT_EQ(static_cast<int>(MemoryPlace::VECOUT), 10);
  EXPECT_EQ(static_cast<int>(MemoryPlace::VECCALC), 11);
  EXPECT_EQ(static_cast<int>(MemoryPlace::GMFlat), 22);
}

TEST(AscendTargetMemoryModelTest, ProvidesVisibilityAndAlignmentRules) {
  llvm::raw_null_ostream os;
  FailureOr<TargetMemoryModel> model =
      TargetMemoryModelBuilder().build(makeCompleteProfile(), os);
  ASSERT_TRUE(succeeded(model));
  EXPECT_TRUE(model->isPlaceVisibleTo(MemoryPlace::A2, ExecutionUnit::Cube));
  EXPECT_FALSE(model->isPlaceVisibleTo(MemoryPlace::A2, ExecutionUnit::Vector));
  ASSERT_TRUE(succeeded(model->getAlignment(MemoryPlace::GM)));
  EXPECT_EQ(model->getAlignment(MemoryPlace::GM)->addressAlignmentBytes, 32);
}

TEST(AscendTargetMemoryModelTest, BuildsRequiredDirectPathGraph) {
  llvm::raw_null_ostream os;
  FailureOr<TargetMemoryModel> model =
      TargetMemoryModelBuilder().build(makeCompleteProfile(), os);
  ASSERT_TRUE(succeeded(model));
  EXPECT_EQ(model->findDirectPaths(MemoryPlace::GM, MemoryPlace::A1).size(), 1u);
  EXPECT_EQ(model->findDirectPaths(MemoryPlace::GM, MemoryPlace::B1).size(), 1u);
  EXPECT_EQ(model->findDirectPaths(MemoryPlace::A1, MemoryPlace::A2).size(), 1u);
  EXPECT_EQ(model->findDirectPaths(MemoryPlace::B1, MemoryPlace::B2).size(), 1u);
  EXPECT_EQ(model->findDirectPaths(MemoryPlace::CO1, MemoryPlace::VECIN).size(), 1u);
  EXPECT_EQ(model->findDirectPaths(MemoryPlace::GM, MemoryPlace::VECIN).size(), 1u);
  EXPECT_EQ(model->findDirectPaths(MemoryPlace::VECOUT, MemoryPlace::GM).size(), 1u);
  EXPECT_EQ(model->findDirectPaths(MemoryPlace::CO1, MemoryPlace::GM).size(), 1u);
  EXPECT_EQ(model->findDirectPaths(MemoryPlace::GM, MemoryPlace::VECCALC).size(), 0u);

  auto edge = model->findDirectPaths(MemoryPlace::CO1, MemoryPlace::VECIN).front();
  ASSERT_TRUE(succeeded(model->getPathKind(edge)));
  EXPECT_EQ(*model->getPathKind(edge), PathKind::QueueTransfer);

  auto fixpipeEdge =
      model->findDirectPaths(MemoryPlace::CO1, MemoryPlace::GM).front();
  ASSERT_TRUE(succeeded(model->getPathKind(fixpipeEdge)));
  EXPECT_EQ(*model->getPathKind(fixpipeEdge), PathKind::FixPipe);
}

TEST(AscendTargetMemoryModelTest, RejectsMissingRequiredPlace) {
  TargetProfile profile = makeCompleteProfile();
  profile.capacityBytes.erase(MemoryPlace::A1);

  std::string message;
  llvm::raw_string_ostream os(message);
  FailureOr<TargetMemoryModel> model =
      TargetMemoryModelBuilder().build(profile, os);
  EXPECT_TRUE(failed(model));
  EXPECT_NE(message.find("missing required memory place A1"),
            std::string::npos);
}

TEST(AscendTargetMemoryModelTest, RejectsZeroCapacityRequiredPlace) {
  TargetProfile profile = makeCompleteProfile();
  profile.capacityBytes[MemoryPlace::VECIN] = 0;

  std::string message;
  llvm::raw_string_ostream os(message);
  FailureOr<TargetMemoryModel> model =
      TargetMemoryModelBuilder().build(profile, os);
  EXPECT_TRUE(failed(model));
  EXPECT_NE(message.find("missing required memory place VECIN"),
            std::string::npos);
}
```

- [ ] **Step 3: Run RED verification in xvm/docker**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build AscendTargetMemoryModelTest'
```

Expected: FAIL because `TargetMemoryModel.h` and `AscendTargetMemoryModelTest` are not implemented yet.

## Task 2: Implement TargetMemoryModel

**Files:**
- Modify: `include/Target/Ascend/TargetProfile.h`
- Create: `include/Target/Ascend/TargetMemoryModel.h`
- Create: `lib/Target/Ascend/TargetMemoryModel.cpp`
- Modify: `lib/Target/Ascend/CMakeLists.txt`

- [ ] **Step 1: Replace physical-only target places with V2-8 logical places**

Update `MemoryPlace` in `TargetProfile.h`:

```cpp
enum class MemoryPlace {
  GM = 0,
  A1 = 1,
  A2 = 2,
  B1 = 3,
  B2 = 4,
  CO1 = 7,
  VECIN = 9,
  VECOUT = 10,
  VECCALC = 11,
  GMFlat = 22
};
```

Keep `GMFlat` out of the normal placement model; it is an ABI-only place for later phases.

- [ ] **Step 2: Add the target memory model public API**

Create `include/Target/Ascend/TargetMemoryModel.h` with:

```cpp
enum class ExecutionUnit { DMA, Cube, Vector };
enum class PathKind { DirectCopy, Load2D, Load2DTranspose, FixPipe, QueueTransfer };

struct PathEdge {
  MemoryPlace srcPlace;
  MemoryPlace dstPlace;
  unsigned pathVariant = 0;
  bool operator==(const PathEdge &other) const;
};

struct CapacityRule {
  int64_t staticCapacityBytes = 0;
  int64_t availableCapacityBytes = 0;
  bool partitionedByUnit = false;
};

struct AlignmentRule {
  int64_t addressAlignmentBytes = 32;
  int64_t strideAlignmentBytes = 32;
  int64_t tileAlignmentElements = 1;
  bool requiresPowerOfTwo = false;
};

struct VisibilityRule {
  bool dma = false;
  bool cube = false;
  bool vector = false;
  bool abiVisible = false;
};

class TargetMemoryModel {
public:
  ArrayRef<MemoryPlace> getMemoryPlaces() const;
  bool supportsMemoryPlace(MemoryPlace place) const;
  FailureOr<CapacityRule> getCapacity(MemoryPlace place) const;
  FailureOr<AlignmentRule> getAlignment(MemoryPlace place) const;
  bool isPlaceVisibleTo(MemoryPlace place, ExecutionUnit unit) const;
  SmallVector<PathEdge> findDirectPaths(MemoryPlace src, MemoryPlace dst) const;
  FailureOr<PathKind> getPathKind(const PathEdge &edge) const;

private:
  friend class TargetMemoryModelBuilder;
  SmallVector<MemoryPlace> memoryPlaces;
  DenseMap<MemoryPlace, CapacityRule> capacity;
  DenseMap<MemoryPlace, AlignmentRule> alignment;
  DenseMap<MemoryPlace, VisibilityRule> visibilityRules;
  DenseMap<MemoryPlace, SmallVector<PathEdge>> pathGraph;
  DenseMap<PathEdge, PathKind> pathKinds;
};

class TargetMemoryModelBuilder {
public:
  FailureOr<TargetMemoryModel> build(const TargetProfile &profile,
                                     raw_ostream &os) const;
};
```

Add `llvm::DenseMapInfo<mlir::ascend::PathEdge>` in the header or source so `DenseMap<PathEdge, PathKind>` works.

- [ ] **Step 3: Implement builder rules**

In `TargetMemoryModel.cpp`:

- Require positive capacities for `GM`, `A1`, `A2`, `B1`, `B2`, `CO1`, `VECIN`, `VECOUT`, and `VECCALC`.
- Exclude `GMFlat` from `memoryPlaces`.
- Use 32-byte default address and stride alignment for every place.
- Visibility:
  - `GM`: DMA-visible and ABI-visible.
  - `A1`, `A2`, `B1`, `B2`, `CO1`: DMA-visible and Cube-visible.
  - `VECIN`, `VECOUT`, `VECCALC`: DMA-visible and Vector-visible.
- Required direct paths:
  - `GM -> A1`: `Load2D`
  - `GM -> B1`: `Load2D`
  - `A1 -> A2`: `DirectCopy`
  - `B1 -> B2`: `DirectCopy`
  - `CO1 -> VECIN`: `QueueTransfer`
  - `GM -> VECIN`: `DirectCopy`
  - `VECOUT -> GM`: `DirectCopy`
- Optional fixpipe path:
  - Add `CO1 -> GM` with `PathKind::FixPipe` only when `profile.hardware.supportFixpipe` is true.
- Intrinsic-backed required-path validation remains deferred to `TargetIntrinsicModel` and `ProfileVerifier`; do not expand intrinsic parsing in this task.

- [ ] **Step 4: Run GREEN unit tests**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build AscendTargetMemoryModelTest && ./build/bin/AscendTargetMemoryModelTest'
```

Expected: unit tests pass.

## Task 3: Connect Loader And Print Pass

**Files:**
- Modify: `lib/Target/Ascend/CannTargetProfileLoader.cpp`
- Modify: `lib/Target/Ascend/TargetProfile.cpp`
- Modify: `test/Target/ascend-target-profile.mlir`

- [ ] **Step 1: Map CANN physical capacities to logical places**

In `CannTargetProfileLoader.cpp`, populate:

- `memory_size` -> `GM`
- `l1_size` -> `A1` and `B1`
- `l0_a_size` -> `A2`
- `l0_b_size` -> `B2`
- `l0_c_size` -> `CO1`
- `ub_size` -> `VECIN`, `VECOUT`, `VECCALC`

Do not store `TargetMemoryModel` by value inside `TargetProfile`. The print pass builds a local model from the loaded profile before printing logical places.

- [ ] **Step 2: Print logical memory places from the memory model**

Update `AscendPrintTargetProfilePass` to build a local `TargetMemoryModel` from `TargetProfile`, then print `getMemoryPlaces()` and `getCapacity(place)`.

Update `stringifyMemoryPlace` cases to return:

```cpp
"GM", "A1", "A2", "B1", "B2", "CO1", "VECIN", "VECOUT", "VECCALC", "GM_FLAT"
```

- [ ] **Step 3: Update target profile LIT expectations**

In `test/Target/ascend-target-profile.mlir`, check logical places:

```mlir
// CHECK: memory_place = "GM"
// CHECK: memory_place = "A1"
// CHECK: memory_place = "A2"
// CHECK: memory_place = "B1"
// CHECK: memory_place = "B2"
// CHECK: memory_place = "CO1"
// CHECK: memory_place = "VECIN"
// CHECK: memory_place = "VECOUT"
// CHECK: memory_place = "VECCALC"
```

- [ ] **Step 4: Run target LIT**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Target/ascend-target-profile.mlir'
```

Expected: pass.

## Task 4: Tracking And Verification

**Files:**
- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [ ] **Step 1: Update Phase 4 table**

Change the `TargetMemoryModel` row status from `Planned` to `Done` and note:

```markdown
| `TargetMemoryModel` | `Done` | logical places、capacity、alignment、visibility、direct path graph；multi-hop routing 与 intrinsic-backed path validation 延后到 TargetRouting / TargetIntrinsicModel / ProfileVerifier | `AscendTargetMemoryModelTest` + target profile lit |
```

- [ ] **Step 2: Run final verification**

```bash
test/tools/check_ascend_no_v2_code_naming.sh
git diff --check
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt AscendTargetMemoryModelTest AscendCommonAttributesTest AscendKernelPatternTest AscendRealizePlannerTest'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ./build/bin/AscendTargetMemoryModelTest && ctest --test-dir build -R "Ascend(CommonAttributes|KernelPattern|RealizePlanner|TargetMemoryModel)Test" --output-on-failure'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Target/ascend-target-profile.mlir'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion --filter="ascend-"'
```

Expected: all commands pass.

- [ ] **Step 3: Commit only related files**

```bash
git status --short
git add include/Target/Ascend/TargetProfile.h include/Target/Ascend/TargetMemoryModel.h lib/Target/Ascend/TargetMemoryModel.cpp lib/Target/Ascend/CannTargetProfileLoader.cpp lib/Target/Ascend/TargetProfile.cpp lib/Target/Ascend/CMakeLists.txt test/unittests/CMakeLists.txt test/unittests/Target/CMakeLists.txt test/unittests/Target/AscendTargetMemoryModelTest.cpp test/Target/ascend-target-profile.mlir docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md docs/superpowers/plans/2026-05-09-ascend-target-memory-model-mvp.md
git commit -m "feat: add Ascend target memory model MVP"
git push
```

Do not stage `AGENTS.md`.
