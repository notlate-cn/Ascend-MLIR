# Ascend Target Cost Model MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue Phase 4 by adding a query-only `TargetCostModel` derived from CANN memory-rate entries and direct memory paths.

**Architecture:** Keep `TargetProfile` as the raw CANN-derived fact carrier by adding `TargetMemoryRateInfo`. Build a local `TargetCostModel` from `TargetProfile` and `TargetMemoryModel`, following the existing `TargetMemoryModelBuilder` and `TargetIntrinsicModelBuilder` pattern. The MVP supports exact memory-rate lookup, preferred-rate lookup, direct-path cost lookup, and transfer-cycle estimation; it does not mutate IR and does not feed scheduling or realization yet.

**Tech Stack:** MLIR/LLVM C++ libraries, LLVM ADT containers, GoogleTest unit tests, LIT/FileCheck, xvm/docker verification via `examples/dev-env.md`.

---

## File Structure

- Modify `include/Target/Ascend/TargetProfile.h`
- Create `include/Target/Ascend/TargetCostModel.h`
- Create `lib/Target/Ascend/TargetCostModel.cpp`
- Modify `lib/Target/Ascend/CannTargetProfileLoader.cpp`
- Modify `lib/Target/Ascend/TargetProfile.cpp`
- Modify `lib/Target/Ascend/CMakeLists.txt`
- Modify `test/unittests/Target/CMakeLists.txt`
- Create `test/unittests/Target/AscendTargetCostModelTest.cpp`
- Modify `test/Target/ascend-target-profile.mlir`
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

Do not modify or stage `AGENTS.md`; it contains unrelated local user edits.

## Design Decisions

- `TargetCostModel` is query-only in this MVP.
- CANN memory rates are stored raw in `TargetProfile::memoryRates` as section/name/value records.
- `CannTargetProfileLoader` parses `[AICoreMemoryRates]` and `[VectorCoreMemoryRates]`.
- Entries whose key starts with `Intrinsic_` remain intrinsic fallback metadata and must not enter `memoryRates`.
- Exact lookup is section-qualified: `getMemoryRate("AICoreMemoryRates", "ddr_read_rate")`.
- Preferred lookup chooses `AICoreMemoryRates` over `VectorCoreMemoryRates`, then any other section by lexical order.
- `TargetCostModelBuilder` creates direct-path costs for every direct edge exposed by `TargetMemoryModel`.
- Path cost lookup is section-aware:
  - vector-side paths such as `GM -> VECIN` and `VECOUT -> GM` try `VectorCoreMemoryRates` before `AICoreMemoryRates`
  - all other MVP direct paths try `AICoreMemoryRates` before `VectorCoreMemoryRates`
- Path-rate mapping for MVP:
  - `A1 -> A2` uses `l1_to_l0_a_rate`
  - `B1 -> B2` uses `l1_to_l0_b_rate`
  - `CO1 -> VECIN` uses `l0_c_to_ub_rate`
  - `VECOUT -> GM` uses `ub_to_ddr_rate`
  - any `src == GM` path prefers `ddr_read_rate`, then `ddr_rate`
  - any `dst == GM` path prefers `ddr_write_rate`, then `ub_to_ddr_rate`, then `ddr_rate`
- Fixed startup cycles for MVP:
  - `DirectCopy`: 1
  - `Load2D`: 2
  - `Load2DTranspose`: 3
  - `FixPipe`: 4
  - `QueueTransfer`: 0
- `QueueTransfer` is marked `overlapsCompute = true`; other path kinds are `false`.
- `estimateTransferCycles(edge, bytes)` returns `startupCycles + ceil(bytes / bytesPerCycle)`.

## Task 1: Red Tests For TargetCostModel

**Files:**
- Create: `test/unittests/Target/AscendTargetCostModelTest.cpp`
- Modify: `test/unittests/Target/CMakeLists.txt`

- [ ] **Step 1: Add the test target**

Append to `test/unittests/Target/CMakeLists.txt`:

```cmake
add_executable(AscendTargetCostModelTest
  AscendTargetCostModelTest.cpp
)

target_link_libraries(AscendTargetCostModelTest PRIVATE
  RuntimeUnitTestSupport
  AscendTargetProfile
)

add_dependencies(RuntimeUnitTests AscendTargetCostModelTest)

add_test(NAME AscendTargetCostModelTest
  COMMAND $<TARGET_FILE:AscendTargetCostModelTest>
)
```

- [ ] **Step 2: Add unit tests for cost queries**

Create `test/unittests/Target/AscendTargetCostModelTest.cpp`:

```cpp
//===- AscendTargetCostModelTest.cpp -------------------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/Ascend/CannTargetProfileLoader.h"
#include "Target/Ascend/TargetCostModel.h"
#include "Target/Ascend/TargetMemoryModel.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace mlir::ascend;
using llvm::FailureOr;
using llvm::failed;
using llvm::succeeded;

namespace {

struct TempDir {
  llvm::SmallString<256> path;

  ~TempDir() {
    if (!path.empty())
      (void)llvm::sys::fs::remove_directories(path);
  }
};

static TargetProfile makeProfileWithRates() {
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
  profile.memoryRates.push_back(
      {"AICoreMemoryRates", "ddr_rate", 32});
  profile.memoryRates.push_back(
      {"AICoreMemoryRates", "ddr_read_rate", 32});
  profile.memoryRates.push_back(
      {"AICoreMemoryRates", "ddr_write_rate", 32});
  profile.memoryRates.push_back(
      {"AICoreMemoryRates", "l1_to_l0_a_rate", 512});
  profile.memoryRates.push_back(
      {"AICoreMemoryRates", "l1_to_l0_b_rate", 256});
  profile.memoryRates.push_back(
      {"AICoreMemoryRates", "l0_c_to_ub_rate", 128});
  profile.memoryRates.push_back(
      {"AICoreMemoryRates", "ub_to_ddr_rate", 64});
  profile.memoryRates.push_back(
      {"VectorCoreMemoryRates", "ddr_read_rate", 48});
  profile.memoryRates.push_back(
      {"VectorCoreMemoryRates", "ub_to_ddr_rate", 96});
  return profile;
}

static FailureOr<TargetMemoryModel>
buildMemoryModel(const TargetProfile &profile) {
  llvm::raw_null_ostream os;
  return TargetMemoryModelBuilder().build(profile, os);
}

static FailureOr<TargetCostModel>
buildCostModel(const TargetProfile &profile,
               const TargetMemoryModel &memoryModel,
               std::string *message = nullptr) {
  if (!message) {
    llvm::raw_null_ostream os;
    return TargetCostModelBuilder().build(profile, memoryModel, os);
  }

  llvm::raw_string_ostream os(*message);
  return TargetCostModelBuilder().build(profile, memoryModel, os);
}

static PathEdge onlyDirectPath(const TargetMemoryModel &model,
                               MemoryPlace src, MemoryPlace dst) {
  SmallVector<PathEdge> paths = model.findDirectPaths(src, dst);
  EXPECT_EQ(paths.size(), 1u);
  return paths.front();
}

static const TargetMemoryRateInfo *
findRate(const TargetProfile &profile, llvm::StringRef section,
         llvm::StringRef name) {
  for (const TargetMemoryRateInfo &rate : profile.memoryRates)
    if (rate.section == section && rate.name == name)
      return &rate;
  return nullptr;
}

static void writeSyntheticCannProfile(llvm::StringRef root) {
  llvm::SmallString<256> configDir(root);
  llvm::sys::path::append(configDir, "aarch64-linux", "data",
                          "platform_config");
  ASSERT_FALSE(llvm::sys::fs::create_directories(configDir));

  llvm::SmallString<256> iniPath(configDir);
  llvm::sys::path::append(iniPath, "SyntheticSoC.ini");

  std::error_code ec;
  llvm::raw_fd_ostream os(iniPath, ec);
  ASSERT_FALSE(ec) << ec.message();

  os << R"ini(
[version]
SoC_version=SyntheticSoC
Short_SoC_version=SYN
NpuArch=SYNTH

[SoCInfo]
ai_core_cnt=1
memory_size=1048576
support_bf16=true

[AICoreSpec]
l1_size=524288
l0_a_size=65536
l0_b_size=65536
l0_c_size=131072
ub_size=196608
support_fixpipe=true

[AICoreMemoryRates]
ddr_rate=32
ddr_read_rate=32
l1_to_l0_a_rate=512
Intrinsic_data_move_out2l1=64

[VectorCoreMemoryRates]
ub_to_ddr_rate=96
)ini";
}

} // namespace

TEST(AscendTargetCostModelTest, LooksUpExactAndPreferredRates) {
  TargetProfile profile = makeProfileWithRates();
  FailureOr<TargetMemoryModel> memoryModel = buildMemoryModel(profile);
  ASSERT_TRUE(succeeded(memoryModel));
  FailureOr<TargetCostModel> costModel = buildCostModel(profile, *memoryModel);
  ASSERT_TRUE(succeeded(costModel));

  ASSERT_TRUE(
      succeeded(costModel->getMemoryRate("AICoreMemoryRates", "ddr_rate")));
  EXPECT_EQ(*costModel->getMemoryRate("AICoreMemoryRates", "ddr_rate"), 32);
  EXPECT_EQ(*costModel->getMemoryRate("VectorCoreMemoryRates",
                                      "ub_to_ddr_rate"),
            96);
  EXPECT_EQ(*costModel->getPreferredMemoryRate("ub_to_ddr_rate"), 64);
  EXPECT_TRUE(failed(costModel->getMemoryRate("AICoreMemoryRates",
                                             "missing_rate")));
}

TEST(AscendTargetCostModelTest, BuildsPathCostsForDirectEdges) {
  TargetProfile profile = makeProfileWithRates();
  FailureOr<TargetMemoryModel> memoryModel = buildMemoryModel(profile);
  ASSERT_TRUE(succeeded(memoryModel));
  FailureOr<TargetCostModel> costModel = buildCostModel(profile, *memoryModel);
  ASSERT_TRUE(succeeded(costModel));

  PathEdge gmToA1 = onlyDirectPath(*memoryModel, MemoryPlace::GM,
                                   MemoryPlace::A1);
  FailureOr<PathCost> gmToA1Cost = costModel->getPathCost(gmToA1);
  ASSERT_TRUE(succeeded(gmToA1Cost));
  EXPECT_EQ(gmToA1Cost->rateName, "ddr_read_rate");
  EXPECT_EQ(gmToA1Cost->bytesPerCycle, 32);
  EXPECT_EQ(gmToA1Cost->startupCycles, 2);
  EXPECT_FALSE(gmToA1Cost->overlapsCompute);
  EXPECT_EQ(*costModel->estimateTransferCycles(gmToA1, 64), 4);

  PathEdge a1ToA2 = onlyDirectPath(*memoryModel, MemoryPlace::A1,
                                   MemoryPlace::A2);
  FailureOr<PathCost> a1ToA2Cost = costModel->getPathCost(a1ToA2);
  ASSERT_TRUE(succeeded(a1ToA2Cost));
  EXPECT_EQ(a1ToA2Cost->rateName, "l1_to_l0_a_rate");
  EXPECT_EQ(a1ToA2Cost->bytesPerCycle, 512);
  EXPECT_EQ(*costModel->estimateTransferCycles(a1ToA2, 1024), 3);

  PathEdge queue = onlyDirectPath(*memoryModel, MemoryPlace::CO1,
                                  MemoryPlace::VECIN);
  FailureOr<PathCost> queueCost = costModel->getPathCost(queue);
  ASSERT_TRUE(succeeded(queueCost));
  EXPECT_EQ(queueCost->rateName, "l0_c_to_ub_rate");
  EXPECT_TRUE(queueCost->overlapsCompute);

  PathEdge gmToVecin = onlyDirectPath(*memoryModel, MemoryPlace::GM,
                                      MemoryPlace::VECIN);
  FailureOr<PathCost> gmToVecinCost = costModel->getPathCost(gmToVecin);
  ASSERT_TRUE(succeeded(gmToVecinCost));
  EXPECT_EQ(gmToVecinCost->rateSection, "VectorCoreMemoryRates");
  EXPECT_EQ(gmToVecinCost->rateName, "ddr_read_rate");
  EXPECT_EQ(gmToVecinCost->bytesPerCycle, 48);

  PathEdge vecoutToGm = onlyDirectPath(*memoryModel, MemoryPlace::VECOUT,
                                       MemoryPlace::GM);
  FailureOr<PathCost> vecoutToGmCost = costModel->getPathCost(vecoutToGm);
  ASSERT_TRUE(succeeded(vecoutToGmCost));
  EXPECT_EQ(vecoutToGmCost->rateSection, "VectorCoreMemoryRates");
  EXPECT_EQ(vecoutToGmCost->rateName, "ub_to_ddr_rate");
  EXPECT_EQ(vecoutToGmCost->bytesPerCycle, 96);
  EXPECT_EQ(*costModel->estimateTransferCycles(vecoutToGm, 192), 3);
}

TEST(AscendTargetCostModelTest, RejectsMissingRequiredPathRate) {
  TargetProfile profile = makeProfileWithRates();
  llvm::erase_if(profile.memoryRates, [](const TargetMemoryRateInfo &rate) {
    return rate.name == "l1_to_l0_b_rate";
  });

  FailureOr<TargetMemoryModel> memoryModel = buildMemoryModel(profile);
  ASSERT_TRUE(succeeded(memoryModel));

  std::string message;
  FailureOr<TargetCostModel> costModel =
      buildCostModel(profile, *memoryModel, &message);
  EXPECT_TRUE(failed(costModel));
  EXPECT_NE(message.find("AICoreMemoryRates.l1_to_l0_b_rate"),
            std::string::npos);
  EXPECT_NE(message.find("B1 -> B2"), std::string::npos);
}

TEST(AscendTargetCostModelTest, LoaderParsesMemoryRates) {
  TempDir temp;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("ascend-cann-cost",
                                                    temp.path));
  writeSyntheticCannProfile(temp.path);

  FailureOr<TargetProfile> profile =
      CannTargetProfileLoader::load(temp.path, "SyntheticSoC");
  ASSERT_TRUE(succeeded(profile));

  const TargetMemoryRateInfo *ddr =
      findRate(*profile, "AICoreMemoryRates", "ddr_rate");
  ASSERT_NE(ddr, nullptr);
  EXPECT_EQ(ddr->bytesPerCycle, 32);

  const TargetMemoryRateInfo *vectorRate =
      findRate(*profile, "VectorCoreMemoryRates", "ub_to_ddr_rate");
  ASSERT_NE(vectorRate, nullptr);
  EXPECT_EQ(vectorRate->bytesPerCycle, 96);

  EXPECT_EQ(findRate(*profile, "AICoreMemoryRates",
                     "Intrinsic_data_move_out2l1"),
            nullptr);
}
```

- [ ] **Step 3: Run RED verification in xvm/docker**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build AscendTargetCostModelTest'
```

Expected: FAIL because `Target/Ascend/TargetCostModel.h` and `TargetMemoryRateInfo` do not exist yet.

## Task 2: Add TargetProfile Memory Rate Carrier And Loader Parsing

**Files:**
- Modify: `include/Target/Ascend/TargetProfile.h`
- Modify: `lib/Target/Ascend/CannTargetProfileLoader.cpp`
- Modify: `lib/Target/Ascend/TargetProfile.cpp`
- Modify: `test/Target/ascend-target-profile.mlir`

- [ ] **Step 1: Add raw memory-rate carrier**

In `include/Target/Ascend/TargetProfile.h`, add:

```cpp
struct TargetMemoryRateInfo {
  std::string section;
  std::string name;
  int64_t bytesPerCycle = 0;
};
```

Then extend `TargetProfile`:

```cpp
struct TargetProfile {
  TargetIdentity identity;
  TargetHardwareInfo hardware;
  DenseMap<MemoryPlace, int64_t> capacityBytes;
  SmallVector<TargetIntrinsicInfo> intrinsics;
  SmallVector<TargetMemoryRateInfo> memoryRates;
};
```

- [ ] **Step 2: Parse memory-rate sections**

In `lib/Target/Ascend/CannTargetProfileLoader.cpp`, add helpers:

```cpp
void appendMemoryRate(TargetProfile &profile, StringRef section,
                      StringRef name, StringRef value) {
  name = trim(name);
  if (name.empty() || name.starts_with("Intrinsic_"))
    return;

  int64_t rate = parseInt64(value);
  if (rate <= 0)
    return;
  profile.memoryRates.push_back({section.str(), name.str(), rate});
}

void parseMemoryRates(TargetProfile &profile, const SectionMap &sections,
                      StringRef section) {
  auto sectionIt = sections.find(section);
  if (sectionIt == sections.end())
    return;
  for (const auto &entry : sectionIt->second)
    appendMemoryRate(profile, section, entry.first(), entry.second);
}
```

After capacity setup and before intrinsic fallback scanning, call:

```cpp
parseMemoryRates(profile, sections, "AICoreMemoryRates");
parseMemoryRates(profile, sections, "VectorCoreMemoryRates");
llvm::sort(profile.memoryRates, [](const TargetMemoryRateInfo &lhs,
                                   const TargetMemoryRateInfo &rhs) {
  return std::tie(lhs.section, lhs.name) < std::tie(rhs.section, rhs.name);
});
```

Keep the existing `AICoreMemoryRates` intrinsic fallback scan for `Intrinsic_*` entries.

- [ ] **Step 3: Print stable target-profile memory rates**

In `lib/Target/Ascend/TargetProfile.cpp`, after printing intrinsics, print sorted raw rates:

```cpp
SmallVector<const ascend::TargetMemoryRateInfo *> memoryRates;
memoryRates.reserve(profile.memoryRates.size());
for (const ascend::TargetMemoryRateInfo &rate : profile.memoryRates)
  memoryRates.push_back(&rate);
llvm::sort(memoryRates, [](const ascend::TargetMemoryRateInfo *lhs,
                           const ascend::TargetMemoryRateInfo *rhs) {
  return std::tie(lhs->section, lhs->name) <
         std::tie(rhs->section, rhs->name);
});

for (const ascend::TargetMemoryRateInfo *rate : memoryRates) {
  llvm::errs() << "  memory_rate = \"" << rate->section << "."
               << rate->name << "\" bytes_per_cycle = "
               << rate->bytesPerCycle << "\n";
}
```

Add `#include <tuple>` if required by the compiler.

- [ ] **Step 4: Add real CANN profile smoke checks**

Update `test/Target/ascend-target-profile.mlir`:

```mlir
// CHECK: memory_rate = "AICoreMemoryRates.ddr_rate" bytes_per_cycle = 32
// CHECK: memory_rate = "AICoreMemoryRates.l1_to_l0_a_rate" bytes_per_cycle = 512
```

## Task 3: Implement TargetCostModel

**Files:**
- Create: `include/Target/Ascend/TargetCostModel.h`
- Create: `lib/Target/Ascend/TargetCostModel.cpp`
- Modify: `lib/Target/Ascend/CMakeLists.txt`

- [ ] **Step 1: Add public API**

Create `include/Target/Ascend/TargetCostModel.h`:

```cpp
//===- TargetCostModel.h - Ascend target cost model ------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_TARGET_ASCEND_TARGET_COST_MODEL_H
#define ASCEND_MLIR_TARGET_ASCEND_TARGET_COST_MODEL_H

#include "Target/Ascend/TargetMemoryModel.h"
#include "Target/Ascend/TargetProfile.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <string>

namespace mlir::ascend {

struct PathCost {
  int64_t startupCycles = 0;
  int64_t bytesPerCycle = 0;
  bool overlapsCompute = false;
  std::string rateSection;
  std::string rateName;
};

class TargetCostModel {
public:
  FailureOr<int64_t> getMemoryRate(StringRef section, StringRef name) const;
  FailureOr<int64_t> getPreferredMemoryRate(StringRef name) const;
  FailureOr<PathCost> getPathCost(const PathEdge &edge) const;
  FailureOr<int64_t> estimateTransferCycles(const PathEdge &edge,
                                            int64_t bytes) const;

private:
  friend class TargetCostModelBuilder;
  llvm::StringMap<int64_t> ratesByQualifiedName;
  llvm::StringMap<int64_t> preferredRatesByName;
  DenseMap<PathEdge, PathCost> pathCosts;
};

class TargetCostModelBuilder {
public:
  FailureOr<TargetCostModel> build(const TargetProfile &profile,
                                   const TargetMemoryModel &memoryModel,
                                   raw_ostream &os) const;
};

} // namespace mlir::ascend

#endif // ASCEND_MLIR_TARGET_ASCEND_TARGET_COST_MODEL_H
```

- [ ] **Step 2: Implement model builder and queries**

Create `lib/Target/Ascend/TargetCostModel.cpp`:

```cpp
//===- TargetCostModel.cpp - Ascend target cost model --------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Target/Ascend/TargetCostModel.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

using namespace mlir;

namespace mlir::ascend {
namespace {

std::string makeRateKey(StringRef section, StringRef name) {
  std::string key = section.str();
  key += ".";
  key += name.str();
  return key;
}

int sectionPriority(StringRef section) {
  if (section == "AICoreMemoryRates")
    return 0;
  if (section == "VectorCoreMemoryRates")
    return 1;
  return 2;
}

int64_t startupCyclesFor(PathKind kind) {
  switch (kind) {
  case PathKind::DirectCopy:
    return 1;
  case PathKind::Load2D:
    return 2;
  case PathKind::Load2DTranspose:
    return 3;
  case PathKind::FixPipe:
    return 4;
  case PathKind::QueueTransfer:
    return 0;
  }
  llvm_unreachable("unknown path kind");
}

bool overlapsCompute(PathKind kind) {
  return kind == PathKind::QueueTransfer;
}

SmallVector<StringRef> getRateCandidates(const PathEdge &edge,
                                         PathKind kind) {
  if (edge.srcPlace == MemoryPlace::A1 && edge.dstPlace == MemoryPlace::A2)
    return {"l1_to_l0_a_rate"};
  if (edge.srcPlace == MemoryPlace::B1 && edge.dstPlace == MemoryPlace::B2)
    return {"l1_to_l0_b_rate"};
  if (edge.srcPlace == MemoryPlace::CO1 && edge.dstPlace == MemoryPlace::VECIN)
    return {"l0_c_to_ub_rate"};
  if (edge.srcPlace == MemoryPlace::VECOUT && edge.dstPlace == MemoryPlace::GM)
    return {"ub_to_ddr_rate"};
  if (edge.srcPlace == MemoryPlace::GM)
    return {"ddr_read_rate", "ddr_rate"};
  if (edge.dstPlace == MemoryPlace::GM)
    return {"ddr_write_rate", "ub_to_ddr_rate", "ddr_rate"};

  switch (kind) {
  case PathKind::DirectCopy:
    return {"ddr_rate"};
  case PathKind::Load2D:
  case PathKind::Load2DTranspose:
    return {"ddr_read_rate", "ddr_rate"};
  case PathKind::FixPipe:
    return {"ddr_write_rate", "ub_to_ddr_rate", "ddr_rate"};
  case PathKind::QueueTransfer:
    return {"l0_c_to_ub_rate"};
  }
  llvm_unreachable("unknown path kind");
}

SmallVector<StringRef> getSectionCandidates(const PathEdge &edge) {
  if ((edge.srcPlace == MemoryPlace::GM &&
       edge.dstPlace == MemoryPlace::VECIN) ||
      (edge.srcPlace == MemoryPlace::VECOUT &&
       edge.dstPlace == MemoryPlace::GM))
    return {"VectorCoreMemoryRates", "AICoreMemoryRates"};
  return {"AICoreMemoryRates", "VectorCoreMemoryRates"};
}

StringRef stringifyPathKind(PathKind kind) {
  switch (kind) {
  case PathKind::DirectCopy:
    return "DirectCopy";
  case PathKind::Load2D:
    return "Load2D";
  case PathKind::Load2DTranspose:
    return "Load2DTranspose";
  case PathKind::FixPipe:
    return "FixPipe";
  case PathKind::QueueTransfer:
    return "QueueTransfer";
  }
  llvm_unreachable("unknown path kind");
}

void printPath(raw_ostream &os, const PathEdge &edge) {
  os << stringifyMemoryPlace(edge.srcPlace) << " -> "
     << stringifyMemoryPlace(edge.dstPlace);
}

void printRateCandidates(raw_ostream &os, ArrayRef<StringRef> sections,
                         ArrayRef<StringRef> names) {
  bool first = true;
  for (StringRef section : sections) {
    for (StringRef name : names) {
      if (!first)
        os << ", ";
      first = false;
      os << section << "." << name;
    }
  }
}

} // namespace

FailureOr<int64_t>
TargetCostModel::getMemoryRate(StringRef section, StringRef name) const {
  auto it = ratesByQualifiedName.find(makeRateKey(section, name));
  if (it == ratesByQualifiedName.end())
    return failure();
  return it->second;
}

FailureOr<int64_t>
TargetCostModel::getPreferredMemoryRate(StringRef name) const {
  auto it = preferredRatesByName.find(name);
  if (it == preferredRatesByName.end())
    return failure();
  return it->second;
}

FailureOr<PathCost>
TargetCostModel::getPathCost(const PathEdge &edge) const {
  auto it = pathCosts.find(edge);
  if (it == pathCosts.end())
    return failure();
  return it->second;
}

FailureOr<int64_t>
TargetCostModel::estimateTransferCycles(const PathEdge &edge,
                                        int64_t bytes) const {
  if (bytes < 0)
    return failure();
  FailureOr<PathCost> cost = getPathCost(edge);
  if (failed(cost) || cost->bytesPerCycle <= 0)
    return failure();
  int64_t payloadCycles =
      bytes == 0 ? 0 : (bytes + cost->bytesPerCycle - 1) / cost->bytesPerCycle;
  return cost->startupCycles + payloadCycles;
}

FailureOr<TargetCostModel>
TargetCostModelBuilder::build(const TargetProfile &profile,
                              const TargetMemoryModel &memoryModel,
                              raw_ostream &os) const {
  TargetCostModel model;
  llvm::StringMap<int> preferredPriorities;

  SmallVector<const TargetMemoryRateInfo *> rates;
  rates.reserve(profile.memoryRates.size());
  for (const TargetMemoryRateInfo &rate : profile.memoryRates)
    rates.push_back(&rate);
  llvm::sort(rates, [](const TargetMemoryRateInfo *lhs,
                       const TargetMemoryRateInfo *rhs) {
    if (sectionPriority(lhs->section) != sectionPriority(rhs->section))
      return sectionPriority(lhs->section) < sectionPriority(rhs->section);
    return std::tie(lhs->section, lhs->name) <
           std::tie(rhs->section, rhs->name);
  });

  for (const TargetMemoryRateInfo *rate : rates) {
    if (rate->bytesPerCycle <= 0)
      continue;

    model.ratesByQualifiedName[makeRateKey(rate->section, rate->name)] =
        rate->bytesPerCycle;
    int priority = sectionPriority(rate->section);
    auto preferredIt = preferredPriorities.find(rate->name);
    if (preferredIt == preferredPriorities.end() ||
        priority < preferredIt->second) {
      preferredPriorities[rate->name] = priority;
      model.preferredRatesByName[rate->name] = rate->bytesPerCycle;
    }
  }

  for (MemoryPlace src : memoryModel.getMemoryPlaces()) {
    for (MemoryPlace dst : memoryModel.getMemoryPlaces()) {
      for (const PathEdge &edge : memoryModel.findDirectPaths(src, dst)) {
        FailureOr<PathKind> kind = memoryModel.getPathKind(edge);
        if (failed(kind))
          continue;

        SmallVector<StringRef> rateCandidates = getRateCandidates(edge, *kind);
        SmallVector<StringRef> sectionCandidates = getSectionCandidates(edge);
        std::optional<StringRef> selectedRateName;
        std::optional<StringRef> selectedRateSection;
        int64_t selectedRate = 0;
        for (StringRef section : sectionCandidates) {
          for (StringRef candidate : rateCandidates) {
            FailureOr<int64_t> rate = model.getMemoryRate(section, candidate);
            if (succeeded(rate)) {
              selectedRateSection = section;
              selectedRateName = candidate;
              selectedRate = *rate;
              break;
            }
          }
          if (selectedRateName)
            break;
        }

        if (!selectedRateName) {
          os << "TargetCostModel verification failed: missing memory rate from "
             << "candidates [";
          printRateCandidates(os, sectionCandidates, rateCandidates);
          os << "]";
          os << " for path ";
          printPath(os, edge);
          os << " kind " << stringifyPathKind(*kind) << "\n";
          return failure();
        }

        model.pathCosts[edge] = PathCost{startupCyclesFor(*kind), selectedRate,
                                         overlapsCompute(*kind),
                                         selectedRateSection->str(),
                                         selectedRateName->str()};
      }
    }
  }

  return model;
}

} // namespace mlir::ascend
```

Add `#include <tuple>` if the compiler requires it.

- [ ] **Step 3: Add source to target library**

Modify `lib/Target/Ascend/CMakeLists.txt`:

```cmake
add_mlir_library(AscendTargetProfile
  TargetProfile.cpp
  TargetMemoryModel.cpp
  TargetIntrinsicModel.cpp
  TargetCostModel.cpp
  CannTargetProfileLoader.cpp
  ...
)
```

- [ ] **Step 4: Run GREEN unit tests**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build AscendTargetCostModelTest && ./build/bin/AscendTargetCostModelTest'
```

Expected: all `AscendTargetCostModelTest` tests pass.

## Task 4: Tracking And Verification

**Files:**
- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [ ] **Step 1: Update Phase 4 tracking row**

Change:

```markdown
| `TargetCostModel` | `Planned` | memory rates / path cost | cost query 测试 |
```

to:

```markdown
| `TargetCostModel MVP` | `Done` | query-only memory-rate lookup、direct-path cost lookup、transfer-cycle estimate；Schedule / Realize consumption 后续继续推进 | `AscendTargetCostModelTest` + target profile lit |
```

- [ ] **Step 2: Run final verification**

```bash
test/tools/check_ascend_no_v2_code_naming.sh
git diff --check -- . ':!AGENTS.md'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt AscendCommonAttributesTest AscendKernelPatternTest AscendRealizePlannerTest AscendTargetMemoryModelTest AscendTargetIntrinsicModelTest AscendTargetCostModelTest'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ctest --test-dir build -R "Ascend(CommonAttributes|KernelPattern|RealizePlanner|TargetMemoryModel|TargetIntrinsicModel|TargetCostModel)Test" --output-on-failure'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Target/ascend-target-profile.mlir'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion --filter="ascend-"'
```

Expected: all commands pass.

- [ ] **Step 3: Commit only related files**

```bash
git status --short
git add docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md docs/superpowers/plans/2026-05-11-ascend-target-cost-model-mvp.md include/Target/Ascend/TargetProfile.h include/Target/Ascend/TargetCostModel.h lib/Target/Ascend/TargetCostModel.cpp lib/Target/Ascend/CannTargetProfileLoader.cpp lib/Target/Ascend/TargetProfile.cpp lib/Target/Ascend/CMakeLists.txt test/unittests/Target/CMakeLists.txt test/unittests/Target/AscendTargetCostModelTest.cpp test/Target/ascend-target-profile.mlir
git commit -m "feat: add Ascend target cost model MVP"
git push
```

Do not stage `AGENTS.md`.
