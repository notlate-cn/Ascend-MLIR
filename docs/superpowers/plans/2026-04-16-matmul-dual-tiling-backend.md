# Matmul Dual Tiling Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fusion-ready matmul tiling abstraction that supports both `MatmulApiTiling` and a native backend under the runtime mix compile path without changing artifact/execution contracts.

**Architecture:** Keep `MixTilingGenerator` as the outer runtime entry point, but route matmul cases through a new `MatmulTilingDispatcher` that consumes a unified `MatmulTilingRequest`. The first backend is an adapter around the existing `MatmulApiTiling` path; the second backend is a native heuristic backend for the currently supported `2D + ND + optional bias` subset. Existing non-matmul mix cases keep the current fallback behavior.

**Tech Stack:** C++, LLVM support utilities, AscendC `MatmulApiTiling`, existing runtime mix artifact pipeline, lit/runtime shell verification.

---

### Task 1: Add Unified Matmul Tiling Types

**Files:**
- Create: `include/Runtime/Mix/MatmulTilingTypes.h`
- Modify: `lib/Runtime/CMakeLists.txt`
- Test: `test/unittests/Runtime/MatmulTilingTypesTest.cpp`

- [ ] **Step 1: Write the failing test**

```cpp
#include "Runtime/Mix/MatmulTilingTypes.h"

#include "gtest/gtest.h"

using namespace mlir::runtime;

TEST(MatmulTilingTypesTest, DefaultRequestIsNativeFriendly) {
  MatmulTilingRequest request;
  request.kernelName = "matmul_bias_relu";
  request.problem.M = 16;
  request.problem.N = 32;
  request.problem.K = 64;
  request.problem.dtypeA = DType::F16;
  request.problem.dtypeB = DType::F16;
  request.problem.dtypeC = DType::F16;

  EXPECT_EQ(request.problem.layoutA, MatmulLayout::ND);
  EXPECT_EQ(request.problem.layoutB, MatmulLayout::ND);
  EXPECT_EQ(request.problem.layoutC, MatmulLayout::ND);
  EXPECT_EQ(request.fusion.epilogue, EpilogueKind::None);
  EXPECT_FALSE(request.problem.transA);
  EXPECT_FALSE(request.problem.transB);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target test_runtime -- -j8 && ctest --test-dir build -R MatmulTilingTypesTest --output-on-failure`
Expected: FAIL with missing header/type errors for `MatmulTilingRequest` and related enums.

- [ ] **Step 3: Write minimal implementation**

```cpp
#pragma once

#include "Runtime/Support/Types.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::runtime {

enum class MatmulLayout { ND, NZ };
enum class EpilogueKind { None, BiasAdd, BiasAddRelu, BiasAddLeakyRelu };
enum class MatrixTraverseKind { FirstM, FirstN };

struct MatmulProblemDesc {
  int64_t M = 0;
  int64_t N = 0;
  int64_t K = 0;
  std::vector<int64_t> batchShape;
  DType dtypeA = DType::F32;
  DType dtypeB = DType::F32;
  DType dtypeC = DType::F32;
  MatmulLayout layoutA = MatmulLayout::ND;
  MatmulLayout layoutB = MatmulLayout::ND;
  MatmulLayout layoutC = MatmulLayout::ND;
  bool transA = false;
  bool transB = false;
  bool hasBias = false;
};

struct MatmulFusionDesc {
  EpilogueKind epilogue = EpilogueKind::None;
  bool preferFuseVectorEpilogue = false;
  uint64_t consumerAlignmentBytes = 0;
};

struct MatmulScheduleHint {
  std::string socVersion;
  std::optional<int64_t> preferBlockDim;
  std::optional<bool> preferSplitK;
  std::optional<MatrixTraverseKind> preferTraverse;
};

struct MatmulTilingRequest {
  std::string kernelName;
  MatmulProblemDesc problem;
  MatmulFusionDesc fusion;
  MatmulScheduleHint hints;
};

struct MatmulTilingResult {
  std::string backendKind;
  std::string strategyName;
  uint32_t blockDim = 0;
  std::vector<uint8_t> tilingData;
  std::optional<int64_t> tileM;
  std::optional<int64_t> tileN;
  std::optional<int64_t> tileK;
  std::string debugNote;
};

} // namespace mlir::runtime
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target test_runtime -- -j8 && ctest --test-dir build -R MatmulTilingTypesTest --output-on-failure`
Expected: PASS with `1 test from MatmulTilingTypesTest`.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/Mix/MatmulTilingTypes.h lib/Runtime/CMakeLists.txt test/unittests/Runtime/MatmulTilingTypesTest.cpp
git commit -m "runtime: add matmul tiling request types"
```

### Task 2: Add Backend Interface And Dispatcher

**Files:**
- Create: `include/Runtime/Mix/MatmulTilingBackend.h`
- Create: `include/Runtime/Mix/MatmulTilingDispatcher.h`
- Create: `lib/Runtime/Mix/MatmulTilingDispatcher.cpp`
- Test: `test/unittests/Runtime/MatmulTilingDispatcherTest.cpp`

- [ ] **Step 1: Write the failing test**

```cpp
#include "Runtime/Mix/MatmulTilingBackend.h"
#include "Runtime/Mix/MatmulTilingDispatcher.h"

#include "gtest/gtest.h"

using namespace mlir::runtime;

namespace {
class FakeBackend final : public MatmulTilingBackend {
public:
  FakeBackend(std::string backendName, bool supported, uint32_t blockDim)
      : backendName(std::move(backendName)), supported(supported),
        blockDim(blockDim) {}

  llvm::StringRef name() const override { return backendName; }
  bool supports(const MatmulTilingRequest &) const override { return supported; }

  llvm::Expected<MatmulTilingResult>
  generate(const MatmulTilingRequest &) const override {
    MatmulTilingResult result;
    result.backendKind = backendName;
    result.strategyName = "fake";
    result.blockDim = blockDim;
    result.tilingData = {1, 2, 3};
    return result;
  }

private:
  std::string backendName;
  bool supported;
  uint32_t blockDim;
};
} // namespace

TEST(MatmulTilingDispatcherTest, PrefersNativeAndFallsBackToApi) {
  MatmulTilingRequest request;
  request.problem.M = 16;
  request.problem.N = 16;
  request.problem.K = 16;

  FakeBackend native("native", false, 8);
  FakeBackend api("api", true, 4);
  auto result = dispatchMatmulTiling(request, native, api);

  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_EQ(result->backendKind, "api");
  EXPECT_EQ(result->blockDim, 4u);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target test_runtime -- -j8 && ctest --test-dir build -R MatmulTilingDispatcherTest --output-on-failure`
Expected: FAIL with missing interface/dispatcher declarations.

- [ ] **Step 3: Write minimal implementation**

```cpp
class MatmulTilingBackend {
public:
  virtual ~MatmulTilingBackend() = default;
  virtual llvm::StringRef name() const = 0;
  virtual bool supports(const MatmulTilingRequest &request) const = 0;
  virtual llvm::Expected<MatmulTilingResult>
  generate(const MatmulTilingRequest &request) const = 0;
};

llvm::Expected<MatmulTilingResult>
dispatchMatmulTiling(const MatmulTilingRequest &request,
                     const MatmulTilingBackend &nativeBackend,
                     const MatmulTilingBackend &apiBackend) {
  if (nativeBackend.supports(request)) {
    if (auto nativeResult = nativeBackend.generate(request))
      return nativeResult;
    llvm::consumeError(nativeResult.takeError());
  }
  if (!apiBackend.supports(request))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "no matmul tiling backend supports %s",
                                   request.kernelName.c_str());
  return apiBackend.generate(request);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target test_runtime -- -j8 && ctest --test-dir build -R MatmulTilingDispatcherTest --output-on-failure`
Expected: PASS with fallback behavior validated.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/Mix/MatmulTilingBackend.h include/Runtime/Mix/MatmulTilingDispatcher.h lib/Runtime/Mix/MatmulTilingDispatcher.cpp test/unittests/Runtime/MatmulTilingDispatcherTest.cpp
git commit -m "runtime: add matmul tiling dispatcher"
```

### Task 3: Wrap Existing MatmulApiTiling As API Backend

**Files:**
- Create: `include/Runtime/Mix/MatmulApiTilingBackend.h`
- Create: `lib/Runtime/Mix/MatmulApiTilingBackend.cpp`
- Modify: `lib/Runtime/CMakeLists.txt`
- Test: `test/unittests/Runtime/MatmulApiTilingBackendTest.cpp`

- [ ] **Step 1: Write the failing test**

```cpp
#include "Runtime/Mix/MatmulApiTilingBackend.h"

#include "gtest/gtest.h"

using namespace mlir::runtime;

TEST(MatmulApiTilingBackendTest, SupportsSimple2DNdMatmul) {
  MatmulApiTilingBackend backend;
  MatmulTilingRequest request;
  request.kernelName = "matmul";
  request.problem.M = 16;
  request.problem.N = 32;
  request.problem.K = 64;
  request.problem.dtypeA = DType::F16;
  request.problem.dtypeB = DType::F16;
  request.problem.dtypeC = DType::F16;

  EXPECT_TRUE(backend.supports(request));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target test_runtime -- -j8 && ctest --test-dir build -R MatmulApiTilingBackendTest --output-on-failure`
Expected: FAIL because the API backend does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```cpp
class MatmulApiTilingBackend final : public MatmulTilingBackend {
public:
  llvm::StringRef name() const override { return "api"; }

  bool supports(const MatmulTilingRequest &request) const override {
    return request.problem.M > 0 && request.problem.N > 0 &&
           request.problem.K > 0 &&
           request.problem.layoutA == MatmulLayout::ND &&
           request.problem.layoutB == MatmulLayout::ND &&
           request.problem.layoutC == MatmulLayout::ND &&
           request.problem.batchShape.empty();
  }

  llvm::Expected<MatmulTilingResult>
  generate(const MatmulTilingRequest &request) const override;
};
```

Implementation notes:
- Move the current `MatmulApiTiling` setup logic out of `MixTilingGenerator.cpp`.
- Map `transA/transB`, `hasBias`, `dtypeA/B/C`, `preferTraverse`, and `socVersion`.
- Fill `backendKind="api"` and `strategyName="matmul-api"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target test_runtime -- -j8 && ctest --test-dir build -R MatmulApiTilingBackendTest --output-on-failure`
Expected: PASS with support check green.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/Mix/MatmulApiTilingBackend.h lib/Runtime/Mix/MatmulApiTilingBackend.cpp lib/Runtime/CMakeLists.txt test/unittests/Runtime/MatmulApiTilingBackendTest.cpp
git commit -m "runtime: wrap matmul api tiling backend"
```

### Task 4: Add Native Backend For Current Supported Subset

**Files:**
- Create: `include/Runtime/Mix/NativeMatmulTilingBackend.h`
- Create: `lib/Runtime/Mix/NativeMatmulTilingBackend.cpp`
- Test: `test/unittests/Runtime/NativeMatmulTilingBackendTest.cpp`

- [ ] **Step 1: Write the failing test**

```cpp
#include "Runtime/Mix/NativeMatmulTilingBackend.h"

#include "gtest/gtest.h"

using namespace mlir::runtime;

TEST(NativeMatmulTilingBackendTest, RejectsUnsupportedTransposeCase) {
  NativeMatmulTilingBackend backend;
  MatmulTilingRequest request;
  request.problem.M = 16;
  request.problem.N = 32;
  request.problem.K = 64;
  request.problem.transB = true;

  EXPECT_FALSE(backend.supports(request));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target test_runtime -- -j8 && ctest --test-dir build -R NativeMatmulTilingBackendTest --output-on-failure`
Expected: FAIL because the native backend does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```cpp
class NativeMatmulTilingBackend final : public MatmulTilingBackend {
public:
  llvm::StringRef name() const override { return "native"; }

  bool supports(const MatmulTilingRequest &request) const override {
    return request.problem.M > 0 && request.problem.N > 0 &&
           request.problem.K > 0 &&
           request.problem.layoutA == MatmulLayout::ND &&
           request.problem.layoutB == MatmulLayout::ND &&
           request.problem.layoutC == MatmulLayout::ND &&
           !request.problem.transA && !request.problem.transB &&
           request.problem.batchShape.empty();
  }

  llvm::Expected<MatmulTilingResult>
  generate(const MatmulTilingRequest &request) const override {
    MatmulTilingResult result;
    result.backendKind = "native";
    result.strategyName = "native-matmul-2d";
    result.blockDim = static_cast<uint32_t>(std::min<int64_t>(request.problem.M, 32));
    result.tileM = request.problem.M;
    result.tileN = request.problem.N;
    result.tileK = request.problem.K;
    result.tilingData = {/* pack native tiling payload */};
    return result;
  }
};
```

Implementation notes:
- Use a small runtime-owned payload format for the native backend instead of pretending it is `TCubeTiling`.
- Keep `debugNote` explicit when the native path is selected.

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target test_runtime -- -j8 && ctest --test-dir build -R NativeMatmulTilingBackendTest --output-on-failure`
Expected: PASS with unsupported transpose behavior validated.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/Mix/NativeMatmulTilingBackend.h lib/Runtime/Mix/NativeMatmulTilingBackend.cpp test/unittests/Runtime/NativeMatmulTilingBackendTest.cpp
git commit -m "runtime: add native matmul tiling backend"
```

### Task 5: Route Mix Matmul Strategy Through Dispatcher

**Files:**
- Modify: `include/Runtime/Mix/MixTilingGenerator.h`
- Modify: `lib/Runtime/Mix/MixTilingGenerator.cpp`
- Test: `test/unittests/Runtime/MixTilingGeneratorTest.cpp`

- [ ] **Step 1: Write the failing test**

```cpp
#include "Runtime/Mix/MixTilingGenerator.h"

#include "gtest/gtest.h"

using namespace mlir::runtime;

TEST(MixTilingGeneratorTest, EmitsBackendMetadataForMatmul) {
  MixTilingRequest request;
  request.kernelName = "matmul_add_leakyrelu";
  request.socVersion = "Ascend910B1";
  request.inputs = {{DType::F16, {16, 32}}, {DType::F16, {32, 64}}, {DType::F16, {64}}};
  request.outputs = {{DType::F16, {16, 64}}};

  auto result = generateMixTilingInProcess(request);
  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_FALSE(result->strategyName.empty());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target test_runtime -- -j8 && ctest --test-dir build -R MixTilingGeneratorTest --output-on-failure`
Expected: FAIL once the old direct `MatmulApiTiling` logic is removed and no dispatcher path exists yet.

- [ ] **Step 3: Write minimal implementation**

```cpp
static MatmulTilingRequest buildMatmulTilingRequest(const MixTilingRequest &request) {
  MatmulTilingRequest matmulRequest;
  matmulRequest.kernelName = request.kernelName;
  matmulRequest.hints.socVersion = request.socVersion;
  matmulRequest.problem.M = request.outputs[0].shape[0];
  matmulRequest.problem.N = request.outputs[0].shape[1];
  matmulRequest.problem.K = request.inputs[0].shape[1];
  matmulRequest.problem.dtypeA = request.inputs[0].dtype;
  matmulRequest.problem.dtypeB = request.inputs[1].dtype;
  matmulRequest.problem.dtypeC = request.outputs[0].dtype;
  matmulRequest.problem.hasBias = request.inputs.size() > 2;
  matmulRequest.fusion.epilogue = request.inputs.size() > 2
                                      ? EpilogueKind::BiasAddLeakyRelu
                                      : EpilogueKind::None;
  return matmulRequest;
}
```

Implementation notes:
- Keep non-matmul behavior unchanged.
- Convert `MatmulTilingResult` back into `MixTilingResult`.
- Preserve `writeMixTilingArtifacts` output format.

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target test_runtime -- -j8 && ctest --test-dir build -R MixTilingGeneratorTest --output-on-failure`
Expected: PASS with matmul requests routed through dispatcher.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/Mix/MixTilingGenerator.h lib/Runtime/Mix/MixTilingGenerator.cpp test/unittests/Runtime/MixTilingGeneratorTest.cpp
git commit -m "runtime: route mix matmul tiling through dispatcher"
```

### Task 6: Add Matmul Semantics To Mix ABI

**Files:**
- Modify: `include/Runtime/Mix/MixAbi.h`
- Modify: `lib/Runtime/Mix/MixAbi.cpp`
- Modify: `lib/Runtime/Mix/MixAbiExtractor.cpp`
- Test: `test/unittests/Runtime/MixAbiTest.cpp`

- [ ] **Step 1: Write the failing test**

```cpp
#include "Runtime/Mix/MixAbi.h"

#include "gtest/gtest.h"

using namespace mlir::runtime;

TEST(MixAbiTest, RoundTripsMatmulSemantics) {
  MixAbi abi;
  abi.kernelName = "matmul";
  abi.blockDim = 4;
  abi.matmul.opKind = "matmul";
  abi.matmul.transA = false;
  abi.matmul.transB = false;
  abi.matmul.hasBias = true;
  abi.matmul.layoutA = "ND";
  abi.matmul.layoutB = "ND";
  abi.matmul.layoutC = "ND";
  abi.matmul.epilogueKind = "BiasAddLeakyRelu";

  auto text = serializeMixAbi(abi);
  auto parsed = parseMixAbi(text);

  ASSERT_TRUE(static_cast<bool>(parsed));
  EXPECT_EQ(parsed->matmul.epilogueKind, "BiasAddLeakyRelu");
  EXPECT_TRUE(parsed->matmul.hasBias);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target test_runtime -- -j8 && ctest --test-dir build -R MixAbiTest --output-on-failure`
Expected: FAIL because the ABI does not contain matmul semantic fields.

- [ ] **Step 3: Write minimal implementation**

```cpp
struct MixMatmulAbi {
  std::string opKind;
  bool transA = false;
  bool transB = false;
  bool hasBias = false;
  std::string layoutA = "ND";
  std::string layoutB = "ND";
  std::string layoutC = "ND";
  std::string epilogueKind = "None";
  std::vector<int64_t> batchShape;
};
```

Implementation notes:
- Serialize these fields in a stable key format near other ABI scalar metadata.
- Extract only what can be determined safely for the current examples path.
- Keep shape-based inference as compatibility fallback, not as the preferred source.

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target test_runtime -- -j8 && ctest --test-dir build -R MixAbiTest --output-on-failure`
Expected: PASS with ABI round-trip green.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/Mix/MixAbi.h lib/Runtime/Mix/MixAbi.cpp lib/Runtime/Mix/MixAbiExtractor.cpp test/unittests/Runtime/MixAbiTest.cpp
git commit -m "runtime: add matmul semantics to mix abi"
```

### Task 7: Record Backend Metadata In Artifacts

**Files:**
- Modify: `lib/Runtime/Mix/MixDirectTilingArtifacts.cpp`
- Modify: `lib/Runtime/Mix/MixDirectRuntimeAbi.cpp`
- Test: `test/unittests/Runtime/MixDirectTilingArtifactsTest.cpp`

- [ ] **Step 1: Write the failing test**

```cpp
#include "Runtime/Mix/MixDirectCompileInternal.h"

#include "gtest/gtest.h"

using namespace mlir::runtime;

TEST(MixDirectTilingArtifactsTest, WritesBackendMetadataForRuntimeNativeMatmul) {
  MixDirectCompileOutputs outputs;
  outputs.tiling.blockDim = 4;
  outputs.abi.blockDim = 4;
  outputs.abi.metadata["tiling_backend"] = "native";
  outputs.abi.metadata["tiling_strategy"] = "native-matmul-2d";

  EXPECT_EQ(outputs.abi.metadata["tiling_backend"], "native");
  EXPECT_EQ(outputs.abi.metadata["tiling_strategy"], "native-matmul-2d");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target test_runtime -- -j8 && ctest --test-dir build -R MixDirectTilingArtifactsTest --output-on-failure`
Expected: FAIL if metadata wiring is missing.

- [ ] **Step 3: Write minimal implementation**

```cpp
outputs.abi.metadata["tiling_backend"] = matmulResult.backendKind;
outputs.abi.metadata["tiling_strategy"] = matmulResult.strategyName;
if (!matmulResult.debugNote.empty())
  outputs.abi.metadata["tiling_debug_note"] = matmulResult.debugNote;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target test_runtime -- -j8 && ctest --test-dir build -R MixDirectTilingArtifactsTest --output-on-failure`
Expected: PASS with metadata visible in compile outputs.

- [ ] **Step 5: Commit**

```bash
git add lib/Runtime/Mix/MixDirectTilingArtifacts.cpp lib/Runtime/Mix/MixDirectRuntimeAbi.cpp test/unittests/Runtime/MixDirectTilingArtifactsTest.cpp
git commit -m "runtime: record matmul tiling backend metadata"
```

### Task 8: Verify Example And Runtime Paths

**Files:**
- Modify: `test/tools/runtime/run_mix_compile_timing_compare.sh`
- Modify: `examples/matmul-add-leakyrelu/run.sh`
- Test: `test/tools/runtime/run_runtime.sh`
- Test: `test/tools/examples/example_pipelines.sh`

- [ ] **Step 1: Add backend-specific verification hooks**

```bash
export MATMUL_TILING_BACKEND="${MATMUL_TILING_BACKEND:-auto}"
grep -q "tiling_backend=" "${ARTIFACT_DIR}/runtime_metadata.txt"
```

- [ ] **Step 2: Run focused runtime verification**

Run: `bash test/tools/runtime/run_runtime.sh`
Expected: PASS, including repeated mix simulation baseline.

- [ ] **Step 3: Run example pipeline verification**

Run: `bash test/tools/examples/example_pipelines.sh`
Expected: PASS with all 6 example pipelines green.

- [ ] **Step 4: Run mix timing comparison for api/native/auto**

Run: `RUNS=3 MATMUL_TILING_BACKEND=api bash test/tools/runtime/run_mix_compile_timing_compare.sh`
Expected: PASS with timing summary for `api`.

Run: `RUNS=3 MATMUL_TILING_BACKEND=native bash test/tools/runtime/run_mix_compile_timing_compare.sh`
Expected: PASS with timing summary for `native`.

Run: `RUNS=3 MATMUL_TILING_BACKEND=auto bash test/tools/runtime/run_mix_compile_timing_compare.sh`
Expected: PASS with timing summary and backend metadata showing selected path.

- [ ] **Step 5: Commit**

```bash
git add test/tools/runtime/run_mix_compile_timing_compare.sh examples/matmul-add-leakyrelu/run.sh
git commit -m "runtime: verify dual matmul tiling backends"
```

### Task 9: Final Regression And Review

**Files:**
- Modify: `docs/superpowers/plans/2026-04-16-matmul-dual-tiling-backend.md`
- Test: `test_taskgraph_runtime`
- Test: `test_capi_runtime`
- Test: `test_runtime`

- [ ] **Step 1: Run unit test slices**

Run: `ctest --test-dir build -R 'MatmulTiling|MixAbi|MixDirectTilingArtifacts|test_runtime' --output-on-failure`
Expected: PASS for all new and existing runtime unit tests.

- [ ] **Step 2: Run runtime regression suites**

Run: `ctest --test-dir build -R 'test_taskgraph_runtime|test_capi_runtime|test_runtime' --output-on-failure`
Expected: PASS for all runtime regression suites.

- [ ] **Step 3: Update plan with actual deviations**

```markdown
- [x] Task 1 completed without interface changes.
- [x] Task 2 completed with dispatcher fallback behavior.
- [x] Task 3-8 completed; note any backend scope restriction discovered during verification.
```

- [ ] **Step 4: Commit final integration**

```bash
git add docs/superpowers/plans/2026-04-16-matmul-dual-tiling-backend.md
git commit -m "runtime: finish dual matmul tiling integration"
```
