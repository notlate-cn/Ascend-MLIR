# FC ReLU Split Wrapperless Helper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a checked-in wrapperless helper source for the split-relu mix sample and verify on xvm that the official AscendC CMake/toolchain no longer fails at the current `auto_gen_*_origin` compile blocker.

**Architecture:** Keep the original `fc_relu_split_mix.cpp` unchanged and add one explicit helper source that preserves the real kernel body and mix metadata but removes the hand-written `auto_gen_fc_relu_split_kernel` wrapper. Validate this helper only as a compile baseline through the official AscendC sample-style CMake flow.

**Tech Stack:** C++17, AscendC sample CMake toolchain, xvm simulator environment, bash, Python.

---

## File Structure

- Create: `examples/matmul-add-relu-sum/fc_relu_split_wrapperless.cpp`
  Purpose: explicit wrapperless compile baseline for the split-relu mix sample.

- Modify: `examples/relu-split-mix-test/README.md`
  Purpose: document that the original sample is still the runtime-facing input while the wrapperless helper exists as a compile/reference baseline.

- Create: `docs/superpowers/specs/2026-04-01-fc-relu-split-wrapperless-helper-design.md`
  Purpose: already written design reference for this slice.

- Create: `docs/superpowers/plans/2026-04-01-fc-relu-split-wrapperless-helper.md`
  Purpose: this implementation plan.

## Task 1: Add the Wrapperless Helper Source

**Files:**
- Create: `examples/matmul-add-relu-sum/fc_relu_split_wrapperless.cpp`

- [ ] **Step 1: Review the current split-relu source and capture the exact sections to keep**

Run:

```bash
rg -n "fc_relu_split_mix_origin|auto_gen_fc_relu_split_kernel|section\\(\" \
  examples/matmul-add-relu-sum/fc_relu_split_mix.cpp
```

Expected:
- identify the body entry to keep
- identify the hand-written wrapper block to remove
- identify the mix metadata sections to preserve

- [ ] **Step 2: Create the new helper source with no hand-written auto-gen wrapper**

Create `examples/matmul-add-relu-sum/fc_relu_split_wrapperless.cpp` with content shaped like:

```cpp
#define __FC_RELU_SPLIT_WRAPPERLESS_KERNEL_FUN_H__

#undef __global__
#define __global__ inline

#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

__aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM) {
    uint64_t *dst = reinterpret_cast<uint64_t *>(tiling);
    auto tiling64 = reinterpret_cast<__gm__ uint64_t *>(tilingGM);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint64_t); ++i) {
        dst[i] = tiling64[i];
    }
}

extern "C" __global__ __aicore__ void fc_relu_split(
    GM_ADDR a, GM_ADDR b, GM_ADDR bias,
    GM_ADDR out, GM_ADDR workspace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;

    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);

    if ASCEND_IS_AIC {
        Matmul<MatmulType<TPosition::GM, CubeFormat::ND, half>,
               MatmulType<TPosition::GM, CubeFormat::ND, half>,
               MatmulType<TPosition::VECIN, CubeFormat::ND, float>,
               MatmulType<TPosition::GM, CubeFormat::ND, float>> mm;

        GlobalTensor<half> aGM, bGM;
        GlobalTensor<float> cGM, biasGM;
        aGM.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(a), tiling.M * tiling.Ka);
        bGM.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b), tiling.Kb * tiling.N);
        cGM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out), tiling.M * tiling.N);
        biasGM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bias), tiling.N);

        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
        mm.SetTensorA(aGM);
        mm.SetTensorB(bGM);
        mm.SetBias(biasGM);
        mm.template IterateAll(cGM);
        mm.End();
        CrossCoreSetFlag<0x2, PIPE_FIX>(3);
    }

    if ASCEND_IS_AIV {
        TQue<TPosition::VECIN, 1> reluInQueue;
        TQue<TPosition::VECOUT, 1> reluOutQueue;

        uint32_t count = (uint32_t)(tiling.singleCoreM * tiling.singleCoreN / 2);
        GlobalTensor<float> cGM;
        cGM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out) + GetBlockIdx() * count);

        pipe.InitBuffer(reluInQueue, 1, count * sizeof(float));
        pipe.InitBuffer(reluOutQueue, 1, count * sizeof(float));

        CrossCoreWaitFlag(3);

        LocalTensor<float> reluInLocal = reluInQueue.AllocTensor<float>();
        DataCopy(reluInLocal, cGM, count);
        reluInQueue.EnQue<float>(reluInLocal);

        LocalTensor<float> inLocal = reluInQueue.DeQue<float>();
        LocalTensor<float> outLocal = reluOutQueue.AllocTensor<float>();
        Relu(outLocal, inLocal, count);
        reluOutQueue.EnQue<float>(outLocal);
        reluInQueue.FreeTensor(inLocal);

        LocalTensor<float> finalLocal = reluOutQueue.DeQue<float>();
        DataCopy(cGM, finalLocal, count);
        reluOutQueue.FreeTensor(finalLocal);
    }
}

#if defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__)
static const struct FunLevelMixCoreType fc_relu_split_aic_section
    __attribute__((used, section(".ascend.meta.fc_relu_split_0_mix_aic"))) = {
        {{F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_MIX_AIC_MAIN},
        {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 2}
    };
#endif

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
static const struct FunLevelMixCoreType fc_relu_split_aiv_section
    __attribute__((used, section(".ascend.meta.fc_relu_split_0_mix_aiv"))) = {
        {{F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_MIX_AIC_MAIN},
        {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 2}
    };
#endif
```

The new file must not contain:
- `auto_gen_fc_relu_split_kernel`
- `__attribute__((cce_kernel))`
- source-level workspace/tiling wrapper glue

- [ ] **Step 3: Sanity-check the helper file shape locally**

Run:

```bash
rg -n "auto_gen_fc_relu_split_kernel|cce_kernel|fc_relu_split\\(|.ascend.meta.fc_relu_split" \
  examples/matmul-add-relu-sum/fc_relu_split_wrapperless.cpp
```

Expected:
- no `auto_gen_fc_relu_split_kernel`
- no `cce_kernel`
- exactly one final entry `fc_relu_split(`
- both mix metadata sections present

- [ ] **Step 4: Commit the helper source**

```bash
git add examples/matmul-add-relu-sum/fc_relu_split_wrapperless.cpp
git commit -m "feat: add wrapperless split relu mix helper"
```

## Task 2: Verify the Official AscendC Compile Baseline on xvm

**Files:**
- Reuse: `examples/matmul-add-relu-sum/fc_relu_split_wrapperless.cpp`

- [ ] **Step 1: Build a minimal sample-style control project on xvm using the new helper**

Run on xvm using the existing BareMix sample skeleton:

```bash
CTRL=/tmp/fc_relu_split_wrapperless_compilecheck
rm -rf "$CTRL"
mkdir -p "$CTRL/cmake"
cp /home/niu/code/samples/BareMixInvocation/cmake/npu_lib.cmake "$CTRL/cmake/npu_lib.cmake"
cp /home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/fc_relu_split_wrapperless.cpp "$CTRL/fc_relu_split_wrapperless.cpp"
cat > "$CTRL/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.16)
project(fc_relu_split_wrapperless_compilecheck)
set(RUN_MODE "sim" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend910B1" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/home/niu/Ascend/latest" CACHE STRING "ASCEND CANN package installation directory")
file(GLOB KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/fc_relu_split_wrapperless.cpp)
set(CUSTOM_ASCEND310P_LIST "Ascend310P1" "Ascend310P3")
include(cmake/npu_lib.cmake)
EOF
cd "$CTRL"
cmake -S . -B build -DRUN_MODE=sim -DSOC_VERSION=Ascend910B1 -DASCEND_CANN_PACKAGE_PATH=/home/niu/Ascend/latest
cmake --build build -j4
```

Expected:
- the build must advance past the previous `auto_gen_*_origin` failure point

- [ ] **Step 2: Capture proof that the previous compile blocker is gone**

Run:

```bash
grep -R "auto_gen_fc_relu_split_kernel_origin" -n /tmp/fc_relu_split_wrapperless_compilecheck/build || true
```

Expected:
- no failing compile log at the previous call site
- if the build later fails, it must fail at a different stage than the old wrapper conflict

- [ ] **Step 3: Record the exact new stopping point if build still does not finish**

Run:

```bash
tail -n 160 /tmp/fc_relu_split_wrapperless.build.log 2>/dev/null || true
```

Expected:
- capture the next real blocker after wrapper conflict removal
- do not fix it in this task

- [ ] **Step 4: Commit only if the helper and evidence gathering scripts changed tracked files**

```bash
git status --short
```

Expected:
- only the helper file and any intentionally updated repo docs should appear

## Task 3: Document the Helper’s Role for Future RuntimeMix Work

**Files:**
- Modify: `examples/relu-split-mix-test/README.md`

- [ ] **Step 1: Add one short note about the wrapperless helper**

Append a concise section to `examples/relu-split-mix-test/README.md` shaped like:

```md
编译基线辅助文件：

- `examples/matmul-add-relu-sum/fc_relu_split_wrapperless.cpp`

该文件只用于验证官方 AscendC / RuntimeMix 编译链能否绕开源码内手写 `auto_gen_*` wrapper 与工具链 auto-gen 的冲突，不替代原始 `fc_relu_split_mix.cpp` 作为语义来源。
```

- [ ] **Step 2: Sanity-check the README wording**

Run:

```bash
sed -n '1,220p' examples/relu-split-mix-test/README.md
```

Expected:
- the helper is clearly documented as compile/reference-only
- the original sample remains the semantic source

- [ ] **Step 3: Commit the doc clarification**

```bash
git add examples/relu-split-mix-test/README.md
git commit -m "docs: clarify split relu wrapperless helper role"
```

## Final Verification

- [ ] **Step 1: Re-run the official xvm compile check from Task 2 after all tracked edits are present**

Run:

```bash
sleep 4 && ssh xvm@orb 'set -euo pipefail
CTRL=/tmp/fc_relu_split_wrapperless_compilecheck
rm -rf "$CTRL"
mkdir -p "$CTRL/cmake"
cp /home/niu/code/samples/BareMixInvocation/cmake/npu_lib.cmake "$CTRL/cmake/npu_lib.cmake"
cp /home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/fc_relu_split_wrapperless.cpp "$CTRL/fc_relu_split_wrapperless.cpp"
cat > "$CTRL/CMakeLists.txt" <<'"'"'EOF'"'"'
cmake_minimum_required(VERSION 3.16)
project(fc_relu_split_wrapperless_compilecheck)
set(RUN_MODE "sim" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend910B1" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/home/niu/Ascend/latest" CACHE STRING "ASCEND CANN package installation directory")
file(GLOB KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/fc_relu_split_wrapperless.cpp)
set(CUSTOM_ASCEND310P_LIST "Ascend310P1" "Ascend310P3")
include(cmake/npu_lib.cmake)
EOF
cd "$CTRL"
cmake -S . -B build -DRUN_MODE=sim -DSOC_VERSION=Ascend910B1 -DASCEND_CANN_PACKAGE_PATH=/home/niu/Ascend/latest >/tmp/fc_relu_split_wrapperless.configure.log 2>&1
cmake --build build -j4 >/tmp/fc_relu_split_wrapperless.build.log 2>&1 || true
tail -n 120 /tmp/fc_relu_split_wrapperless.build.log'
```

Expected:
- the old `auto_gen_fc_relu_split_kernel_origin` compile failure must be absent
- if there is still a later failure, it becomes the new documented blocker

- [ ] **Step 2: Save the observed result in your final summary with exact xvm evidence**

Expected summary points:
- whether the old compile blocker disappeared
- what the new stopping point is, if any
- that no execution claims are made in this stage
