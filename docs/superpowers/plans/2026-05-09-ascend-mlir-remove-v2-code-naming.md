# Remove V2 Code Naming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `V2` / `v2` from Ascend pipeline code names and IR contracts while keeping V2 only as a documentation/version label.

**Architecture:** Treat this as a naming-contract migration, not a behavior change. The public pass names remain `--ascend-normalize`, `--ascend-kernelize`, `--ascend-schedule`, and `--ascend-realize`; the C++ source tree, CMake target, namespaces, MLIR attributes, debug output, and tests move to versionless Ascend names.

**Tech Stack:** MLIR C++ passes, TableGen pass definitions, CMake/Ninja, llvm-lit, gtest, xvm/docker verification via `ssh xvm@orb`.

---

## Scope

Change code and tests:

- `include/Conversion/AscendV2/**` -> `include/Conversion/Ascend/**`
- `lib/Conversion/AscendV2/**` -> `lib/Conversion/Ascend/**`
- `AscendV2Conversion` -> `AscendConversion`
- `mlir::afir::ascend::v2` -> `mlir::afir::ascend`
- `mlir::ascend::v2` debug helpers -> `mlir::afir::ascend::debug`
- `ascend.v2.*` attributes -> `ascend.*`
- `test/Conversion/ascend-v2-pipeline-mvp.mlir` -> `test/Conversion/ascend-pipeline-mvp.mlir`
- `AscendV2KernelPatternTest` -> `AscendKernelPatternTest`
- `AscendV2CommonAttributesTest` -> `AscendCommonAttributesTest`
- `Ascend V2 <stage> report` debug output -> `Ascend <stage> report`

Do not change:

- `docs/Ascend-MLIR-Detailed-Implementation-V2*.zh.md`
- historical plan filenames under `docs/superpowers/plans/*v2*.md`
- `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md` filename
- third-party or CANN API names containing `V2`
- numpy BF16 `"<V2"` dtype strings
- dirty `AGENTS.md`

## Files

- Create: `test/tools/check_ascend_no_v2_code_naming.sh`
- Rename: `include/Conversion/AscendV2/**` to `include/Conversion/Ascend/**`
- Rename: `lib/Conversion/AscendV2/**` to `lib/Conversion/Ascend/**`
- Rename: `test/Conversion/ascend-v2-pipeline-mvp.mlir` to `test/Conversion/ascend-pipeline-mvp.mlir`
- Rename: `test/unittests/Conversion/AscendV2KernelPatternTest.cpp` to `test/unittests/Conversion/AscendKernelPatternTest.cpp`
- Rename: `test/unittests/Conversion/AscendV2CommonAttributesTest.cpp` to `test/unittests/Conversion/AscendCommonAttributesTest.cpp`
- Modify: `include/Conversion/Passes.h`
- Modify: `include/Conversion/Passes.td`
- Modify: `lib/Conversion/CMakeLists.txt`
- Modify: `lib/CAPI/Dialect/CMakeLists.txt`
- Modify: `tools/afir-opt/CMakeLists.txt`
- Modify: `test/unittests/Conversion/CMakeLists.txt`
- Modify: `test/Conversion/ascend-*.mlir`
- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

## Task 1: Add Guard Test For Versionless Code Naming

**Files:**
- Create: `test/tools/check_ascend_no_v2_code_naming.sh`

- [ ] **Step 1: Add the failing guard script**

Create `test/tools/check_ascend_no_v2_code_naming.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

pattern='AscendV2|ascend[.]v2|::v2([^[:alnum:]_]|$)|(^|[^[:alnum:]_])v2::|(^|[^[:alnum:]_])V2([^[:alnum:]_]|$)|ascend-v2-pipeline'
paths=(
  include/Conversion
  lib/Conversion
  tools/afir-opt
  test/Conversion
  test/unittests/Conversion
)

search_contents() {
  if command -v rg >/dev/null 2>&1; then
    rg -n "${pattern}" "${paths[@]}" --glob '!build/**'
  else
    find "${paths[@]}" -path '*/build/*' -prune -o -type f \
      -exec grep -nE "${pattern}" {} +
  fi
}

search_paths() {
  find "${paths[@]}" -path '*/build/*' -prune -o -print | grep -E "${pattern}"
}

found=0

if search_contents; then
  found=1
else
  result=$?
  if [[ ${result} -gt 1 ]]; then
    exit "${result}"
  fi
fi

if search_paths; then
  found=1
else
  result=$?
  if [[ ${result} -gt 1 ]]; then
    exit "${result}"
  fi
fi

if [[ ${found} -eq 1 ]]; then
  echo "found V2/v2 code naming in Ascend pipeline sources" >&2
  exit 1
fi
```

- [ ] **Step 2: Verify RED on the current tree**

Run on host:

```bash
bash test/tools/check_ascend_no_v2_code_naming.sh
```

Expected: non-zero exit and matches such as `include/Conversion/AscendV2`, `ascend.v2.kernel`, and `AscendV2Conversion`.

- [ ] **Step 3: Commit the guard script with the plan**

Run:

```bash
git add docs/superpowers/plans/2026-05-09-ascend-mlir-remove-v2-code-naming.md \
  test/tools/check_ascend_no_v2_code_naming.sh
git commit -m "docs: plan Ascend versionless code naming"
```

## Task 2: Rename Source Tree, CMake Target, And C++ Namespaces

**Files:**
- Rename: `include/Conversion/AscendV2/**` -> `include/Conversion/Ascend/**`
- Rename: `lib/Conversion/AscendV2/**` -> `lib/Conversion/Ascend/**`
- Modify: `include/Conversion/Passes.h`
- Modify: `include/Conversion/Passes.td`
- Modify: `lib/Conversion/CMakeLists.txt`
- Modify: `lib/CAPI/Dialect/CMakeLists.txt`
- Modify: `tools/afir-opt/CMakeLists.txt`
- Modify: `test/unittests/Conversion/CMakeLists.txt`

- [ ] **Step 1: Move directories**

Run:

```bash
git mv include/Conversion/AscendV2 include/Conversion/Ascend
git mv lib/Conversion/AscendV2 lib/Conversion/Ascend
```

- [ ] **Step 2: Apply mechanical source-tree renames**

Run:

```bash
perl -pi -e 's#Conversion/AscendV2/#Conversion/Ascend/#g;
             s/AscendV2Conversion/AscendConversion/g;
             s/add_subdirectory\\(AscendV2\\)/add_subdirectory(Ascend)/g;
             s/ASCEND_MLIR_CONVERSION_ASCENDV2_/ASCEND_MLIR_CONVERSION_ASCEND_/g;
             s/Ascend V2/Ascend/g' \
  $(rg -l 'Conversion/AscendV2/|AscendV2Conversion|add_subdirectory\\(AscendV2\\)|ASCEND_MLIR_CONVERSION_ASCENDV2_|Ascend V2' \
    include/Conversion lib/Conversion tools/afir-opt lib/CAPI/Dialect test/unittests/Conversion)
```

- [ ] **Step 3: Rename namespaces**

Run:

```bash
perl -pi -e 's/mlir::afir::ascend::v2::/mlir::afir::ascend::/g;
             s/mlir::afir::ascend::v2/mlir::afir::ascend/g;
             s/namespace mlir::ascend::v2/namespace mlir::afir::ascend::debug/g;
             s/} \\/\\/ namespace mlir::ascend::v2/} \\/\\/ namespace mlir::afir::ascend::debug/g;
             s/::mlir::ascend::v2::/::mlir::afir::ascend::debug::/g;
             s/ascend::v2::/ascend::debug::/g' \
  $(rg -l 'mlir::afir::ascend::v2|mlir::ascend::v2|::mlir::ascend::v2::|ascend::v2::' \
    include/Conversion/Ascend lib/Conversion/Ascend test/unittests/Conversion)
```

- [ ] **Step 4: Verify target and include references**

Run:

```bash
rg -n 'AscendV2|Conversion/AscendV2|ASCEND_MLIR_CONVERSION_ASCENDV2|mlir::afir::ascend::v2|mlir::ascend::v2' \
  include/Conversion lib/Conversion tools/afir-opt lib/CAPI/Dialect test/unittests/Conversion
```

Expected: no matches.

## Task 3: Migrate IR Attribute Prefixes And Lit Tests

**Files:**
- Modify: `include/Conversion/Ascend/Common/Attributes.h`
- Modify: `include/Conversion/Ascend/Kernelize/KernelizeTypes.h`
- Modify: `include/Conversion/Ascend/Schedule/ScheduleTypes.h`
- Modify: `lib/Conversion/Ascend/Kernelize/KernelizePass.cpp`
- Modify: `test/Conversion/ascend-*.mlir`
- Rename: `test/Conversion/ascend-v2-pipeline-mvp.mlir` -> `test/Conversion/ascend-pipeline-mvp.mlir`

- [ ] **Step 1: Rename the pipeline smoke test file**

Run:

```bash
git mv test/Conversion/ascend-v2-pipeline-mvp.mlir \
  test/Conversion/ascend-pipeline-mvp.mlir
```

- [ ] **Step 2: Apply attribute prefix changes**

Run:

```bash
perl -pi -e 's/ascend\\.v2\\./ascend./g;
             s/ascend-v2-pipeline/ascend-pipeline/g' \
  $(rg -l 'ascend\\.v2\\.|ascend-v2-pipeline' \
    include/Conversion/Ascend lib/Conversion/Ascend test/Conversion test/unittests/Conversion)
```

- [ ] **Step 3: Verify old attribute prefix is gone from code/tests**

Run:

```bash
rg -n 'ascend\\.v2|ascend-v2-pipeline' include/Conversion lib/Conversion test/Conversion test/unittests/Conversion
```

Expected: no matches.

## Task 4: Rename Unit Tests And Debug Output

**Files:**
- Rename: `test/unittests/Conversion/AscendV2KernelPatternTest.cpp` -> `test/unittests/Conversion/AscendKernelPatternTest.cpp`
- Rename: `test/unittests/Conversion/AscendV2CommonAttributesTest.cpp` -> `test/unittests/Conversion/AscendCommonAttributesTest.cpp`
- Modify: `test/unittests/Conversion/CMakeLists.txt`
- Modify: `lib/Conversion/Ascend/Debug/DebugOptions.cpp`
- Modify: `test/Conversion/ascend-realize-mvp.mlir`

- [ ] **Step 1: Rename unit test files**

Run:

```bash
git mv test/unittests/Conversion/AscendV2KernelPatternTest.cpp \
  test/unittests/Conversion/AscendKernelPatternTest.cpp
git mv test/unittests/Conversion/AscendV2CommonAttributesTest.cpp \
  test/unittests/Conversion/AscendCommonAttributesTest.cpp
```

- [ ] **Step 2: Rename unit test targets and test names**

Run:

```bash
perl -pi -e 's/AscendV2KernelPatternTest/AscendKernelPatternTest/g;
             s/AscendV2CommonAttributesTest/AscendCommonAttributesTest/g;
             s/AscendV2KernelPatternEdgeKeyTest/AscendKernelPatternEdgeKeyTest/g;
             s/AscendV2CommonAttributesTest/AscendCommonAttributesTest/g' \
  test/unittests/Conversion/CMakeLists.txt \
  test/unittests/Conversion/AscendKernelPatternTest.cpp \
  test/unittests/Conversion/AscendCommonAttributesTest.cpp
```

- [ ] **Step 3: Update debug report checks**

Run:

```bash
perl -pi -e 's/Ascend V2 /Ascend /g' \
  lib/Conversion/Ascend/Debug/DebugOptions.cpp \
  test/Conversion/ascend-realize-mvp.mlir
```

- [ ] **Step 4: Run the guard script**

Run:

```bash
bash test/tools/check_ascend_no_v2_code_naming.sh
```

Expected: exit 1 only if remaining matches are real violations. If matches remain only in explicitly allowed files outside the script scope, keep the script scope unchanged and proceed.

## Task 5: Update Tracking And Verify In xvm/docker

**Files:**
- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [ ] **Step 1: Add a tracking entry**

Add a Phase 3 follow-up row stating:

```markdown
| 代码命名去版本化 | `Done` | 将源码目录、namespace、CMake target、IR attrs、测试名从 `V2` / `v2` 迁移为版本无关 Ascend 命名；文档版本名保留 | guard script passed；xvm focused build/lit/unit passed |
```

- [ ] **Step 2: Sync source changes to xvm**

Run from host:

```bash
rsync -av --delete \
  --exclude build \
  --exclude build-v2-verify \
  --exclude .git \
  ./ xvm@orb:/home/niu/code/Ascend-MLIR/
```

- [ ] **Step 3: Build and run focused tests in xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && \
  ninja -C build afir-opt AscendCommonAttributesTest AscendKernelPatternTest && \
  source examples/env.sh && \
  bash test/tools/check_ascend_no_v2_code_naming.sh && \
  ./build/bin/AscendCommonAttributesTest && \
  ./build/bin/AscendKernelPatternTest && \
  ctest --test-dir build -R "Ascend(CommonAttributes|KernelPattern)Test" --output-on-failure && \
  files=$(cd test/Conversion && printf "build/test/Conversion/%s " ascend-*.mlir) && \
  /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v $files'
```

Expected:

- `afir-opt`, `AscendCommonAttributesTest`, `AscendKernelPatternTest` build succeeds.
- guard script prints no matches and exits 0.
- both unit binaries pass.
- `ctest` reports 2/2 passed for the renamed tests.
- all `test/Conversion/ascend-*.mlir` tests pass.

- [ ] **Step 4: Static checks and commit**

Run on host:

```bash
git diff --check -- . ':(exclude)AGENTS.md'
git status -sb
```

Then commit:

```bash
git add include/Conversion/Ascend lib/Conversion/Ascend \
  include/Conversion/Passes.h include/Conversion/Passes.td \
  lib/Conversion/CMakeLists.txt lib/CAPI/Dialect/CMakeLists.txt tools/afir-opt/CMakeLists.txt \
  test/Conversion test/unittests/Conversion test/tools/check_ascend_no_v2_code_naming.sh \
  docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md
git add -u include/Conversion/AscendV2 lib/Conversion/AscendV2
git commit -m "refactor: remove V2 from Ascend code naming"
git push origin dev-nyh
```

## Self-Review Checklist

- Code-visible `V2` / `v2` names are removed from Ascend pipeline source and tests.
- Pass names stay stable and versionless.
- Documentation version names are not rewritten wholesale.
- `AGENTS.md` remains dirty but unstaged.
- xvm/docker build and focused tests provide the completion evidence.
