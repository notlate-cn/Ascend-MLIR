# RuntimeMix Second Sample Wrapperless Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将第二个 RuntimeMix 验证样例的编译输入切到 `fc_relu_split_wrapperless.cpp`，并在 xvm 上重新验证 compile、runner、direct-packed 和精度闭环。

**Architecture:** 保持 `RuntimeMix` direct backend 现状不变，只调整 `examples/relu-split-mix-test/` 的入口输入和说明文档。验证继续复用现有 `mix-compiler`、`mix-validator`、artifact-backed tiling 和 runner/direct 双路径，不在本阶段新增 wrapper 兼容逻辑。

**Tech Stack:** Bash, C++, LLVM Support, RuntimeMix, AscendC simulator, Python/numpy

---

### Task 1: Switch Split-ReLU Example To Wrapperless Input

**Files:**
- Modify: `examples/relu-split-mix-test/run.sh`
- Modify: `examples/relu-split-mix-test/README.md`

- [ ] **Step 1: Update the second-sample kernel source path**

Modify `examples/relu-split-mix-test/run.sh` so the example compiles the wrapperless helper instead of the original wrapper-bearing source:

```bash
KERNEL_SRC="${REPO_ROOT}/examples/matmul-add-relu-sum/fc_relu_split_wrapperless.cpp"
KERNEL_NAME="${KERNEL_NAME:-fc_relu_split}"
```

Do not change:
- data generation script
- artifact directory layout
- runner validation command
- direct-packed validation command

- [ ] **Step 2: Sanity-check the edited script**

Run:

```bash
sed -n '1,40p' examples/relu-split-mix-test/run.sh
```

Expected:
- `KERNEL_SRC` points at `fc_relu_split_wrapperless.cpp`
- `KERNEL_NAME` remains `fc_relu_split`

- [ ] **Step 3: Update the example README to match the new input**

Edit `examples/relu-split-mix-test/README.md` so it explicitly states:

```md
- 语义来源仍然是 `examples/matmul-add-relu-sum/fc_relu_split_mix.cpp`
- 当前 RuntimeMix / 官方工具链验证入口使用
  `examples/matmul-add-relu-sum/fc_relu_split_wrapperless.cpp` 作为编译输入基线
```

Keep the README focused on:
- why wrapperless helper exists
- what `run.sh` now compiles
- what success looks like on xvm

- [ ] **Step 4: Re-read the changed files**

Run:

```bash
sed -n '1,220p' examples/relu-split-mix-test/run.sh
sed -n '1,220p' examples/relu-split-mix-test/README.md
```

Expected:
- no stale references claiming `run.sh` compiles `fc_relu_split_mix.cpp`
- README and script agree on the wrapperless helper path

- [ ] **Step 5: Commit the entrypoint-only change**

```bash
git add examples/relu-split-mix-test/run.sh \
        examples/relu-split-mix-test/README.md
git commit -m "test: point split relu mix example at wrapperless input"
```

Expected:
- commit records only example entrypoint/doc changes for this task

### Task 2: Validate RuntimeMix Split-ReLU Closure On xvm

**Files:**
- Verify: `examples/relu-split-mix-test/run.sh`
- Verify: `examples/matmul-add-relu-sum/fc_relu_split_wrapperless.cpp`
- Verify: `build/runtime-mix-relu-split/out/manifest.txt`
- Verify: `build/runtime-mix-relu-split/out/tiling.bin`

- [ ] **Step 1: Force a fresh tool bootstrap and artifact rebuild on xvm**

Run:

```bash
sleep 4 && ssh xvm@orb 'set -euo pipefail
cd /home/niu/code/Ascend-MLIR
rm -f build/runtime-mix-bootstrap/bin/mix-compiler
rm -f build/runtime-mix-bootstrap/bin/mix-validator
rm -rf build/runtime-mix-relu-split
bash examples/relu-split-mix-test/run.sh'
```

Expected:
- the script rebuilds `mix-compiler` / `mix-validator`
- compile succeeds using `fc_relu_split_wrapperless.cpp`
- runner path executes
- direct-packed path executes
- final output includes `test pass`

- [ ] **Step 2: Verify artifact-backed tiling and manifest output**

Run:

```bash
sleep 4 && ssh xvm@orb 'set -euo pipefail
cd /home/niu/code/Ascend-MLIR
test -f build/runtime-mix-relu-split/out/tiling.bin
sed -n "1,220p" build/runtime-mix-relu-split/out/manifest.txt'
```

Expected manifest evidence:
- `kernel_name=fc_relu_split`
- `abi_tiling_mode=generated_file`
- `abi_tiling_source=out/tiling.bin`

- [ ] **Step 3: Verify result files and md5**

Run:

```bash
sleep 4 && ssh xvm@orb 'set -euo pipefail
cd /home/niu/code/Ascend-MLIR
md5sum build/runtime-mix-relu-split/testdata/output/golden.bin \
       build/runtime-mix-relu-split/testdata/output/actual.bin \
       build/runtime-mix-relu-split/testdata/output/direct-actual.bin'
```

Expected:
- all three files exist
- all three md5 values are identical

- [ ] **Step 4: If validation fails, capture the first failing layer instead of patching blindly**

Run the narrowest reproduction first:

```bash
sleep 4 && ssh xvm@orb 'set -euo pipefail
cd /home/niu/code/Ascend-MLIR
./build/runtime-mix-bootstrap/bin/mix-validator \
  --artifact-root build/runtime-mix-relu-split \
  --input-dir build/runtime-mix-relu-split/testdata/input \
  --golden build/runtime-mix-relu-split/testdata/output/golden.bin \
  --output-file build/runtime-mix-relu-split/testdata/output/direct-actual.bin \
  --soc Ascend910B1 \
  --force-direct-packed'
```

Expected:
- either PASS, or a single concrete failure mode to debug next
- do not implement any fix in this task; only record the exact failing boundary

- [ ] **Step 5: Commit verification-backed changes**

```bash
git add examples/relu-split-mix-test/run.sh \
        examples/relu-split-mix-test/README.md
git commit -m "test: validate split relu mix with wrapperless input"
```

Expected:
- if the run passes, commit captures the validated wrapperless-input switch
- if the run fails, do not commit implementation guesses; stop with evidence
