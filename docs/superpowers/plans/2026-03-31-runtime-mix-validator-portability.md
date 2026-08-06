# RuntimeMix Validator Portability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove machine-specific `/home/niu/...` fallback paths from `tools/mix-validator/mix_validator_main.cpp`, align its runtime/toolkit discovery with the existing `lib/Runtime` style, and keep xvm baremix validation passing.

**Architecture:** Keep the current `mix-validator` entrypoint and behavior, but tighten its discovery helpers around a small ordered set of standard roots and standard layout probes. This is a portability cleanup only: no runner ABI work, no CLI changes, and no shared-helper extraction in this phase.

**Tech Stack:** C++17, LLVM Support, existing `RuntimeMix` validator tool, Ascend simulator/runtime libraries, xvm verification.

---

## File Map

- Modify: `tools/mix-validator/mix_validator_main.cpp`
  Purpose: replace `/home/niu/...` fallback logic with standard environment-variable and standard-layout probing.

- Verify: `examples/baremix-test/run.sh`
  Purpose: acceptance entrypoint only; no functional changes planned in this phase.

- Verify: `lib/Runtime/Executor.cpp`
  Purpose: reference for existing runtime-style simulator path probing.

- Verify: `lib/Runtime/HostRunnerGen.cpp`
  Purpose: reference for existing runtime-style simulator loading behavior.

- Create: `docs/superpowers/specs/2026-03-31-runtime-mix-validator-portability-design.md`
  Purpose: already written spec for this cleanup.

- Create: `docs/superpowers/plans/2026-03-31-runtime-mix-validator-portability.md`
  Purpose: this implementation plan.

## Task 1: Align Validator Path Discovery With Runtime Style

**Files:**
- Modify: `tools/mix-validator/mix_validator_main.cpp`
- Verify: `lib/Runtime/Executor.cpp`
- Verify: `lib/Runtime/HostRunnerGen.cpp`
- Test: direct xvm `mix-validator` rebuild plus baremix acceptance

- [ ] **Step 1: Replace machine-specific `findAscendHome()` fallbacks**

In `tools/mix-validator/mix_validator_main.cpp`, rewrite `findAscendHome()` so it removes all `/home/niu/...` candidates and uses this order only:

```cpp
static std::string findAscendHome() {
  auto hasRuntimeLibs = [](const std::string &root) {
    return llvm::sys::fs::exists(root + "/lib64/libplatform.so") &&
           llvm::sys::fs::exists(root + "/lib64/libunified_dlog.so") &&
           llvm::sys::fs::exists(root + "/lib64/libmmpa.so") &&
           llvm::sys::fs::exists(root + "/lib64/libc_sec.so");
  };

  if (const char *home = std::getenv("ASCEND_HOME_PATH");
      home && *home && hasRuntimeLibs(home))
    return home;
  if (const char *home = std::getenv("ASCEND_TOOLKIT_HOME");
      home && *home && hasRuntimeLibs(home))
    return home;

  if (const char *userHome = std::getenv("HOME")) {
    const std::string latest = std::string(userHome) + "/Ascend/latest";
    if (hasRuntimeLibs(latest))
      return latest;
    const std::string toolkitLatest =
        std::string(userHome) + "/Ascend/ascend-toolkit/latest";
    if (hasRuntimeLibs(toolkitLatest))
      return toolkitLatest;
  }

  const std::string sysDefault = "/usr/local/Ascend/ascend-toolkit/latest";
  if (hasRuntimeLibs(sysDefault))
    return sysDefault;
  return "";
}
```

Do not leave any `/home/niu/...` literal fallback in this function.

- [ ] **Step 2: Replace `findAscendLib64()` and `findSimulatorLibDir()` with standard-layout probing**

Still in `tools/mix-validator/mix_validator_main.cpp`, update the helper logic so it only probes standard layouts derived from `ascendHome`:

```cpp
static std::string findAscendLib64(const std::string &ascendHome) {
  const std::string candidates[] = {
      ascendHome + "/lib64",
      ascendHome + "/aarch64-linux/lib64",
      ascendHome + "/arm64-linux/lib64",
  };
  ...
}

static std::string findSimulatorLibDir(const std::string &ascendHome,
                                       const std::string &socVersion) {
  const std::string candidates[] = {
      ascendHome + "/aarch64-linux/simulator/" + socVersion + "/lib",
      ascendHome + "/arm64-linux/simulator/" + socVersion + "/lib",
      ascendHome + "/tools/simulator/" + socVersion + "/lib",
  };
  ...
}
```

Requirements:
- remove all `/home/niu/...` candidates
- prefer exact library existence checks (`libruntime_camodel.so`, `libplatform.so`)
- return `""` on failure instead of guessing a private path

- [ ] **Step 3: Tighten error handling in `configureRuntimeEnv()`**

In `tools/mix-validator/mix_validator_main.cpp`, make missing path discovery fail explicitly before env vars are set:

```cpp
static llvm::Error configureRuntimeEnv(... ) {
  const std::string ascendHome = findAscendHome();
  if (ascendHome.empty())
    return llvm::createStringError(..., "Cannot find Ascend toolkit root; set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME");

  const std::string ascendLib64 = findAscendLib64(ascendHome);
  if (ascendLib64.empty())
    return llvm::createStringError(..., "Cannot find Ascend lib64 under %s", ascendHome.c_str());

  const std::string simulatorLibDir = findSimulatorLibDir(ascendHome, socVersion);
  if (simulatorLibDir.empty())
    return llvm::createStringError(..., "Cannot find simulator libs for %s under %s", socVersion.c_str(), ascendHome.c_str());
  ...
}
```

Then update the call site to handle the returned `llvm::Error` and emit a clear validator error.

- [ ] **Step 4: Rebuild validator and verify xvm acceptance**

Run:

```bash
sleep 2 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  rm -f build/runtime-mix-bootstrap/bin/mix-validator &&
  bash examples/baremix-test/run.sh &&
  md5sum build/runtime-mix-baremix/testdata/output/golden.bin \
         build/runtime-mix-baremix/testdata/output/actual.bin
'
```

Expected:
- `mix-validator` still runs successfully without `/home/niu/...` fallbacks
- `PASS`
- `test pass`
- identical md5 values

- [ ] **Step 5: Sanity-check no private paths remain**

Run:

```bash
rg -n "/home/niu/" tools/mix-validator/mix_validator_main.cpp
```

Expected:
- no matches

- [ ] **Step 6: Commit**

```bash
git add tools/mix-validator/mix_validator_main.cpp
git commit -m "refactor: remove machine-specific RuntimeMix validator fallbacks"
```

## Self-Review

- Spec coverage:
  - `findAscendHome()` cleanup is covered by Task 1 Steps 1 and 5.
  - `findAscendLib64()` / `findSimulatorLibDir()` cleanup is covered by Task 1 Step 2.
  - explicit error handling is covered by Task 1 Step 3.
  - xvm regression coverage is covered by Task 1 Step 4.

- Placeholder scan:
  - No `TODO`, `TBD`, or vague “handle appropriately” placeholders remain.
  - Every required code path and command is explicit.

- Type consistency:
  - The plan keeps `mix-validator` CLI unchanged.
  - No shared-helper extraction or ABI work is introduced by accident.
