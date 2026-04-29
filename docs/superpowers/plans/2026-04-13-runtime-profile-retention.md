# Runtime Profile Retention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound retained simulator profile artifacts under `/tmp/ascendc-runtime-profiles` while preserving valid `session.profile[*]` outputs after successful simulator runs.

**Architecture:** Keep the existing simulator `_Exit(0)` strategy, but add a small retention policy around the retained profile root. `runtime-session` will continue copying current session artifacts into a durable directory, then prune older retained session directories down to a fixed cap. Tests cover both the low-level pruning behavior and end-to-end xvm verification.

**Tech Stack:** C++17, LLVM support utilities, existing `Runtime/ProfileUtils`, `runtime-session`, xvm shell verification scripts

---

### Task 1: Add failing retention tests

**Files:**
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing tests**

Add two focused tests next to the existing `testRetainProfileArtifactsForCli()` coverage:

```cpp
static void testRetainProfileArtifactsPrunesOldSessions() {
  const std::filesystem::path retainRoot = makeTempDir("profile-retain-root");
  std::filesystem::create_directories(retainRoot);

  for (int i = 0; i < 3; ++i) {
    const std::filesystem::path dir =
        retainRoot / ("runtime-session--old" + std::to_string(i));
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "trace.json") << i;
    std::filesystem::last_write_time(
        dir, std::filesystem::file_time_type::clock::now() +
                 std::chrono::seconds(i));
  }

  auto err = pruneRetainedProfileDirectoriesForTest(retainRoot.string(), 2);
  EXPECT(!err, "retention pruning succeeds");
  if (err)
    llvm::consumeError(std::move(err));

  EXPECT(std::filesystem::exists(retainRoot / "runtime-session--old1"),
         "retention keeps second-newest directory");
  EXPECT(std::filesystem::exists(retainRoot / "runtime-session--old2"),
         "retention keeps newest directory");
  EXPECT(!std::filesystem::exists(retainRoot / "runtime-session--old0"),
         "retention prunes oldest directory");

  std::error_code ec;
  std::filesystem::remove_all(retainRoot, ec);
}

static void testRetainProfileArtifactsIgnoresNonDirectories() {
  const std::filesystem::path retainRoot = makeTempDir("profile-retain-files");
  std::filesystem::create_directories(retainRoot);
  std::ofstream(retainRoot / "README.txt") << "keep me";
  std::filesystem::create_directories(retainRoot / "runtime-session--keep");

  auto err = pruneRetainedProfileDirectoriesForTest(retainRoot.string(), 1);
  EXPECT(!err, "retention pruning ignores non-directories");
  if (err)
    llvm::consumeError(std::move(err));

  EXPECT(std::filesystem::exists(retainRoot / "README.txt"),
         "retention does not touch non-directory files");
  EXPECT(std::filesystem::exists(retainRoot / "runtime-session--keep"),
         "retention keeps the single retained session directory");

  std::error_code ec;
  std::filesystem::remove_all(retainRoot, ec);
}
```

Register them in `main()`:

```cpp
  testRetainProfileArtifactsForCli();
  testRetainProfileArtifactsPrunesOldSessions();
  testRetainProfileArtifactsIgnoresNonDirectories();
```

Also add the forward declaration near the existing testing helper declarations:

```cpp
llvm::Error pruneRetainedProfileDirectoriesForTest(llvm::StringRef root,
                                                   size_t keepCount);
```

- [ ] **Step 2: Run test to verify it fails**

Run on xvm:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
source scripts/resolve_llvm_env.sh
LLVM_BUILD=$(require_llvm_build_dir)
LLVM_SOURCE_INCLUDE=$(cd "${LLVM_BUILD}/.." && pwd)/include
cmake --build build --target AscendCRuntime -j2
g++ -std=c++17 -I include/ -I "$LLVM_BUILD/include" -I "$LLVM_SOURCE_INCLUDE" \
  test/tools/runtime/test_taskgraph_runtime.cpp build/lib/libAscendCRuntime.a \
  $("$LLVM_BUILD/bin/llvm-config" --ldflags --libs support --system-libs) -ldl \
  -o /tmp/test_taskgraph_runtime.retention
```

Expected: link failure complaining about missing
`pruneRetainedProfileDirectoriesForTest(...)`.

- [ ] **Step 3: Commit the red test**

```bash
git add test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "test: add runtime profile retention coverage"
```

### Task 2: Implement bounded retention in ProfileUtils

**Files:**
- Modify: `include/Runtime/ProfileUtils.h`
- Modify: `lib/Runtime/ProfileUtils.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Add the retention API**

In `include/Runtime/ProfileUtils.h`, add:

```cpp
llvm::Error pruneRetainedProfileDirectories(llvm::StringRef root,
                                            size_t keepCount);
llvm::Error pruneRetainedProfileDirectoriesForTest(llvm::StringRef root,
                                                   size_t keepCount);
```

- [ ] **Step 2: Implement the minimal pruning logic**

In `lib/Runtime/ProfileUtils.cpp`, add a small helper and the two exported
functions:

```cpp
namespace {

struct RetainedProfileDirectory {
  std::string path;
  std::filesystem::file_time_type mtime;
};

llvm::Error pruneRetainedProfileDirectoriesImpl(llvm::StringRef root,
                                                size_t keepCount) {
  std::error_code ec;
  if (!std::filesystem::exists(root.str(), ec))
    return llvm::Error::success();
  if (ec)
    return llvm::createStringError(ec, "cannot inspect retained profile root: %s",
                                   root.str().c_str());

  std::vector<RetainedProfileDirectory> dirs;
  for (const std::filesystem::directory_entry &entry :
       std::filesystem::directory_iterator(root.str(), ec)) {
    if (ec)
      return llvm::createStringError(ec,
                                     "cannot iterate retained profile root: %s",
                                     root.str().c_str());
    if (!entry.is_directory())
      continue;
    std::error_code timeEc;
    auto mtime = entry.last_write_time(timeEc);
    if (timeEc)
      mtime = std::filesystem::file_time_type::min();
    dirs.push_back({entry.path().string(), mtime});
  }

  std::sort(dirs.begin(), dirs.end(),
            [](const RetainedProfileDirectory &lhs,
               const RetainedProfileDirectory &rhs) {
              return lhs.mtime > rhs.mtime;
            });

  for (size_t i = keepCount; i < dirs.size(); ++i) {
    std::error_code removeEc;
    std::filesystem::remove_all(dirs[i].path, removeEc);
  }
  return llvm::Error::success();
}

} // namespace

llvm::Error pruneRetainedProfileDirectories(llvm::StringRef root,
                                            size_t keepCount) {
  return pruneRetainedProfileDirectoriesImpl(root, keepCount);
}

llvm::Error pruneRetainedProfileDirectoriesForTest(llvm::StringRef root,
                                                   size_t keepCount) {
  return pruneRetainedProfileDirectoriesImpl(root, keepCount);
}
```

Keep the behavior best-effort on old directory deletion failures. The current
run must not fail because an old directory could not be removed.

- [ ] **Step 3: Run the focused test to verify it passes**

Run the same command as Task 1 Step 2, then:

```bash
/tmp/test_taskgraph_runtime.retention
```

Expected: all tests pass, including the two new retention tests.

- [ ] **Step 4: Commit**

```bash
git add include/Runtime/ProfileUtils.h lib/Runtime/ProfileUtils.cpp \
  test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "feat: add retained runtime profile pruning"
```

### Task 3: Wire retention into runtime-session

**Files:**
- Modify: `tools/runtime-session/runtime_session_main.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Add a small failing assertion around pruning**

Extend the existing retained-profile test area with a direct helper check that
the newest current session survives a prune call:

```cpp
static void testRetainProfileArtifactsKeepsCurrentSession() {
  const std::filesystem::path retainRoot = makeTempDir("profile-retain-current");
  std::filesystem::create_directories(retainRoot);
  const std::filesystem::path current = retainRoot / "runtime-session--current";
  const std::filesystem::path old = retainRoot / "runtime-session--old";
  std::filesystem::create_directories(old);
  std::filesystem::create_directories(current);

  std::filesystem::last_write_time(
      old, std::filesystem::file_time_type::clock::now() -
               std::chrono::seconds(10));
  std::filesystem::last_write_time(
      current, std::filesystem::file_time_type::clock::now());

  auto err = pruneRetainedProfileDirectoriesForTest(retainRoot.string(), 1);
  EXPECT(!err, "retention pruning with current session succeeds");
  if (err)
    llvm::consumeError(std::move(err));

  EXPECT(std::filesystem::exists(current),
         "retention keeps current session directory");
  EXPECT(!std::filesystem::exists(old),
         "retention prunes older session directory");

  std::error_code ec;
  std::filesystem::remove_all(retainRoot, ec);
}
```

Register it in `main()` before backend/session tests.

- [ ] **Step 2: Verify the new test is green before wiring CLI**

Re-run:

```bash
/tmp/test_taskgraph_runtime.retention
```

Expected: PASS. This confirms the helper logic is correct before touching the CLI.

- [ ] **Step 3: Apply pruning in runtime-session**

In `tools/runtime-session/runtime_session_main.cpp`, add:

```cpp
static constexpr size_t RetainedProfileSessionLimit = 20;
```

Then, after successful `retainProfileArtifactsForCli(...)` on the simulator path,
call:

```cpp
    if (auto pruneErr = pruneRetainedProfileDirectories(
            std::filesystem::path(*retainRootOr).parent_path().string(),
            RetainedProfileSessionLimit)) {
      llvm::consumeError(std::move(pruneErr));
    }
```

Do this before `runSession.reset();`.

This preserves current-run success semantics while bounding retained directories.

- [ ] **Step 4: Build and run focused verification**

Run on xvm:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_runtime.sh
```

Expected:

- `test_taskgraph_runtime` passes
- `test_capi_runtime` passes
- `test_runtime` passes
- smoke examples pass

- [ ] **Step 5: Commit**

```bash
git add tools/runtime-session/runtime_session_main.cpp \
  include/Runtime/ProfileUtils.h lib/Runtime/ProfileUtils.cpp \
  test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "fix: bound retained runtime profile sessions"
```

### Task 4: Verify bounded retention on xvm

**Files:**
- Modify: none
- Test: `test/tools/runtime/run_runtime.sh`

- [ ] **Step 1: Run repeated xvm verification**

Run:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
for i in 1 2 3; do
  echo "=== RUN $i ==="
  bash test/tools/runtime/run_runtime.sh
done
```

Expected:

- all three runs pass
- smoke summary still shows valid retained profile paths

- [ ] **Step 2: Inspect retained profile root**

Run:

```bash
find /tmp/ascendc-runtime-profiles -maxdepth 1 -mindepth 1 -type d | wc -l
du -sh /tmp/ascendc-runtime-profiles
find /tmp/ascendc-runtime-profiles -maxdepth 2 -name '*.json' | tail -n 10
```

Expected:

- retained session directory count is bounded near the configured cap
- retained JSON artifacts still exist and are readable

- [ ] **Step 3: Commit verification notes**

No code change. Do not create a commit. Capture the exact xvm results in the
task handoff message.

---

## Self-Review

### Spec coverage

- deterministic newest-`N` retention: covered by Tasks 1-3
- non-directory safety: covered by Task 1
- preserved current CLI behavior: covered by Task 3 + Task 4
- repeated xvm bounded verification: covered by Task 4

### Placeholder scan

- No `TODO`/`TBD`
- Each task has explicit files, commands, and expected outcomes

### Type consistency

- retention helper names are used consistently:
  - `pruneRetainedProfileDirectories`
  - `pruneRetainedProfileDirectoriesForTest`
- runtime-session still uses `retainProfileArtifactsForCli`
