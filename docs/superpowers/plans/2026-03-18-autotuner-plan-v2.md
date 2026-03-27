# AutoTuner CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an `autotuner` CLI tool that enumerates tiling configurations from `tiling_space.json`, validates each via `SimValidator`, and emits `tiling_func.cpp` (typed C++ tiling function) for the best-performing configuration.

**Architecture:** Single-file CLI `tools/autotuner/autotuner_main.cpp` linked against the existing `AscendCRuntime` library. Extends `SimValidator::Result` with `cycle_count` (parsed from `core*_summary_log`). Reads `tiling_space.json` (extended format with `min/max/step`, `shape_key`, `block_dim_expr`), builds Cartesian product of search variables, runs `SimValidator::Validate()` per config, ranks by `cycle_count`, emits `tiling_func.cpp`.

**Tech Stack:** C++17, LLVM Support (JSON, CommandLine, FileSystem), existing `AscendCRuntime` (SimValidator, NpyIO, Compiler, Executor). No new libraries.

---

## Environment & Build Notes

- Code written on macOS, auto-synced to OrbStack VM `xvm` at `/home/niu/code/Ascend-MLIR`
- Build in xvm: `ssh xvm@orb && cd /home/niu/code/Ascend-MLIR && ./scripts/build.sh --build-project --llvm-build-dir ~/code/llvm-project/build`
- Wait **1 second** after local edits before building in xvm
- Run autotuner in xvm `sim/` dir with env set:
  ```bash
  export ASCEND_HOME_PATH=/home/niu/Ascend/latest
  export SOC_VERSION=Ascend910B1
  export ASCEND_CPU_SIMULATION=1
  export ASCEND_DEVICE_ID=0
  source /home/niu/code/Ascend-MLIR/examples/env.sh
  source ${ASCEND_HOME_PATH}/../set_env.sh
  export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/runtime/lib64/stub/linux/$(arch):${ASCEND_HOME_PATH}/$(arch)-linux/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
  ```
- Simulator produces `core*_summary_log` files in cwd when `ASCEND_CPU_SIMULATION=1`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `include/Runtime/SimValidator.h` | Modify | Add `cycle_count` field to `Result` |
| `lib/Runtime/SimValidator.cpp` | Modify | Parse `core*_summary_log` after Run, fill `cycle_count` |
| `examples/broadcast-add-reduce/tiling_space.json` | Modify | Extend format: add `block_dim_expr`, `shape_key`, `min/max/step` |
| `tools/autotuner/autotuner_main.cpp` | Create | Full CLI: parse JSON, enumerate, validate, emit tiling_func.cpp |
| `tools/autotuner/CMakeLists.txt` | Create | Build rules (mirrors sim-validator pattern) |
| `CMakeLists.txt` | Modify | Add `add_subdirectory(tools/autotuner)` |

---

## Task 1: Extend SimValidator::Result with cycle_count

**Files:**
- Modify: `include/Runtime/SimValidator.h`
- Modify: `lib/Runtime/SimValidator.cpp`

Background: The simulator writes `core0_summary_log`, `core1_summary_log`, ... in cwd when `ASCEND_CPU_SIMULATION=1`. Format:
```
kernal total ticks : 11404
system total ticks : 11767
```
`cycle_count` = max of `kernal total ticks` across all cores. Value `-1` means not available (real device or log missing).

- [ ] **Step 1: Add `cycle_count` to `SimValidator::Result`**

In `include/Runtime/SimValidator.h`, change:
```cpp
struct Result {
  bool        passed        = false;
  double      max_abs_diff  = 0.0;
  double      mean_abs_diff = 0.0;
  int64_t     cycle_count   = -1;   // -1 = not available (real hw or log missing)
  std::string error_msg;
};
```

- [ ] **Step 2: Add `ParseCycleCounts()` helper in `lib/Runtime/SimValidator.cpp`**

Add before `Validate()`:
```cpp
// Parse cycle counts from simulator summary logs in the given directory.
// Returns max "kernal total ticks" across all core*_summary_log files, or -1.
static int64_t ParseCycleCounts(const std::string& sim_dir) {
  int64_t max_ticks = -1;
  std::error_code ec;
  for (llvm::sys::fs::directory_iterator it(sim_dir, ec), end;
       !ec && it != end; it.increment(ec)) {
    llvm::StringRef name = llvm::sys::path::filename(it->path());
    // Only process "core*_summary_log" files (not other *_summary_log files)
    if (!name.starts_with("core") || !name.ends_with("_summary_log")) continue;
    auto buf = llvm::MemoryBuffer::getFile(it->path());
    if (!buf) continue;
    llvm::StringRef content = (*buf)->getBuffer();
    // Find "kernal total ticks : N"
    llvm::SmallVector<llvm::StringRef> lines;
    content.split(lines, '\n');
    for (auto line : lines) {
      line = line.trim();
      if (!line.starts_with("kernal total ticks")) continue;
      auto colon = line.rfind(':');
      if (colon == llvm::StringRef::npos) continue;
      int64_t ticks = 0;
      if (line.substr(colon + 1).trim().getAsInteger(10, ticks)) continue;
      if (ticks > max_ticks) max_ticks = ticks;
    }
  }
  return max_ticks;
}
```

Add required includes at top of `lib/Runtime/SimValidator.cpp`:
```cpp
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
```

- [ ] **Step 3: Call `ParseCycleCounts` in `Validate()` after `RunFile`**

In `lib/Runtime/SimValidator.cpp`, after the `executor.RunFile(...)` block and before the comparison loop, add:
```cpp
  // Parse cycle count from simulator logs (cwd is the sim run directory)
  {
    llvm::SmallString<256> cwd;
    llvm::sys::fs::current_path(cwd);
    r.cycle_count = ParseCycleCounts(cwd.str().str());
  }
```

- [ ] **Step 4: Build and verify**

```bash
# In xvm:
cd /home/niu/code/Ascend-MLIR
sleep 1
./scripts/build.sh --build-project --llvm-build-dir ~/code/llvm-project/build 2>&1 | tail -5
```
Expected: clean build, no errors.

- [ ] **Step 5: Smoke-test sim-validator still works**

```bash
cd /home/niu/code/Ascend-MLIR/sim
# use env from Environment section above
sim-validator \
  --kernel   ../examples/broadcast-add-reduce/step8_kernel.cpp \
  --name     broadcast_add_reducesum \
  --tiling-params "TB_M=16,TB_N=16,dim_arg0_0=32,dim_arg1_1=32,dim_arg0_1=32,dim_arg1_0=32" \
  --tiling-layout "int64,int64,int64,int64,int64,int64" \
  --inputs   /tmp/input_a.npy,/tmp/input_b.npy \
  --expected /tmp/expected.npy \
  --block-dim 2
```
Expected: PASS (cycle_count is now parsed but not printed by sim-validator — that's fine).

- [ ] **Step 6: Commit**

```bash
git add include/Runtime/SimValidator.h lib/Runtime/SimValidator.cpp
git commit -m "feat(runtime): add cycle_count to SimValidator::Result from sim summary logs"
```

---

## Task 2: Extend tiling_space.json format

**Files:**
- Modify: `examples/broadcast-add-reduce/tiling_space.json`

New format supports:
- Search variables: `"values": [...]` (explicit list) OR `"min"/"max"/"step"` (expanded to list)
- Fixed params: `"fixed": true, "shape_key": "M"` — value filled from `--shape M=32,N=32`
- `"block_dim_expr"`: simple expression, supports `"ceil(X/Y)"` and `"X/Y"` (integer division)

- [ ] **Step 1: Rewrite `examples/broadcast-add-reduce/tiling_space.json`**

```json
{
  "kernel": "broadcast_add_reducesum",
  "kernel_file": "step8_kernel.cpp",
  "soc": "Ascend910B1",
  "block_dim_expr": "ceil(M / TB_M)",
  "tiling_params": [
    {
      "name": "TB_M",
      "type": "int64",
      "min": 16, "max": 32, "step": 16,
      "note": "DataCopy half requires >= 16 elements (32B alignment)"
    },
    {
      "name": "TB_N",
      "type": "int64",
      "values": [16, 32],
      "note": "N must be multiple of 32 for DataCopy alignment"
    },
    {"name": "dim_arg0_0", "type": "int64", "fixed": true, "shape_key": "M"},
    {"name": "dim_arg1_1", "type": "int64", "fixed": true, "shape_key": "N"},
    {"name": "dim_arg0_1", "type": "int64", "fixed": true, "shape_key": "M"},
    {"name": "dim_arg1_0", "type": "int64", "fixed": true, "shape_key": "N"}
  ]
}
```

- [ ] **Step 2: Commit**

```bash
git add examples/broadcast-add-reduce/tiling_space.json
git commit -m "feat: extend tiling_space.json with min/max/step, shape_key, block_dim_expr"
```

---

## Task 3: autotuner CLI tool

**Files:**
- Create: `tools/autotuner/CMakeLists.txt`
- Create: `tools/autotuner/autotuner_main.cpp`
- Modify: `CMakeLists.txt` (root, add `add_subdirectory(tools/autotuner)`)

### Step 1: Write `tools/autotuner/CMakeLists.txt`

- [ ] **Create `tools/autotuner/CMakeLists.txt`**

```cmake
# tools/autotuner/CMakeLists.txt
set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(autotuner
  autotuner_main.cpp
)

target_link_libraries(autotuner PRIVATE
  AscendCRuntime
)

target_include_directories(autotuner PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
```

### Step 2: Wire into root CMakeLists.txt

- [ ] **Add `add_subdirectory(tools/autotuner)` to root `CMakeLists.txt`**

Find the line `add_subdirectory(tools/sim-validator)` and add after it:
```cmake
add_subdirectory(tools/autotuner)
```

### Step 3: Write `tools/autotuner/autotuner_main.cpp`

This is the main implementation. Write it in sections:

- [ ] **Write includes and CLI option declarations**

```cpp
// tools/autotuner/autotuner_main.cpp
#include "Runtime/NpyIO.h"
#include "Runtime/SimValidator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

using namespace mlir::runtime;
using namespace llvm;

static cl::opt<std::string> SpaceFile("space",
    cl::desc("Path to tiling_space.json"), cl::Required);
static cl::opt<std::string> KernelFile("kernel",
    cl::desc("Kernel .cpp source file (overrides kernel_file in JSON)"), cl::init(""));
static cl::opt<std::string> InputFiles("inputs",
    cl::desc("Comma-separated input .npy files"), cl::Required);
static cl::opt<std::string> ExpectedFile("expected",
    cl::desc("Expected output .npy file"), cl::Required);
static cl::opt<std::string> ShapeStr("shape",
    cl::desc("Comma-separated KEY=VALUE shape pairs: M=32,N=32"), cl::Required);
static cl::opt<std::string> OutputFile("output",
    cl::desc("Output tiling_func.cpp path"), cl::init("tiling_func.cpp"));
static cl::opt<std::string> SocVersion("soc",
    cl::desc("SoC version (overrides JSON; default: Ascend910B1)"), cl::init(""));
static cl::opt<double> Atol("atol", cl::desc("Absolute tolerance"), cl::init(1.0));
static cl::opt<double> Rtol("rtol", cl::desc("Relative tolerance"), cl::init(1e-2));
```

- [ ] **Write helper functions**

```cpp
static std::vector<std::string> splitComma(const std::string& s) {
  std::vector<std::string> parts;
  std::istringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) parts.push_back(tok);
  return parts;
}

// Parse "KEY=VALUE,..." into map
static std::map<std::string, int64_t> parseKV(const std::string& s) {
  std::map<std::string, int64_t> m;
  for (auto& tok : splitComma(s)) {
    auto eq = tok.find('=');
    if (eq == std::string::npos) continue;
    m[tok.substr(0, eq)] = std::stoll(tok.substr(eq + 1));
  }
  return m;
}

// Evaluate block_dim_expr: supports "ceil(X/Y)" and "X/Y"
// vars: map of variable name → value (includes both shape keys and tiling params)
static int64_t evalBlockDimExpr(const std::string& expr,
                                 const std::map<std::string, int64_t>& vars) {
  // Try "ceil(A/B)"
  bool is_ceil = false;
  std::string inner = expr;
  {
    std::string e = expr;
    // trim spaces
    e.erase(std::remove(e.begin(), e.end(), ' '), e.end());
    if (e.size() > 5 && e.substr(0, 5) == "ceil(" && e.back() == ')') {
      inner = e.substr(5, e.size() - 6);
      is_ceil = true;
    } else {
      inner = e;
    }
  }
  // Parse "A/B"
  auto slash = inner.find('/');
  if (slash == std::string::npos) {
    // single variable
    auto it = vars.find(inner);
    return it != vars.end() ? it->second : 1;
  }
  std::string lhs = inner.substr(0, slash);
  std::string rhs = inner.substr(slash + 1);
  auto lookup = [&](const std::string& name) -> int64_t {
    auto it = vars.find(name);
    if (it != vars.end()) return it->second;
    // try as literal integer
    try { return std::stoll(name); } catch (...) { return 1; }
  };
  int64_t a = lookup(lhs), b = lookup(rhs);
  if (b == 0) return 1;
  if (is_ceil) return (a + b - 1) / b;
  return a / b;
}

// Pack tiling bytes from ordered param list
static std::vector<uint8_t> packTiling(
    const std::vector<std::pair<std::string, int64_t>>& params,
    const std::vector<std::string>& types) {
  std::vector<uint8_t> bytes;
  for (size_t i = 0; i < params.size(); ++i) {
    std::string type = i < types.size() ? types[i] : "int64";
    int64_t val = params[i].second;
    if (type == "int32" || type == "int32_t") {
      int32_t v = static_cast<int32_t>(val);
      uint8_t buf[4]; std::memcpy(buf, &v, 4);
      bytes.insert(bytes.end(), buf, buf + 4);
    } else {
      uint8_t buf[8]; std::memcpy(buf, &val, 8);
      bytes.insert(bytes.end(), buf, buf + 8);
    }
  }
  return bytes;
}
```

- [ ] **Write `TilingConfig` struct and JSON parsing**

```cpp
struct TilingParam {
  std::string name;
  std::string type;         // "int64" or "int32"
  bool        fixed = false;
  std::string shape_key;    // for fixed params: which shape variable to use
  std::vector<int64_t> values; // search candidates (from "values" or expanded min/max/step)
};

struct TilingSpace {
  std::string kernel_name;
  std::string kernel_file;
  std::string soc;
  std::string block_dim_expr;
  std::vector<TilingParam> params;
};

static llvm::Expected<TilingSpace> loadTilingSpace(const std::string& path) {
  auto buf = llvm::MemoryBuffer::getFile(path);
  if (!buf)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot open %s", path.c_str());
  auto json = llvm::json::parse((*buf)->getBuffer());
  if (!json)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "JSON parse error in %s", path.c_str());

  TilingSpace ts;
  auto& obj = *json->getAsObject();
  if (auto* v = obj.getString("kernel"))       ts.kernel_name    = v->str();
  if (auto* v = obj.getString("kernel_file"))  ts.kernel_file    = v->str();
  if (auto* v = obj.getString("soc"))          ts.soc            = v->str();
  if (auto* v = obj.getString("block_dim_expr")) ts.block_dim_expr = v->str();

  auto* params = obj.getArray("tiling_params");
  if (!params)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Missing tiling_params in %s", path.c_str());

  for (auto& pv : *params) {
    auto* po = pv.getAsObject();
    if (!po) continue;
    TilingParam p;
    if (auto* v = po->getString("name")) p.name = v->str();
    if (auto* v = po->getString("type")) p.type = v->str();
    else p.type = "int64";
    if (auto v = po->getBoolean("fixed")) p.fixed = *v;
    if (auto* v = po->getString("shape_key")) p.shape_key = v->str();

    if (!p.fixed) {
      // Explicit values list
      if (auto* arr = po->getArray("values")) {
        for (auto& av : *arr)
          if (auto iv = av.getAsInteger()) p.values.push_back(*iv);
      }
      // min/max/step range
      if (p.values.empty()) {
        int64_t mn = 0, mx = 0, st = 1;
        if (auto v = po->getInteger("min"))  mn = *v;
        if (auto v = po->getInteger("max"))  mx = *v;
        if (auto v = po->getInteger("step")) st = *v;
        if (st > 0)
          for (int64_t x = mn; x <= mx; x += st) p.values.push_back(x);
      }
    }
    ts.params.push_back(p);
  }
  return ts;
}
```

- [ ] **Write enumeration and validation loop**

```cpp
struct SearchResult {
  std::vector<std::pair<std::string, int64_t>> config; // name → value (all params)
  int64_t   block_dim   = 1;
  int64_t   cycle_count = -1;
  double    max_abs_diff = 0.0;
  bool      passed      = false;
};

// Cartesian product of search variable indices
static void enumerate(
    const std::vector<TilingParam>& search_vars,
    size_t idx,
    std::vector<int64_t>& current,
    std::vector<std::vector<int64_t>>& results) {
  if (idx == search_vars.size()) {
    results.push_back(current);
    return;
  }
  for (int64_t v : search_vars[idx].values) {
    current[idx] = v;
    enumerate(search_vars, idx + 1, current, results);
  }
}

static std::vector<SearchResult> runSearch(
    const TilingSpace& ts,
    const std::map<std::string, int64_t>& shape,
    const std::string& kernel_path,
    const std::string& soc,
    RunArgs& args_template,          // inputs filled; outputs pre-alloc'd; tiling empty
    const std::vector<NDArray>& expected,
    double atol, double rtol) {

  // Separate search vars from fixed
  std::vector<TilingParam> search_vars;
  for (auto& p : ts.params)
    if (!p.fixed && !p.values.empty()) search_vars.push_back(p);

  // Cartesian product
  std::vector<std::vector<int64_t>> combos;
  std::vector<int64_t> cur(search_vars.size(), 0);
  enumerate(search_vars, 0, cur, combos);

  int total = static_cast<int>(combos.size());
  std::vector<SearchResult> results;
  Compiler::Config cc;
  cc.soc_version = soc;

  for (int ci = 0; ci < total; ++ci) {
    // Build var map: shape + this combo
    std::map<std::string, int64_t> vars = shape;
    for (size_t si = 0; si < search_vars.size(); ++si)
      vars[search_vars[si].name] = combos[ci][si];

    // Build ordered param list (preserving JSON order for byte packing)
    std::vector<std::pair<std::string, int64_t>> param_vals;
    std::vector<std::string> param_types;
    bool param_error = false;
    for (auto& p : ts.params) {
      int64_t val = 0;
      if (p.fixed) {
        if (p.shape_key.empty()) {
          llvm::errs() << "Error: fixed param '" << p.name
                       << "' has no shape_key in tiling_space.json\n";
          param_error = true;
          break;
        }
        auto it = shape.find(p.shape_key);
        if (it == shape.end()) {
          llvm::errs() << "Error: shape key '" << p.shape_key
                       << "' not found in --shape (needed by param '" << p.name << "')\n";
          param_error = true;
          break;
        }
        val = it->second;
      } else {
        auto it = vars.find(p.name);
        val = it != vars.end() ? it->second : 0;
      }
      param_vals.push_back({p.name, val});
      param_types.push_back(p.type);
    }
    if (param_error) break;

    int64_t block_dim = ts.block_dim_expr.empty() ? 1
                        : evalBlockDimExpr(ts.block_dim_expr, vars);

    // Print progress
    llvm::outs() << "[" << (ci + 1) << "/" << total << "]";
    for (auto& [n, v] : param_vals)
      if (!n.starts_with("dim_")) llvm::outs() << " " << n << "=" << v;
    llvm::outs() << " block_dim=" << block_dim << " ...";
    llvm::outs().flush();

    // Run — zero the output buffer before each call to avoid stale data
    // (args_template.outputs[0].data is a shared allocation; shallow copy below
    //  keeps the same pointer, so we must memset before each Validate call.
    //  Input data is never modified by Executor, so shallow copy of inputs is safe.)
    std::memset(args_template.outputs[0].data,
                0, args_template.outputs[0].nbytes());

    RunArgs args = args_template;
    args.tiling    = packTiling(param_vals, param_types);
    args.block_dim = static_cast<int>(block_dim);

    SimValidator validator;
    auto res = validator.Validate(kernel_path, ts.kernel_name, args,
                                   expected, atol, rtol, cc);

    SearchResult sr;
    sr.config      = param_vals;
    sr.block_dim   = block_dim;
    sr.cycle_count = res.cycle_count;
    sr.max_abs_diff = res.max_abs_diff;
    sr.passed      = res.passed;

    if (res.passed)
      llvm::outs() << " PASS  cycles=" << res.cycle_count
                   << " max_diff=" << res.max_abs_diff << "\n";
    else
      llvm::outs() << " FAIL  " << res.error_msg << "\n";

    results.push_back(sr);
  }
  return results;
}
```

- [ ] **Write `emitTilingFunc()` code generator**

```cpp
static void emitTilingFunc(
    const std::string& out_path,
    const TilingSpace& ts,
    const SearchResult& best,
    const std::map<std::string, int64_t>& shape) {

  // Build shape param list (unique shape_keys from fixed params)
  std::vector<std::string> shape_params;
  for (auto& p : ts.params)
    if (p.fixed && !p.shape_key.empty()) {
      bool found = false;
      for (auto& s : shape_params) if (s == p.shape_key) { found = true; break; }
      if (!found) shape_params.push_back(p.shape_key);
    }

  std::string soc = ts.soc;

  std::ofstream f(out_path);

  f << "// Auto-generated by autotuner\n";
  f << "// kernel: " << ts.kernel_name << "  soc: " << soc << "\n";
  f << "// best config:";
  for (auto& [n, v] : best.config)
    if (!n.starts_with("dim_")) f << "  " << n << "=" << v;
  f << "\n";
  f << "// cycle_count=" << best.cycle_count
    << "  max_abs_diff=" << best.max_abs_diff << "\n\n";

  f << "#include <cstdint>\n\n";

  // TilingData struct
  f << "struct TilingData {\n";
  for (auto& p : ts.params)
    f << "  " << (p.type == "int32" ? "int32_t" : "int64_t")
      << " " << p.name << ";\n";
  f << "};\n\n";

  // Shape param list for function signatures
  std::string shape_sig;
  for (size_t i = 0; i < shape_params.size(); ++i) {
    if (i) shape_sig += ", ";
    shape_sig += "int64_t " + shape_params[i];
  }

  // get_tiling()
  f << "void get_tiling(" << shape_sig << ", TilingData* out) {\n";
  for (auto& [n, v] : best.config) {
    // fixed params use shape variable; search params use constant
    bool is_fixed = false;
    std::string sk;
    for (auto& p : ts.params)
      if (p.name == n && p.fixed) { is_fixed = true; sk = p.shape_key; break; }
    if (is_fixed)
      f << "  out->" << n << " = " << sk << ";\n";
    else
      f << "  out->" << n << " = " << v << ";\n";
  }
  f << "}\n\n";

  // get_block_dim()
  // Strategy: substitute search-var constants into block_dim_expr, then rewrite
  // "ceil(A/B)" → "(A + B - 1) / B" so the emitted C++ uses integer arithmetic
  // (not std::ceil which would require floating point and truncation issues).
  // Shape variables (e.g. "M", "N") are left as-is — they become function params.
  // IMPORTANT: substitute longer names before shorter ones to avoid partial matches
  // (e.g. substitute "TB_M" before "M" so "M" doesn't match inside "TB_M").
  f << "int get_block_dim(" << shape_sig << ") {\n";
  if (!ts.block_dim_expr.empty()) {
    std::string expr = ts.block_dim_expr;
    // trim spaces
    expr.erase(std::remove(expr.begin(), expr.end(), ' '), expr.end());

    // Collect (name, value) for non-fixed params, sort longest name first
    std::vector<std::pair<std::string, int64_t>> subst_list;
    for (auto& [n, v] : best.config) {
      bool is_fixed = false;
      for (auto& p : ts.params) if (p.name == n && p.fixed) { is_fixed = true; break; }
      if (!is_fixed) subst_list.push_back({n, v});
    }
    std::sort(subst_list.begin(), subst_list.end(),
              [](auto& a, auto& b){ return a.first.size() > b.first.size(); });

    // Substitute each search var with its best constant value
    for (auto& [from, val] : subst_list) {
      std::string to = std::to_string(val);
      size_t pos = 0;
      while ((pos = expr.find(from, pos)) != std::string::npos) {
        bool pre_ok  = (pos == 0) || (!std::isalnum(expr[pos-1]) && expr[pos-1] != '_');
        bool post_ok = (pos + from.size() >= expr.size()) ||
                       (!std::isalnum(expr[pos+from.size()]) && expr[pos+from.size()] != '_');
        if (pre_ok && post_ok) {
          expr.replace(pos, from.size(), to);
          pos += to.size();
        } else {
          pos += from.size();
        }
      }
    }

    // Rewrite "ceil(A/B)" → "(A + B - 1) / B" for correct integer ceiling
    // Only handles the simple single-division case we use in block_dim_expr.
    std::string emit_expr = expr;
    {
      std::string e = expr;
      if (e.size() > 5 && e.substr(0, 5) == "ceil(" && e.back() == ')') {
        std::string inner = e.substr(5, e.size() - 6);
        auto slash = inner.find('/');
        if (slash != std::string::npos) {
          std::string lhs = inner.substr(0, slash);
          std::string rhs = inner.substr(slash + 1);
          // emit "(lhs + rhs - 1) / rhs"
          emit_expr = "(" + lhs + " + " + rhs + " - 1) / " + rhs;
        }
      }
    }

    f << "  // block_dim_expr: " << ts.block_dim_expr << "\n";
    f << "  return static_cast<int>(" << emit_expr << ");\n";
  } else {
    f << "  return " << best.block_dim << ";\n";
  }
  f << "}\n";

  llvm::outs() << "Emitted: " << out_path << "\n";
}

- [ ] **Write `main()`**

```cpp
int main(int argc, char** argv) {
  cl::ParseCommandLineOptions(argc, argv, "AscendC AutoTuner\n");

  // Load tiling space
  // Note: all _exit() calls below bypass destructors to avoid simulator thread
  // race on process exit (same reason as sim-validator tool).
  auto ts_or = loadTilingSpace(SpaceFile);
  if (!ts_or) {
    llvm::errs() << "Error: " << llvm::toString(ts_or.takeError()) << "\n";
    _exit(1);
  }
  TilingSpace ts = std::move(*ts_or);

  // Resolve kernel path
  std::string kernel_path = KernelFile.empty() ? ts.kernel_file : KernelFile.getValue();
  if (kernel_path.empty()) {
    llvm::errs() << "Error: kernel file not specified (--kernel or kernel_file in JSON)\n";
    _exit(1);
  }
  // If kernel_path is relative, resolve relative to SpaceFile's directory
  if (!llvm::sys::path::is_absolute(kernel_path)) {
    llvm::SmallString<256> base(SpaceFile.getValue());
    llvm::sys::path::remove_filename(base);
    llvm::sys::path::append(base, kernel_path);
    kernel_path = base.str().str();
  }

  // Resolve SOC
  std::string soc = SocVersion.empty() ? ts.soc : SocVersion.getValue();
  if (soc.empty()) soc = "Ascend910B1";
  ts.soc = soc;

  // Parse shape
  auto shape = parseKV(ShapeStr);

  // Load inputs
  std::vector<NDArray> inputs;
  for (auto& path : splitComma(InputFiles)) {
    auto arr_or = LoadNpy(path);
    if (!arr_or) {
      llvm::errs() << "Error loading " << path << ": "
                   << llvm::toString(arr_or.takeError()) << "\n";
      _exit(1);
    }
    inputs.push_back(*arr_or);
  }

  // Load expected
  auto exp_or = LoadNpy(ExpectedFile);
  if (!exp_or) {
    llvm::errs() << "Error loading expected: " << llvm::toString(exp_or.takeError()) << "\n";
    _exit(1);
  }
  std::vector<NDArray> expected_arrs = {*exp_or};

  // Pre-alloc output
  NDArray out_buf;
  out_buf.shape = expected_arrs[0].shape;
  out_buf.dtype = expected_arrs[0].dtype;
  out_buf.data  = new uint8_t[out_buf.nbytes()]();

  // Build args template (inputs + output pre-filled; tiling set per-config in runSearch)
  RunArgs args_tmpl;
  args_tmpl.inputs  = inputs;
  args_tmpl.outputs = {out_buf};

  // Run search
  llvm::outs() << "Searching " << ts.kernel_name << " on " << soc << "\n";
  auto results = runSearch(ts, shape, kernel_path, soc,
                            args_tmpl, expected_arrs, Atol, Rtol);

  // Free allocations
  delete[] static_cast<uint8_t*>(out_buf.data);
  for (auto& inp : inputs) delete[] static_cast<uint8_t*>(inp.data);
  delete[] static_cast<uint8_t*>(expected_arrs[0].data);

  // Filter PASS results
  std::vector<SearchResult*> passed;
  for (auto& r : results) if (r.passed) passed.push_back(&r);

  if (passed.empty()) {
    llvm::errs() << "Error: no configuration passed validation\n";
    _exit(1);
  }

  // Sort by cycle_count (ascending); configs with -1 go last
  std::sort(passed.begin(), passed.end(), [](SearchResult* a, SearchResult* b) {
    if (a->cycle_count < 0) return false;
    if (b->cycle_count < 0) return true;
    return a->cycle_count < b->cycle_count;
  });

  SearchResult& best = *passed[0];
  llvm::outs() << "\nBest config: cycles=" << best.cycle_count
               << " max_diff=" << best.max_abs_diff << "\n";
  for (auto& [n, v] : best.config)
    if (!n.starts_with("dim_")) llvm::outs() << "  " << n << "=" << v << "\n";

  emitTilingFunc(OutputFile, ts, best, shape);

  llvm::outs().flush();
  llvm::errs().flush();
  // Use _exit (not return/exit) to skip C++ destructors and atexit handlers:
  // SimValidator invokes libruntime_camodel which leaves background simulator
  // threads running; normal exit() races those threads and segfaults.
  _exit(0);
}
```

- [ ] **Step 4: Add `add_subdirectory(tools/autotuner)` to root `CMakeLists.txt`**

After `add_subdirectory(tools/sim-validator)`, add:
```cmake
add_subdirectory(tools/autotuner)
```

- [ ] **Step 5: Build**

```bash
# In xvm:
sleep 1
cd /home/niu/code/Ascend-MLIR
./scripts/build.sh --build-project --llvm-build-dir ~/code/llvm-project/build 2>&1 | tail -10
```
Expected: `autotuner` binary in `build/bin/`.

- [ ] **Step 6: Commit**

```bash
git add tools/autotuner/ CMakeLists.txt
git commit -m "feat(autotuner): add autotuner CLI tool"
```

---

## Task 4: Integration test

**Files:** none (test only)

- [ ] **Step 1: Generate test inputs (if not already present)**

```bash
# In xvm:
python3 -c "
import numpy as np
M, N = 32, 32
np.random.seed(42)
a = np.random.rand(M).astype(np.float16)
b = np.random.rand(M, N).astype(np.float16)
exp = (a.astype(np.float32)[:, None] + b.astype(np.float32)).sum(axis=1).astype(np.float16)
np.save('/tmp/input_a.npy', a)
np.save('/tmp/input_b.npy', b)
np.save('/tmp/expected.npy', exp)
print('saved')
"
```

- [ ] **Step 2: Run autotuner**

```bash
# In xvm sim/ dir, with env set (see Environment section):
cd /home/niu/code/Ascend-MLIR/sim
autotuner \
  --space    ../examples/broadcast-add-reduce/tiling_space.json \
  --kernel   ../examples/broadcast-add-reduce/step8_kernel.cpp \
  --inputs   /tmp/input_a.npy,/tmp/input_b.npy \
  --expected /tmp/expected.npy \
  --shape    "M=32,N=32" \
  --output   /tmp/tiling_func.cpp
```

Expected output:
```
Searching broadcast_add_reducesum on Ascend910B1
[1/4] TB_M=16 TB_N=16 block_dim=2 ... PASS  cycles=<N>  max_diff=<d>
[2/4] TB_M=16 TB_N=32 block_dim=2 ... PASS  cycles=<N>  max_diff=<d>
[3/4] TB_M=32 TB_N=16 block_dim=1 ... PASS  cycles=<N>  max_diff=<d>
[4/4] TB_M=32 TB_N=32 block_dim=1 ... PASS  cycles=<N>  max_diff=<d>

Best config: cycles=<lowest>  max_diff=<d>
  TB_M=<best>
  TB_N=<best>
Emitted: /tmp/tiling_func.cpp
```

- [ ] **Step 3: Verify generated `tiling_func.cpp`**

```bash
cat /tmp/tiling_func.cpp
```

Expected: valid C++ with `TilingData` struct, `get_tiling(int64_t M, int64_t N, TilingData*)`, `get_block_dim(int64_t M, int64_t N)`.

Compile-check:
```bash
g++ -std=c++17 -c /tmp/tiling_func.cpp -o /dev/null && echo "OK"
```

- [ ] **Step 4: Commit**

```bash
git add examples/broadcast-add-reduce/tiling_space.json  # already staged from Task 2
git commit -m "test(autotuner): integration test PASS for broadcast-add-reduce"
```

---

## Acceptance Criteria

- `autotuner --help` runs without error
- Running on `broadcast-add-reduce` with M=32,N=32 enumerates 4 configs, all PASS
- Best config selected by cycle_count
- Generated `tiling_func.cpp` compiles with `g++ -std=c++17`
- `get_tiling()` parameters match fixed `shape_key` values in JSON
- `get_block_dim()` returns `ceil(M / TB_M_best)`
