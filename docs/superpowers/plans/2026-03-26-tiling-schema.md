# TilingSchema Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a validated `TilingSchema` helper that reads `tiling_space.json` and packs tiling parameters with named-field validation in both C++ and Python, without changing any existing runtime interfaces.

**Architecture:** Three independent deliverables in dependency order: (1) C++ `TilingSchema` class in `lib/Runtime`, (2) `--tiling-schema` option in the validator CLI, (3) pure-Python `TilingSchema` in `python/runtime`. All share the same JSON fixture from `examples/broadcast-add-reduce/tiling_space.json`. The existing `buildTiling()` and `--tiling-layout` are left untouched.

**Tech Stack:** C++17, `llvm::json` (already linked via `LLVMSupport`), Python 3 stdlib (`json`, `struct`), LLVM `cl::opt` for CLI.

---

## File Map

| Change | File | Responsibility |
|--------|------|----------------|
| Create | `include/Runtime/TilingSchema.h` | `TilingField` struct + `TilingSchema` class declaration |
| Create | `lib/Runtime/TilingSchema.cpp` | `fromJson()` + `pack()` implementation |
| Modify | `lib/Runtime/CMakeLists.txt` | Add `TilingSchema.cpp` to `AscendCRuntime` sources |
| Create | `test/tools/runtime/test_tiling_schema.cpp` | C++ unit tests (no simulator required) |
| Modify | `tools/validator/validator_main.cpp` | Add `--tiling-schema` option |
| Create | `python/runtime/tiling_schema.py` | Pure-Python `TilingSchema` |
| Modify | `python/runtime/__init__.py` | Export `TilingSchema` |
| Create | `python/test/test_tiling_schema.py` | Python unit tests |

---

## Task 1: C++ header and implementation

**Files:**
- Create: `include/Runtime/TilingSchema.h`
- Create: `lib/Runtime/TilingSchema.cpp`
- Modify: `lib/Runtime/CMakeLists.txt`

- [ ] **Step 1: Write the header**

Create `include/Runtime/TilingSchema.h`:

```cpp
//===- TilingSchema.h - Validated tiling parameter packing ------*- C++ -*-===//
//
// Reads the `tiling_params` array from a tiling_space.json file and packs
// named parameters into a little-endian byte vector suitable for RunArgs::tiling.
//
// Only `name` and `type` fields are read; autotuner fields (fixed, shape_key,
// min, max, step, values) are ignored.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <string>
#include <utility>
#include <vector>

namespace mlir {
namespace runtime {

struct TilingField {
  std::string name;
  std::string type; // "int32" | "int64" (unknown defaults to int64)
};

class TilingSchema {
public:
  /// Load schema from a tiling_space.json file.
  /// Returns error if the file cannot be read or is malformed.
  static llvm::Expected<TilingSchema> fromJson(llvm::StringRef path);

  /// Pack named parameters into a little-endian byte vector.
  ///
  /// params must be supplied in field-declaration order and each name must
  /// match the corresponding schema field name positionally.
  ///
  /// Returns error if:
  ///   - params.size() != fields_.size()
  ///   - params[i].first != fields_[i].name for any i
  llvm::Expected<std::vector<uint8_t>>
  pack(llvm::ArrayRef<std::pair<std::string, int64_t>> params) const;

  llvm::ArrayRef<TilingField> fields() const { return fields_; }
  size_t size() const { return fields_.size(); }

private:
  std::vector<TilingField> fields_;
};

} // namespace runtime
} // namespace mlir
```

- [ ] **Step 2: Write the implementation**

Create `lib/Runtime/TilingSchema.cpp`:

```cpp
//===- TilingSchema.cpp ---------------------------------------------------===//
#include "Runtime/TilingSchema.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <string>

namespace mlir {
namespace runtime {

llvm::Expected<TilingSchema> TilingSchema::fromJson(llvm::StringRef path) {
  auto bufOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufOrErr)
    return llvm::createStringError(bufOrErr.getError(),
                                   "cannot open '%s'", path.data());

  auto parsed = llvm::json::parse((*bufOrErr)->getBuffer());
  if (!parsed)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "JSON parse error in '%s': %s",
                                   path.data(),
                                   llvm::toString(parsed.takeError()).c_str());

  auto *root = parsed->getAsObject();
  if (!root)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "'%s': root must be a JSON object",
                                   path.data());

  auto *paramsArr = root->getArray("tiling_params");
  if (!paramsArr)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "'%s': missing 'tiling_params' array",
                                   path.data());

  TilingSchema schema;
  for (auto &elem : *paramsArr) {
    auto *obj = elem.getAsObject();
    if (!obj)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "'%s': tiling_params entry is not an object",
                                     path.data());
    auto nameStr = obj->getString("name");
    if (!nameStr)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "'%s': tiling_params entry missing 'name'",
                                     path.data());
    TilingField f;
    f.name = nameStr->str();
    if (auto typeStr = obj->getString("type"))
      f.type = typeStr->str();
    else
      f.type = "int64";
    schema.fields_.push_back(std::move(f));
  }
  return schema;
}

llvm::Expected<std::vector<uint8_t>>
TilingSchema::pack(
    llvm::ArrayRef<std::pair<std::string, int64_t>> params) const {
  if (params.size() != fields_.size()) {
    std::string msg;
    llvm::raw_string_ostream os(msg);
    os << "tiling param count mismatch: got " << params.size()
       << ", expected " << fields_.size() << " (";
    for (size_t i = 0; i < fields_.size(); ++i) {
      if (i) os << ", ";
      os << fields_[i].name;
    }
    os << ")";
    return llvm::createStringError(llvm::inconvertibleErrorCode(), os.str());
  }

  std::vector<uint8_t> bytes;
  for (size_t i = 0; i < fields_.size(); ++i) {
    if (params[i].first != fields_[i].name) {
      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << "tiling param name mismatch at position " << i
         << ": got '" << params[i].first
         << "', expected '" << fields_[i].name << "'";
      return llvm::createStringError(llvm::inconvertibleErrorCode(), os.str());
    }
    int64_t val = params[i].second;
    if (fields_[i].type == "int32" || fields_[i].type == "int32_t") {
      int32_t v = static_cast<int32_t>(val);
      uint8_t buf[4];
      std::memcpy(buf, &v, 4);
      bytes.insert(bytes.end(), buf, buf + 4);
    } else {
      uint8_t buf[8];
      std::memcpy(buf, &val, 8);
      bytes.insert(bytes.end(), buf, buf + 8);
    }
  }
  return bytes;
}

} // namespace runtime
} // namespace mlir
```

- [ ] **Step 3: Add TilingSchema.cpp to CMakeLists.txt**

Edit `lib/Runtime/CMakeLists.txt`. Change:

```cmake
add_mlir_library(AscendCRuntime
  Compiler.cpp
  Executor.cpp
  HostRunnerGen.cpp
  SimValidator.cpp
  NpyIO.cpp
```

To:

```cmake
add_mlir_library(AscendCRuntime
  Compiler.cpp
  Executor.cpp
  HostRunnerGen.cpp
  SimValidator.cpp
  NpyIO.cpp
  TilingSchema.cpp
```

- [ ] **Step 4: Build to verify it compiles**

On xvm:
```bash
cd /home/niu/code/Ascend-MLIR/build
cmake --build . --target AscendCRuntime -j4 2>&1 | tail -5
```
Expected: `[100%] Built target AscendCRuntime` with no errors.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/TilingSchema.h lib/Runtime/TilingSchema.cpp lib/Runtime/CMakeLists.txt
git commit -m "feat(runtime): add TilingSchema – validated tiling parameter packing"
```

---

## Task 2: C++ unit tests

**Files:**
- Create: `test/tools/runtime/test_tiling_schema.cpp`

The test uses the same hand-rolled framework as `test_runtime.cpp` (custom `EXPECT` macro, `g_pass`/`g_fail` counters, no external test framework). It uses `examples/broadcast-add-reduce/tiling_space.json` as the fixture, which has 6 fields all of type `int64`:
`TB_M, TB_N, dim_arg0_0, dim_arg1_1, dim_arg0_1, dim_arg1_0`.

- [ ] **Step 1: Write the test file**

Create `test/tools/runtime/test_tiling_schema.cpp`:

```cpp
// test/tools/runtime/test_tiling_schema.cpp
//
// Unit tests for TilingSchema (fromJson + pack).
// Does NOT require a simulator or .bin file.
//
// Build (on xvm):
//   LLVM_BUILD=~/code/llvm-project/llvm/build
//   cd /home/niu/code/Ascend-MLIR/build
//   cmake --build . --target AscendCRuntime -j4
//   cd ..
//   g++ -std=c++17 -I include/ -I $LLVM_BUILD/include \
//       -I ~/code/llvm-project/llvm/include \
//       test/tools/runtime/test_tiling_schema.cpp \
//       build/lib/libAscendCRuntime.a \
//       $($LLVM_BUILD/bin/llvm-config --ldflags --libs support) \
//       -ldl -o /tmp/test_tiling_schema
//   /tmp/test_tiling_schema
//
// The fixture used is examples/broadcast-add-reduce/tiling_space.json.
// It defines 6 int64 fields: TB_M, TB_N, dim_arg0_0, dim_arg1_1, dim_arg0_1, dim_arg1_0.
//
#include "Runtime/TilingSchema.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace mlir::runtime;

static int g_pass = 0, g_fail = 0;

#define EXPECT(cond, msg)                                              \
  do {                                                                 \
    if (cond) {                                                        \
      llvm::outs() << "  PASS: " << (msg) << "\n";                    \
      ++g_pass;                                                        \
    } else {                                                           \
      llvm::errs() << "  FAIL: " << (msg) << "\n";                    \
      ++g_fail;                                                        \
    }                                                                  \
  } while (0)

// Write a minimal tiling_space.json with mixed types to a temp file.
// Returns the file path.
static std::string writeMixedJson(const std::string& path) {
  std::ofstream f(path);
  f << R"({
  "kernel": "test_kernel",
  "tiling_params": [
    {"name": "A", "type": "int64"},
    {"name": "B", "type": "int32"},
    {"name": "C", "type": "int64"}
  ]
})";
  return path;
}

static std::string writeMalformedJson(const std::string& path) {
  std::ofstream f(path);
  f << "{ not valid json";
  return path;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

static void testFromJsonRoundtrip(const std::string& fixture) {
  llvm::outs() << "\n[fromJson roundtrip]\n";
  auto s = TilingSchema::fromJson(fixture);
  EXPECT(!s.takeError(), "fromJson succeeds on valid fixture");

  // Re-load to inspect
  auto s2 = TilingSchema::fromJson(fixture);
  if (!s2) { llvm::consumeError(s2.takeError()); return; }

  EXPECT(s2->size() == 6, "fixture has 6 fields");
  EXPECT(s2->fields()[0].name == "TB_M",       "fields[0].name == TB_M");
  EXPECT(s2->fields()[0].type == "int64",      "fields[0].type == int64");
  EXPECT(s2->fields()[1].name == "TB_N",       "fields[1].name == TB_N");
  EXPECT(s2->fields()[2].name == "dim_arg0_0", "fields[2].name == dim_arg0_0");
  EXPECT(s2->fields()[5].name == "dim_arg1_0", "fields[5].name == dim_arg1_0");
}

static void testPackHappyPath(const std::string& fixture) {
  llvm::outs() << "\n[pack happy path]\n";
  auto s = TilingSchema::fromJson(fixture);
  if (!s) { llvm::consumeError(s.takeError()); return; }

  // 6 int64 params: TB_M=16, TB_N=16, dim_arg0_0=64, dim_arg1_1=64, dim_arg0_1=64, dim_arg1_0=64
  using P = std::pair<std::string, int64_t>;
  auto result = s->pack({
    P{"TB_M",       16},
    P{"TB_N",       16},
    P{"dim_arg0_0", 64},
    P{"dim_arg1_1", 64},
    P{"dim_arg0_1", 64},
    P{"dim_arg1_0", 64},
  });
  EXPECT(!result.takeError(), "pack succeeds with correct params");

  // Re-pack to inspect bytes
  auto result2 = TilingSchema::fromJson(fixture);
  if (!result2) { llvm::consumeError(result2.takeError()); return; }
  auto bytes = result2->pack({
    P{"TB_M",       16},
    P{"TB_N",       16},
    P{"dim_arg0_0", 64},
    P{"dim_arg1_1", 64},
    P{"dim_arg0_1", 64},
    P{"dim_arg1_0", 64},
  });
  if (!bytes) { llvm::consumeError(bytes.takeError()); return; }

  // 6 fields × 8 bytes each = 48 bytes
  EXPECT(bytes->size() == 48, "6 int64 fields → 48 bytes");

  // Verify first field TB_M=16 is at offset 0
  int64_t tb_m = 0;
  std::memcpy(&tb_m, bytes->data(), 8);
  EXPECT(tb_m == 16, "bytes[0..7] == 16 (TB_M)");

  // Verify last field dim_arg1_0=64 is at offset 40
  int64_t last = 0;
  std::memcpy(&last, bytes->data() + 40, 8);
  EXPECT(last == 64, "bytes[40..47] == 64 (dim_arg1_0)");
}

static void testPackWrongCount(const std::string& fixture) {
  llvm::outs() << "\n[pack wrong count]\n";
  auto s = TilingSchema::fromJson(fixture);
  if (!s) { llvm::consumeError(s.takeError()); return; }

  using P = std::pair<std::string, int64_t>;
  // Only 2 params instead of 6
  auto result = s->pack({P{"TB_M", 16}, P{"TB_N", 16}});
  auto err = result.takeError();
  EXPECT((bool)err, "pack returns error when count mismatches");
  std::string msg = llvm::toString(std::move(err));
  EXPECT(msg.find("count mismatch") != std::string::npos,
         "error message contains 'count mismatch'");
}

static void testPackWrongName(const std::string& fixture) {
  llvm::outs() << "\n[pack wrong name]\n";
  auto s = TilingSchema::fromJson(fixture);
  if (!s) { llvm::consumeError(s.takeError()); return; }

  using P = std::pair<std::string, int64_t>;
  // First field name wrong: "WRONG" instead of "TB_M"
  auto result = s->pack({
    P{"WRONG",      16},
    P{"TB_N",       16},
    P{"dim_arg0_0", 64},
    P{"dim_arg1_1", 64},
    P{"dim_arg0_1", 64},
    P{"dim_arg1_0", 64},
  });
  auto err = result.takeError();
  EXPECT((bool)err, "pack returns error when name mismatches");
  std::string msg = llvm::toString(std::move(err));
  EXPECT(msg.find("name mismatch") != std::string::npos,
         "error message contains 'name mismatch'");
  EXPECT(msg.find("WRONG") != std::string::npos,
         "error message contains the bad name 'WRONG'");
  EXPECT(msg.find("TB_M") != std::string::npos,
         "error message contains the expected name 'TB_M'");
}

static void testFromJsonFileNotFound() {
  llvm::outs() << "\n[fromJson file not found]\n";
  auto s = TilingSchema::fromJson("/nonexistent/path/schema.json");
  auto err = s.takeError();
  EXPECT((bool)err, "fromJson returns error for missing file");
  llvm::consumeError(std::move(err));
}

static void testFromJsonMalformed() {
  llvm::outs() << "\n[fromJson malformed JSON]\n";
  std::string path = "/tmp/malformed_tiling.json";
  writeMalformedJson(path);
  auto s = TilingSchema::fromJson(path);
  auto err = s.takeError();
  EXPECT((bool)err, "fromJson returns error for malformed JSON");
  llvm::consumeError(std::move(err));
}

static void testPackMixedTypes() {
  llvm::outs() << "\n[pack mixed int32/int64 types]\n";
  std::string path = "/tmp/mixed_tiling.json";
  writeMixedJson(path);
  auto s = TilingSchema::fromJson(path);
  if (!s) { llvm::consumeError(s.takeError()); return; }
  EXPECT(s->size() == 3, "mixed fixture has 3 fields");
  EXPECT(s->fields()[0].type == "int64", "A is int64");
  EXPECT(s->fields()[1].type == "int32", "B is int32");
  EXPECT(s->fields()[2].type == "int64", "C is int64");

  using P = std::pair<std::string, int64_t>;
  auto bytes = s->pack({P{"A", 1}, P{"B", 2}, P{"C", 3}});
  if (!bytes) { llvm::consumeError(bytes.takeError()); return; }

  // A(int64)=8B + B(int32)=4B + C(int64)=8B = 20 bytes
  EXPECT(bytes->size() == 20, "int64+int32+int64 = 20 bytes");

  int64_t a = 0; std::memcpy(&a, bytes->data() + 0, 8);
  int32_t b = 0; std::memcpy(&b, bytes->data() + 8, 4);
  int64_t c = 0; std::memcpy(&c, bytes->data() + 12, 8);
  EXPECT(a == 1, "A=1");
  EXPECT(b == 2, "B=2");
  EXPECT(c == 3, "C=3");
}

int main(int argc, char** argv) {
  // argv[1] must be path to examples/broadcast-add-reduce/tiling_space.json
  if (argc < 2) {
    llvm::errs() << "Usage: " << argv[0]
                 << " <path/to/tiling_space.json>\n";
    return 2;
  }
  std::string fixture = argv[1];

  testFromJsonRoundtrip(fixture);
  testPackHappyPath(fixture);
  testPackWrongCount(fixture);
  testPackWrongName(fixture);
  testFromJsonFileNotFound();
  testFromJsonMalformed();
  testPackMixedTypes();

  llvm::outs() << "\n" << g_pass << " passed, " << g_fail << " failed\n";
  return g_fail > 0 ? 1 : 0;
}
```

- [ ] **Step 2: Build and run the tests**

On xvm:
```bash
LLVM_BUILD=~/code/llvm-project/llvm/build
cd /home/niu/code/Ascend-MLIR/build
cmake --build . --target AscendCRuntime -j4
cd ..
g++ -std=c++17 -I include/ -I $LLVM_BUILD/include \
    -I ~/code/llvm-project/llvm/include \
    test/tools/runtime/test_tiling_schema.cpp \
    build/lib/libAscendCRuntime.a \
    $($LLVM_BUILD/bin/llvm-config --ldflags --libs support) \
    -ldl -o /tmp/test_tiling_schema
/tmp/test_tiling_schema examples/broadcast-add-reduce/tiling_space.json
```

Expected output:
```
[fromJson roundtrip]
  PASS: fromJson succeeds on valid fixture
  PASS: fixture has 6 fields
  PASS: fields[0].name == TB_M
  PASS: fields[0].type == int64
  PASS: fields[1].name == TB_N
  PASS: fields[2].name == dim_arg0_0
  PASS: fields[5].name == dim_arg1_0

[pack happy path]
  PASS: pack succeeds with correct params
  PASS: 6 int64 fields → 48 bytes
  PASS: bytes[0..7] == 16 (TB_M)
  PASS: bytes[40..47] == 64 (dim_arg1_0)

[pack wrong count]
  PASS: pack returns error when count mismatches
  PASS: error message contains 'count mismatch'

[pack wrong name]
  PASS: pack returns error when name mismatches
  PASS: error message contains 'name mismatch'
  PASS: error message contains the bad name 'WRONG'
  PASS: error message contains the expected name 'TB_M'

[fromJson file not found]
  PASS: fromJson returns error for missing file

[fromJson malformed JSON]
  PASS: fromJson returns error for malformed JSON

[pack mixed int32/int64 types]
  PASS: mixed fixture has 3 fields
  PASS: A is int64
  PASS: B is int32
  PASS: C is int64
  PASS: int64+int32+int64 = 20 bytes
  PASS: A=1
  PASS: B=2
  PASS: C=3

20 passed, 0 failed
```

- [ ] **Step 3: Commit**

```bash
git add test/tools/runtime/test_tiling_schema.cpp
git commit -m "test(runtime): add TilingSchema C++ unit tests"
```

---

## Task 3: validator `--tiling-schema` option

**Files:**
- Modify: `tools/validator/validator_main.cpp`

This task adds `--tiling-schema <path>` to the validator. When supplied, it replaces the `buildTiling()` + `--tiling-layout` path. The existing `--tiling-layout` path is left intact.

- [ ] **Step 1: Add the include and new CLI option**

In `tools/validator/validator_main.cpp`, add `#include "Runtime/TilingSchema.h"` after the existing includes (line 7), and add a new `cl::opt` after the `TilingLayout` opt (after line 29):

The includes block (lines 1–14) becomes:
```cpp
// tools/validator/validator_main.cpp
#include "Runtime/Executor.h"
#include "Runtime/NpyIO.h"
#include "Runtime/SimValidator.h"
#include "Runtime/TilingSchema.h"
#include "Runtime/Types.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
```

Add after the `TilingLayout` opt (after line 29):
```cpp
static cl::opt<std::string> TilingSchema("tiling-schema",
    cl::desc("Path to tiling_space.json; validates and packs --tiling-params by name"),
    cl::init(""));
```

- [ ] **Step 2: Replace the tiling-build section in main()**

Find the tiling-build block in `main()` (lines 142–145 in the original):

```cpp
  // Build tiling bytes
  std::vector<uint8_t> tiling;
  if (!TilingParams.empty()) {
    if (!buildTiling(TilingParams, TilingLayout, tiling)) _Exit(4);
  }
```

Replace it with:

```cpp
  // Build tiling bytes
  std::vector<uint8_t> tiling;
  if (!TilingParams.empty()) {
    if (!TilingSchema.empty()) {
      // Schema-validated path: load schema, pack by name
      auto schemaOrErr = mlir::runtime::TilingSchema::fromJson(TilingSchema);
      if (!schemaOrErr) {
        llvm::errs() << "Error: --tiling-schema: "
                     << llvm::toString(schemaOrErr.takeError()) << "\n";
        _Exit(4);
      }
      // Parse KEY=VALUE params into ordered pairs preserving declaration order
      auto pvec = splitComma(TilingParams);
      std::vector<std::pair<std::string, int64_t>> namedParams;
      // Build a name→value map first, then reorder by schema field order
      std::map<std::string, int64_t> pmap;
      for (auto& token : pvec) {
        auto eq = token.find('=');
        if (eq == std::string::npos) {
          llvm::errs() << "Error: --tiling-params token missing '=': " << token << "\n";
          _Exit(4);
        }
        pmap[token.substr(0, eq)] = std::stoll(token.substr(eq + 1));
      }
      for (auto& field : schemaOrErr->fields()) {
        auto it = pmap.find(field.name);
        if (it == pmap.end()) {
          llvm::errs() << "Error: --tiling-params missing field '" << field.name
                       << "' required by schema\n";
          _Exit(4);
        }
        namedParams.push_back({field.name, it->second});
      }
      // Warn about extra params not in schema
      for (auto& kv : pmap) {
        bool found = false;
        for (auto& f : schemaOrErr->fields())
          if (f.name == kv.first) { found = true; break; }
        if (!found)
          llvm::errs() << "Warning: --tiling-params field '" << kv.first
                       << "' not in schema (ignored)\n";
      }
      auto bytesOrErr = schemaOrErr->pack(namedParams);
      if (!bytesOrErr) {
        llvm::errs() << "Error: tiling pack: "
                     << llvm::toString(bytesOrErr.takeError()) << "\n";
        _Exit(4);
      }
      tiling = std::move(*bytesOrErr);
    } else {
      // Legacy path: positional layout string
      if (!buildTiling(TilingParams, TilingLayout, tiling)) _Exit(4);
    }
  }
```

Also add `#include <map>` to the includes block.

- [ ] **Step 3: Build the validator**

On xvm:
```bash
cd /home/niu/code/Ascend-MLIR/build
cmake --build . --target afir-validator -j4 2>&1 | tail -5
```
Expected: `[100%] Built target afir-validator` with no errors.

- [ ] **Step 4: Smoke-test --tiling-schema error path**

The validator needs a .bin file to actually run, but we can test the error path without one by triggering a schema mismatch:

```bash
source /home/niu/code/Ascend-MLIR/examples/env.sh
# This should print an error about missing field, not segfault
afir-validator \
  --bin /nonexistent.bin \
  --name test_kernel \
  --inputs /dev/null \
  --expected /dev/null \
  --tiling-schema examples/broadcast-add-reduce/tiling_space.json \
  --tiling-params "TB_M=16,TB_N=16" 2>&1 | head -3
```
Expected output contains: `missing field 'dim_arg0_0'` (exit 4, not a crash).

- [ ] **Step 5: Commit**

```bash
git add tools/validator/validator_main.cpp
git commit -m "feat(validator): add --tiling-schema option for validated tiling packing"
```

---

## Task 4: Python TilingSchema

**Files:**
- Create: `python/runtime/tiling_schema.py`
- Modify: `python/runtime/__init__.py`

- [ ] **Step 1: Write tiling_schema.py**

Create `python/runtime/tiling_schema.py`:

```python
# python/runtime/tiling_schema.py
"""
TilingSchema: load a tiling_space.json and pack named tiling parameters
into little-endian bytes suitable for execute_kernel(tiling_data=...).

Only 'name' and 'type' fields in tiling_params[] are used.
Autotuner fields (fixed, shape_key, min, max, step, values) are ignored.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Union


class TilingSchema:
    """
    Validated tiling parameter packer loaded from a tiling_space.json file.

    Example::

        schema = TilingSchema.from_json("step8.tiling_space.json")
        tiling_bytes = schema.pack(TB_M=16, TB_N=4, M=64, N=64)
        execute_kernel(..., tiling_data=tiling_bytes)
    """

    def __init__(self, fields: list[dict]) -> None:
        # fields: list of {"name": str, "type": "int32"|"int64"}
        self._fields = fields

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "TilingSchema":
        """
        Load schema from a tiling_space.json file.

        Raises:
            FileNotFoundError: if the file does not exist.
            ValueError: if the JSON is malformed or missing 'tiling_params'.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"tiling_space.json not found: {path}")
        with path.open() as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parse error in '{path}': {e}") from e
        params = data.get("tiling_params")
        if params is None:
            raise ValueError(f"'{path}': missing 'tiling_params' array")
        if not isinstance(params, list):
            raise ValueError(f"'{path}': 'tiling_params' must be a list")
        fields = []
        for i, entry in enumerate(params):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"'{path}': tiling_params[{i}] must be an object"
                )
            name = entry.get("name")
            if not name:
                raise ValueError(
                    f"'{path}': tiling_params[{i}] missing 'name'"
                )
            fields.append({
                "name": name,
                "type": entry.get("type", "int64"),
            })
        return cls(fields)

    def pack(self, **kwargs: int) -> bytes:
        """
        Pack named tiling parameters into little-endian bytes.

        Parameters must cover every schema field exactly (no extras, no missing).
        Field order in the output follows the schema declaration order.

        Args:
            **kwargs: one keyword argument per schema field, e.g.
                      schema.pack(TB_M=16, TB_N=4, M=64, N=64)

        Returns:
            bytes: packed little-endian representation

        Raises:
            ValueError: if fields are missing, extra, or names do not match.
        """
        expected = [f["name"] for f in self._fields]
        given = set(kwargs.keys())
        expected_set = set(expected)

        missing = sorted(expected_set - given)
        extra = sorted(given - expected_set)

        errors = []
        if missing:
            errors.append(f"missing tiling fields: {missing}")
        if extra:
            errors.append(f"unexpected tiling fields: {extra}")
        if errors:
            raise ValueError(
                "\n".join(errors) + f"\n  expected: {expected}"
            )

        result = bytearray()
        for field in self._fields:
            val = int(kwargs[field["name"]])
            if field["type"] in ("int32", "int32_t"):
                result += struct.pack("<i", val)
            else:
                result += struct.pack("<q", val)
        return bytes(result)

    @property
    def field_names(self) -> list[str]:
        """Ordered list of field names as declared in the schema."""
        return [f["name"] for f in self._fields]

    @property
    def fields(self) -> list[dict]:
        """Ordered list of {'name': str, 'type': str} dicts."""
        return list(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __repr__(self) -> str:
        return f"TilingSchema({self.field_names})"
```

- [ ] **Step 2: Export TilingSchema from __init__.py**

In `python/runtime/__init__.py`, add the import after the `# 执行器模块` block (after line 112). Add:

```python
# Tiling schema 模块
from .tiling_schema import TilingSchema
```

Also add `"TilingSchema"` to the `__all__` list, in the `# 便捷函数` section (after `"compile_kernel"`):

```python
    # Tiling schema
    "TilingSchema",
```

- [ ] **Step 3: Verify import works**

On xvm (no build needed, pure Python):
```bash
cd /home/niu/code/Ascend-MLIR
python3 -c "from python.runtime import TilingSchema; print(TilingSchema)"
```
Expected: `<class 'python.runtime.tiling_schema.TilingSchema'>`

- [ ] **Step 4: Commit**

```bash
git add python/runtime/tiling_schema.py python/runtime/__init__.py
git commit -m "feat(python): add TilingSchema – validated tiling packing from JSON"
```

---

## Task 5: Python unit tests

**Files:**
- Create: `python/test/test_tiling_schema.py`

- [ ] **Step 1: Write the test file**

Create `python/test/test_tiling_schema.py`:

```python
# python/test/test_tiling_schema.py
"""
Unit tests for TilingSchema.

Run:
    cd /home/niu/code/Ascend-MLIR
    python3 -m pytest python/test/test_tiling_schema.py -v

The fixture is examples/broadcast-add-reduce/tiling_space.json.
It has 6 int64 fields: TB_M, TB_N, dim_arg0_0, dim_arg1_1, dim_arg0_1, dim_arg1_0.
"""
import json
import struct
import tempfile
from pathlib import Path

import pytest

# Adjust import path so tests can be run from repo root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from python.runtime.tiling_schema import TilingSchema

FIXTURE = (
    Path(__file__).parent.parent.parent
    / "examples/broadcast-add-reduce/tiling_space.json"
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def make_json(params: list[dict]) -> Path:
    """Write a temporary tiling_space.json and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        suffix=".json", mode="w", delete=False
    )
    json.dump({"kernel": "test", "tiling_params": params}, tmp)
    tmp.close()
    return Path(tmp.name)


# ── from_json ─────────────────────────────────────────────────────────────────

def test_from_json_roundtrip():
    schema = TilingSchema.from_json(FIXTURE)
    assert schema.field_names == [
        "TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1", "dim_arg0_1", "dim_arg1_0"
    ]
    assert len(schema) == 6
    assert all(f["type"] == "int64" for f in schema.fields)


def test_from_json_file_not_found():
    with pytest.raises(FileNotFoundError, match="not found"):
        TilingSchema.from_json("/nonexistent/path/schema.json")


def test_from_json_malformed_json():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write("{ not valid json")
        path = Path(f.name)
    with pytest.raises(ValueError, match="JSON parse error"):
        TilingSchema.from_json(path)


def test_from_json_missing_tiling_params():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump({"kernel": "test"}, f)
        path = Path(f.name)
    with pytest.raises(ValueError, match="missing 'tiling_params'"):
        TilingSchema.from_json(path)


def test_from_json_type_defaults_to_int64():
    path = make_json([{"name": "X"}])  # no "type" key
    schema = TilingSchema.from_json(path)
    assert schema.fields[0]["type"] == "int64"


# ── pack ──────────────────────────────────────────────────────────────────────

def test_pack_happy_path_all_int64():
    schema = TilingSchema.from_json(FIXTURE)
    data = schema.pack(
        TB_M=16, TB_N=16,
        dim_arg0_0=64, dim_arg1_1=64,
        dim_arg0_1=64, dim_arg1_0=64,
    )
    # 6 × 8 bytes = 48
    assert len(data) == 48
    # Check each field value using struct.unpack
    vals = struct.unpack_from("<6q", data)
    assert vals == (16, 16, 64, 64, 64, 64)


def test_pack_int32_vs_int64():
    path = make_json([
        {"name": "A", "type": "int64"},
        {"name": "B", "type": "int32"},
        {"name": "C", "type": "int64"},
    ])
    schema = TilingSchema.from_json(path)
    data = schema.pack(A=1, B=2, C=3)
    # int64(8) + int32(4) + int64(8) = 20 bytes
    assert len(data) == 20
    a = struct.unpack_from("<q", data, 0)[0]
    b = struct.unpack_from("<i", data, 8)[0]
    c = struct.unpack_from("<q", data, 12)[0]
    assert (a, b, c) == (1, 2, 3)


def test_pack_int32_alias():
    """int32_t is accepted as a type alias for int32."""
    path = make_json([{"name": "X", "type": "int32_t"}])
    schema = TilingSchema.from_json(path)
    data = schema.pack(X=42)
    assert len(data) == 4
    assert struct.unpack_from("<i", data)[0] == 42


def test_pack_missing_field():
    schema = TilingSchema.from_json(FIXTURE)
    with pytest.raises(ValueError) as exc:
        schema.pack(TB_M=16, TB_N=16)  # missing 4 fields
    msg = str(exc.value)
    assert "missing tiling fields" in msg
    assert "dim_arg0_0" in msg
    assert "expected" in msg


def test_pack_extra_field():
    schema = TilingSchema.from_json(FIXTURE)
    with pytest.raises(ValueError) as exc:
        schema.pack(
            TB_M=16, TB_N=16,
            dim_arg0_0=64, dim_arg1_1=64,
            dim_arg0_1=64, dim_arg1_0=64,
            EXTRA=99,  # not in schema
        )
    msg = str(exc.value)
    assert "unexpected tiling fields" in msg
    assert "EXTRA" in msg


def test_pack_field_order_in_output_matches_schema():
    """Output byte order follows schema declaration order, not kwargs order."""
    path = make_json([
        {"name": "FIRST", "type": "int64"},
        {"name": "SECOND", "type": "int64"},
    ])
    schema = TilingSchema.from_json(path)
    # Intentionally pass kwargs in reverse order
    data = schema.pack(SECOND=200, FIRST=100)
    first, second = struct.unpack_from("<2q", data)
    assert first == 100
    assert second == 200


def test_repr():
    schema = TilingSchema.from_json(FIXTURE)
    r = repr(schema)
    assert "TB_M" in r
    assert "TilingSchema" in r
```

- [ ] **Step 2: Run the tests**

On xvm:
```bash
cd /home/niu/code/Ascend-MLIR
python3 -m pytest python/test/test_tiling_schema.py -v 2>&1
```

Expected: all tests pass (13 tests, 0 failures).

- [ ] **Step 3: Commit**

```bash
git add python/test/test_tiling_schema.py
git commit -m "test(python): add TilingSchema unit tests"
```

---

## Task 6: Update broadcast-add-reduce example

**Files:**
- Modify: `examples/broadcast-add-reduce/run.sh`

This task updates the existing example to use `--tiling-schema` instead of `--tiling-layout`, demonstrating the new API and verifying end-to-end integration.

- [ ] **Step 1: Find the validator invocation in run.sh**

The relevant validator call is around line 234–245 of `examples/broadcast-add-reduce/run.sh`. It currently uses `--tiling-layout` (implicit, no explicit flag) or passes the layout via some mechanism. Check:

```bash
grep -n "tiling" examples/broadcast-add-reduce/run.sh
```

- [ ] **Step 2: Update the validator call**

Find the validator invocation that includes `--tiling-params` and add `--tiling-schema`:

Replace any line of the form:
```bash
"$VALIDATOR" \
  --bin "$BUILD_DIR/broadcast_add_reducesum.bin" \
  --name broadcast_add_reducesum \
  --inputs "$DIR/input_a.npy,$DIR/input_b.npy" \
  --expected "$DIR/output_c.npy" \
  --tiling-params 'TB_M=16,TB_N=16,dim_arg0_0=64,dim_arg1_1=64,dim_arg0_1=64,dim_arg1_0=64' \
  --block-dim 4 \
  --atol 1e-3 \
  --rtol 1e-3
```

With:
```bash
"$VALIDATOR" \
  --bin "$BUILD_DIR/broadcast_add_reducesum.bin" \
  --name broadcast_add_reducesum \
  --inputs "$DIR/input_a.npy,$DIR/input_b.npy" \
  --expected "$DIR/output_c.npy" \
  --tiling-schema "$DIR/tiling_space.json" \
  --tiling-params 'TB_M=16,TB_N=16,dim_arg0_0=64,dim_arg1_1=64,dim_arg0_1=64,dim_arg1_0=64' \
  --block-dim 4 \
  --atol 1e-3 \
  --rtol 1e-3
```

(The `--tiling-layout` flag, if present, can be removed since `--tiling-schema` takes precedence.)

- [ ] **Step 3: Run the full example to verify**

```bash
cd /home/niu/code/Ascend-MLIR
bash examples/broadcast-add-reduce/run.sh 2>&1 | tail -10
```

Expected: final output includes `PASS` with no new errors.

- [ ] **Step 4: Commit**

```bash
git add examples/broadcast-add-reduce/run.sh
git commit -m "example(broadcast-add-reduce): use --tiling-schema in validator call"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|-----------------|------------|
| C++ `TilingSchema::fromJson()` | Task 1 |
| C++ `TilingSchema::pack()` with positional validation | Task 1 |
| `--tiling-schema` validator option, `--tiling-layout` remains | Task 3 |
| Pure-Python `TilingSchema.from_json()` | Task 4 |
| Pure-Python `TilingSchema.pack(**kwargs)` with set-based validation | Task 4 |
| No new C API binding | Tasks 4 (pure Python, no C layer) |
| `RunArgs::tiling`, `execute_kernel` unchanged | No tasks modify them ✓ |
| C++ unit tests: roundtrip, happy path, wrong count, wrong name, file not found, malformed | Task 2 |
| Python unit tests: roundtrip, happy path, missing, extra, int32/int64, order | Task 5 |
| Integration: broadcast-add-reduce updated | Task 6 |

**Placeholder scan:** None found.

**Type consistency check:**
- `TilingSchema::fromJson` → declared in Task 1 header, used in Tasks 2 and 3 ✓
- `TilingSchema::pack(ArrayRef<pair<string,int64_t>>)` → declared Task 1, used Task 2 ✓
- `TilingSchema.from_json()` / `.pack(**kwargs)` → defined Task 4, tested Task 5 ✓
- `schema.fields` (list of dict), `schema.field_names` (list of str), `len(schema)` → defined Task 4, used Task 5 ✓

Note on C++ validator (Task 3): the `--tiling-schema` path uses a map to resolve
by-name, then reorders to schema field order before calling `pack()`. This means the user
can pass `--tiling-params` in any order (e.g. alphabetical) and it still works correctly.
The positional validation in `pack()` is satisfied because we build `namedParams` in
schema order.
