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
//   /tmp/test_tiling_schema <path/to/tiling_space.json>
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

static void testFromJsonRoundtrip(const std::string& fixture) {
  llvm::outs() << "\n[fromJson roundtrip]\n";
  auto s = TilingSchema::fromJson(fixture);
  EXPECT(!s.takeError(), "fromJson succeeds on valid fixture");

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

  EXPECT(bytes->size() == 48, "6 int64 fields -> 48 bytes");

  int64_t tb_m = 0;
  std::memcpy(&tb_m, bytes->data(), 8);
  EXPECT(tb_m == 16, "bytes[0..7] == 16 (TB_M)");

  int64_t last = 0;
  std::memcpy(&last, bytes->data() + 40, 8);
  EXPECT(last == 64, "bytes[40..47] == 64 (dim_arg1_0)");
}

static void testPackWrongCount(const std::string& fixture) {
  llvm::outs() << "\n[pack wrong count]\n";
  auto s = TilingSchema::fromJson(fixture);
  if (!s) { llvm::consumeError(s.takeError()); return; }

  using P = std::pair<std::string, int64_t>;
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

  EXPECT(bytes->size() == 20, "int64+int32+int64 = 20 bytes");

  int64_t a = 0; std::memcpy(&a, bytes->data() + 0, 8);
  int32_t b = 0; std::memcpy(&b, bytes->data() + 8, 4);
  int64_t c = 0; std::memcpy(&c, bytes->data() + 12, 8);
  EXPECT(a == 1, "A=1");
  EXPECT(b == 2, "B=2");
  EXPECT(c == 3, "C=3");
}

int main(int argc, char** argv) {
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
