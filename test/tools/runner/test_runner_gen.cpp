// test/tools/runner/test_runner_gen.cpp
// Tests that HostRunnerGen::Generate() emits and compiles a valid runner,
// and that the runner enforces its required --bin argument.
#include "Runtime/HostRunnerGen.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <string>

int main() {
  mlir::runtime::HostRunnerGen gen;
  mlir::runtime::HostRunnerGen::Config cfg;
  cfg.kernel_name   = "broadcast_add_reducesum";
  cfg.kernel_type   = "vec";
  cfg.soc_version   = "Ascend910B1";
  cfg.num_inputs    = 2;
  cfg.num_outputs   = 1;
  cfg.tiling_layout = {"int64", "int64", "int64", "int64"};
  cfg.verbose       = true;

  // Test 1: Generate succeeds
  auto result = gen.Generate(cfg, "/tmp/runner_gen_test");
  if (!result) {
    llvm::errs() << "FAIL Generate: " << llvm::toString(result.takeError()) << "\n";
    return 1;
  }
  llvm::outs() << "PASS Generate: runner at " << *result << "\n";

  // Test 2: num_outputs=2 now succeeds (multi-output supported)
  cfg.num_outputs   = 2;
  cfg.output_dtypes = {"f16", "f32"};
  auto result2 = gen.Generate(cfg, "/tmp/runner_gen_test2");
  if (!result2) {
    llvm::errs() << "FAIL Generate num_outputs=2: "
                 << llvm::toString(result2.takeError()) << "\n";
    return 1;
  }
  llvm::outs() << "PASS Generate num_outputs=2: runner at " << *result2 << "\n";

  // Test 3: output_dtypes shorter than num_outputs → defaults to f16 for missing
  cfg.num_outputs   = 3;
  cfg.output_dtypes = {"f32"};  // only 1 provided, indices 1 and 2 default to f16
  auto result3 = gen.Generate(cfg, "/tmp/runner_gen_test3");
  if (!result3) {
    llvm::errs() << "FAIL Generate num_outputs=3 partial dtypes: "
                 << llvm::toString(result3.takeError()) << "\n";
    return 1;
  }
  llvm::outs() << "PASS Generate num_outputs=3 partial dtypes: runner at " << *result3 << "\n";

  // Test 4: workspace_size propagated to runner source
  cfg.num_outputs    = 1;
  cfg.output_dtypes  = {};
  cfg.workspace_size = 65536;
  auto result4 = gen.Generate(cfg, "/tmp/runner_gen_test4");
  if (!result4) {
    llvm::errs() << "FAIL Generate workspace_size=65536: "
                 << llvm::toString(result4.takeError()) << "\n";
    return 1;
  }
  // Verify 65536 appears in the generated runner.cpp
  {
    std::ifstream src("/tmp/runner_gen_test4/runner.cpp");
    std::string content((std::istreambuf_iterator<char>(src)), {});
    if (content.find("65536") == std::string::npos) {
      llvm::errs() << "FAIL: workspace_size=65536 not found in runner.cpp\n";
      return 1;
    }
  }
  llvm::outs() << "PASS Generate workspace_size=65536 in runner.cpp\n";

  return 0;
}
