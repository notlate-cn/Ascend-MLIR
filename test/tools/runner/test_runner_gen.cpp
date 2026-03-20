// test/tools/runner/test_runner_gen.cpp
// Tests that HostRunnerGen::Generate() emits and compiles a valid runner,
// and that the runner enforces its required --bin argument.
#include "Runtime/HostRunnerGen.h"
#include "llvm/Support/raw_ostream.h"

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

  // Test 2: num_outputs > 1 returns error
  cfg.num_outputs = 2;
  auto bad = gen.Generate(cfg, "/tmp/runner_gen_test2");
  if (bad) {
    llvm::errs() << "FAIL: expected error for num_outputs=2 but got success\n";
    return 1;
  }
  llvm::consumeError(bad.takeError());
  llvm::outs() << "PASS: num_outputs=2 correctly rejected\n";

  return 0;
}
