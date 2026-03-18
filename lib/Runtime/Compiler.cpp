// lib/Runtime/Compiler.cpp
#include "Runtime/Compiler.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

namespace mlir::runtime {

Compiler::Compiler(const Config& cfg) : cfg_(cfg) {}

static std::string getAscendHome() {
  const char* home = std::getenv("ASCEND_HOME_PATH");
  return home ? home : "/usr/local/Ascend/ascend-toolkit/latest";
}

llvm::Error Compiler::RunProcess(const std::vector<std::string>& args) {
  std::vector<llvm::StringRef> argv;
  argv.reserve(args.size());
  for (auto& a : args) argv.push_back(a);

  std::string err_msg;
  std::optional<llvm::StringRef> redirects[3];

  int ret = llvm::sys::ExecuteAndWait(argv[0], argv,
                                      /*env=*/std::nullopt,
                                      redirects,
                                      /*secondsToWait=*/300,
                                      /*memoryLimit=*/0,
                                      &err_msg);
  if (ret != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Process failed (exit %d): %s", ret, err_msg.c_str());
  return llvm::Error::success();
}

llvm::Expected<std::string> Compiler::Compile(const std::string& src_file,
                                               const std::string& output_dir,
                                               const std::string& kernel_name) {
  ::setenv("SOC_VERSION", cfg_.soc_version.c_str(), 1);

  std::string ascend_home = getAscendHome();
  std::string bisheng     = ascend_home + "/compiler/ccec_compiler/bin/bisheng";
  std::string lld         = ascend_home + "/compiler/ccec_compiler/bin/ld.lld";
  std::string tikcpp      = ascend_home + "/compiler/tikcpp";

  std::string obj_file = output_dir + "/" + kernel_name + ".o";
  std::string bin_file = output_dir + "/" + kernel_name + ".bin";

  // Get src directory and filename (bisheng requires cwd=src_dir, filename only)
  llvm::SmallString<256> src_path(src_file);
  llvm::sys::fs::make_absolute(src_path);
  if (!llvm::sys::fs::exists(src_path))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Source file not found: %s", src_path.c_str());
  std::string src_dir  = llvm::sys::path::parent_path(src_path).str();
  std::string src_name = llvm::sys::path::filename(src_path).str();

  // Step 1: compile to .o
  // bisheng must be invoked with cwd=src_dir and src_name (not full path)
  std::string compile_cmd =
      bisheng +
      " -c -x cce"
      " -O" + std::to_string(cfg_.opt_level) +
      " " + src_name +
      " --cce-aicore-arch=" + cfg_.arch +
      " --cce-aicore-only"
      " -std=c++17"
      " --cce-disable-kernel-global-attr-check"
      " -mllvm -cce-aicore-stack-size=0x8000"
      " -mllvm -cce-aicore-function-stack-size=0x8000"
      " -mllvm -cce-aicore-dcci-insert-for-scalar=false"
      " -I " + tikcpp + "/tikcfw"
      " -I " + tikcpp + "/tikcfw/impl"
      " -I " + tikcpp + "/tikcfw/interface"
      " -DASCENDC_DUMP=0"
      " -D__NPU_TILING__"
      " -DTILING_KEY_VAR=0"
      " -o " + obj_file;

  // Run via sh -c to set cwd
  std::string wrapped = "cd " + src_dir + " && " + compile_cmd;
  std::vector<std::string> sh_compile = {"/bin/sh", "-c", wrapped};
  if (auto err = RunProcess(sh_compile)) return std::move(err);

  // Step 2: link to .bin
  std::string link_cmd =
      lld + " -m aicorelinux -Ttext=0 " + obj_file + " -static -o " + bin_file;
  std::vector<std::string> sh_link = {"/bin/sh", "-c", link_cmd};
  if (auto err = RunProcess(sh_link)) return std::move(err);

  return bin_file;
}

} // namespace mlir::runtime
