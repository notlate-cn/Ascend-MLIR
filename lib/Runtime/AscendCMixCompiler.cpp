// lib/RuntimeMix/AscendCMixCompiler.cpp
#include "Runtime/AscendCMixCompiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include <cstdlib>
#include <fstream>
#include <optional>
#include <vector>

namespace mlir::runtime {

static llvm::Error runProcess(const std::vector<std::string>& args,
                              const std::string& cwd = "") {
  std::vector<llvm::StringRef> argv;
  argv.reserve(args.size());
  for (const auto& a : args)
    argv.push_back(a);

  std::string err_msg;
  std::optional<llvm::StringRef> redirects[3];
  std::vector<std::string> env_storage;
  std::vector<llvm::StringRef> env_refs;
  if (!cwd.empty()) {
    env_storage.push_back("PWD=" + cwd);
    env_refs.push_back(env_storage.back());
  }

  int ret = llvm::sys::ExecuteAndWait(argv[0], argv,
                                      env_refs.empty() ? std::nullopt
                                                       : std::optional<llvm::ArrayRef<llvm::StringRef>>(env_refs),
                                      redirects, 300, 0, &err_msg);
  if (ret != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Process failed (exit %d): %s", ret,
                                   err_msg.c_str());
  return llvm::Error::success();
}

static llvm::Error writeFile(const std::string& path, const std::string& content) {
  std::ofstream os(path);
  if (!os)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot write file: %s", path.c_str());
  os << content;
  return llvm::Error::success();
}

static std::string getAscendHome() {
  const char* home = std::getenv("ASCEND_HOME_PATH");
  if (home)
    return home;
  if (const char* home2 = std::getenv("ASCEND_TOOLKIT_HOME"))
    return home2;
  if (llvm::sys::fs::exists("/home/niu/Ascend/latest"))
    return "/home/niu/Ascend/latest";
  if (llvm::sys::fs::exists("/home/niu/Ascend/ascend-toolkit/latest"))
    return "/home/niu/Ascend/ascend-toolkit/latest";
  if (const char* userHome = std::getenv("HOME")) {
    std::string latest = std::string(userHome) + "/Ascend/latest";
    if (llvm::sys::fs::exists(latest))
      return latest;
    std::string toolkitLatest =
        std::string(userHome) + "/Ascend/ascend-toolkit/latest";
    if (llvm::sys::fs::exists(toolkitLatest))
      return toolkitLatest;
  }
  return "/usr/local/Ascend/ascend-toolkit/latest";
}

static std::string escapeForCxx(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c == '\\' || c == '"')
      out.push_back('\\');
    out.push_back(c);
  }
  return out;
}

static std::string emitDataUtils() {
  return R"cpp(#pragma once
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "acl/acl.h"

#define CHECK_ACL(x) do { aclError __ret = (x); if (__ret != ACL_ERROR_NONE) { \
  std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
} } while (0)

static bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize) {
  struct stat sBuf;
  if (stat(filePath.data(), &sBuf) == -1) return false;
  if (S_ISREG(sBuf.st_mode) == 0) return false;
  std::ifstream file(filePath, std::ios::binary);
  if (!file.is_open()) return false;
  std::filebuf *buf = file.rdbuf();
  size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
  if (size == 0 || size > bufferSize) return false;
  buf->pubseekpos(0, std::ios::in);
  buf->sgetn(static_cast<char *>(buffer), size);
  fileSize = size;
  return true;
}

static bool WriteFile(const std::string &filePath, const void *buffer, size_t size) {
  if (buffer == nullptr) return false;
  int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
  if (fd < 0) return false;
  size_t writeSize = write(fd, buffer, size);
  (void)close(fd);
  return writeSize == size;
}
)cpp";
}

static std::string emitTilingCpp() {
  return R"cpp(#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

using namespace matmul_tiling;

extern "C" void GenerateTiling(const char *socVersion, uint8_t *tilingBuf) {
  int M = 128;
  int N = 128;
  int K = 256;
  optiling::TCubeTiling tilingData;
  auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
  MatmulApiTiling tilingApi(*ascendcPlatform);

  tilingApi.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
  tilingApi.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
  tilingApi.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
  tilingApi.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
  tilingApi.SetOrgShape(M, N, K);
  tilingApi.SetShape(M, N, K);
  tilingApi.SetBias(true);
  tilingApi.SetTraverse(MatrixTraverse::FIRSTM);
  tilingApi.SetFixSplit(128, 128, -1);
  tilingApi.SetBufferSpace(-1, -1, -1);
  (void)tilingApi.GetTiling(tilingData);
  tilingData.SaveToBuffer(tilingBuf, tilingData.GetDataSize());
}
)cpp";
}

static std::string emitMainCpp(const std::string& kernel_name) {
  return "#include \"data_utils.h\"\n"
         "#include \"kernel_tiling/kernel_tiling.h\"\n"
         "#include \"tiling/platform/platform_ascendc.h\"\n"
         "#include \"acl/acl.h\"\n"
         "#include \"aclrtlaunch_" + kernel_name + ".h\"\n"
         "#include <cstdlib>\n"
         "#include <cstring>\n"
         "#include <string>\n\n"
         "extern \"C\" void GenerateTiling(const char *socVersion, uint8_t *tilingBuf);\n\n"
         "int main(int argc, char *argv[]) {\n"
         "  std::string inputDir = \"./input\";\n"
         "  std::string outputFile = \"./output/output.bin\";\n"
         "  for (int i = 1; i < argc; ++i) {\n"
         "    std::string arg = argv[i];\n"
         "    if (arg == \"--input-dir\" && i + 1 < argc) inputDir = argv[++i];\n"
         "    else if (arg == \"--output-file\" && i + 1 < argc) outputFile = argv[++i];\n"
         "  }\n\n"
         "  const char *socVersion = SOC_VERSION;\n"
         "  auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);\n"
         "  size_t aFileSize = 32768 * sizeof(int16_t);\n"
         "  size_t bFileSize = 32768 * sizeof(int16_t);\n"
         "  size_t cFileSize = 16384 * sizeof(float);\n"
         "  size_t biasFileSize = 640 * sizeof(float);\n"
         "  size_t tilingFileSize = sizeof(TCubeTiling);\n"
         "  size_t workspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());\n"
         "  uint8_t *tilingBuf = static_cast<uint8_t *>(malloc(tilingFileSize));\n"
         "  GenerateTiling(socVersion, tilingBuf);\n"
         "  uint32_t blockDim = 1;\n\n"
         "  CHECK_ACL(aclInit(nullptr));\n"
         "  int32_t deviceId = 0;\n"
         "  CHECK_ACL(aclrtSetDevice(deviceId));\n"
         "  aclrtStream stream = nullptr;\n"
         "  CHECK_ACL(aclrtCreateStream(&stream));\n\n"
         "  auto readHostToDevice = [&](const std::string& path, size_t bytes, uint8_t** host, uint8_t** device) {\n"
         "    size_t fileSize = 0;\n"
         "    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(host), bytes));\n"
         "    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(device), bytes, ACL_MEM_MALLOC_HUGE_FIRST));\n"
         "    if (!ReadFile(path, fileSize, *host, bytes)) return false;\n"
         "    CHECK_ACL(aclrtMemcpy(*device, bytes, *host, bytes, ACL_MEMCPY_HOST_TO_DEVICE));\n"
         "    return true;\n"
         "  };\n\n"
         "  uint8_t *inputAHost = nullptr, *inputADevice = nullptr;\n"
         "  uint8_t *inputBHost = nullptr, *inputBDevice = nullptr;\n"
         "  uint8_t *inputBiasHost = nullptr, *inputBiasDevice = nullptr;\n"
         "  uint8_t *outputCHost = nullptr, *outputCDevice = nullptr;\n"
         "  uint8_t *tilingHost = nullptr, *tilingDevice = nullptr;\n"
         "  uint8_t *workspaceDevice = nullptr;\n\n"
         "  if (!readHostToDevice(inputDir + \"/x1_gm.bin\", aFileSize, &inputAHost, &inputADevice)) return 2;\n"
         "  if (!readHostToDevice(inputDir + \"/x2_gm.bin\", bFileSize, &inputBHost, &inputBDevice)) return 2;\n"
         "  if (!readHostToDevice(inputDir + \"/bias.bin\", biasFileSize, &inputBiasHost, &inputBiasDevice)) return 2;\n\n"
         "  CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&outputCHost), cFileSize));\n"
         "  CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&outputCDevice), cFileSize, ACL_MEM_MALLOC_HUGE_FIRST));\n"
         "  CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&tilingHost), tilingFileSize));\n"
         "  CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&tilingDevice), tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST));\n"
         "  CHECK_ACL(aclrtMemcpy(tilingHost, tilingFileSize, tilingBuf, tilingFileSize, ACL_MEMCPY_HOST_TO_HOST));\n"
         "  CHECK_ACL(aclrtMemcpy(tilingDevice, tilingFileSize, tilingHost, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));\n"
         "  CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&workspaceDevice), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));\n\n"
         "  ACLRT_LAUNCH_KERNEL(" + kernel_name + ")(blockDim, stream, inputADevice, inputBDevice, inputBiasDevice, outputCDevice, workspaceDevice, tilingDevice);\n"
         "  CHECK_ACL(aclrtSynchronizeStream(stream));\n"
         "  CHECK_ACL(aclrtMemcpy(outputCHost, cFileSize, outputCDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST));\n\n"
         "  size_t lastSlash = outputFile.find_last_of('/');\n"
         "  if (lastSlash != std::string::npos) {\n"
         "    std::string outDir = outputFile.substr(0, lastSlash);\n"
         "    std::string mkdirCmd = \"mkdir -p \" + outDir;\n"
         "    (void)std::system(mkdirCmd.c_str());\n"
         "  }\n"
         "  if (!WriteFile(outputFile, outputCHost, cFileSize)) return 3;\n\n"
         "  CHECK_ACL(aclrtFree(inputADevice));\n"
         "  CHECK_ACL(aclrtFreeHost(inputAHost));\n"
         "  CHECK_ACL(aclrtFree(inputBDevice));\n"
         "  CHECK_ACL(aclrtFreeHost(inputBHost));\n"
         "  CHECK_ACL(aclrtFree(outputCDevice));\n"
         "  CHECK_ACL(aclrtFreeHost(outputCHost));\n"
         "  CHECK_ACL(aclrtFree(inputBiasDevice));\n"
         "  CHECK_ACL(aclrtFreeHost(inputBiasHost));\n"
         "  CHECK_ACL(aclrtFree(tilingDevice));\n"
         "  CHECK_ACL(aclrtFreeHost(tilingHost));\n"
         "  CHECK_ACL(aclrtFree(workspaceDevice));\n"
         "  CHECK_ACL(aclrtDestroyStream(stream));\n"
         "  CHECK_ACL(aclrtResetDevice(deviceId));\n"
         "  CHECK_ACL(aclFinalize());\n"
         "  free(tilingBuf);\n"
         "  return 0;\n"
         "}\n";
}

static std::string emitCMakeLists(const std::string& repo_root,
                                  const std::string& kernel_src,
                                  const std::string&) {
  return "cmake_minimum_required(VERSION 3.16)\n"
         "project(runtime_mix_generated LANGUAGES CXX)\n\n"
         "set(RUN_MODE \"sim\" CACHE STRING \"sim only\")\n"
         "set(SOC_VERSION \"Ascend910B1\" CACHE STRING \"system on chip type\")\n"
         "set(ASCEND_CANN_PACKAGE_PATH \"" + escapeForCxx(getAscendHome()) +
         "\" CACHE STRING \"ASCEND CANN package installation directory\")\n\n"
         "if(NOT CMAKE_BUILD_TYPE)\n"
         "  set(CMAKE_BUILD_TYPE \"Debug\" CACHE STRING \"Build type Release/Debug (default Debug)\" FORCE)\n"
         "endif()\n"
         "if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)\n"
         "  set(CMAKE_INSTALL_PREFIX \"${CMAKE_CURRENT_LIST_DIR}/out\" CACHE STRING \"path for install()\" FORCE)\n"
         "endif()\n\n"
         "set(SOC_SIM_LIB_DIR \"${ASCEND_CANN_PACKAGE_PATH}/tools/simulator/${SOC_VERSION}/lib\")\n"
         "set(DAV_SIM_LIB_DIR \"${ASCEND_CANN_PACKAGE_PATH}/tools/simulator/dav_3002/lib\")\n\n"
         "set(KERNEL_FILES \"" + escapeForCxx(kernel_src) + "\")\n"
         "include(\"" + escapeForCxx(repo_root) +
         "/examples/baremix-test/cmake/npu_lib.cmake\")\n\n"
         "link_directories(${SOC_SIM_LIB_DIR} ${DAV_SIM_LIB_DIR})\n\n"
         "add_executable(mix_runner\n"
         "  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp\n"
         "  ${CMAKE_CURRENT_SOURCE_DIR}/baremix_custom_tiling.cpp\n"
         ")\n\n"
         "target_compile_options(mix_runner PRIVATE\n"
         "  -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Werror\n"
         ")\n\n"
         "target_compile_definitions(mix_runner PRIVATE\n"
         "  SOC_VERSION=\"${SOC_VERSION}\"\n"
         ")\n\n"
         "target_link_options(mix_runner PRIVATE\n"
         "  -Wl,-rpath-link,${ASCEND_CANN_PACKAGE_PATH}/lib64\n"
         "  -Wl,-rpath-link,${SOC_SIM_LIB_DIR}\n"
         "  -Wl,-rpath-link,${DAV_SIM_LIB_DIR}\n"
         ")\n\n"
         "target_include_directories(mix_runner PRIVATE\n"
         "  ${CMAKE_CURRENT_SOURCE_DIR}\n"
         "  ${CMAKE_CURRENT_BINARY_DIR}\n"
         "  ${CMAKE_INSTALL_PREFIX}/include/ascendc_kernels_sim\n"
         ")\n\n"
         "target_link_libraries(mix_runner PRIVATE\n"
         "  host_intf_pub\n"
         "  ascendc_kernels_sim\n"
          "  tiling_api\n"
          "  register\n"
          "  platform\n"
          "  ascendalog\n"
          "  unified_dlog\n"
         "  dl\n"
         "  runtime_camodel\n"
         "  npu_drv\n"
         "  stars\n"
         "  model_top\n"
         "  ascendcl\n"
         "  tiling_api\n"
         "  register\n"
         "  platform\n"
         "  error_manager\n"
         "  profapi\n"
         "  ge_common_base\n"
         "  mmpa\n"
         "  ascend_dump\n"
         "  c_sec\n"
         "  unified_dlog\n"
         "  dl\n"
         ")\n\n"
         "add_dependencies(mix_runner ascendc_kernels_sim)\n\n"
         "install(TARGETS mix_runner\n"
         "  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}\n"
         ")\n";
}

llvm::Expected<MixArtifact>
AscendCMixCompiler::Compile(const AscendCMixCompileConfig& cfg) {
  if (cfg.kernel_src.empty() || cfg.kernel_name.empty() || cfg.output_dir.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "kernel_src, kernel_name and output_dir are required");

  llvm::SmallString<256> kernel_src_abs(cfg.kernel_src);
  llvm::sys::fs::make_absolute(kernel_src_abs);
  if (!llvm::sys::fs::exists(kernel_src_abs))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Kernel source not found: %s",
                                   kernel_src_abs.c_str());

  llvm::SmallString<256> repo_root = llvm::sys::path::parent_path(
      llvm::sys::path::parent_path(
          llvm::sys::path::parent_path(llvm::StringRef(__FILE__))));
  llvm::SmallString<256> output_dir_abs(cfg.output_dir);
  llvm::sys::fs::make_absolute(output_dir_abs);
  llvm::SmallString<256> work_dir(output_dir_abs);
  llvm::sys::path::append(work_dir, "work");
  llvm::SmallString<256> build_dir(output_dir_abs);
  llvm::sys::path::append(build_dir, "build");
  llvm::SmallString<256> install_dir(output_dir_abs);
  llvm::sys::path::append(install_dir, "out");

  if (auto ec = llvm::sys::fs::create_directories(work_dir))
    return llvm::createStringError(ec, "Cannot create work dir");
  if (auto ec = llvm::sys::fs::create_directories(build_dir))
    return llvm::createStringError(ec, "Cannot create build dir");
  if (auto ec = llvm::sys::fs::create_directories(install_dir))
    return llvm::createStringError(ec, "Cannot create install dir");

  std::string cmakePath = (work_dir + "/CMakeLists.txt").str();
  std::string mainPath = (work_dir + "/main.cpp").str();
  std::string tilingPath = (work_dir + "/baremix_custom_tiling.cpp").str();
  std::string utilsPath = (work_dir + "/data_utils.h").str();
  if (auto err = writeFile(cmakePath, emitCMakeLists(repo_root.str().str(),
                                                     kernel_src_abs.str().str(),
                                                     cfg.kernel_name)))
    return std::move(err);
  if (auto err = writeFile(mainPath, emitMainCpp(cfg.kernel_name)))
    return std::move(err);
  if (auto err = writeFile(tilingPath, emitTilingCpp()))
    return std::move(err);
  if (auto err = writeFile(utilsPath, emitDataUtils()))
    return std::move(err);

  std::string soc = cfg.soc_version.empty() ? "Ascend910B1" : cfg.soc_version;
  std::string ascendHome = getAscendHome();
  std::string envPrefix = llvm::formatv(
      "export ASCEND_HOME_PATH=\"{0}\" ASCEND_TOOLKIT_HOME=\"{0}\" && source \"{0}/bin/setenv.bash\" >/dev/null 2>&1 && ",
      ascendHome).str();
  std::string configureCmd = envPrefix + llvm::formatv(
      "cmake -S \"{0}\" -B \"{1}\" -DSOC_VERSION={2} -DCMAKE_INSTALL_PREFIX=\"{3}\" -DASCEND_CANN_PACKAGE_PATH=\"{4}\"",
      work_dir.str(), build_dir.str(), soc, install_dir.str(), ascendHome).str();
  std::string buildCmd = envPrefix +
      llvm::formatv("cmake --build \"{0}\" --target install -j4", build_dir.str()).str();
  if (auto err = runProcess({"/bin/bash", "-lc", configureCmd}))
    return std::move(err);
  if (auto err = runProcess({"/bin/bash", "-lc", buildCmd}))
    return std::move(err);

  llvm::SmallString<256> kernel_so(install_dir);
  llvm::sys::path::append(kernel_so, "lib", "libascendc_kernels_sim.so");
  llvm::SmallString<256> header_dir(install_dir);
  llvm::sys::path::append(header_dir, "include", "ascendc_kernels_sim");
  llvm::SmallString<256> runner_path(install_dir);
  llvm::sys::path::append(runner_path, "bin", "mix_runner");
  if (!llvm::sys::fs::exists(kernel_so) || !llvm::sys::fs::exists(runner_path))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Expected outputs missing under %s",
                                   install_dir.c_str());

  llvm::SmallString<256> manifest_path(output_dir_abs);
  llvm::sys::path::append(manifest_path, "mix-artifact.txt");
  std::string manifest = llvm::formatv(
      "kernel_name={0}\nsoc_version={1}\nwork_dir={2}\nbuild_dir={3}\ninstall_dir={4}\nkernel_so_path={5}\nlauncher_header_dir={6}\nhost_runner_path={7}\n",
      cfg.kernel_name, soc, work_dir.str(), build_dir.str(), install_dir.str(),
      kernel_so.str(), header_dir.str(), runner_path.str()).str();
  if (auto err = writeFile(manifest_path.str().str(), manifest))
    return std::move(err);

  MixArtifact artifact;
  artifact.kernel_name = cfg.kernel_name;
  artifact.soc_version = soc;
  artifact.work_dir = work_dir.str().str();
  artifact.build_dir = build_dir.str().str();
  artifact.install_dir = install_dir.str().str();
  artifact.kernel_so_path = kernel_so.str().str();
  artifact.launcher_header_dir = header_dir.str().str();
  artifact.host_runner_path = runner_path.str().str();
  artifact.manifest_path = manifest_path.str().str();
  return artifact;
}

} // namespace mlir::runtime
