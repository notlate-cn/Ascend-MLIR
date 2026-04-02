#include "Runtime/MixStubTemplate.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <sstream>

namespace mlir::runtime {

namespace {

static llvm::Error writeTextFile(llvm::StringRef path, llvm::StringRef content) {
  llvm::SmallString<256> pathBuf(path);
  llvm::sys::path::remove_filename(pathBuf);
  if (!pathBuf.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(pathBuf))
      return llvm::createStringError(ec, "Cannot create directory: %s",
                                     pathBuf.c_str());
  }

  std::ofstream os(path.str(), std::ios::binary);
  if (!os)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Cannot write file: %s", path.str().c_str());
  os << content.str();
  if (!os)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to write file: %s",
                                   path.str().c_str());
  return llvm::Error::success();
}

static std::string headerBasename(llvm::StringRef headerPath) {
  return llvm::sys::path::filename(headerPath).str();
}

static std::string toLowerAscii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

static std::string normalizeSocForSymbol(std::string value) {
  std::replace(value.begin(), value.end(), '-', '_');
  return value;
}

static uint64_t ceil4(uint64_t value) {
  return (value + 3u) & ~uint64_t(3u);
}

static std::string emitMixHeader(const MixStubTemplateArgs &args) {
  std::string guard = "HEADER_ACLRTLAUNCH_" + toLowerAscii(args.kernelName) + "_H";
  std::transform(guard.begin(), guard.end(), guard.begin(),
                 [](unsigned char c) {
                   if (c >= 'a' && c <= 'z')
                     return static_cast<char>(std::toupper(c));
                   if (c == '.')
                     return '_';
                   return static_cast<char>(c);
                 });
  std::ostringstream os;
  os << "#ifndef " << guard << "\n"
     << "#define " << guard << "\n"
     << "#include \"acl/acl_base.h\"\n\n"
     << "#ifndef ACLRT_LAUNCH_KERNEL\n"
     << "#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func\n"
     << "#endif\n\n"
     << "extern \"C\" uint32_t " << args.launcherSymbol
     << "(uint32_t blockDim, aclrtStream stream, void *arg0, void *arg1, "
        "void *arg2, void *arg3, void *workspace, void *tiling);\n";
  os << "#endif\n";
  return os.str();
}

static std::string emitMixHostStub(const MixStubTemplateArgs &args) {
  const std::string socSymbol = normalizeSocForSymbol(toLowerAscii(args.socVersion));
  std::ostringstream os;
  os << "#include \"" << headerBasename(args.launcherHeaderPath) << "\"\n"
     << "#include <algorithm>\n"
     << "#include <cctype>\n"
     << "#include <cstddef>\n"
     << "#include <cstdio>\n"
     << "#include <cstdlib>\n"
     << "#include <fstream>\n"
     << "#include <iostream>\n"
     << "#include <string>\n"
     << "#include <securec.h>\n\n"
     << "#define CHECK_ACL(x) do { aclError __ret = (x); if (__ret != ACL_ERROR_NONE) { \\\n"
     << "  std::cerr << __FILE__ << \":\" << __LINE__ << \" aclError:\" << __ret << std::endl; \\\n"
     << "} } while (0)\n\n"
     << "static char ascendcErrMsg[1024] = {0};\n"
     << "static void *g_kernel_handle = nullptr;\n"
     << "// Packed by " << args.targetName << "\n"
     << "struct ascend_kernels {\n"
     << "  uint32_t version;\n"
     << "  uint32_t type_cnt;\n"
     << "  uint32_t mix_type;\n"
     << "  uint32_t mix_len;\n"
     << "  uint32_t mix_file_len;\n"
     << "  uint8_t mix_buf[" << args.mixLen << "];\n"
     << "} __ascend_kernel_" << socSymbol << "_" << args.targetName
     << " __attribute__((section(\".ascend.kernel." << socSymbol << "."
     << args.targetName << "\"))) = {1, 1, 0, " << args.mixLen << ", "
     << args.mixFileLen << ", {0}};\n\n"
     << "extern \"C\" {\n"
     << "uint32_t RegisterAscendBinary(const char *fileBuf, size_t fileSize, uint32_t type, void **handle);\n"
     << "uint32_t LaunchAscendKernel(void *handle, const uint64_t key, const uint32_t numBlocks, void **args, uint32_t size, const void *stream);\n"
     << "uint32_t GetAscendCoreSyncAddr(void **addr);\n"
     << "int UnregisterAscendBinary(void *hdl);\n"
     << "uint32_t AllocAscendMemDevice(void **devMem, uint64_t size);\n"
     << "uint32_t FreeAscendMemDevice(void *devMem);\n"
     << "bool AscendCheckSoCVersion(const char *socVersion, char *errMsg);\n"
     << "void AscendProfRegister();\n"
     << "bool GetAscendProfStatus();\n"
     << "uint32_t GetCoreNumForMixVectorCore(uint32_t *aiCoreNum, uint32_t *vectorCoreNum);\n"
     << "uint32_t LaunchAscendKernelForVectorCore(const char *opType, void *handle, const uint64_t key, void **args, uint32_t size,\n"
     << "    const void *stream, bool enableProf, uint32_t aicNumBlocks, uint32_t aivNumBlocks, uint32_t aivNumBlocksOffset);\n"
     << "}\n\n"
     << "class KernelHandleGradUnregister {\n"
     << "private:\n"
     << "  KernelHandleGradUnregister() = default;\n"
     << "public:\n"
     << "  KernelHandleGradUnregister(const KernelHandleGradUnregister&) = delete;\n"
     << "  KernelHandleGradUnregister& operator=(const KernelHandleGradUnregister&) = delete;\n"
     << "  static KernelHandleGradUnregister& GetInstance() {\n"
     << "    static KernelHandleGradUnregister instance;\n"
     << "    return instance;\n"
     << "  }\n"
     << "  ~KernelHandleGradUnregister() {\n"
     << "    if (g_kernel_handle) {\n"
     << "      UnregisterAscendBinary(g_kernel_handle);\n"
     << "      g_kernel_handle = nullptr;\n"
     << "    }\n"
     << "  }\n"
     << "};\n\n"
     << "static void __register_kernels(void) __attribute__((constructor));\n"
     << "static void __register_kernels(void) {\n"
     << "  if (!AscendCheckSoCVersion(\"" << socSymbol
     << "\", ascendcErrMsg)) {\n"
     << "    (void)strcpy_s(ascendcErrMsg, sizeof(ascendcErrMsg), \"AscendCheckSoCVersion failed\");\n"
     << "    return;\n"
     << "  }\n"
     << "  uint32_t ret = RegisterAscendBinary(reinterpret_cast<const char *>(__ascend_kernel_"
     << socSymbol << "_" << args.targetName << ".mix_buf), __ascend_kernel_"
     << socSymbol << "_" << args.targetName
     << ".mix_file_len, 0, &g_kernel_handle);\n"
     << "  if (ret != 0) {\n"
     << "    (void)snprintf(ascendcErrMsg, sizeof(ascendcErrMsg), \"RegisterAscendBinary failed: %u\", ret);\n"
     << "    g_kernel_handle = nullptr;\n"
     << "  }\n"
     << "  AscendProfRegister();\n"
     << "}\n\n"
     << "extern \"C\" uint32_t " << args.launcherSymbol
     << "(uint32_t numBlocks, aclrtStream stream, void *arg0, void *arg1, "
        "void *arg2, void *arg3, void *workspace, void *tilingGm) {\n"
     << "  struct {\n"
     << "    alignas(((alignof(void *) + 3) >> 2) << 2) void *ffts_addr;\n"
     << "    alignas(((alignof(void *) + 3) >> 2) << 2) void *arg0;\n"
     << "    alignas(((alignof(void *) + 3) >> 2) << 2) void *arg1;\n"
     << "    alignas(((alignof(void *) + 3) >> 2) << 2) void *arg2;\n"
     << "    alignas(((alignof(void *) + 3) >> 2) << 2) void *arg3;\n"
     << "    alignas(((alignof(void *) + 3) >> 2) << 2) void *workspace;\n"
     << "    alignas(((alignof(void *) + 3) >> 2) << 2) void *tilingGm;\n"
     << "    alignas(((alignof(void *) + 3) >> 2) << 2) void *overflow;\n"
     << "  } args;\n"
     << "  if (g_kernel_handle == nullptr) {\n"
     << "    printf(\"[ERROR] %s\\n\", ascendcErrMsg);\n"
     << "    return 1;\n"
     << "  }\n"
     << "  uint32_t ret = AllocAscendMemDevice(&args.overflow, 8);\n"
     << "  if (ret != 0) {\n"
     << "    printf(\"AllocAscendMemDevice ret %u\\n\", ret);\n"
     << "    return ret;\n"
     << "  }\n"
     << "  ret = GetAscendCoreSyncAddr(&args.ffts_addr);\n"
     << "  if (ret != 0) {\n"
     << "    printf(\"GetAscendCoreSyncAddr ret %u\\n\", ret);\n"
     << "    FreeAscendMemDevice(args.overflow);\n"
     << "    return ret;\n"
     << "  }\n"
     << "  args.arg0 = arg0;\n"
     << "  args.arg1 = arg1;\n"
     << "  args.arg2 = arg2;\n"
     << "  args.arg3 = arg3;\n"
     << "  args.workspace = workspace;\n"
     << "  args.tilingGm = tilingGm;\n"
     << "  uint32_t aicNumBlocks = 0;\n"
     << "  uint32_t aivNumBlocks = 0;\n"
     << "  ret = GetCoreNumForMixVectorCore(&aicNumBlocks, &aivNumBlocks);\n"
     << "  if (ret != 0) {\n"
     << "    printf(\"GetCoreNumForMixVectorCore ret %u\\n\", ret);\n"
     << "    FreeAscendMemDevice(args.overflow);\n"
     << "    return ret;\n"
     << "  }\n"
     << (args.aivOnly
             ? std::string("  {\n"
                           "    const bool profStatus = GetAscendProfStatus();\n"
                           "    ret = LaunchAscendKernelForVectorCore(\"") +
                   args.kernelName +
                   std::string("\", g_kernel_handle, 0,\n"
                               "        reinterpret_cast<void **>(&args), sizeof(args), stream, profStatus,\n"
                               "        0, numBlocks, 0);\n"
                               "  }\n")
             : std::string("  if (numBlocks <= aicNumBlocks) {\n"
                           "    ret = LaunchAscendKernel(g_kernel_handle, 0, numBlocks, reinterpret_cast<void **>(&args), sizeof(args), stream);\n"
                           "  } else {\n"
                           "    uint32_t totalCoreNum = aicNumBlocks + aivNumBlocks;\n"
                           "    if (numBlocks > totalCoreNum && totalCoreNum != 0) {\n"
                           "      aicNumBlocks = (numBlocks * aicNumBlocks + totalCoreNum - 1U) / totalCoreNum;\n"
                           "    }\n"
                           "    aivNumBlocks = numBlocks - aicNumBlocks;\n"
                           "    const bool profStatus = GetAscendProfStatus();\n"
                           "    const uint32_t aivNumBlocksOffset = aicNumBlocks;\n"
                           "    ret = LaunchAscendKernelForVectorCore(\"") +
                   args.kernelName +
                   std::string("\", g_kernel_handle, 0,\n"
                               "        reinterpret_cast<void **>(&args), sizeof(args), stream, profStatus,\n"
                               "        aicNumBlocks, aivNumBlocks, aivNumBlocksOffset);\n"
                               "  }\n"))
     << "  KernelHandleGradUnregister::GetInstance();\n"
     << "  if (ret != 0) {\n"
     << "    printf(\"LaunchAscendKernel ret %u\\n\", ret);\n"
     << "  }\n"
     << "  FreeAscendMemDevice(args.overflow);\n"
     << "  return ret;\n"
     << "}\n";
  return os.str();
}

} // namespace

llvm::Error writeMixStubTemplate(const MixStubTemplateArgs &args) {
  if (args.kernelName.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "MixStubTemplateArgs.kernelName is required");
  if (args.targetName.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "MixStubTemplateArgs.targetName is required");
  if (args.socVersion.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "MixStubTemplateArgs.socVersion is required");
  if (args.launcherSymbol.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "MixStubTemplateArgs.launcherSymbol is required");
  if (args.launcherHeaderPath.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "MixStubTemplateArgs.launcherHeaderPath is required");
  if (args.hostStubSourcePath.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "MixStubTemplateArgs.hostStubSourcePath is required");
  if (args.mixLen == 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "MixStubTemplateArgs.mixLen must be greater than zero");
  if (args.mixFileLen == 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "MixStubTemplateArgs.mixFileLen must be greater than zero");

  if (auto err = writeTextFile(args.launcherHeaderPath, emitMixHeader(args)))
    return err;
  if (auto err = writeTextFile(args.hostStubSourcePath, emitMixHostStub(args)))
    return err;

  return llvm::Error::success();
}

} // namespace mlir::runtime
