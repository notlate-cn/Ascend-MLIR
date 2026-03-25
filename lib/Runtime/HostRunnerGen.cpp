// lib/Runtime/HostRunnerGen.cpp
#include "Runtime/HostRunnerGen.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <fstream>

namespace mlir::runtime {

static uint32_t magicForType(const std::string& kernel_type) {
  if (kernel_type == "cube") return 0x41494343u;
  return 0x41415246u; // "vec" and "mix" both use AIVEC
}

static std::string escapeCppStr(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (c == '\\') out += "\\\\";
    else if (c == '"') out += "\\\"";
    else out += c;
  }
  return out;
}

// Returns the complete runner.cpp source as a string.
static std::string emitRunnerCpp(const HostRunnerGen::Config& cfg) {
  int n_out = cfg.num_outputs;

  // Build tiling layout push_back lines
  std::string layout_init;
  for (size_t i = 0; i < cfg.tiling_layout.size(); ++i)
    layout_init += "    tiling_layout.push_back(\"" +
                   escapeCppStr(cfg.tiling_layout[i]) + "\");\n";

  char magic_buf[32];
  std::snprintf(magic_buf, sizeof(magic_buf), "0x%08Xu",
                magicForType(cfg.kernel_type));

  // Build per-output compiled-in dtype defaults
  // dtype string → (npy descr, dtypeBytes, dtype enum int)
  // enum: 0=f16, 1=bf16, 2=f32, 3=i8, 4=i32, 5=i64
  auto dtypeEnum = [](const std::string& s) -> int {
    if (s == "f16")  return 0;
    if (s == "bf16") return 1;
    if (s == "f32")  return 2;
    if (s == "i8")   return 3;
    if (s == "i32")  return 4;
    if (s == "i64")  return 5;
    return 0; // default f16
  };

  // Build compiled-in output dtype defaults string
  std::string output_dtype_defaults;
  for (int i = 0; i < n_out; ++i) {
    std::string dt = (i < (int)cfg.output_dtypes.size())
                         ? cfg.output_dtypes[i]
                         : "f16";
    output_dtype_defaults += "  output_dtypes_default.push_back(" +
                             std::to_string(dtypeEnum(dt)) + "); // " + dt + "\n";
  }

  // Build output arg parsing (--outputN, --output-shapeN, --output-dtypeN)
  std::string output_arg_decls;
  for (int i = 0; i < n_out; ++i) {
    std::string si = std::to_string(i);
    output_arg_decls +=
        "  std::string output_path_" + si + " = \"/dev/null\";\n"
        "  std::string output_shape_str_" + si + ";\n"
        "  int         output_dtype_" + si + " = -1; // -1 = use compiled-in default\n";
  }
  std::string output_arg_parsing;
  for (int i = 0; i < n_out; ++i) {
    std::string si = std::to_string(i);
    output_arg_parsing +=
        "    else if (a == \"--output" + si + "\")       output_path_" + si + "  = next();\n"
        "    else if (a == \"--output-shape" + si + "\") output_shape_str_" + si + " = next();\n"
        "    else if (a == \"--output-dtype" + si + "\") {\n"
        "      std::string ds = next();\n"
        "      if      (ds==\"f16\")  output_dtype_" + si + " = 0;\n"
        "      else if (ds==\"bf16\") output_dtype_" + si + " = 1;\n"
        "      else if (ds==\"f32\")  output_dtype_" + si + " = 2;\n"
        "      else if (ds==\"i8\")   output_dtype_" + si + " = 3;\n"
        "      else if (ds==\"i32\")  output_dtype_" + si + " = 4;\n"
        "      else if (ds==\"i64\")  output_dtype_" + si + " = 5;\n"
        "      else { std::cerr << \"Unknown dtype: \" << ds << \"\\n\"; return 4; }\n"
        "    }\n";
  }

  // Build per-output NDArray alloc
  std::string output_alloc;
  for (int i = 0; i < n_out; ++i) {
    std::string si = std::to_string(i);
    output_alloc +=
        "  NDArray output_" + si + ";\n"
        "  {\n"
        "    int dt = (output_dtype_" + si + " >= 0) ? output_dtype_" + si +
        "               : output_dtypes_default[" + si + "];\n"
        "    output_" + si + ".dtype = dt;\n"
        "    if (!output_shape_str_" + si + ".empty()) {\n"
        "      auto sp = splitComma(output_shape_str_" + si + ");\n"
        "      for (auto& s : sp) if (!s.empty()) output_" + si + ".shape.push_back(std::stoll(s));\n"
        "    } else {\n"
        "      output_" + si + ".shape = inputs[0].shape;\n"
        "    }\n"
        "    output_" + si + ".data = new uint8_t[output_" + si + ".nbytes()]();\n"
        "  }\n";
  }

  // Build free lambda body for outputs
  std::string output_free;
  for (int i = 0; i < n_out; ++i) {
    std::string si = std::to_string(i);
    output_free += "    delete[] (uint8_t*)output_" + si + ".data; output_" + si + ".data = nullptr;\n";
  }

  // Build out_ptrs push and workspace size
  std::string out_ptrs_push;
  for (int i = 0; i < n_out; ++i) {
    std::string si = std::to_string(i);
    out_ptrs_push += "  void* out_ptr_" + si + " = doAlloc(output_" + si + ".nbytes());\n"
                     "  out_ptrs.push_back(out_ptr_" + si + ");\n";
  }
  std::string ws_size = std::to_string(cfg.workspace_size);

  // Build D2H output loop
  std::string d2h_outputs;
  for (int i = 0; i < n_out; ++i) {
    std::string si = std::to_string(i);
    d2h_outputs +=
        "  {\n"
        "    const uint8_t* src_d = (const uint8_t*)out_ptr_" + si + ";\n"
        "    uint8_t* dst_d = (uint8_t*)output_" + si + ".data;\n"
        "    size_t nb = output_" + si + ".nbytes();\n"
        "    for (size_t off = 0; off < nb; off += 4) {\n"
        "      uint8_t buf[4] = {};\n"
        "      rtMemcpy(buf, 4, src_d + off, 4, 2);\n"
        "      size_t chunk = std::min<size_t>(4, nb - off);\n"
        "      std::memcpy(dst_d + off, buf, chunk);\n"
        "    }\n"
        "  }\n";
  }

  // Build saveNpy calls
  std::string save_outputs;
  for (int i = 0; i < n_out; ++i) {
    std::string si = std::to_string(i);
    save_outputs += "  saveNpy(output_path_" + si + ", output_" + si + ");\n";
  }

  // clang-format off
  return std::string(R"cpp(
// Auto-generated by HostRunnerGen. DO NOT EDIT.
// Compile: g++ -O2 -std=c++17 runner.cpp -ldl -o runner
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

// ── dtype encoding: 0=f16 1=bf16 2=f32 3=i8 4=i32 5=i64 ─────────────────────

static size_t dtypeBytes(int dt) {
  switch (dt) {
    case 0: return 2; // f16
    case 1: return 2; // bf16
    case 2: return 4; // f32
    case 3: return 1; // i8
    case 4: return 4; // i32
    case 5: return 8; // i64
  }
  return 2;
}

// ── Minimal .npy I/O ──────────────────────────────────────────────────────────

struct NDArray {
  void*                data  = nullptr;
  std::vector<int64_t> shape;
  int                  dtype = 0;
  size_t numElements() const {
    size_t n = 1;
    for (auto s : shape) n *= (size_t)s;
    return n;
  }
  size_t nbytes() const { return numElements() * dtypeBytes(dtype); }
};

static bool loadNpy(const std::string& path, NDArray& arr) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { std::cerr << "Cannot open: " << path << "\n"; return false; }
  char magic[6]; f.read(magic, 6);
  if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
    std::cerr << "Not a .npy file: " << path << "\n"; return false;
  }
  uint8_t major, minor;
  f.read((char*)&major, 1); f.read((char*)&minor, 1);
  uint32_t hlen = 0;
  if (major == 1) { uint16_t h; f.read((char*)&h, 2); hlen = h; }
  else { f.read((char*)&hlen, 4); }
  std::string header(hlen, '\0');
  f.read(header.data(), hlen);
  std::regex re_shape(R"('shape'\s*:\s*\(([^)]*)\))");
  std::smatch m;
  if (!std::regex_search(header, m, re_shape)) {
    std::cerr << "Cannot parse shape in: " << path << "\n"; return false;
  }
  std::istringstream ss(m[1].str()); std::string tok;
  while (std::getline(ss, tok, ',')) {
    tok.erase(0, tok.find_first_not_of(" \t"));
    tok.erase(tok.find_last_not_of(" \t,") + 1);
    if (!tok.empty()) arr.shape.push_back(std::stoll(tok));
  }
  std::regex re_dtype(R"('descr'\s*:\s*'([^']+)')");
  if (!std::regex_search(header, m, re_dtype)) {
    std::cerr << "Cannot parse dtype in: " << path << "\n"; return false;
  }
  std::string descr = m[1].str();
  if      (descr=="<f2"||descr=="=f2") arr.dtype = 0;
  else if (descr=="<V2"||descr=="=V2") arr.dtype = 1; // bf16
  else if (descr=="<f4"||descr=="=f4") arr.dtype = 2;
  else if (descr=="|i1"||descr=="<i1") arr.dtype = 3;
  else if (descr=="<i4"||descr=="=i4") arr.dtype = 4;
  else if (descr=="<i8"||descr=="=i8") arr.dtype = 5;
  else { std::cerr << "Unsupported dtype: " << descr << "\n"; return false; }
  arr.data = new uint8_t[arr.nbytes()];
  f.read((char*)arr.data, (std::streamsize)arr.nbytes());
  return true;
}

static bool saveNpy(const std::string& path, const NDArray& arr) {
  if (path == "/dev/null") return true;
  std::ofstream f(path, std::ios::binary);
  if (!f) { std::cerr << "Cannot write: " << path << "\n"; return false; }
  const char* descr = nullptr;
  switch (arr.dtype) {
    case 0: descr = "<f2"; break;
    case 1: descr = "<V2"; break;
    case 2: descr = "<f4"; break;
    case 3: descr = "|i1"; break;
    case 4: descr = "<i4"; break;
    case 5: descr = "<i8"; break;
    default: descr = "<f2"; break;
  }
  std::string shape_str = "(";
  for (size_t i = 0; i < arr.shape.size(); ++i) {
    shape_str += std::to_string(arr.shape[i]);
    if (arr.shape.size()==1 || i+1 < arr.shape.size()) shape_str += ",";
  }
  shape_str += ")";
  std::string dict = "{'descr': '" + std::string(descr) +
                     "', 'fortran_order': False, 'shape': " + shape_str + ", }";
  size_t total = 10 + dict.size() + 1;
  size_t pad   = (64 - total % 64) % 64;
  dict.append(pad, ' '); dict += '\n';
  uint16_t hlen = (uint16_t)dict.size();
  f.write("\x93NUMPY", 6); f.put(1); f.put(0);
  f.write((char*)&hlen, 2); f.write(dict.data(), dict.size());
  f.write((char*)arr.data, arr.nbytes());
  return true;
}

// ── Runtime API ───────────────────────────────────────────────────────────────

struct DevBinary {
  uint32_t magic; uint32_t version; const char* data; uint64_t length;
};

// ── Helpers ───────────────────────────────────────────────────────────────────

static std::vector<std::string> splitComma(const std::string& s) {
  std::vector<std::string> v;
  std::istringstream ss(s); std::string t;
  while (std::getline(ss, t, ',')) v.push_back(t);
  return v;
}

static std::vector<uint8_t> buildTiling(const std::string& params,
                                         const std::vector<std::string>& layout) {
  std::vector<uint8_t> bytes;
  auto pvec = splitComma(params);
  for (size_t i = 0; i < pvec.size(); ++i) {
    auto eq = pvec[i].find('=');
    int64_t val = (eq == std::string::npos)
                      ? std::stoll(pvec[i])
                      : std::stoll(pvec[i].substr(eq + 1));
    std::string type = i < layout.size() ? layout[i] : "int64";
    if (type == "int32" || type == "int32_t") {
      int32_t v = (int32_t)val; uint8_t buf[4];
      std::memcpy(buf, &v, 4); bytes.insert(bytes.end(), buf, buf+4);
    } else {
      uint8_t buf[8]; std::memcpy(buf, &val, 8);
      bytes.insert(bytes.end(), buf, buf+8);
    }
  }
  return bytes;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  std::string bin_path, tiling_params, tiling_layout_str, inputs_str;
  int block_dim = 1;

)cpp") + output_arg_decls + R"cpp(
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i+1 >= argc) { std::cerr << "Missing arg after " << a << "\n"; exit(4); }
      return argv[++i];
    };
    if      (a == "--bin")           bin_path          = next();
    else if (a == "--tiling-params") tiling_params     = next();
    else if (a == "--tiling-layout") tiling_layout_str = next();
    else if (a == "--inputs")        inputs_str        = next();
    else if (a == "--block-dim")     block_dim         = std::stoi(next());
)cpp" + output_arg_parsing + R"cpp(  }
  if (bin_path.empty())  { std::cerr << "--bin required\n";    return 4; }
  if (inputs_str.empty()){ std::cerr << "--inputs required\n"; return 4; }

  // Compiled-in output dtype defaults (can be overridden via --output-dtypeN)
  std::vector<int> output_dtypes_default;
)cpp" + output_dtype_defaults + R"cpp(
  // tiling layout: command-line overrides compiled-in default
  std::vector<std::string> tiling_layout;
  if (!tiling_layout_str.empty())
    tiling_layout = splitComma(tiling_layout_str);
  else {
)cpp" + layout_init + R"cpp(  }

  std::vector<uint8_t> tiling_bytes;
  if (!tiling_params.empty())
    tiling_bytes = buildTiling(tiling_params, tiling_layout);

  // Load inputs
  auto input_paths = splitComma(inputs_str);
  std::vector<NDArray> inputs(input_paths.size());
  for (size_t i = 0; i < input_paths.size(); ++i)
    if (!loadNpy(input_paths[i], inputs[i])) return 4;

  // Allocate outputs
)cpp" + output_alloc + R"cpp(
  auto freeArrays = [&]() {
    for (auto& inp : inputs) { delete[] (uint8_t*)inp.data; inp.data = nullptr; }
)cpp" + output_free + R"cpp(  };

  // dlopen camodel
  const char* home = std::getenv("ASCEND_HOME_PATH");
  std::string home_str = home ? home : "/usr/local/Ascend/ascend-toolkit/latest";
)cpp"
    + "  std::string soc_ver = \"" + escapeCppStr(cfg.soc_version) + "\";\n"
    + R"cpp(  std::string lib_path = home_str + "/aarch64-linux/simulator/" + soc_ver + "/lib/libruntime_camodel.so";
  void* lib = dlopen(lib_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  if (!lib) {
    lib_path = home_str + "/tools/simulator/" + soc_ver + "/lib/libruntime_camodel.so";
    lib = dlopen(lib_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  }
  if (!lib) {
    lib_path = home_str + "/runtime/lib64/libruntime_camodel.so";
    lib = dlopen(lib_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  }
  if (!lib) { std::cerr << "dlopen failed: " << dlerror() << "\n"; freeArrays(); return 3; }

#define LOAD(name, T) \
  auto name = reinterpret_cast<T>(dlsym(lib, #name)); \
  if (!name) { std::cerr << "dlsym " #name " failed\n"; freeArrays(); dlclose(lib); return 3; }
  LOAD(rtSetDevice,         int(*)(int32_t))
  LOAD(rtDevBinaryRegister, int(*)(const DevBinary*, void**))
  LOAD(rtFunctionRegister,  int(*)(void*, void*, const char*, void*, uint32_t))
  LOAD(rtMalloc,            int(*)(void**, uint64_t, uint32_t, uint16_t))
  LOAD(rtFree,              int(*)(void*))
  LOAD(rtMemcpy,            int(*)(void*, uint64_t, const void*, uint64_t, uint32_t))
  LOAD(rtStreamCreate,      int(*)(void**, int32_t))
  LOAD(rtStreamDestroy,     int(*)(void*))
  LOAD(rtKernelLaunch,      int(*)(void*, uint32_t, void*, uint32_t, void*, void*))
  LOAD(rtDeviceSynchronize, int(*)())
#undef LOAD

  if (rtSetDevice(0) != 0) {
    std::cerr << "rtSetDevice failed\n"; freeArrays(); dlclose(lib); return 3;
  }

  // Read binary
  std::ifstream bf(bin_path, std::ios::binary);
  if (!bf) { std::cerr << "Cannot open bin: " << bin_path << "\n"; freeArrays(); dlclose(lib); return 4; }
  std::vector<uint8_t> bin_data((std::istreambuf_iterator<char>(bf)), {});

  // Register binary + function
  DevBinary dev_bin;
)cpp"
    + "  dev_bin.magic   = " + std::string(magic_buf) + ";\n"
    + R"cpp(  dev_bin.version = 0;
  dev_bin.data    = (const char*)bin_data.data();
  dev_bin.length  = bin_data.size();
  void* bin_handle = nullptr;
  if (rtDevBinaryRegister(&dev_bin, &bin_handle) != 0) {
    std::cerr << "rtDevBinaryRegister failed\n"; freeArrays(); dlclose(lib); return 3;
  }
)cpp"
    + "  const char* fn_name = \"" + escapeCppStr(cfg.kernel_name) + "\";\n"
    + R"cpp(  void* fn_ptr = const_cast<char*>(fn_name);
  if (rtFunctionRegister(bin_handle, fn_ptr, fn_name, fn_ptr, 0) != 0) {
    std::cerr << "rtFunctionRegister failed\n"; freeArrays(); dlclose(lib); return 3;
  }

  // Alloc helper: rtMalloc(n+512), align to 512 bytes
  struct AllocRec { void* raw; void* aligned; };
  std::vector<AllocRec> allocs;
  auto doAlloc = [&](size_t n) -> void* {
    void* raw = nullptr;
    rtMalloc(&raw, (uint64_t)(n + 512), 0u, (uint16_t)33u);
    uintptr_t a = ((uintptr_t)raw + 511) & ~511ULL;
    allocs.push_back({raw, (void*)a});
    return (void*)a;
  };
  auto freeAll = [&]() {
    for (auto& r : allocs) rtFree(r.raw);
    allocs.clear();
  };

  // H2D inputs (4096-byte chunks)
  std::vector<void*> in_ptrs, out_ptrs;
  for (auto& inp : inputs) {
    void* p = doAlloc(inp.nbytes());
    in_ptrs.push_back(p);
    const uint8_t* src = (const uint8_t*)inp.data;
    for (size_t off = 0; off < inp.nbytes(); off += 4096) {
      size_t chunk = std::min<size_t>(4096, inp.nbytes() - off);
      rtMemcpy((uint8_t*)p + off, chunk, src + off, chunk, 1);
    }
  }
)cpp" + out_ptrs_push
    + "  void* ws_ptr = doAlloc(" + ws_size + ");\n"
    + R"cpp(
  // Build launch args: [inputs..., outputs..., workspace, tiling_words...]
  std::vector<uint64_t> launch_args;
  for (auto* p : in_ptrs)  launch_args.push_back((uint64_t)p);
  for (auto* p : out_ptrs) launch_args.push_back((uint64_t)p);
  launch_args.push_back((uint64_t)ws_ptr);
  for (size_t i = 0; i < tiling_bytes.size(); i += 8) {
    uint64_t w = 0;
    std::memcpy(&w, tiling_bytes.data() + i,
                std::min<size_t>(8, tiling_bytes.size() - i));
    launch_args.push_back(w);
  }

  void* stream = nullptr;
  rtStreamCreate(&stream, 0);
  int rc = rtKernelLaunch(fn_ptr, (uint32_t)block_dim,
                          launch_args.data(),
                          (uint32_t)(launch_args.size() * 8),
                          nullptr, stream);
  if (rc != 0) {
    std::cerr << "rtKernelLaunch failed: rc=" << rc << "\n";
    rtStreamDestroy(stream); freeAll(); freeArrays(); dlclose(lib); return 3;
  }
  if (rtDeviceSynchronize() != 0) {
    std::cerr << "rtDeviceSynchronize failed\n";
    rtStreamDestroy(stream); freeAll(); freeArrays(); dlclose(lib); return 3;
  }
  rtStreamDestroy(stream);

  // D2H outputs (4-byte chunks)
)cpp" + d2h_outputs + R"cpp(
  freeAll();
)cpp" + save_outputs + R"cpp(  freeArrays();
  dlclose(lib);
  return 0;
}
)cpp";
  // clang-format on
}

llvm::Expected<std::string> HostRunnerGen::Generate(const Config& cfg,
                                                     const std::string& output_dir) {
  // Ensure output dir exists
  if (auto ec = llvm::sys::fs::create_directories(output_dir))
    return llvm::createStringError(ec, "Cannot create output dir: %s",
                                   output_dir.c_str());

  std::string src_path = output_dir + "/runner.cpp";
  std::string exe_path = output_dir + "/runner";

  // Write runner.cpp
  {
    std::ofstream f(src_path);
    if (!f)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Cannot write runner.cpp to: %s",
                                     src_path.c_str());
    f << emitRunnerCpp(cfg);
  }

  if (cfg.verbose)
    llvm::errs() << "[HostRunnerGen] Written: " << src_path << "\n";

  // Compile with g++ via /bin/sh -c
  std::string cmd = "g++ -O2 -std=c++17 \"" + src_path + "\" -ldl -o \"" + exe_path + "\"";
  std::vector<std::string> args = {"/bin/sh", "-c", cmd};
  std::vector<llvm::StringRef> argv;
  argv.reserve(args.size());
  for (auto& a : args) argv.push_back(a);

  std::string err_msg;
  std::optional<llvm::StringRef> redirects[3];
  int ret = llvm::sys::ExecuteAndWait(argv[0], argv, std::nullopt, redirects,
                                      /*secondsToWait=*/300, /*memoryLimit=*/0,
                                      &err_msg);
  if (ret != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "g++ failed (exit %d): %s\n"
                                   "  cmd: %s",
                                   ret, err_msg.c_str(), cmd.c_str());

  if (cfg.verbose)
    llvm::errs() << "[HostRunnerGen] Compiled: " << exe_path << "\n";

  return exe_path;
}

} // namespace mlir::runtime
