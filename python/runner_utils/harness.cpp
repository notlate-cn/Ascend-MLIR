// runner_utils/harness.cpp – generic network harness for network_runner.py Phase 3.
//
// Reads arbitrary .npy inputs, calls network(), writes .npy outputs.
// The network_host.cpp compiled alongside this file declares network() and
// network_set_dump_dir().
//
// Compile (managed by python/runner_utils/build_host.py):
//   g++ -std=c++17 -O2
//       -I<repo>/include
//       -I<ASCEND_HOME_PATH>/x86_64-linux/include
//       harness.cpp <network_host>.cpp
//       -L<repo>/build/lib -lAscendCRuntime
//       -L<llvm-build>/lib -lLLVMSupport
//       -L<CANN>/x86_64-linux/lib64   -lascendcl
//       -L<CANN>/x86_64-linux/devlib/linux/x86_64  -lascend_hal
//       -Wl,-rpath,...
//       -o network_test_default
//
// CLI:
//   harness --input a.npy [--input b.npy ...]
//           --output out0.npy [--output out1.npy ...]
//           [--dump-intermediates DIR]

#include "Runtime/AclnnOps.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using TensorInfo = mlir::runtime::aclnn::TensorInfo;

// Declared by the generated network_host.cpp compiled alongside this file.
extern "C" void network(TensorInfo inputs[], int numInputs,
                        TensorInfo outputs[], int numOutputs,
                        void *stream);
extern "C" void network_set_dump_dir(const char *dir);
extern "C" void network_set_profile_dir(const char *dir);

// ---------------------------------------------------------------------------
// Minimal .npy v1 reader (little-endian, fortran_order=False, numeric dtypes)
// Dtype encoding matches TensorInfo.dtype (aclDataType values):
//   0 = float32, 1 = float16, 27 = bfloat16, 2 = int8, 3 = int32, 9 = int64
// ---------------------------------------------------------------------------
struct NpyArray {
  std::vector<int64_t> shape;
  int  dtype    = 0;   // aclDataType value
  int  elemBytes = 4;
  std::vector<uint8_t> data;

  size_t numElems() const {
    size_t n = 1;
    for (auto s : shape) n *= (size_t)s;
    return n;
  }
  size_t nbytes() const { return numElems() * (size_t)elemBytes; }
};

// Parse the dtype descriptor string from a .npy header.
// Returns aclDataType id and element byte size.
static bool parseDtype(const std::string &descr, int &dtype, int &elemBytes) {
  // Strip endian prefix: '<', '>', '=', '|'
  std::string d = descr;
  if (!d.empty() && (d[0] == '<' || d[0] == '>' || d[0] == '=' || d[0] == '|'))
    d = d.substr(1);
  if (d == "f2") { dtype = 1;  elemBytes = 2; return true; }  // float16
  if (d == "f4") { dtype = 0;  elemBytes = 4; return true; }  // float32
  if (d == "f8") { dtype = 11; elemBytes = 8; return true; }  // float64 (ACL_DOUBLE)
  if (d == "V2") { dtype = 27; elemBytes = 2; return true; }  // bfloat16
  if (d == "i1" || d == "u1") { dtype = 2;  elemBytes = 1; return true; }  // int8
  if (d == "i4" || d == "u4") { dtype = 3;  elemBytes = 4; return true; }  // int32
  if (d == "i8" || d == "u8") { dtype = 9;  elemBytes = 8; return true; }  // int64
  return false;
}

static NpyArray loadNpy(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    fprintf(stderr, "[harness] Cannot open: %s\n", path.c_str());
    exit(1);
  }

  // Magic: \x93NUMPY
  char magic[6];
  f.read(magic, 6);
  if (magic[0] != '\x93' || std::string(magic+1, 5) != "NUMPY") {
    fprintf(stderr, "[harness] Not a .npy file: %s\n", path.c_str());
    exit(1);
  }

  uint8_t major, minor;
  f.read((char*)&major, 1);
  f.read((char*)&minor, 1);

  uint32_t hlen = 0;
  if (major == 1) {
    uint16_t h;
    f.read((char*)&h, 2);
    hlen = h;
  } else {
    f.read((char*)&hlen, 4);
  }
  std::string header(hlen, '\0');
  f.read(header.data(), hlen);

  NpyArray arr;
  // Parse dtype
  std::regex reDt(R"('descr'\s*:\s*'([^']+)')");
  std::smatch m;
  if (std::regex_search(header, m, reDt)) {
    if (!parseDtype(m[1].str(), arr.dtype, arr.elemBytes)) {
      fprintf(stderr, "[harness] Unsupported dtype '%s' in %s\n",
              m[1].str().c_str(), path.c_str());
      exit(1);
    }
  }
  // Parse shape
  std::regex reSh(R"('shape'\s*:\s*\(([^)]*)\))");
  if (std::regex_search(header, m, reSh)) {
    std::istringstream ss(m[1].str());
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      tok.erase(0, tok.find_first_not_of(" \t"));
      auto last = tok.find_last_not_of(" \t,");
      if (last != std::string::npos) tok = tok.substr(0, last + 1);
      if (!tok.empty()) arr.shape.push_back(std::stoll(tok));
    }
  }

  arr.data.resize(arr.nbytes());
  f.read((char*)arr.data.data(), (std::streamsize)arr.nbytes());
  return arr;
}

// ---------------------------------------------------------------------------
// Save an NDArray as .npy v1.0 (little-endian, fortran_order=False).
// ---------------------------------------------------------------------------
static void saveNpy(const std::string &path, void *data,
                    const int64_t *shape, int rank, int dtype, int elemBytes) {
  // Build dtype descriptor
  std::string descr;
  switch (dtype) {
  case 0:  descr = "<f4"; break;  // float32
  case 1:  descr = "<f2"; break;  // float16
  case 2:  descr = "|i1"; break;  // int8
  case 3:  descr = "<i4"; break;  // int32
  case 9:  descr = "<i8"; break;  // int64
  case 11: descr = "<f8"; break;  // float64
  case 27: descr = "<V2"; break;  // bfloat16
  default: descr = "<f4"; break;
  }

  // Build shape string
  std::string shapeStr = "(";
  for (int i = 0; i < rank; ++i) {
    shapeStr += std::to_string(shape[i]);
    shapeStr += ",";
  }
  shapeStr += ")";

  std::string dictStr = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shapeStr + ", }";
  // Pad to 64-byte alignment (v1 header: 10 bytes fixed + 2 bytes hlen + dictStr + '\n')
  size_t hdrBodyLen = dictStr.size() + 1; // +1 for '\n'
  size_t padTotal = ((10 + 2 + hdrBodyLen + 63) / 64) * 64;
  size_t padding = padTotal - (10 + 2 + hdrBodyLen);
  dictStr.append(padding, ' ');
  dictStr += '\n';

  uint16_t hlen = (uint16_t)dictStr.size();

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    fprintf(stderr, "[harness] Cannot write: %s\n", path.c_str());
    exit(1);
  }
  // Magic + version
  out.write("\x93NUMPY", 6);
  out.put(1); out.put(0);
  out.write((char*)&hlen, 2);
  out.write(dictStr.data(), (std::streamsize)dictStr.size());

  // Data
  size_t nelems = 1;
  for (int i = 0; i < rank; ++i) nelems *= (size_t)shape[i];
  out.write((char*)data, (std::streamsize)(nelems * (size_t)elemBytes));
}

// ---------------------------------------------------------------------------
// Build a TensorInfo from a loaded NpyArray (host buffer).
// ---------------------------------------------------------------------------
static TensorInfo makeTensor(const NpyArray &arr) {
  TensorInfo ti;
  ti.rank  = (int)arr.shape.size();
  ti.dtype = arr.dtype;
  for (int i = 0; i < ti.rank; ++i) ti.shape[i] = arr.shape[i];
  mlir::runtime::aclnn::rowMajorStrides(ti.shape, ti.rank, ti.strides);

  size_t nb = arr.nbytes();
  ti.data = ::operator new(nb);
  std::memcpy(ti.data, arr.data.data(), nb);
  return ti;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  std::vector<std::string> inputPaths;
  std::vector<std::string> outputPaths;
  std::string dumpDir;
  std::string profileDir;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 < argc) return argv[++i];
      fprintf(stderr, "[harness] Missing argument after %s\n", a.c_str());
      exit(1);
    };
    if      (a == "--input")               inputPaths.push_back(next());
    else if (a == "--output")              outputPaths.push_back(next());
    else if (a == "--dump-intermediates")  dumpDir = next();
    else if (a == "--profile-dir")         profileDir = next();
    else {
      fprintf(stderr, "[harness] Unknown argument: %s\n", a.c_str());
      return 1;
    }
  }

  if (inputPaths.empty()) {
    fprintf(stderr, "Usage: harness --input a.npy [--input b.npy ...] "
                    "--output out.npy [--output out1.npy ...] "
                    "[--dump-intermediates DIR] [--profile-dir DIR]\n");
    return 1;
  }

  if (!dumpDir.empty())
    network_set_dump_dir(dumpDir.c_str());
  if (!profileDir.empty())
    network_set_profile_dir(profileDir.c_str());

  // Force aclnn host-mode CPU reference. The mixed-network case (aclnn op
  // sandwiched between AscendC kernels) needs aclnn to compute on host
  // buffers; the real aclnn library may "succeed" on the camodel sim's
  // libascendcl but write into NPU/device memory pointers, which then
  // crash on the next dumpTensorIfEnabled (memcpy of bogus pointer).
  // Setting host-mode here is idempotent with the generated network()'s
  // own aclInit-fallback path.
  if (std::getenv("NETWORK_RUNNER_FORCE_HOST_MODE") ||
      !std::getenv("NETWORK_RUNNER_REAL_ACLNN")) {
    mlir::runtime::aclnn::setHostMode(true);
  }

  // Load inputs
  std::vector<NpyArray> inputArrays;
  inputArrays.reserve(inputPaths.size());
  for (const auto &p : inputPaths)
    inputArrays.push_back(loadNpy(p));

  std::vector<TensorInfo> inputs(inputArrays.size());
  for (size_t i = 0; i < inputArrays.size(); ++i)
    inputs[i] = makeTensor(inputArrays[i]);

  // Allocate output TensorInfos (zero-initialized; network() fills them)
  std::vector<TensorInfo> outputs(outputPaths.empty() ? 1 : outputPaths.size());

  network(inputs.data(), (int)inputs.size(),
          outputs.data(), (int)outputs.size(),
          /*stream=*/nullptr);

  // Save outputs
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (outputs[i].data == nullptr) {
      fprintf(stderr, "[harness] output[%zu].data is null after network()\n", i);
      continue;
    }
    int elemBytes = 4;
    switch (outputs[i].dtype) {
    case 2:                  elemBytes = 1; break;  // int8
    case 1: case 27:         elemBytes = 2; break;  // f16/bf16
    case 9: case 11:         elemBytes = 8; break;  // i64/f64
    default:                 elemBytes = 4; break;  // f32/i32 etc.
    }
    const std::string &outPath = (i < outputPaths.size())
        ? outputPaths[i]
        : ("output_" + std::to_string(i) + ".npy");
    saveNpy(outPath, outputs[i].data,
            outputs[i].shape, outputs[i].rank,
            outputs[i].dtype, elemBytes);
    printf("[harness] wrote output[%zu] → %s\n", i, outPath.c_str());
  }

  // Cleanup inputs
  for (auto &ti : inputs)
    ::operator delete(ti.data);

  // The CANN sim runtime occasionally crashes during global destructor
  // teardown (camodel threads racing aclFinalize). Outputs are already
  // flushed at this point; use _exit to skip destructors and exit cleanly.
  std::fflush(stdout);
  std::fflush(stderr);
  std::_Exit(0);
}
