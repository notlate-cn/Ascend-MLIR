// harness.cpp – end-to-end accuracy driver for the aclnn attention path.
//
// Compile (after run.sh generates network_host.cpp):
//   g++ -std=c++17 -O2 \
//       -I${PROJECT_ROOT}/include \
//       -I${ASCEND_HOME_PATH}/x86_64-linux/include \
//       harness.cpp network_host.cpp ${PROJECT_ROOT}/lib/Runtime/AclnnOps.cpp \
//       -L${ASCEND_HOME_PATH}/x86_64-linux/lib64 -lascendcl \
//       -L${ASCEND_HOME_PATH}/x86_64-linux/lib64/stub -lopapi \
//       -laclnn_flash_attention_score \
//       -o test_attention
//
// Usage:
//   source examples/env.sh          # sets SIM_LIB → transparent sim mode
//   ./test_attention [--q q.npy] [--k k.npy] [--v v.npy] \
//                    [--mask mask.npy] [--init init.npy] \
//                    [--expected expected.npy] [--atol 0.1] [--rtol 0.05]
//
// On success: prints "PASS" + max/mean abs-diff.
// On failure: prints "FAIL" + first mismatch.

#include "Runtime/AclnnOps.h"

// CANN runtime headers (acl_rt.h pulled in transitively via AclnnOps.h)
#include "aclnn/acl_meta.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using TensorInfo = mlir::runtime::aclnn::TensorInfo;

// Declared by generated network_host.cpp
extern "C" void network(TensorInfo inputs[], int numInputs,
                         TensorInfo outputs[], int numOutputs,
                         aclrtStream stream);

// ---------------------------------------------------------------------------
// Minimal .npy reader (supports fortran_order=False, numeric types only)
// ---------------------------------------------------------------------------
struct NpyArray {
  std::vector<int64_t> shape;
  int dtype = 0;   // 0=float32, 1=float16
  std::vector<uint8_t> data;

  size_t numElems() const {
    size_t n = 1;
    for (auto s : shape) n *= (size_t)s;
    return n;
  }
  size_t elemBytes() const { return dtype == 1 ? 2 : 4; }
  size_t nbytes() const { return numElems() * elemBytes(); }
};

static NpyArray loadNpy(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "Cannot open: %s\n", path.c_str()); exit(1); }

  char magic[6]; f.read(magic, 6);
  uint8_t major, minor; f.read((char*)&major, 1); f.read((char*)&minor, 1);
  uint32_t hlen = 0;
  if (major == 1) { uint16_t h; f.read((char*)&h, 2); hlen = h; }
  else { f.read((char*)&hlen, 4); }
  std::string header(hlen, '\0');
  f.read(header.data(), hlen);

  NpyArray arr;
  // parse dtype
  std::regex re_dt(R"('descr'\s*:\s*'([^']+)')");
  std::smatch m;
  if (std::regex_search(header, m, re_dt)) {
    std::string descr = m[1].str();
    if (descr == "<f2" || descr == "=f2" || descr == "|f2") arr.dtype = 1;
    else arr.dtype = 0;  // treat everything else as float32
  }
  // parse shape
  std::regex re_sh(R"('shape'\s*:\s*\(([^)]*)\))");
  if (std::regex_search(header, m, re_sh)) {
    std::istringstream ss(m[1].str());
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      tok.erase(0, tok.find_first_not_of(" \t"));
      tok.erase(tok.find_last_not_of(" \t,") + 1);
      if (!tok.empty()) arr.shape.push_back(std::stoll(tok));
    }
  }

  arr.data.resize(arr.nbytes());
  f.read((char*)arr.data.data(), (std::streamsize)arr.nbytes());
  return arr;
}

// ---------------------------------------------------------------------------
// Convert uint16 half to float32
// ---------------------------------------------------------------------------
static float halfToFloat(uint16_t h) {
  uint32_t sign = (h >> 15) & 1;
  uint32_t exp  = (h >> 10) & 0x1f;
  uint32_t frac = h & 0x3ff;
  uint32_t f;
  if (exp == 0)       f = (sign << 31) | (frac << 13);
  else if (exp == 31) f = (sign << 31) | 0x7f800000u | (frac << 13);
  else                f = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
  float v; memcpy(&v, &f, 4); return v;
}

// ---------------------------------------------------------------------------
// Build a TensorInfo from a host NpyArray.
// In device mode: aclrtMalloc + aclrtMemcpy to device.
// In host mode:   malloc + memcpy (stays in host memory).
// ---------------------------------------------------------------------------
static TensorInfo makeTensor(const NpyArray &arr) {
  TensorInfo ti;
  ti.rank  = (int)arr.shape.size();
  ti.dtype = arr.dtype;
  for (int i = 0; i < ti.rank; ++i) ti.shape[i] = arr.shape[i];
  mlir::runtime::aclnn::rowMajorStrides(ti.shape, ti.rank, ti.strides);

  size_t nb = arr.nbytes();
  if (mlir::runtime::aclnn::isHostMode()) {
    ti.data = ::operator new(nb);
    std::memcpy(ti.data, arr.data.data(), nb);
  } else {
    aclrtMalloc(&ti.data, nb, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpy(ti.data, nb, arr.data.data(), nb, ACL_MEMCPY_HOST_TO_DEVICE);
  }
  return ti;
}

// ---------------------------------------------------------------------------
// Compare output tensor against expected host array.
// In host mode the tensor data pointer is already host-accessible.
// ---------------------------------------------------------------------------
static bool compare(TensorInfo out, const NpyArray &expected,
                    double atol, double rtol,
                    double &maxDiff, double &meanDiff) {
  size_t nb = expected.nbytes();
  std::vector<uint8_t> host(nb);
  if (mlir::runtime::aclnn::isHostMode())
    std::memcpy(host.data(), out.data, nb);
  else
    aclrtMemcpy(host.data(), nb, out.data, nb, ACL_MEMCPY_DEVICE_TO_HOST);

  size_t n = expected.numElems();
  maxDiff = 0; meanDiff = 0;
  int failures = 0;

  for (size_t i = 0; i < n; ++i) {
    float actual, ref;
    if (expected.dtype == 1) {
      uint16_t a, e;
      memcpy(&a, host.data()     + i * 2, 2);
      memcpy(&e, expected.data.data() + i * 2, 2);
      actual = halfToFloat(a);
      ref    = halfToFloat(e);
    } else {
      memcpy(&actual, host.data()     + i * 4, 4);
      memcpy(&ref,    expected.data.data() + i * 4, 4);
    }
    double diff = std::abs((double)actual - (double)ref);
    double thresh = atol + rtol * std::abs((double)ref);
    if (diff > thresh && failures < 5) {
      fprintf(stderr, "  mismatch[%zu]: actual=%.6g  expected=%.6g  diff=%.4g\n",
              i, (double)actual, (double)ref, diff);
      failures++;
    }
    if (diff > maxDiff) maxDiff = diff;
    meanDiff += diff;
  }
  meanDiff /= (double)n;
  return failures == 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  std::string qPath, kPath, vPath, maskPath, initPath, expPath;
  double atol = 0.1, rtol = 0.05;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nextArg = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
    if      (a == "--q")        qPath    = nextArg();
    else if (a == "--k")        kPath    = nextArg();
    else if (a == "--v")        vPath    = nextArg();
    else if (a == "--mask")     maskPath = nextArg();
    else if (a == "--init")     initPath = nextArg();
    else if (a == "--expected") expPath  = nextArg();
    else if (a == "--atol")     atol     = std::stod(nextArg());
    else if (a == "--rtol")     rtol     = std::stod(nextArg());
  }

  if (qPath.empty() || expPath.empty()) {
    fprintf(stderr, "Usage: test_attention --q q.npy --k k.npy --v v.npy "
                    "--mask mask.npy --init init.npy --expected expected.npy\n");
    return 1;
  }

  // Initialize ACL; fall back to CPU reference if hardware is unavailable.
  bool hostMode = false;
  aclrtStream stream = nullptr;
  if (aclInit(nullptr) != 0) {
    fprintf(stderr, "[harness] aclInit failed — running CPU reference implementation\n");
    hostMode = true;
    mlir::runtime::aclnn::setHostMode(true);
  } else {
    if (aclrtSetDevice(0) != 0) {
      fprintf(stderr, "[harness] aclrtSetDevice(0) failed — running CPU reference implementation\n");
      hostMode = true;
      mlir::runtime::aclnn::setHostMode(true);
    } else {
      aclrtCreateStream(&stream);
    }
  }

  // Load inputs
  NpyArray qArr    = loadNpy(qPath);
  NpyArray kArr    = loadNpy(kPath);
  NpyArray vArr    = loadNpy(vPath);
  NpyArray maskArr = loadNpy(maskPath);
  NpyArray initArr = loadNpy(initPath);
  NpyArray expArr  = loadNpy(expPath);

  TensorInfo ins[5];
  ins[0] = makeTensor(qArr);
  ins[1] = makeTensor(kArr);
  ins[2] = makeTensor(vArr);
  ins[3] = makeTensor(maskArr);
  ins[4] = makeTensor(initArr);

  TensorInfo outs[1] = {};

  network(ins, 5, outs, 1, stream);
  if (!hostMode) aclrtSynchronizeStream(stream);

  // Compare
  double maxDiff, meanDiff;
  bool ok = compare(outs[0], expArr, atol, rtol, maxDiff, meanDiff);
  printf("%s  max_abs_diff=%.4g  mean_abs_diff=%.4g\n",
         ok ? "PASS" : "FAIL", maxDiff, meanDiff);

  // Cleanup
  if (hostMode) {
    for (auto &ti : ins) ::operator delete(ti.data);
  } else {
    for (auto &ti : ins) aclrtFree(ti.data);
  }
  mlir::runtime::aclnn::freeTensor(&outs[0]);

  if (!hostMode) {
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
  }

  return ok ? 0 : 1;
}