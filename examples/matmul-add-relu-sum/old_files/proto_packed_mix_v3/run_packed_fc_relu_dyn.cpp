#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <vector>

static std::vector<char> readFile(const char* path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<char>((std::istreambuf_iterator<char>(f)), {});
}

static void trimNpy(std::vector<char>& v) {
  if (v.size() > 128) v.erase(v.begin(), v.begin() + 128);
}

int main() {
  void* rt = dlopen("/home/niu/Ascend/latest/tools/simulator/Ascend910B1/lib/libruntime_camodel.so", RTLD_NOW | RTLD_GLOBAL);
  if (!rt) {
    std::cerr << dlerror() << "\n";
    return 2;
  }

  auto rtSetDevice = (int(*)(int32_t))dlsym(rt, "rtSetDevice");
  auto rtMalloc = (int(*)(void**, uint64_t, uint32_t, uint16_t))dlsym(rt, "rtMalloc");
  auto rtMemcpy = (int(*)(void*, uint64_t, const void*, uint64_t, uint32_t))dlsym(rt, "rtMemcpy");
  auto rtStreamCreate = (int(*)(void**, int32_t))dlsym(rt, "rtStreamCreate");
  auto rtStreamSynchronize = (int(*)(void*))dlsym(rt, "rtStreamSynchronize");
  auto rtStreamDestroy = (int(*)(void*))dlsym(rt, "rtStreamDestroy");
  if (!rtSetDevice || !rtMalloc || !rtMemcpy || !rtStreamCreate || !rtStreamSynchronize || !rtStreamDestroy) {
    std::cerr << "missing runtime symbols\n";
    return 3;
  }

  auto allocAligned = [&](size_t n) {
    void* raw = nullptr;
    if (rtMalloc(&raw, n + 512, 0u, (uint16_t)33u) != 0) return (void*)nullptr;
    return (void*)((((uintptr_t)raw) + 511) & ~511ULL);
  };

  auto A = readFile("/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/runner_probe2/data/input_a.npy");
  auto B = readFile("/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/runner_probe2/data/input_b.npy");
  auto Z = readFile("/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/runner_probe2/data/input_bias.npy");
  auto T = readFile("/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/proto_packed_mix_v3/t.bin");
  trimNpy(A);
  trimNpy(B);
  trimNpy(Z);

  rtSetDevice(0);
  void* stream = nullptr;
  rtStreamCreate(&stream, 0);

  void* a = allocAligned(A.size());
  void* b = allocAligned(B.size());
  void* bias = allocAligned(Z.size());
  void* c = allocAligned(128 * 128 * 4);
  void* ws = allocAligned(16777216);
  void* tiling = allocAligned(T.size());

  rtMemcpy(a, A.size(), A.data(), A.size(), 1);
  rtMemcpy(b, B.size(), B.data(), B.size(), 1);
  rtMemcpy(bias, Z.size(), Z.data(), Z.size(), 1);
  rtMemcpy(tiling, T.size(), T.data(), T.size(), 1);

  void* lib = dlopen("/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/proto_packed_mix_v2/out/libfc_relu_packed.so", RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    std::cerr << dlerror() << "\n";
    return 4;
  }

  using Fn = uint32_t(*)(uint32_t, void*, void*, void*, void*, void*, void*, void*);
  auto fn = (Fn)dlsym(lib, "aclrtlaunch_fc_relu");
  if (!fn) {
    std::cerr << dlerror() << "\n";
    return 5;
  }

  uint32_t rc = fn(1, stream, a, b, bias, c, ws, tiling);
  std::cout << "launch_rc=" << rc << "\n";
  rtStreamSynchronize(stream);

  float out[4] = {};
  rtMemcpy(out, sizeof(out), c, sizeof(out), 2);
  std::cout << out[0] << " " << out[1] << " " << out[2] << " " << out[3] << "\n";

  rtStreamDestroy(stream);
  return 0;
}
