#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"

extern "C" uint32_t aclrtlaunch_fc_relu(uint32_t numBlocks, aclrtStream stream,
                                         void* a, void* b, void* bias, void* c,
                                         void* workspace, void* tilingGm);

static std::vector<char> readFile(const char* path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<char>((std::istreambuf_iterator<char>(f)), {});
}

#define CHECK_ACL(x) do { int rc = (x); if (rc != 0) { std::cerr << #x << " rc=" << rc << "\n"; return rc; } } while (0)

int main() {
  auto A = readFile("/tmp/packed_raw_inputs/fc_relu_input_a.bin");
  auto B = readFile("/tmp/packed_raw_inputs/fc_relu_input_b.bin");
  auto Bias = readFile("/tmp/packed_raw_inputs/fc_relu_input_bias.bin");
  auto T = readFile("/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/proto_packed_mix_v3/t.bin");

  CHECK_ACL(aclInit(nullptr));
  CHECK_ACL(aclrtSetDevice(0));
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  uint8_t *aHost, *bHost, *biasHost, *tilingHost, *outHost;
  uint8_t *aDev, *bDev, *biasDev, *tilingDev, *outDev, *workspaceDev;

  CHECK_ACL(aclrtMallocHost((void**)&aHost, A.size()));
  CHECK_ACL(aclrtMallocHost((void**)&bHost, B.size()));
  CHECK_ACL(aclrtMallocHost((void**)&biasHost, Bias.size()));
  CHECK_ACL(aclrtMallocHost((void**)&tilingHost, T.size()));
  CHECK_ACL(aclrtMallocHost((void**)&outHost, 128 * 128 * 4));
  std::memcpy(aHost, A.data(), A.size());
  std::memcpy(bHost, B.data(), B.size());
  std::memcpy(biasHost, Bias.data(), Bias.size());
  std::memcpy(tilingHost, T.data(), T.size());

  CHECK_ACL(aclrtMalloc((void**)&aDev, A.size(), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&bDev, B.size(), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&biasDev, Bias.size(), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&tilingDev, T.size(), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&outDev, 128 * 128 * 4, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&workspaceDev, 16777216, ACL_MEM_MALLOC_HUGE_FIRST));

  CHECK_ACL(aclrtMemcpy(aDev, A.size(), aHost, A.size(), ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(bDev, B.size(), bHost, B.size(), ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(biasDev, Bias.size(), biasHost, Bias.size(), ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(tilingDev, T.size(), tilingHost, T.size(), ACL_MEMCPY_HOST_TO_DEVICE));

  uint32_t rc = aclrtlaunch_fc_relu(1, stream, aDev, bDev, biasDev, outDev, workspaceDev, tilingDev);
  std::cout << "launch_rc=" << rc << "\n";
  CHECK_ACL(aclrtSynchronizeStream(stream));
  int d2h_rc = aclrtMemcpy(outHost, 128 * 128 * 4, outDev, 128 * 128 * 4, ACL_MEMCPY_DEVICE_TO_HOST);
  std::cout << "d2h_rc=" << d2h_rc << "\n";

  float* out = reinterpret_cast<float*>(outHost);
  std::cout << out[0] << " " << out[1] << " " << out[2] << " " << out[3] << "\n";
  std::ofstream dump("/tmp/fc_relu_host_out.bin", std::ios::binary);
  dump.write(reinterpret_cast<const char*>(outHost), 128 * 128 * 4);
  dump.close();

  aclrtFree(aDev); aclrtFree(bDev); aclrtFree(biasDev); aclrtFree(tilingDev); aclrtFree(outDev); aclrtFree(workspaceDev);
  aclrtFreeHost(aHost); aclrtFreeHost(bHost); aclrtFreeHost(biasHost); aclrtFreeHost(tilingHost); aclrtFreeHost(outHost);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  return 0;
}
