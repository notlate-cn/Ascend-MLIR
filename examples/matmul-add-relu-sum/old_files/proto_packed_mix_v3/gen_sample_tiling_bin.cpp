#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace matmul_tiling;
int main() {
  const char* socVersion = "Ascend910B1";
  int M = 128, N = 128, K = 256;
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
  auto res = tilingApi.GetTiling(tilingData);
  if (res == -1) return 1;
  std::vector<uint8_t> buf(tilingData.GetDataSize());
  tilingData.SaveToBuffer(buf.data(), buf.size());
  std::ofstream out("/tmp/sample_tiling.bin", std::ios::binary);
  out.write(reinterpret_cast<const char*>(buf.data()), buf.size());
  std::cout << buf.size() << "\n";
  return 0;
}
