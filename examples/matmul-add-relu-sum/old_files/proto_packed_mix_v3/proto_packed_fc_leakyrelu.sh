#!/usr/bin/env bash
set -euo pipefail
source /home/niu/code/Ascend-MLIR/examples/env.sh
export LD_LIBRARY_PATH=/home/niu/Ascend/latest/lib64:/home/niu/Ascend/latest/tools/simulator/Ascend910B1/lib:/home/niu/Ascend/20260323_newest/cann-9.0.0/aarch64-linux/lib64:/home/niu/Ascend/20260323_newest/cann-9.0.0/aarch64-linux/devlib:/home/niu/Ascend/20260323_newest/cann-9.0.0/aarch64-linux/lib64/device/lib64:/home/niu/Ascend/20260323_newest/cann-9.0.0/aarch64-linux/devlib/linux/aarch64:/home/niu/Ascend/20260323_newest/cann-9.0.0/aarch64-linux/simulator/dav_2201/lib:${LD_LIBRARY_PATH:-}
ASCEND_HOME=${ASCEND_HOME_PATH:-/home/niu/Ascend/latest}
BISHENG=${ASCEND_HOME}/toolkit/tools/ccec_compiler/bin/bisheng
LLD=${ASCEND_HOME}/aarch64-linux/ccec_compiler/bin/ld.lld
TIKCPP=${ASCEND_HOME}/toolkit/tools/tikcpp
MERGE_MIX=${ASCEND_HOME}/compiler/tikcpp/ascendc_kernel_cmake/legacy_modules/util/merge_mix_obj.sh
PACK_SH=${ASCEND_HOME}/compiler/tikcpp/ascendc_kernel_cmake/legacy_modules/util/ascendc_pack_kernel.sh
PACK_TOOL=${ASCEND_HOME}/bin/ascendc_pack_kernel
ROOT=/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/proto_fc_leakyrelu_packed
SRC=/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/fc_leakyrelu_mix.cpp
KERNEL=fc_leakyrelu
rm -rf "$ROOT"
mkdir -p "$ROOT/auto" "$ROOT/mix_aic" "$ROOT/mix_aiv" "$ROOT/mix_merge" "$ROOT/out"
cat > "$ROOT/auto/auto_gen_${KERNEL}.cpp" <<'CPP'
#include "/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/fc_leakyrelu_mix.cpp"
CPP
(
  cd "$ROOT/auto"
  "$BISHENG" -c -x cce -O3 auto_gen_${KERNEL}.cpp --cce-aicore-arch=dav-c220-cube --cce-aicore-only -std=c++17 --cce-disable-kernel-global-attr-check -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false -I "$TIKCPP/tikcfw" -I "$TIKCPP/tikcfw/impl" -I "$TIKCPP/tikcfw/include" -I "$TIKCPP/tikcfw/interface" -DASCENDC_DUMP=0 -D__NPU_TILING__ -DTILING_KEY_VAR=0 -DHAVE_WORKSPACE -DHAVE_TILING -D__MIX_CORE_MACRO__=1 -Dauto_gen_fc_leakyrelu_kernel=fc_leakyrelu_0_mix_aic -D__DAV_C220_CUBE__ -o "$ROOT/out/${KERNEL}_aic.o"
  "$BISHENG" -c -x cce -O3 auto_gen_${KERNEL}.cpp --cce-aicore-arch=dav-c220-vec --cce-aicore-only -std=c++17 --cce-disable-kernel-global-attr-check -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-dcci-insert-for-scalar=false -I "$TIKCPP/tikcfw" -I "$TIKCPP/tikcfw/impl" -I "$TIKCPP/tikcfw/include" -I "$TIKCPP/tikcfw/interface" -DASCENDC_DUMP=0 -D__NPU_TILING__ -DTILING_KEY_VAR=0 -DHAVE_WORKSPACE -DHAVE_TILING -D__MIX_CORE_MACRO__=1 -Dauto_gen_fc_leakyrelu_kernel=fc_leakyrelu_0_mix_aiv -D__DAV_C220_VEC__ -o "$ROOT/out/${KERNEL}_aiv.o"
)
cp "$ROOT/out/${KERNEL}_aic.o" "$ROOT/mix_aic/device.o"
cp "$ROOT/out/${KERNEL}_aiv.o" "$ROOT/mix_aiv/device.o"
touch "$ROOT/mix_aic/mix_build.flag" "$ROOT/mix_aiv/mix_build.flag"
bash "$MERGE_MIX" -l "$LLD" -o "$ROOT/mix_merge" --aic-dir "$ROOT/mix_aic" --aiv-dir "$ROOT/mix_aiv" --build-type Debug
MIX_LEN=$(stat -c %s "$ROOT/mix_merge/device.o")
cat > "$ROOT/auto/host_stub.cpp" <<CPP
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
static char ascendcErrMsg[1024] = {0};
static void *g_kernel_handle = nullptr;
struct ascend_kernels { uint32_t version; uint32_t type_cnt; uint32_t mix_type; uint32_t mix_len; uint32_t mix_file_len; uint8_t mix_buf[${MIX_LEN}]; } __ascend_kernel_ascend910b1_${KERNEL} __attribute__ ((section (".ascend.kernel.ascend910b1.${KERNEL}"))) = {1,1,0,${MIX_LEN},${MIX_LEN},{0}};
extern "C" {
uint32_t RegisterAscendBinary(const char *fileBuf, size_t fileSize, uint32_t type, void **handle);
uint32_t LaunchAscendKernel(void *handle, const uint64_t key, const uint32_t numBlocks, void **args, uint32_t size, const void *stream);
uint32_t GetAscendCoreSyncAddr(void **addr);
uint32_t AllocAscendMemDevice(void **devMem, uint64_t size);
uint32_t FreeAscendMemDevice(void *devMem);
bool AscendCheckSoCVersion(const char *socVersion, char* errMsg);
}
static void __register_kernels(void) __attribute__((constructor));
void __register_kernels(void) {
  if (!AscendCheckSoCVersion("ascend910b1", ascendcErrMsg)) return;
  RegisterAscendBinary((const char *)__ascend_kernel_ascend910b1_${KERNEL}.mix_buf, __ascend_kernel_ascend910b1_${KERNEL}.mix_file_len, 0, &g_kernel_handle);
}
extern "C" uint32_t aclrtlaunch_${KERNEL}(uint32_t numBlocks, void* stream, void* a, void* b, void* bias, void* c, void* workspace, void* tilingGm) {
  struct {
    alignas(((alignof(void*) + 3) >> 2) << 2) void* ffts_addr;
    alignas(((alignof(void*) + 3) >> 2) << 2) void* a;
    alignas(((alignof(void*) + 3) >> 2) << 2) void* b;
    alignas(((alignof(void*) + 3) >> 2) << 2) void* bias;
    alignas(((alignof(void*) + 3) >> 2) << 2) void* c;
    alignas(((alignof(void*) + 3) >> 2) << 2) void* workspace;
    alignas(((alignof(void*) + 3) >> 2) << 2) void* tilingGm;
    alignas(((alignof(void*) + 3) >> 2) << 2) void* overflow;
  } args;
  AllocAscendMemDevice(&args.overflow, 8);
  if (GetAscendCoreSyncAddr(&args.ffts_addr) != 0) return 1;
  args.a = a; args.b = b; args.bias = bias; args.c = c; args.workspace = workspace; args.tilingGm = tilingGm;
  uint32_t ret = LaunchAscendKernel(g_kernel_handle, 0, numBlocks, (void**)&args, sizeof(args), stream);
  FreeAscendMemDevice(args.overflow);
  return ret;
}
CPP
g++ -fPIC -std=c++17 -c "$ROOT/auto/host_stub.cpp" -o "$ROOT/out/host_stub.o"
bash "$PACK_SH" --pack_tool "$PACK_TOOL" --elf_in "$ROOT/out/host_stub.o" --add_dir "$ROOT/mix_merge"
g++ -fPIC -shared -o "$ROOT/out/lib${KERNEL}_packed.so" "$ROOT/out/host_stub.o" -L"$ASCEND_HOME/lib64" -L"$ASCEND_HOME/tools/simulator/Ascend910B1/lib" "$ASCEND_HOME/lib64/libascendc_runtime.a" -lascend_dump -lc_sec
cat > "$ROOT/host_main.cpp" <<'CPP'
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"
extern "C" uint32_t aclrtlaunch_fc_leakyrelu(uint32_t numBlocks, aclrtStream stream, void* a, void* b, void* bias, void* c, void* workspace, void* tilingGm);
static std::vector<char> readFile(const char* path){ std::ifstream f(path, std::ios::binary); return std::vector<char>((std::istreambuf_iterator<char>(f)), {}); }
#define CHECK_ACL(x) do { int rc = (x); if (rc != 0) { std::cerr << #x << " rc=" << rc << "\n"; return rc; } } while (0)
int main(){
 auto A=readFile("/tmp/fc_leakyrelu_split_data/input_a.npy"); auto B=readFile("/tmp/fc_leakyrelu_split_data/input_b.npy"); auto Bias=readFile("/tmp/fc_leakyrelu_split_data/input_bias.npy"); auto T=readFile("/tmp/fc_leakyrelu_tiling.bin"); if(A.size()>128) A.erase(A.begin(),A.begin()+128); if(B.size()>128) B.erase(B.begin(),B.begin()+128); if(Bias.size()>128) Bias.erase(Bias.begin(),Bias.begin()+128);
 CHECK_ACL(aclInit(nullptr)); CHECK_ACL(aclrtSetDevice(0)); aclrtStream stream=nullptr; CHECK_ACL(aclrtCreateStream(&stream));
 uint8_t *aHost,*bHost,*biasHost,*tilingHost,*outHost,*aDev,*bDev,*biasDev,*tilingDev,*outDev,*workspaceDev;
 CHECK_ACL(aclrtMallocHost((void**)&aHost, A.size())); CHECK_ACL(aclrtMallocHost((void**)&bHost, B.size())); CHECK_ACL(aclrtMallocHost((void**)&biasHost, Bias.size())); CHECK_ACL(aclrtMallocHost((void**)&tilingHost, T.size())); CHECK_ACL(aclrtMallocHost((void**)&outHost, 128*128*4));
 memcpy(aHost,A.data(),A.size()); memcpy(bHost,B.data(),B.size()); memcpy(biasHost,Bias.data(),Bias.size()); memcpy(tilingHost,T.data(),T.size());
 CHECK_ACL(aclrtMalloc((void**)&aDev, A.size(), ACL_MEM_MALLOC_HUGE_FIRST)); CHECK_ACL(aclrtMalloc((void**)&bDev, B.size(), ACL_MEM_MALLOC_HUGE_FIRST)); CHECK_ACL(aclrtMalloc((void**)&biasDev, Bias.size(), ACL_MEM_MALLOC_HUGE_FIRST)); CHECK_ACL(aclrtMalloc((void**)&tilingDev, T.size(), ACL_MEM_MALLOC_HUGE_FIRST)); CHECK_ACL(aclrtMalloc((void**)&outDev, 128*128*4, ACL_MEM_MALLOC_HUGE_FIRST)); CHECK_ACL(aclrtMalloc((void**)&workspaceDev, 16777216, ACL_MEM_MALLOC_HUGE_FIRST));
 CHECK_ACL(aclrtMemcpy(aDev, A.size(), aHost, A.size(), ACL_MEMCPY_HOST_TO_DEVICE)); CHECK_ACL(aclrtMemcpy(bDev, B.size(), bHost, B.size(), ACL_MEMCPY_HOST_TO_DEVICE)); CHECK_ACL(aclrtMemcpy(biasDev, Bias.size(), biasHost, Bias.size(), ACL_MEMCPY_HOST_TO_DEVICE)); CHECK_ACL(aclrtMemcpy(tilingDev, T.size(), tilingHost, T.size(), ACL_MEMCPY_HOST_TO_DEVICE));
 uint32_t rc=aclrtlaunch_fc_leakyrelu(1, stream, aDev, bDev, biasDev, outDev, workspaceDev, tilingDev); std::cout << "launch_rc=" << rc << "\n"; CHECK_ACL(aclrtSynchronizeStream(stream)); int d2h_rc=aclrtMemcpy(outHost,128*128*4,outDev,128*128*4,ACL_MEMCPY_DEVICE_TO_HOST); std::cout << "d2h_rc=" << d2h_rc << "\n"; float* out=(float*)outHost; std::cout << out[0] << " " << out[1] << " " << out[2] << " " << out[3] << "\n"; return 0; }
CPP
c++ -g -pie -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack "$ROOT/host_main.cpp" -o "$ROOT/host_main" -I/home/niu/Ascend/20260323_newest/cann-9.0.0/aarch64-linux/include -L/home/niu/Ascend/latest/lib64 -L/home/niu/Ascend/latest/tools/simulator/Ascend910B1/lib -L/home/niu/Ascend/20260323_newest/cann-9.0.0/aarch64-linux/lib64 "$ROOT/out/lib${KERNEL}_packed.so" -ltiling_api -lregister -lplatform -lascendalog -lunified_dlog -ldl -lruntime_camodel -lnpu_drv -lascendcl -lregister -lplatform -lerror_manager -lprofapi -lge_common_base -lmmpa -lascend_dump -lc_sec -lunified_dlog -ldl
timeout 150 "$ROOT/host_main" > "$ROOT/host.log" 2>&1 || true
echo "=== HOST LOG ==="
tail -n 160 "$ROOT/host.log"
