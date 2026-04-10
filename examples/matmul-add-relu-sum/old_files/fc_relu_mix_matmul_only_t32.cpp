#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

__aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
}

extern "C" __global__ __aicore__ void auto_gen_fc_relu_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR bias,
    GM_ADDR out, GM_ADDR workspace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    TPipe pipe;

    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, float>,
           MatmulType<TPosition::GM, CubeFormat::ND, float>,
           MatmulType<TPosition::GM, CubeFormat::ND, float>> mm;

    REGIST_CUBE_OBJ((&pipe), workspace, mm);

    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);

    int32_t M_  = tiling.M;
    int32_t N_  = tiling.N;
    int32_t Ka_ = tiling.Ka;
    int32_t Kb_ = tiling.Kb;

    GlobalTensor<float> aGM, bGM, cGM;
    aGM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a), M_ * Ka_);
    bGM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b), Kb_ * N_);
    cGM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out), M_ * N_);

    mm.Init(&tiling, &pipe);
    mm.SetTensorA(aGM);
    mm.SetTensorB(bGM);
    mm.template IterateAll(cGM);
    mm.End();
}
