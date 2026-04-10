#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

static __aicore__ inline void CopyTilingFromGM_8B(
    TCubeTiling& dst, const __gm__ uint64_t* src)
{
    uint64_t* d = reinterpret_cast<uint64_t*>(&dst);
    for (int i = 0; i < 25; i++) d[i] = src[i];
}

extern "C" __global__ __aicore__ void fc_relu_kernel(
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
    CopyTilingFromGM_8B(tiling,
        reinterpret_cast<const __gm__ uint64_t*>(tilingGm));

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

    CrossCoreSetFlag<0x2, PIPE_FIX>(EVENT_ID0);

    int32_t baseM_ = tiling.baseM > 0 ? tiling.baseM : M_;

    GlobalTensor<float> biasGM, outGM;
    biasGM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bias), M_ * N_);
    outGM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out), M_ * N_);

    int64_t bufBytes = (int64_t)baseM_ * N_ * sizeof(float);
    TQue<TPosition::VECIN,  1> vecinQ, biasQ;
    TQue<TPosition::VECCALC,1> zeroQ;
    TQue<TPosition::VECOUT, 1> vecoutQ;
    pipe.InitBuffer(vecinQ,  1, bufBytes);
    pipe.InitBuffer(biasQ,   1, bufBytes);
    pipe.InitBuffer(zeroQ,   1, bufBytes);
    pipe.InitBuffer(vecoutQ, 1, bufBytes);

    CrossCoreWaitFlag<0x2, PIPE_MTE2>(EVENT_ID0);

    uint32_t count = (uint32_t)(baseM_ * N_);
    for (int32_t row = 0; row < M_; row += baseM_) {
        uint32_t offset = (uint32_t)(row * N_);

        LocalTensor<float> cLocal    = vecinQ.AllocTensor<float>();
        LocalTensor<float> biasLocal = biasQ.AllocTensor<float>();
        DataCopy(cLocal,    cGM[offset],    count);
        DataCopy(biasLocal, biasGM[offset], count);
        vecinQ.EnQue(cLocal);
        biasQ.EnQue(biasLocal);

        LocalTensor<float> cv = vecinQ.DeQue<float>();
        LocalTensor<float> bv = biasQ.DeQue<float>();
        Add(cv, cv, bv, count);
        vecinQ.FreeTensor(bv);

        LocalTensor<float> zv = zeroQ.AllocTensor<float>();
        Duplicate(zv, (float)0.0f, count);
        zeroQ.EnQue(zv);
        LocalTensor<float> zeroT = zeroQ.DeQue<float>();

        LocalTensor<float> outLocal = vecoutQ.AllocTensor<float>();
        Max(outLocal, cv, zeroT, count);
        vecoutQ.EnQue(outLocal);
        vecinQ.FreeTensor(cv);
        zeroQ.FreeTensor(zeroT);

        LocalTensor<float> ov = vecoutQ.DeQue<float>();
        DataCopy(outGM[offset], ov, count);
        vecoutQ.FreeTensor(ov);
    }
}
