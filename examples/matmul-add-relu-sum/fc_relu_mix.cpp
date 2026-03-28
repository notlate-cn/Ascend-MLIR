// examples/matmul-add-relu-sum/fc_relu_mix.cpp
// Mix kernel: fc_relu(A[M,K], B[K,N], bias[M,N]) → relu(A×B + bias)
// AIC: high-level Matmul<> API; AIV: Add+ReLU per-tile.
#include "kernel_operator.h"

using namespace AscendC;

// -----------------------------------------------------------------------
// Tiling copy helper (used by both AIC and AIV sides)
// -----------------------------------------------------------------------
__aicore__ inline void CopyTiling(TCubeTiling *dst, GM_ADDR src) {
    auto *p  = reinterpret_cast<uint32_t *>(dst);
    auto *gm = reinterpret_cast<__gm__ uint32_t *>(src);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); ++i)
        p[i] = gm[i];
}

// -----------------------------------------------------------------------
// AIC side: Matmul<GM,GM,GM> + CrossCoreSetFlag
// -----------------------------------------------------------------------
#ifdef __DAV_CUBE__
#include "lib/matmul_intf.h"
using namespace matmul;

class FcAicKernel {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                const TCubeTiling &tiling, TPipe *pipe) {
        tiling_ = tiling;
        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a),
                                 tiling.M * tiling.Ka);
        bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b),
                                 tiling.Kb * tiling.N);
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c),
                                 tiling.M * tiling.N);
    }
    __aicore__ inline void Process() {
        matmulObj_.SetTensorA(aGlobal_);
        matmulObj_.SetTensorB(bGlobal_);
        matmulObj_.template IterateAll(cGlobal_);
        matmulObj_.End();
        CrossCoreSetFlag<0x2, PIPE_FIX>(3);
    }

    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, float>,
           MatmulType<TPosition::GM, CubeFormat::ND, float>,
           MatmulType<TPosition::GM, CubeFormat::ND, float>>
        matmulObj_;
    GlobalTensor<float> aGlobal_, bGlobal_, cGlobal_;
    TCubeTiling tiling_;
};

extern "C" __global__ __aicore__ void auto_gen_fc_relu_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR bias,
    GM_ADDR out, GM_ADDR workspace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);
    FcAicKernel kernel;
    kernel.Init(a, b, out, tiling, &pipe);
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), kernel.matmulObj_, &tiling);
    kernel.Process();
}
#endif  // __DAV_CUBE__

// -----------------------------------------------------------------------
// AIV side: DataCopy C from GM, Add bias, ReLU, DataCopy to out GM
// -----------------------------------------------------------------------
#ifdef __DAV_VEC__

class FcAivKernel {
public:
    __aicore__ inline void Init(GM_ADDR c, GM_ADDR bias, GM_ADDR out,
                                const TCubeTiling &tiling, TPipe *pipe) {
        M_     = tiling.M;
        N_     = tiling.N;
        baseM_ = tiling.baseM;
        int64_t bufBytes = (int64_t)baseM_ * N_ * sizeof(float);
        pipe->InitBuffer(vecinQ_,  1, bufBytes);
        pipe->InitBuffer(biasQ_,   1, bufBytes);
        pipe->InitBuffer(zeroQ_,   1, bufBytes);
        pipe->InitBuffer(vecoutQ_, 1, bufBytes);
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c),    M_ * N_);
        biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bias), M_ * N_);
        outGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out), M_ * N_);
    }

    __aicore__ inline void Process() {
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(3);

        uint32_t count = (uint32_t)(baseM_ * N_);
        for (int32_t row = 0; row < M_; row += baseM_) {
            uint32_t offset = (uint32_t)(row * N_);

            LocalTensor<float> cLocal    = vecinQ_.AllocTensor<float>();
            LocalTensor<float> biasLocal = biasQ_.AllocTensor<float>();
            DataCopy(cLocal,    cGlobal_[offset],    count);
            DataCopy(biasLocal, biasGlobal_[offset], count);
            vecinQ_.EnQue(cLocal);
            biasQ_.EnQue(biasLocal);

            LocalTensor<float> cv = vecinQ_.DeQue<float>();
            LocalTensor<float> bv = biasQ_.DeQue<float>();
            Add(cv, cv, bv, count);
            vecinQ_.FreeTensor(bv);

            LocalTensor<float> zv = zeroQ_.AllocTensor<float>();
            Duplicate(zv, (float)0.0f, count);
            zeroQ_.EnQue(zv);
            LocalTensor<float> zeroT = zeroQ_.DeQue<float>();

            LocalTensor<float> outLocal = vecoutQ_.AllocTensor<float>();
            Max(outLocal, cv, zeroT, count);
            vecoutQ_.EnQue(outLocal);
            vecinQ_.FreeTensor(cv);
            zeroQ_.FreeTensor(zeroT);

            LocalTensor<float> ov = vecoutQ_.DeQue<float>();
            DataCopy(outGlobal_[offset], ov, count);
            vecoutQ_.FreeTensor(ov);
        }
    }

    GlobalTensor<float> cGlobal_, biasGlobal_, outGlobal_;
    TQue<TPosition::VECIN,  1> vecinQ_;
    TQue<TPosition::VECIN,  1> biasQ_;
    TQue<TPosition::VECCALC,1> zeroQ_;
    TQue<TPosition::VECOUT, 1> vecoutQ_;
    int32_t M_, N_, baseM_;
};

extern "C" __global__ __aicore__ void auto_gen_fc_relu_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR bias,
    GM_ADDR out, GM_ADDR workspace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);
    FcAivKernel kernel;
    kernel.Init(out, bias, out, tiling, &pipe);
    kernel.Process();
}
#endif  // __DAV_VEC__
