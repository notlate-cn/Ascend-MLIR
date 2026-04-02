#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;
using namespace AscendC;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
}

template <typename aType, typename bType, typename cType>
class MatmulReluKernel {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                const TCubeTiling &tiling);
    __aicore__ inline void Process(AscendC::TPipe *pipe);
    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling,
                                      int32_t &offsetA, int32_t &offsetB,
                                      int32_t &offsetC);

    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
           MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, cType>> matmulObj;

    AscendC::GlobalTensor<aType> aGlobal;
    AscendC::GlobalTensor<bType> bGlobal;
    AscendC::GlobalTensor<cType> cGlobal;
    TCubeTiling tiling;
};

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulReluKernel<aType, bType, cType>::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, const TCubeTiling &tiling)
{
    this->tiling = tiling;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling.M * tiling.N);

    int32_t offsetA, offsetB, offsetC;
    CalcOffset(AscendC::GetBlockIdx(), tiling, offsetA, offsetB, offsetC);
    aGlobal = aGlobal[offsetA];
    bGlobal = bGlobal[offsetB];
    cGlobal = cGlobal[offsetC];
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulReluKernel<aType, bType, cType>::Process(AscendC::TPipe *pipe)
{
    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.template IterateAll(cGlobal);
    matmulObj.End();
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(3);
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulReluKernel<aType, bType, cType>::CalcOffset(
    int32_t blockIdx, const TCubeTiling &tiling,
    int32_t &offsetA, int32_t &offsetB, int32_t &offsetC)
{
    auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
    auto mCoreIndx = blockIdx % mSingleBlocks;
    auto nCoreIndx = blockIdx / mSingleBlocks;
    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    offsetB = nCoreIndx * tiling.singleCoreN;
    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
}

template <typename cType>
class ReluKernel {
public:
    __aicore__ inline void Init(GM_ADDR c, GM_ADDR bias, const TCubeTiling &tiling, AscendC::TPipe *pipe);
    __aicore__ inline void Process(AscendC::TPipe *pipe);

    AscendC::GlobalTensor<cType> cGlobal;
    AscendC::GlobalTensor<cType> biasGlobal;
    TCubeTiling tiling;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> reluInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> biasQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> reluOutQueue_;
};

template <typename cType>
__aicore__ inline void ReluKernel<cType>::Init(GM_ADDR c, GM_ADDR bias, const TCubeTiling &tiling, AscendC::TPipe *pipe)
{
    this->tiling = tiling;
    uint32_t count = tiling.singleCoreM * tiling.singleCoreN / 2;
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c) + AscendC::GetBlockIdx() * count);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(bias) + AscendC::GetBlockIdx() * count);
    pipe->InitBuffer(reluInQueue_, 1, count * sizeof(cType));
    pipe->InitBuffer(biasQueue_, 1, count * sizeof(cType));
    pipe->InitBuffer(reluOutQueue_, 1, count * sizeof(cType));
}

template <typename cType>
__aicore__ inline void ReluKernel<cType>::Process(AscendC::TPipe *pipe)
{
    uint32_t count = tiling.singleCoreM * tiling.singleCoreN / 2;
    AscendC::CrossCoreWaitFlag(3);

    AscendC::LocalTensor<float> reluInLocal = reluInQueue_.AllocTensor<float>();
    AscendC::LocalTensor<float> biasLocal = biasQueue_.AllocTensor<float>();
    AscendC::DataCopy(reluInLocal, cGlobal, count);
    AscendC::DataCopy(biasLocal, biasGlobal, count);
    reluInQueue_.EnQue<float>(reluInLocal);
    biasQueue_.EnQue<float>(biasLocal);

    AscendC::LocalTensor<float> inLocal = reluInQueue_.DeQue<float>();
    AscendC::LocalTensor<float> bLocal = biasQueue_.DeQue<float>();
    AscendC::Add(inLocal, inLocal, bLocal, count);
    biasQueue_.FreeTensor(bLocal);

    AscendC::LocalTensor<float> outLocal = reluOutQueue_.AllocTensor<float>();
    AscendC::Relu(outLocal, inLocal, count);
    reluOutQueue_.EnQue<float>(outLocal);
    reluInQueue_.FreeTensor(inLocal);

    AscendC::LocalTensor<float> finalLocal = reluOutQueue_.DeQue<float>();
    AscendC::DataCopy(cGlobal, finalLocal, count);
    reluOutQueue_.FreeTensor(finalLocal);
}

extern "C" __global__ __aicore__ void baremix_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                      GM_ADDR workspace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);

    if ASCEND_IS_AIC {
        MatmulReluKernel<half, half, float> kernel;
        kernel.Init(a, b, c, tiling);
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), kernel.matmulObj, &kernel.tiling);
        kernel.Process(&pipe);
    }
    if ASCEND_IS_AIV {
        ReluKernel<float> kernel;
        kernel.Init(c, bias, tiling, &pipe);
        kernel.Process(&pipe);
    }
}
