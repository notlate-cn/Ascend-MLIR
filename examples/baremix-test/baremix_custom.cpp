/**
 * @file baremix_custom.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

/**
  * @brief  Copy tiling data to TCubeTiling ptr from tiling gm addr.
  * @param  tiling: TCubeTiling ptr which needs to copy tiling data.
  * @param  tilingGM: tiling gm addr.
  * @retval None
  */
__aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

template <typename aType, typename bType, typename cType, typename biasType> class MatmulLeakyKernel {
public:
    __aicore__ inline MatmulLeakyKernel(){};
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace,
                                const TCubeTiling &tiling, AscendC::TPipe *pipe);
    __aicore__ inline void Process(AscendC::TPipe *pipe);

    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB,
                                      int32_t &offsetC, int32_t &offsetBias);

    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
           MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, cType>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>>
        matmulObj;

    AscendC::GlobalTensor<aType> aGlobal;
    AscendC::GlobalTensor<bType> bGlobal;
    AscendC::GlobalTensor<cType> cGlobal;
    AscendC::GlobalTensor<biasType> biasGlobal;
    TCubeTiling tiling;
};

/**
  * @brief  Set matmulLeaky input and output gm addr of current core.
  * @param  a: A matrix gm addr.
  * @param  b: B matrix gm addr.
  * @param  bias: Bias gm addr.
  * @param  c: C matrix gm addr.
  * @param  workspace: Temporary gm space addr required by matmul calc.
  * @param  tiling: matmul tiling data.
  * @param  pipe: Global memory and sync management TPipe object.
  * @retval None
  */
template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias,
                                                                              GM_ADDR c, GM_ADDR workspace,
                                                                              const TCubeTiling &tiling, AscendC::TPipe *pipe)
{
    this->tiling = tiling;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling.M * tiling.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling.N);

    int32_t offsetA, offsetB, offsetC, offsetBias;
    CalcOffset(AscendC::GetBlockIdx(), tiling, offsetA, offsetB, offsetC, offsetBias); // Calculate the gm offset based on the blockidx.
    aGlobal = aGlobal[offsetA];
    bGlobal = bGlobal[offsetB];
    cGlobal = cGlobal[offsetC];
    biasGlobal = biasGlobal[offsetBias];
}

/**
  * @brief  Main process of matmul calculation
  * @param  pipe: Global memory and sync management TPipe object.
  * @retval None
  */
template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::Process(AscendC::TPipe *pipe)
{
    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.SetBias(biasGlobal);

    matmulObj.template IterateAll(cGlobal);
    matmulObj.End();
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(3);
}

/**
  * @brief  Calculate the gm offset based on the blockidx.
  * @param  blockIdx: Current Core blockidx.
  * @param  tiling: Matmul tiling data.
  * @param  offsetA: Gm offset of A matrix.
  * @param  offsetB: Gm offset of B matrix.
  * @param  offsetC: Gm offset of C matrix.
  * @param  offsetBias: Gm offset of Bias matrix.
  * @retval None
  */
template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulLeakyKernel<aType, bType, cType, biasType>::CalcOffset(int32_t blockIdx, const TCubeTiling &tiling,
                                                             int32_t &offsetA, int32_t &offsetB, int32_t &offsetC,
                                                             int32_t &offsetBias)
{
    auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
    auto mCoreIndx = blockIdx % mSingleBlocks;
    auto nCoreIndx = blockIdx / mSingleBlocks;

    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    offsetB = nCoreIndx * tiling.singleCoreN;
    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
    offsetBias = nCoreIndx * tiling.singleCoreN;
}

template <typename cType> class LeakyReluKernel {
public:
    __aicore__ inline LeakyReluKernel(){};
    __aicore__ inline void Init(GM_ADDR c, const TCubeTiling &tiling, AscendC::TPipe *pipe);
    __aicore__ inline void Process(AscendC::TPipe *pipe);

    __aicore__ inline void LeakyReluCopyIn(const TCubeTiling &tiling);
    __aicore__ inline void LeakyReluCompute(const TCubeTiling &tiling);
    __aicore__ inline void LeakyReluCopyOut(const TCubeTiling &tiling);

    AscendC::GlobalTensor<cType> cGlobal;

    AscendC::LocalTensor<cType> reluInLocal;
    AscendC::LocalTensor<cType> reluOutLocal;
    TCubeTiling tiling;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> reluInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> reluOutQueue_;
};

/**
  * @brief  Set matmulLeaky input and output gm addr of current core.
  * @param  a: A matrix gm addr.
  * @param  b: B matrix gm addr.
  * @param  bias: Bias gm addr.
  * @param  c: C matrix gm addr.
  * @param  workspace: Temporary gm space addr required by matmul calc.
  * @param  tiling: matmul tiling data.
  * @param  pipe: Global memory and sync management TPipe object.
  * @retval None
  */
template <typename cType>
__aicore__ inline void LeakyReluKernel<cType>::Init(GM_ADDR c, const TCubeTiling &tiling, AscendC::TPipe *pipe)
{
    this->tiling = tiling;
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c) + AscendC::GetBlockIdx() * tiling.M * tiling.N / 2); //c:v = 1:2, split into 2 parts, for vector calculation

    pipe->InitBuffer(reluInQueue_, 1, tiling.singleCoreM * tiling.singleCoreN * sizeof(cType) /2); // Init input buffer.
    pipe->InitBuffer(reluOutQueue_, 1, tiling.singleCoreM * tiling.singleCoreN * sizeof(cType)/2); // Init output buffer.
}

template <typename cType>
__aicore__ inline void LeakyReluKernel<cType>::Process(AscendC::TPipe *pipe)
{
    AscendC::CrossCoreWaitFlag(3);
    LeakyReluCopyIn(tiling);
    LeakyReluCompute(tiling);
    LeakyReluCopyOut(tiling);
}
template <typename cType>
__aicore__ inline void LeakyReluKernel<cType>::LeakyReluCopyIn(const TCubeTiling &tiling)
{
    AscendC::LocalTensor<float> reluInLocal = reluInQueue_.AllocTensor<float>();
    AscendC::DataCopy(reluInLocal, cGlobal, tiling.singleCoreM * tiling.singleCoreN / 2);
    reluInQueue_.EnQue<float>(reluInLocal);
}

template <typename cType>
__aicore__ inline void LeakyReluKernel<cType>::LeakyReluCompute(const TCubeTiling &tiling)
{
    AscendC::LocalTensor<float> reluInLocal = reluInQueue_.DeQue<float>();
    AscendC::LocalTensor<float> reluOutLocal = reluOutQueue_.AllocTensor<float>();
    AscendC::LeakyRelu(reluOutLocal, reluInLocal, (float)0.001, tiling.singleCoreM * tiling.singleCoreN /2);
    reluOutQueue_.EnQue<float>(reluOutLocal);
    reluInQueue_.FreeTensor(reluInLocal);
}

template <typename cType>
__aicore__ inline void LeakyReluKernel<cType>::LeakyReluCopyOut(const TCubeTiling &tiling)
{
    AscendC::LocalTensor<float> reluOutLocal = reluOutQueue_.DeQue<float>();
    AscendC::DataCopy(cGlobal, reluOutLocal, tiling.singleCoreM * tiling.singleCoreN / 2);
    reluOutQueue_.FreeTensor(reluOutLocal);
}

/**
  * @brief  baremix kernel function entry
  * @param  a: A matrix gm addr.
  * @param  b: B matrix gm addr.
  * @param  bias: Bias gm addr.
  * @param  c: Out gm addr.
  * @param  workspace: Temporary gm space addr required by matmul calc.
  * @param  tilingGm: Tiling data addr. 
  * @retval None
  */
extern "C" __global__ __aicore__ void baremix_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                              GM_ADDR workspace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);

    if ASCEND_IS_AIC {
        MatmulLeakyKernel<half, half, float, float> matmulLeakyKernel;
        matmulLeakyKernel.Init(a, b, bias, c, workspace, tiling, &pipe);
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulLeakyKernel.matmulObj, &matmulLeakyKernel.tiling); // Initialize the matmul object.
        matmulLeakyKernel.Process(&pipe);
    }
    if ASCEND_IS_AIV {
        LeakyReluKernel<float> leakyReluKernel;
        leakyReluKernel.Init(c, tiling, &pipe);
        leakyReluKernel.Process(&pipe);
    }
}