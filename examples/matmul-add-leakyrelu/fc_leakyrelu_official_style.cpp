#define __FC_LEAKYRELU_WRAPPERLESS_KERNEL_FUN_H__

#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

__aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM) {
  uint64_t *dst = reinterpret_cast<uint64_t *>(tiling);
  auto tiling64 = reinterpret_cast<__gm__ uint64_t *>(tilingGM);
  for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint64_t); ++i)
    dst[i] = tiling64[i];
}

extern "C" __global__ __aicore__ void fc_leakyrelu(
    GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR out, GM_ADDR workspace,
    GM_ADDR tilingGm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  TPipe pipe;
  (void)workspace;

  TCubeTiling tiling;
  CopyTiling(&tiling, tilingGm);

  if ASCEND_IS_AIC {
    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, half>,
           MatmulType<TPosition::GM, CubeFormat::ND, half>,
           MatmulType<TPosition::VECIN, CubeFormat::ND, float>,
           MatmulType<TPosition::GM, CubeFormat::ND, float>>
        mm;

    GlobalTensor<half> aGM, bGM;
    GlobalTensor<float> cGM, biasGM;
    aGM.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(a),
                        tiling.M * tiling.Ka);
    bGM.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b),
                        tiling.Kb * tiling.N);
    cGM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out),
                        tiling.M * tiling.N);
    biasGM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bias), tiling.N);

    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
    mm.SetTensorA(aGM);
    mm.SetTensorB(bGM);
    mm.SetBias(biasGM);
    mm.template IterateAll(cGM);
    mm.End();
    CrossCoreSetFlag<0x2, PIPE_FIX>(3);
  }

  if ASCEND_IS_AIV {
    TQue<TPosition::VECIN, 1> reluInQueue;
    TQue<TPosition::VECOUT, 1> reluOutQueue;

    uint32_t count = static_cast<uint32_t>(tiling.singleCoreM *
                                           tiling.singleCoreN / 2);
    GlobalTensor<float> cGM;
    cGM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out) +
                            GetBlockIdx() * count,
                        count);

    pipe.InitBuffer(reluInQueue, 1, count * sizeof(float));
    pipe.InitBuffer(reluOutQueue, 1, count * sizeof(float));

    CrossCoreWaitFlag(3);

    LocalTensor<float> reluInLocal = reluInQueue.AllocTensor<float>();
    DataCopy(reluInLocal, cGM, count);
    reluInQueue.EnQue<float>(reluInLocal);

    LocalTensor<float> inLocal = reluInQueue.DeQue<float>();
    LocalTensor<float> outLocal = reluOutQueue.AllocTensor<float>();
    LeakyRelu(outLocal, inLocal, static_cast<float>(0.001f), count);
    reluOutQueue.EnQue<float>(outLocal);
    reluInQueue.FreeTensor(inLocal);

    LocalTensor<float> finalLocal = reluOutQueue.DeQue<float>();
    DataCopy(cGM, finalLocal, count);
    reluOutQueue.FreeTensor(finalLocal);
  }
}
