#ifndef __FC_LEAKYRELU_MIX_KERNEL_FUN_H__
#define __FC_LEAKYRELU_MIX_KERNEL_FUN_H__

// Sample-style device wrapper experiment.
// Make the real kernel body inline-first, then wrap it with workspace/metadata.

#undef __global__
#define __global__ inline

#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

__aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM)
{
    // Read tiling in 8-byte chunks to avoid simulator alignment faults seen with
    // 32-bit GM loads in our runtime path.
    uint64_t *dst = reinterpret_cast<uint64_t *>(tiling);
    auto tiling64 = reinterpret_cast<__gm__ uint64_t *>(tilingGM);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint64_t); ++i) {
        dst[i] = tiling64[i];
    }
}

extern "C" __global__ __aicore__ void fc_leakyrelu_mix_origin(
    GM_ADDR a, GM_ADDR b, GM_ADDR bias,
    GM_ADDR out, GM_ADDR workspace, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;

    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);

    if ASCEND_IS_AIC {
        Matmul<MatmulType<TPosition::GM, CubeFormat::ND, half>,
               MatmulType<TPosition::GM, CubeFormat::ND, half>,
               MatmulType<TPosition::VECIN, CubeFormat::ND, float>,
               MatmulType<TPosition::GM, CubeFormat::ND, float>> mm;

        GlobalTensor<half> aGM, bGM;
        GlobalTensor<float> cGM, biasGM;
        aGM.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(a), tiling.M * tiling.Ka);
        bGM.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b), tiling.Kb * tiling.N);
        cGM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out), tiling.M * tiling.N);
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

        uint32_t count = (uint32_t)(tiling.singleCoreM * tiling.singleCoreN / 2);
        GlobalTensor<float> cGM;
        cGM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out) + GetBlockIdx() * count);

        pipe.InitBuffer(reluInQueue, 1, count * sizeof(float));
        pipe.InitBuffer(reluOutQueue, 1, count * sizeof(float));

        CrossCoreWaitFlag(3);

        LocalTensor<float> reluInLocal = reluInQueue.AllocTensor<float>();
        DataCopy(reluInLocal, cGM, count);
        reluInQueue.EnQue<float>(reluInLocal);

        LocalTensor<float> inLocal = reluInQueue.DeQue<float>();
        LocalTensor<float> outLocal = reluOutQueue.AllocTensor<float>();
        LeakyRelu(outLocal, inLocal, (float)0.001f, count);
        reluOutQueue.EnQue<float>(outLocal);
        reluInQueue.FreeTensor(inLocal);

        LocalTensor<float> finalLocal = reluOutQueue.DeQue<float>();
        DataCopy(cGM, finalLocal, count);
        reluOutQueue.FreeTensor(finalLocal);
    }
}

#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

extern "C" __global__ __aicore__ void auto_gen_fc_leakyrelu_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR bias,
    GM_ADDR out, GM_ADDR workspace, GM_ADDR tilingGm)
{
#if defined(HAVE_WORKSPACE)
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
#if defined(HAVE_TILING)
    workspace_param = workspace;
#else
    workspace_param = tilingGm;
#endif
    if (workspace_param == nullptr) {
        return;
    }
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(REGIST_MATMUL_OBJ) || defined(__MIX_CORE_MACRO__)
    if constexpr (g_coreType == AscendC::AIC) {
        matmul::clearWorkspace(workspace_param);
    }
#endif
#if defined(HAVE_TILING)
    workspace = workspace_usr;
#else
    tilingGm = workspace_usr;
#endif
#endif
    fc_leakyrelu_mix_origin(a, b, bias, out, workspace, tilingGm);
}

#if defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__)
static const struct FunLevelMixCoreType fc_leakyrelu_mix_aic_section
    __attribute__((used, section(".ascend.meta.fc_leakyrelu_0_mix_aic"))) = {
        {{F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_MIX_AIC_MAIN},
        {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 2}
    };
#endif

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
static const struct FunLevelMixCoreType fc_leakyrelu_mix_aiv_section
    __attribute__((used, section(".ascend.meta.fc_leakyrelu_0_mix_aiv"))) = {
        {{F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_MIX_AIC_MAIN},
        {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 2}
    };
#endif

#endif