#ifndef __BAREMIX_CUSTOM_WRAP__KERNEL_FUN_H__
#define __BAREMIX_CUSTOM_WRAP__KERNEL_FUN_H__

// Minimal device-side wrapper matching sample auto_gen_baremix_custom.cpp.
// Keep the original kernel logic unchanged; only provide the runtime contract
// our toolchain was missing: workspace remap + mix metadata sections.

#undef __global__
#define __global__ inline
#define baremix_custom baremix_custom_origin
#include "/Users/niu/code/samples/operator/ascendc/0_introduction/22_baremix_kernellaunch/BareMixInvocation/baremix_custom.cpp"

#undef baremix_custom
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

extern "C" __global__ __aicore__ void auto_gen_baremix_custom_kernel(
    GM_ADDR a,
    GM_ADDR b,
    GM_ADDR bias,
    GM_ADDR c,
    GM_ADDR workspace,
    GM_ADDR tilingGm) {
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

    baremix_custom_origin(a, b, bias, c, workspace, tilingGm);
}

#if defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__)
static const struct FunLevelMixCoreType baremix_custom_mix_aic_section
    __attribute__((used, section(".ascend.meta.baremix_custom_0_mix_aic"))) = {
        {{F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_MIX_AIC_MAIN},
        {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 2}
    };
#endif

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
static const struct FunLevelMixCoreType baremix_custom_mix_aiv_section
    __attribute__((used, section(".ascend.meta.baremix_custom_0_mix_aiv"))) = {
        {{F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_MIX_AIC_MAIN},
        {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 2}
    };
#endif

#endif