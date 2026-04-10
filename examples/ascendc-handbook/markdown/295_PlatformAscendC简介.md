<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_1026.html -->
<!-- 下载时间: 2026-03-04 11:39:03 -->

# PlatformAscendC简介

# PlatformAscendC简介

在实现Host侧的Tiling函数时，可能需要获取一些硬件平台的信息，来支撑Tiling的计算，比如获取硬件平台的核数等信息。PlatformAscendC类提供获取这些平台信息的功能。

#### 需要包含的头文件

使用该功能需要包含"tiling/platform/platform_ascendc.h"头文件。样例如下：
    
    
    #include "tiling/platform/platform_ascendc.h"
    

#### Public成员函数
    
    
    [PlatformAscendC](atlasascendc_api_07_1027.html)() = delete
    [~PlatformAscendC](atlasascendc_api_07_1027.html)() = default
    explicit [PlatformAscendC](atlasascendc_api_07_1027.html)(fe::PlatFormInfos *platformInfo): platformInfo_(platformInfo) {}
    uint32_t [GetCoreNum](atlasascendc_api_07_1028.html)(void) const
    SocVersion [GetSocVersion](atlasascendc_api_07_1029.html)(void) const
    NpuArch [GetCurNpuArch](atlasascendc_api_07_00199.html)(void)const
    uint32_t [GetCoreNumAic](atlasascendc_api_07_1030.html)(void) const
    uint32_t [GetCoreNumAiv](atlasascendc_api_07_1031.html)(void) const
    uint32_t [GetCoreNumVector](atlasascendc_api_07_1032.html)(void) const
    uint32_t [CalcTschBlockDim](atlasascendc_api_07_1033.html)(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum) const
    void [GetCoreMemSize](atlasascendc_api_07_1034.html)(const CoreMemType &memType, uint64_t &size) const
    void [GetCoreMemBw](atlasascendc_api_07_1035.html)(const CoreMemType &memType, uint64_t &bwSize) const
    uint32_t [GetLibApiWorkSpaceSize](atlasascendc_api_07_1036.html)(void) const
    uint32_t [GetResGroupBarrierWorkSpaceSize](atlasascendc_api_07_1037.html)(void) const
    uint32_t [GetResCubeGroupWorkSpaceSize](atlasascendc_api_07_1038.html)(void) const

**父主题：** [PlatformAscendC](atlasascendc_api_07_1026.html)
