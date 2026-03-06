<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0254.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# SetFixpipePreQuantFlag

# SetFixpipePreQuantFlag

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | x  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

[DataCopy](atlasascendc_api_07_00130.html)（CO1->GM、CO1->A1）过程中进行随路量化时，通过调用该接口设置量化流程中标量量化参数。

#### 函数原型
    
    
    template<template T>
    __aicore__ inline void SetFixpipePreQuantFlag(uint64_t config)
    

#### 参数说明

表1 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
config | 输入 | 量化过程中使用到的标量量化参数。  
  
#### 返回值说明

无

#### 约束说明

无

#### 调用示例

完整示例可参考[完整示例](atlasascendc_api_07_00130.html#ZH-CN_TOPIC_0000002521039734__li178441955134010)。
    
    
    float tmp = (float)0.5;
    // 将float的tmp转换成uint64_t的deqScalar
    uint64_t deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp)); 
    AscendC::SetFixpipePreQuantFlag(deqScalar);  // 设置量化参数
    AscendC::PipeBarrier<PIPE_FIX>();
    

**父主题：** [数据搬运](atlasascendc_api_07_00171.html)
