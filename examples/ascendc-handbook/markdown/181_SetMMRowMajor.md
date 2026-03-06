<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0260.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# SetMMRowMajor

# SetMMRowMajor

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

设置Mmad计算时优先通过N方向，CUBE将首先通过N方向，然后通过M方向生成结果。

#### 函数原型
    
    
    __aicore__ inline void SetMMRowMajor()
    

#### 参数说明

无

#### 返回值说明

无

#### 约束说明

无

#### 调用示例
    
    
    AscendC::SetMMRowMajor();// 设置Mmad优先计算输出矩阵的N方向，再计算M方向。
    

**父主题：** [矩阵计算](atlasascendc_api_07_00172.html)
