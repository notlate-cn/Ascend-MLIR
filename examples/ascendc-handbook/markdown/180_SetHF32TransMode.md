<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0259.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# SetHF32TransMode

# SetHF32TransMode

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

设置HF32模式取整的具体方式，需要先使用[SetHF32Mode](atlasascendc_api_07_0258.html)开启HF32取整模式。

#### 函数原型
    
    
    __aicore__ inline void SetHF32TransMode(HF32TransMode mode)
    

#### 参数说明

表1 参数说明

展开

参数名 | 输入/输出 | 描述  
---|---|---  
mode | 输入 | Mmad HF32取整模式控制入参，HF32TransMode类型。支持如下两种取值：

  * NEAREST_ZERO：则FP32将以向零靠近的方式四舍五入为HF32。
  * NEAREST_EVEN：则FP32将以最接近偶数的方式四舍五入为HF32。

  
  
#### 返回值说明

无

#### 约束说明

无

#### 调用示例
    
    
    AscendC::SetHF32TransMode(HF32TransMode::NEAREST_ZERO);  
    

**父主题：** [矩阵计算](atlasascendc_api_07_00172.html)
