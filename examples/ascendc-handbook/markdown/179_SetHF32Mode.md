<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0258.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# SetHF32Mode

# SetHF32Mode

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

用于设置Mmad计算是否开启HF32模式，开启该模式后L0A/L0B中的FP32数据将在参与Mmad计算之前被舍入为HF32。

#### 函数原型
    
    
    __aicore__ inline void SetHF32Mode(HF32Mode mode)
    

#### 参数说明

表1 参数说明

展开

参数名 | 输入/输出 | 描述  
---|---|---  
mode | 输入 | Mmad HF32模式控制入参，HF32Mode枚举类型。支持如下两种取值：

  * ENABLE：L0A/L0B中的FP32数据将在矩阵乘法之前被舍入为HF32。
  * DISABLE：将执行常规的FP32矩阵乘法。

  
  
#### 返回值说明

无

#### 约束说明

无

#### 调用示例
    
    
    AscendC::SetHF32Mode(HF32Mode::ENABLE); // 控制mmad计算时是否使用HF32精度进行计算。
    

**父主题：** [矩阵计算](atlasascendc_api_07_00172.html)
