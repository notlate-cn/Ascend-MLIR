<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0248.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# SetLoadDataPaddingValue

# SetLoadDataPaddingValue

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | √  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

用于调用[Load3Dv1接口](atlasascendc_api_07_00170.html#ZH-CN_TOPIC_0000002552079741__li1135744819417)/[Load3Dv2接口](atlasascendc_api_07_00170.html#ZH-CN_TOPIC_0000002552079741__li83241850104315)时设置Pad填充的数值。Load3Dv1/Load3Dv2的模板参数isSetPadding设置为true时，用户需要通过本接口设置Pad填充的数值，设置为false时，本接口设置的填充值不生效。

#### 函数原型
    
    
    template <typename T>
    __aicore__ inline void SetLoadDataPaddingValue(const T padValue)
    

#### 参数说明

表1 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
padValue | 输入 | Pad填充值的数值。 Atlas 推理系列产品AI Core，支持的数据类型为：int8_t/uint8_t/half/int16_t/uint16_t Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型：int8_t/uint8_t/half/int16_t/uint16_t/bfloat16_t/int32_t/uint32_t/float Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型：int8_t/uint8_t/half/int16_t/uint16_t/bfloat16_t/int32_t/uint32_t/float Atlas 200I/500 A2 推理产品， 支持数据类型：int8_t/uint8_t/half/int16_t/uint16_t/bfloat16_t/int32_t/uint32_t/float  
  
#### 约束说明

无

#### 调用示例

参考[调用示例](atlasascendc_api_07_0245.html#ZH-CN_TOPIC_0000002552079805__section642mcpsimp)

**父主题：** [数据搬运](atlasascendc_api_07_00171.html)
