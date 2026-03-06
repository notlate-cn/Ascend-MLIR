<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0255.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# SetFixPipeClipRelu

# SetFixPipeClipRelu

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | x  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

[DataCopy](atlasascendc_api_07_00130.html)（CO1->GM）过程中进行随路量化后，通过调用该接口设置ClipRelu操作的最大值。

ClipRelu计算公式为min(clipReluMaxVal，srcData)，clipReluMaxVal为通过该接口设置的最大值，srcData为源数据。

#### 函数原型
    
    
    __aicore__ inline void SetFixPipeClipRelu(uint64_t config)
    

#### 参数说明

表1 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
config | 输入 | clipReluMaxVal，ClipRelu操作中的最大值。clipReluMaxVal只占用0-15bit，必须大于0，不能为INF/NAN。  
  
#### 约束说明

使能Relu的情况下，先进行Relu操作，之后再进行ClipRelu。

#### 返回值说明

无

#### 调用示例

完整示例可参考[完整示例](atlasascendc_api_07_00130.html#ZH-CN_TOPIC_0000002521039734__li152921471718)。
    
    
    // 使能Relu的情况下，先进行Relu操作，之后再进行ClipRelu。value 1, half类型转换成uint64_t类型
    uint64_t clipReluMaxVal = 0x3c00;
    SetFixPipeClipRelu(clipReluMaxVal);
    

**父主题：** [数据搬运](atlasascendc_api_07_00171.html)
