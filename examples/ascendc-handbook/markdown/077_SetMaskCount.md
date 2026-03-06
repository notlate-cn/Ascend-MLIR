<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0094.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# SetMaskCount

# SetMaskCount

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

设置mask模式为Counter模式。该模式下，不需要开发者去感知迭代次数、处理非对齐的尾块等操作，可直接传入计算数据量，实际迭代次数由Vector计算单元自动推断。

#### 函数原型
    
    
    __aicore__ inline void SetMaskCount()
    

#### 参数说明

无

#### 返回值说明

无

#### 约束说明

设置为Counter模式的场景需要在矢量计算使用完之后调用[SetMaskNorm](atlasascendc_api_07_0095.html)将mask模式恢复为Normal模式。

#### 调用示例

请参考[Counter模式调用示例](atlasascendc_api_07_0096.html#ZH-CN_TOPIC_0000002520880642__li4954135522812)。

**父主题：** [掩码操作](atlasascendc_api_07_0093.html)
