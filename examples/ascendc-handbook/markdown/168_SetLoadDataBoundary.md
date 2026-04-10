<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0246.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# SetLoadDataBoundary

# SetLoadDataBoundary

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | x  
Atlas 推理系列产品AI Core | √  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

设置[Load3D](atlasascendc_api_07_0238.html)时A1/B1边界值。

如果Load3D指令在处理源操作数时，源操作数在A1/B1上的地址超出设置的边界，则会从A1/B1起始地址开始读取数据。

#### 函数原型
    
    
    __aicore__ inline void SetLoadDataBoundary(uint32_t boundaryValue)
    

#### 参数说明

表1 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
boundaryValue | 输入 | 边界值。 Load3Dv1指令：单位是32字节。 Load3Dv2指令：单位是字节。  
  
#### 约束说明

  * 用于Load3Dv1时，boundaryValue的最小值是16（单位：32字节）；用于Load3Dv2时，boundaryValue的最小值是1024（单位：字节）。
  * 如果使用SetLoadDataBoundary接口设置了边界值，配合Load3D指令使用时，Load3D指令的A1/B1初始地址要在设置的边界内。
  * 如果boundaryValue设置为0，则表示无边界，可使用整个A1/B1。
  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。



#### 调用示例

参考[调用示例](atlasascendc_api_07_0245.html#ZH-CN_TOPIC_0000002552079805__section642mcpsimp)。

**父主题：** [数据搬运](atlasascendc_api_07_00171.html)
