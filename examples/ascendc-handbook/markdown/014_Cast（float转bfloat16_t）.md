<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0021.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# Cast（float转bfloat16_t）

# Cast（float转bfloat16_t）

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | x  
Atlas 推理系列产品AI Core | x  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

float类型标量数据转换成bfloat16_t类型标量数据。

#### 函数原型
    
    
    __aicore__ inline bfloat16_t Cast(const float& fVal)
    

#### 参数说明

表1 接口参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
fVal | 输入 | float类型标量数据。  
  
#### 返回值说明

转换后的bfloat16_t类型标量数据。

#### 约束说明

无

#### 调用示例
    
    
    float m = 3.0f;
    bfloat16_t n = AscendC::Cast(m);  // n = 3.0  bfloat16_t和 float指数和尾数表达不同，可以通过截断进行转换
    

**父主题：** [标量计算](atlasascendc_api_07_0015.html)
