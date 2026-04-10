<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0281.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# GetSubBlockIdx(ISASI)

# GetSubBlockIdx(ISASI)

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

获取AI Core上Vector核的ID。

#### 函数原型
    
    
    __aicore__ inline int64_t GetSubBlockIdx()
    

#### 参数说明

无

#### 返回值说明

返回Vector核ID。

#### 约束说明

无

#### 调用示例
    
    
    int64_t subBlockID = AscendC::GetSubBlockIdx();
    

**父主题：** [系统变量访问](atlasascendc_api_07_0183.html)
