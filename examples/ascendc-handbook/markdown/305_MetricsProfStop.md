<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_1215.html -->
<!-- 下载时间: 2026-03-04 11:39:03 -->

# MetricsProfStop

# MetricsProfStop

#### 产品支持情况

展开

产品 |  是否支持  
---|---  
Atlas A3 训练系列产品 / Atlas A3 推理系列产品  |  √  
Atlas A2 训练系列产品 / Atlas A2 推理系列产品  |  √  
Atlas 200I/500 A2 推理产品  |  x  
Atlas 推理系列产品 AI Core |  x  
Atlas 推理系列产品 Vector Core |  x  
Atlas 训练系列产品  |  x  
  
#### 功能说明

设置性能数据采集信号停止，和MetricsProfStart配合使用。使用[msProf](/document/detail/zh/CANNCommunityEdition/900beta1/devaids/optool/atlasopdev_16_0081.html)工具进行算子上板调优时，可在kernel侧代码段前后分别调用MetricsProfStart和MetricsProfStop来指定需要调优的代码段范围。

#### 函数原型
    
    
    __aicore__ inline void MetricsProfStop()
    

#### 参数说明

无

#### 返回值说明

无

#### 约束说明

无

#### 调用示例
    
    
    MetricsProfStop();
    

**父主题：** [性能统计](atlasascendc_api_07_00180.html)
