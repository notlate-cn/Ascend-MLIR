<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0277.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# GetICachePreloadStatus(ISASI)

# GetICachePreloadStatus(ISASI)

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

获取ICACHE的PreLoad的状态。

#### 函数原型
    
    
    __aicore__ inline int64_t GetICachePreloadStatus()
    

#### 参数说明

无

#### 返回值说明

int64_t类型，0表示空闲，1表示忙。

#### 约束说明

无

#### 调用示例
    
    
    // 获取预取状态，0-空闲，1-忙
    int64_t cachePreloadStatus = AscendC::GetICachePreloadStatus();
    

**父主题：** [缓存控制](atlasascendc_api_07_0275.html)
