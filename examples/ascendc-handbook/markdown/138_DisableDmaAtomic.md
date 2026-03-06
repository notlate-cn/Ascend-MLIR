<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0212.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# DisableDmaAtomic

# DisableDmaAtomic

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | √  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | √  
  
#### 功能说明

清空原子操作的状态。

#### 函数原型
    
    
    __aicore__ inline void DisableDmaAtomic()
    

#### 参数说明

无

#### 返回值说明

无

#### 约束说明

无

#### 调用示例
    
    
    constexpr uint32_t totalLength = 256;    // 参与搬运的元素个数
    AscendC::LocalTensor<float> src0Local;
    AscendC::GlobalTensor<float> dstGlobal;
    AscendC::SetAtomicAdd<float>();    // 开启原子累加，累加数据类型为float
    AscendC::DataCopy(dstGlobal, src0Local, totalLength * sizeof(float));
    AscendC::DisableDmaAtomic();    // 清空原子操作的状态
    

**父主题：** [原子操作](atlasascendc_api_07_0209.html)
