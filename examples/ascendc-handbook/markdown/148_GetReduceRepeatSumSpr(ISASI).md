<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0225.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# GetReduceRepeatSumSpr(ISASI)

# GetReduceRepeatSumSpr(ISASI)

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

获取[ReduceSum](atlasascendc_api_07_0078.html#ZH-CN_TOPIC_0000002552079835__li6452183633914)接口（Tensor前n个数据计算接口，n为接口的count参数）的计算结果。

#### 函数原型
    
    
    template <typename T>
    __aicore__ inline T GetReduceRepeatSumSpr()
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
T | ReduceSum指令的数据类型，支持half、float。  
  
#### 返回值说明

ReduceSum接口（Tensor前n个数据计算接口，n为接口的count参数）的计算结果。

#### 约束说明

无。

#### 调用示例
    
    
    AscendC::LocalTensor<float> src;
    AscendC::LocalTensor<float> work;
    AscendC::LocalTensor<float> dst;
    AscendC::ReduceSum(dst, src, work, 128);
    float res = AscendC::GetReduceRepeatSumSpr<float>();
    

**父主题：** [归约计算](atlasascendc_api_07_0075.html)
