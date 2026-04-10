<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0276.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# ICachePreLoad(ISASI)

# ICachePreLoad(ISASI)

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

从指令所在DDR地址预加载指令到ICache中。

#### 函数原型
    
    
    __aicore__ inline void ICachePreLoad(const int64_t preFetchLen)
    

#### 参数说明

表1 参数说明

展开

参数名 | 输入/输出 | 描述  
---|---|---  
preFetchLen | 输入 | 预取长度。 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品：preFetchLen参数单位为2K Byte, 取值应小于ICache的大小/2K。AIC和AIV的ICache大小分别为32KB和16KB。 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品：preFetchLen参数单位为2K Byte, 取值应小于ICache的大小/2K。AIC和AIV的ICache大小分别为32KB和16KB。 针对Atlas 推理系列产品AI Core：传入该参数无效，预取长度均为128Byte。  
  
#### 返回值说明

无

#### 约束说明

无

#### 调用示例
    
    
    int64_t preFetchLen = 2; // 预取指令长度
    AscendC::ICachePreLoad(preFetchLen);
    

**父主题：** [缓存控制](atlasascendc_api_07_0275.html)
