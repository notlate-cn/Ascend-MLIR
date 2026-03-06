<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0016.html -->
<!-- 下载时间: 2026-03-04 11:39:01 -->

# GetBitCount

# GetBitCount

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

获取一个uint64_t类型数字的二进制中0或者1的个数。

#### 函数原型
    
    
    template <int countValue> 
    __aicore__ inline int64_t GetBitCount(uint64_t valueIn)
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
countValue | 指定统计0还是统计1的个数。 只能输入0或1。  
  
表2 参数说明

展开

参数名 | 输入/输出 | 描述  
---|---|---  
valueIn | 输入 | 被统计的二进制数字。  
  
#### 返回值说明

valueIn中0或者1的个数。

#### 约束说明

无

#### 调用示例
    
    
    uint64_t valueIn = 0xffff;    // 二进制格式中有16个1
    constexpr int countValue = 1;    // 统计valueIn二进制格式中1的个数
    // 输出数据oneCount: 16
    int64_t oneCount = AscendC::GetBitCount<countValue>(valueIn);
    

**父主题：** [标量计算](atlasascendc_api_07_0015.html)
