<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0018.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# Cast（float转half、int32_t）

# Cast（float转half、int32_t）

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

对标量的数据类型进行转换。

#### 函数原型
    
    
    template <typename T, typename U, RoundMode roundMode>
    __aicore__ inline U Cast(T valueIn)
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
T | valueIn的数据类型，支持float。  
U | 转换后的数据类型，支持half、int32_t。  
roundMode | 精度转换处理模式，类型是RoundMode。 RoundMode为枚举类型，用以控制精度转换处理模式，具体定义为：
    
    
    enum class RoundMode {
        CAST_NONE = 0,  // 在转换有精度损失时表示CAST_RINT模式，不涉及精度损失时表示不取整
        CAST_RINT,      // rint，四舍六入五成双取整
        CAST_FLOOR,     // floor，向负无穷取整
        CAST_CEIL,      // ceil，向正无穷取整
        CAST_ROUND,     // round，四舍五入取整
        CAST_TRUNC,     // trunc，向零取整
        CAST_ODD,       // Von Neumann rounding，最近邻奇数舍入
    };
    

对于Cast，转换类型仅支持float转half(f322f16)与float转int32_t(f322s32)，相应支持的RoundMode如下：

  * f322f16：CAST_ODD；
  * f322s32：CAST_ROUND、CAST_CEIL、CAST_FLOOR、CAST_RINT。

Cast的精度转换规则具体可参考[表1](atlasascendc_api_07_0073.html#ZH-CN_TOPIC_0000002520880724__table235404962912)。  
  
表2 参数说明

展开

参数名 | 输入/输出 | 描述  
---|---|---  
valueIn | 输入 | 被转换数据类型的标量。  
  
#### 返回值说明

U类型的valueIn。

#### 约束说明

无

#### 调用示例
    
    
    float valueIn = 2.5;
    // 输出数据valueOut：3， 2.5向上取整为3
    int32_t valueOut = AscendC::Cast<float, int32_t, AscendC::RoundMode::CAST_ROUND>(valueIn);
    

**父主题：** [标量计算](atlasascendc_api_07_0015.html)
