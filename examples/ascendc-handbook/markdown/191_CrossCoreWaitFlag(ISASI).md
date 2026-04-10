<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0274.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# CrossCoreWaitFlag(ISASI)

# CrossCoreWaitFlag(ISASI)

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

面向分离模式的核间同步控制接口。该接口和[CrossCoreSetFlag](atlasascendc_api_07_0273.html)接口配合使用。具体使用方法请参考[CrossCoreSetFlag](atlasascendc_api_07_0273.html)。

#### 函数原型
    
    
    template <uint8_t modeId = 0, pipe_t pipe = PIPE_S>
    __aicore__ inline void CrossCoreWaitFlag(uint16_t flagId)
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
modeId | 核间同步的模式，取值如下：

  * 模式0：AI Core核间的同步控制。
  * 模式1：AI Core内部，Vector核（AIV）之间的同步控制。
  * 模式2：AI Core内部，Cube核（AIC）与Vector核（AIV）之间的同步控制。

  
pipe | 设置这条指令所在的流水类型，流水类型可参考[硬件流水类型](atlasascendc_api_07_0179.html#ZH-CN_TOPIC_0000002521039978__section1272612276459)。  
  
表2 参数说明

展开

参数名 | 输入/输出 | 描述  
---|---|---  
flagId | 输入 | 核间同步的标记。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，取值范围是0-10。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，取值范围是0-10。  
  
#### 返回值说明

无

#### 约束说明

  * 使用该同步接口时，需要按照如下规则[设置Kernel类型](atlasascendc_api_07_0218.html)：
    * 在纯Vector/Cube场景下，需设置Kernel类型为KERNEL_TYPE_MIX_AIV_1_0或KERNEL_TYPE_MIX_AIC_1_0。
    * 对于Vector和Cube混合场景，需根据实际情况灵活配置Kernel类型。


  * CrossCoreWaitFlag必须与[CrossCoreSetFlag](atlasascendc_api_07_0273.html)接口配合使用，避免计算核一直处于阻塞阶段。
  * 如果执行CrossCoreWaitFlag时该flagId的计数器的值为0，则CrossCoreWaitFlag之后的所有指令都将被阻塞，直到该flagId的计数器的值不为0。同一个flagId的计数器最多设置15次。



#### 调用示例

请参考[调用示例](atlasascendc_api_07_0273.html#ZH-CN_TOPIC_0000002552119769__section837496171220)。

**父主题：** [核间同步](atlasascendc_api_07_0268.html)
