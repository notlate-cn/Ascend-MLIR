<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0780.html -->
<!-- 下载时间: 2026-03-04 11:39:03 -->

# Silu

# Silu

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

按元素做Silu运算，计算公式如下：

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_formulaimage_0000002552122315.png)

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_formulaimage_0000002552122313.png)

#### 函数原型
    
    
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void Silu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint32_t dataSize)
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
T | 操作数的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half、float。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half、float。 Atlas 推理系列产品AI Core，支持的数据类型为：half、float。  
isReuseSource | 是否允许修改源操作数。该参数预留，传入默认值false即可。  
  
表2 接口参数说明

展开

参数名 | 输入/输出 | 描述  
---|---|---  
dstLocal | 输出 | 目的操作数。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。  
srcLocal | 输入 | 源操作数。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 源操作数的数据类型需要与目的操作数保持一致。  
dataSize | 输入 | 实际计算数据元素个数。  
  
#### 返回值说明

无

#### 约束说明

  * 操作数地址偏移对齐要求请参见[通用说明和约束](atlasascendc_api_07_0004.html)。
  * **不支持源操作数与目的操作数地址重叠。**
  * 当前仅支持ND格式的输入，不支持其他格式。



#### 调用示例
    
    
    #include "kernel_operator.h"
    
    AscendC::LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();
    AscendC::LocalTensor<srcType> srcLocal = inQueueX.DeQue<srcType>();
    AscendC::(dstLocal, srcLocal, dataSize);
    outQueue.EnQue<srcType>(dstLocal);
    inQueueX.FreeTensor(srcLocal);
    

结果示例如下：
    
    
    输入数据(srcLocal):[3.304723 1.04788 ... -1.0512]
    输出数据(dstLocal): [3.185546875 0.77587890625 ... -0.272216796875]
    

**父主题：** [Silu接口](atlasascendc_api_07_0779.html)
