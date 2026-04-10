<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0783.html -->
<!-- 下载时间: 2026-03-04 11:39:03 -->

# Swish

# Swish

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

在神经网络中，Swish是一个重要的激活函数。计算公式如下，其中β为常数：

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_formulaimage_0000002521040800.png)

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_formulaimage_0000002521040798.png)

#### 函数原型
    
    
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void Swish(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint32_t dataSize, const T scalarValue)
    

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
srcLocal | 输入 | 源操作数。 源操作数的数据类型需要与目的操作数保持一致。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。  
dataSize | 输入 | 实际计算数据元素个数。  
scalarValue | 输入 | 激活函数中的β参数。支持的数据类型为：half、float。 β参数的数据类型需要与源操作数和目的操作数保持一致。  
  
#### 返回值说明

无

#### 约束说明

  * 操作数地址偏移对齐要求请参见[通用说明和约束](atlasascendc_api_07_0004.html)。
  * **不支持源操作数与目的操作数地址重叠。**
  * 当前仅支持ND格式的输入，不支持其他格式。



#### 调用示例
    
    
    #include "kernel_operator.h"
    
    AscendC::LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
    AscendC::LocalTensor<T> srcLocal = inQueueX.DeQue<T>();
    AscendC::Swish(dstLocal, srcLocal, dataSize, scalarValue);
    outQueue.EnQue<T>(dstLocal);
    inQueueX.FreeTensor(srcLocal);
    

结果示例如下：
    
    
    输入数据(srcLocal): 
    [ 0.5312  -3.654   -2.92     3.787   -3.059    3.77     0.571   -0.668
     -0.09534  0.5454  -1.801   -1.791    1.563    0.878    3.973    1.799
      2.023    1.018    3.082   -3.814    2.254   -3.717    0.4675  -0.4631
     -2.47     0.9814  -0.854    3.31     3.256    3.764    1.867   -1.773]
    输出数据(dstLocal): 
    [ 0.3784   -0.007263 -0.02016   3.78     -0.01666   3.762     0.414
     -0.1622   -0.04382   0.3909   -0.0803   -0.08105   1.461     0.717
      3.969     1.719     1.96      0.8647    3.066    -0.00577   2.207
     -0.006626  0.3223   -0.1448   -0.03622   0.8257   -0.1617    3.297
      3.244     3.756     1.792    -0.0825]
    

**父主题：** [Swish接口](atlasascendc_api_07_0782.html)
