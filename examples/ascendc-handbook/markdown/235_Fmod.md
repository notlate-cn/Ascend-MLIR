<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0608.html -->
<!-- 下载时间: 2026-03-04 11:39:03 -->

# Fmod

# Fmod

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

按元素计算两个浮点数a, b相除后的余数。计算公式如下：

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_formulaimage_0000002520880882.png)

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_formulaimage_0000002552120861.png)

其中，Trunc为向零取整操作。举例如下：

Fmod(2.0, 1.5) = 0.5

Fmod(-3.0, 1.1) = -0.8

#### 函数原型

  * 通过sharedTmpBuffer入参传入临时空间
    * 源操作数Tensor全部/部分参与计算
          
          template <typename T, bool isReuseSource = false>
          __aicore__ inline void Fmod(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
          

    * 源操作数Tensor全部参与计算
          
          template <typename T, bool isReuseSource = false>
          __aicore__ inline void Fmod(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer)
          



  * 接口框架申请临时空间
    * 源操作数Tensor全部/部分参与计算
          
          template <typename T, bool isReuseSource = false>
          __aicore__ inline void Fmod(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const LocalTensor<T>& src1Tensor, const uint32_t calCount)
          

    * 源操作数Tensor全部参与计算
          
          template <typename T, bool isReuseSource = false>
          __aicore__ inline void Fmod(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const LocalTensor<T>& src1Tensor)
          




由于该接口的内部实现中涉及精度转换。需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请** 和开发者**通过sharedTmpBuffer入参传入** 两种方式。

  * 接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。


  * 通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。



接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式如下：通过[GetFmodMaxMinTmpSize](atlasascendc_api_07_0609.html)中提供的接口获取需要预留空间的大小。

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
dstTensor | 输出 | 目的操作数。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。  
src0Tensor、src1Tensor | 输入 | 源操作数。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 源操作数的数据类型需要与目的操作数保持一致。  
sharedTmpBuffer | 输入 | 临时空间。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 用于Fmod内部复杂计算时存储中间变量，由开发者提供。 临时空间大小BufferSize的获取方式请参考[GetFmodMaxMinTmpSize](atlasascendc_api_07_0609.html)。  
calCount | 输入 | 参与计算的元素个数。  
  
#### 返回值说明

无

#### 约束说明

  * 针对Atlas 推理系列产品AI Core，输入数据限制在[-2147483647.0, 2147483647.0]范围内。
  * 源操作数src0Tensor与src1Tensor的数据长度必须保持一致。
  * **不支持源操作数与目的操作数地址重叠。**
  * 不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。



#### 调用示例
    
    
    // dstLocal: 存放Fmod计算结果的输出Tensor
    // src0Local: 存放Fmod计算除数的输入Tensor
    // src1Local: 存放Fmod计算被除数的输入Tensor
    // sharedTmpBuffer: 存放Fmod计算过程中临时缓存的Tensor
    
    // 算子输入的数据类型为half, 需要参与计算的元素个数为512
    AscendC::Fmod(dstLocal, src0Local, src1Local, sharedTmpBuffer, 512);
    

结果示例如下：
    
    
    输入数据(src0Local): [ 0.5 -6.3 5.5 ... 11.1 -11.6]
    输入数据(src1Local): [ 2.1 3.0 -0.3 ...  5.6   5.9]
    输出数据(dstLocal):  [ 0.5 -0.3 0.1 ...  5.5  -5.7]
    

**父主题：** [Fmod接口](atlasascendc_api_07_0607.html)
