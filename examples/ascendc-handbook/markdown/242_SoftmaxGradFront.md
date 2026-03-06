<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0759.html -->
<!-- 下载时间: 2026-03-04 11:39:03 -->

# SoftmaxGradFront

# SoftmaxGradFront

#### 产品支持情况

展开

产品 |  是否支持  
---|---  
Atlas A3 训练系列产品 / Atlas A3 推理系列产品  |  √  
Atlas A2 训练系列产品 / Atlas A2 推理系列产品  |  √  
Atlas 200I/500 A2 推理产品  |  √  
Atlas 推理系列产品 AI Core |  √  
Atlas 推理系列产品 Vector Core |  x  
Atlas 训练系列产品  |  x  
  
#### 功能说明

将输入tensor[m0, m1, ...mt, n]（t大于等于0）的非尾轴长度相乘的结果看作m，则输入tensor的shape看作[m, n]。对输入tensor[m,n]按行做gradfront反向计算，计算公式如下：

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_formulaimage_0000002552082811.png)

当输入shape为ND格式时，内部的reduce过程按last轴进行；当输入shape为NZ格式时，内部的reduce过程按照last轴和first轴进行，reduce过程可以参考[SoftMax](atlasascendc_api_07_0754.html)中的图示说明。

为方便理解，通过Python脚本实现的方式，表达其计算公式如下，其中dx、y是源操作数（输入），d为目的操作数（输出）。
    
    
    def softmax_grad_front(dx, y, is_fp16=False):
        dx = dx.astype(np.float32)
        y = y.astype(np.float32)
    
        d = (dx * y).sum(axis=-1, keepdims=True)  ###[1024,1]
        if is_fp16:
        d = d.astype(np.float16)
        return d
    

#### 实现原理

以float类型，ND格式，shape为[m, k]的输入Tensor为例，描述SoftmaxGradFront高阶API内部算法框图，如下图所示。

**图1** SoftmaxGradFront算法框图   


![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002552122811.png)

计算过程分为如下几步，均在Vector上进行：

  1. mul步骤：对输入x和y所有数据相乘，计算结果会保存到一个临时空间temp中；
  2. reducesum步骤：对temp中的数据([m, k])每一行数据求和得到[m, 1]，计算结果保存到临时空间中；
  3. broadcast步骤：对[m, 1]做一个按datablock为单位的填充，比如float类型下，把[m, 1]扩展成[m, 8]，并输出结果z。



#### 函数原型

  * 接口框架申请临时空间 
        
        template <typename T, bool isBasicBlock = false, bool isDataFormatNZ = false>
        __aicore__ inline void SoftmaxGradFront(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor, const LocalTensor<T>& srcTensor, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
        



  * 通过sharedTmpBuffer入参传入临时空间 
        
        template <typename T, bool isBasicBlock = false, bool isDataFormatNZ = false>
        __aicore__ inline void SoftmaxGradFront(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
        




由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请** 和开发者**通过sharedTmpBuffer入参传入** 两种方式。

  * 接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。


  * 通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。



接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式如下：通过[SoftmaxGrad Tiling接口](atlasascendc_api_07_0764.html)中提供的GetSoftMaxGradMaxTmpSize/GetSoftMaxGradMinTmpSize接口获取所需最小和最大临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

#### 参数说明

表1 模板参数说明

展开

参数名 |  描述  
---|---  
T |  操作数的数据类型。 Atlas A3 训练系列产品 / Atlas A3 推理系列产品 ，支持的数据类型为：half、float。 Atlas A2 训练系列产品 / Atlas A2 推理系列产品 ，支持的数据类型为：half、float。 Atlas 200I/500 A2 推理产品 ，支持的数据类型为：half、float。 Atlas 推理系列产品 AI Core，支持的数据类型为：half、float。  
isBasicBlock |  srcTensor和gradTensor的shape信息和Tiling切分策略满足基本块要求的情况下，可以使能该参数用于提升性能，默认不使能。是否满足基本块的要求，可以采用如下两种方式之一判断：

  * srcTensor和dstTensor的shape信息[m,n]需要满足如下条件： 
    * 尾轴长度n小于2048并且大于等于256/sizeof(T)（即half场景下n最小为128，float场景下n最小为64），同时n是64的倍数；
    * 非尾轴长度的乘积m为8的倍数。


  * 在Tiling实现中，通过调用[IsBasicBlockInSoftMax](atlasascendc_api_07_0766.html)判断Tiling切分策略是否满足基本块的切分要求。

针对 Atlas 200I/500 A2 推理产品 ，该参数为预留参数，暂未启用，为后续的功能扩展做保留，保持默认值即可。  
isDataFormatNZ |  当前输入输出的数据格式是否为NZ格式，默认数据格式为ND，即默认取值为false。 针对 Atlas 200I/500 A2 推理产品 ，不支持配置为NZ格式。  
  
表2 接口参数说明

展开

参数名 |  输入/输出 |  描述  
---|---|---  
dstTensor |  输出 |  目的操作数。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 last轴长度固定32Byte即一个datablock长度，并且该datablock中的所有数据为同一个值。比如half数据类型下，该datablock里的16个数均为相同的值，非last轴长度需要和srcTensor保持一致。  
gradTensor |  输入 |  源操作数。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 last轴长度需要32Byte对齐，gradTensor的shape与srcTensor的shape一致。  
srcTensor |  输入 |  源操作数。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 last轴长度需要32Byte对齐，srcTensor的shape与gradTensor的shape一致。  
sharedTmpBuffer |  输入 |  临时空间。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 该操作数的数据类型固定uint8_t。 用于接口内部复杂计算时存储中间变量，由开发者提供。 临时空间大小BufferSize的获取方式请参考[SoftmaxGrad Tiling接口](atlasascendc_api_07_0764.html)。  
tiling |  输入 |  softmaxgradfront计算所需tiling信息，Tiling信息的获取请参考[SoftmaxGrad Tiling接口](atlasascendc_api_07_0764.html)。  
softmaxShapeInfo |  输入 |  srcTensor的shape信息。SoftMaxShapeInfo类型，具体定义如下：
    
    
    struct SoftMaxShapeInfo {
        uint32_t srcM; // 非尾轴乘积长度
        uint32_t srcK; // 尾轴长度，必须32Byte对齐
        uint32_t oriSrcM; // 原始非尾轴乘积长度
        uint32_t oriSrcK;  // 原始尾轴长度
    };
    

需要注意，当输入输出的数据格式为NZ格式时，尾轴长度为reduce轴长度即[图2](atlasascendc_api_07_0754.html#ZH-CN_TOPIC_0000002552079755__fig0172155842215)中的W0*W1，非尾轴为H0*H1。  
  
#### 返回值说明

无

#### 约束说明

  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。
  * 不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
  * 当参数softmaxShapeInfo中srcM != oriSrcM 或者 srcK != oriSrcK时，开发者需要对GM上的原始输入(oriSrcM, oriSrcK)在M或K方向补齐数据到(srcM, srcK)，补齐的数据会参与部分运算，在输入输出复用的场景下，API的计算结果会覆盖srcTensor中补齐的原始数据，在输入输出不复用的场景下，API的计算结果会覆盖dstTensor中对应srcTensor补齐位置的数据。



#### 调用示例

本样例输入srcTensor的Shape大小为[128,64]，输入gradtensor的Shape大小为[128,64]，输出dstTensor的Shape大小为[128,16]，数据类型均为half，输入输出的数据排布格式为ND，不使能基本块。 
    
    
    #include "kernel_operator.h"
    
    AscendC::LocalTensor<T> srcLocal1 = inQueueSrc1.DeQue<T>();
    AscendC::LocalTensor<T> srcLocal2 = inQueueSrc2.DeQue<T>();
    AscendC::LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();
    AscendC::SoftMaxShapeInfo srcShape = { height, width, height, width };
    AscendC::SoftmaxGradFront<T>(dstLocal, srcLocal2, srcLocal1, tiling, srcShape);
    outQueueDst.EnQue<T>(dstLocal);
    inQueueSrc1.FreeTensor(srcLocal1);
    inQueueSrc2.FreeTensor(srcLocal2);
    

**父主题：** [SoftMax接口](atlasascendc_api_07_0753.html)
