<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0238.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# Load2D

# Load2D

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

Load2D支持如下数据通路的搬运：

GM->A1; GM->B1; GM->A2; GM->B2;

A1->A2; B1->B2。

#### 函数原型

  * Load2D接口
        
        template <typename T>
        __aicore__ inline void LoadData(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LoadData2DParams& loadDataParams)
        template <typename T> 
        __aicore__ inline void LoadData(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const LoadData2DParams& loadDataParams)
        




#### 参数说明

表1 模板参数说明

展开

参数名称 | 含义  
---|---  
T | 源操作数和目的操作数的数据类型。

  * **Load2D接口** Atlas 训练系列产品，支持的数据类型为：uint8_t/int8_t/uint16_t/int16_t/half Atlas 推理系列产品AI Core，支持的数据类型为：uint8_t/int8_t/uint16_t/int16_t/half Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型为：uint8_t/int8_t/uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/float Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型为：uint8_t/int8_t/uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/float Atlas 200I/500 A2 推理产品，支持数据类型为：uint8_t/int8_t/uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/float

  
  
表2 通用参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
dst | 输出 | 目的操作数，类型为LocalTensor。 数据连续排列顺序由目的操作数所在TPosition决定，具体约束如下：

  * A2：ZZ格式；对应的分形大小为16 * (32B / sizeof(T))。
  * B2：ZN格式；对应的分形大小为 (32B / sizeof(T)) * 16。
  * A1/B1：无格式要求，一般情况下为NZ格式。NZ格式下，对应的分形大小为16 * (32B / sizeof(T))。

  
src | 输入 | 源操作数，类型为LocalTensor或GlobalTensor。 数据类型需要与dst保持一致。  
loadDataParams | 输入 | LoadData参数结构体，类型为：

  * LoadData2DParams，具体参考[表3](#ZH-CN_TOPIC_0000002520880136__table8955841508)。

上述结构体参数定义请参考${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_mm.h，${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。  
  
表3 LoadData2DParams结构体内参数说明

展开

参数名称 | 含义  
---|---  
startIndex | 分形矩阵ID，说明搬运起始位置为源操作数中第几个分形（0为源操作数中第1个分形矩阵）。取值范围：startIndex∈[0, 65535] 。单位：512B。默认为0。  
repeatTimes | 迭代次数，每个迭代可以处理512B数据。取值范围：repeatTimes∈[1, 255]。  
srcStride | 相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔，单位：512B。取值范围：src_stride∈[0, 65535]。默认为0。  
sid | 预留参数，配置为0即可。  
dstGap | 相邻迭代间，目的操作数前一个分形结束地址与后一个分形起始地址的间隔，单位：512B。取值范围：dstGap∈[0, 65535]。默认为0。 注：Atlas 训练系列产品此参数不使能。  
ifTranspose | 是否启用转置功能，对每个分形矩阵进行转置，默认为false:

  * true：启用
  * false：不启用

注意：只有A1->A2和B1->B2通路才能使能转置，使能转置功能时，源操作数、目的操作数仅支持uint16_t/int16_t/half数据类型。  
addrMode | 控制地址更新方式，默认为false：

  * true：递减，每次迭代在前一个地址的基础上减去srcStride。
  * false：递增，每次迭代在前一个地址的基础上加上srcStride。

  
  
#### 约束说明

  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。



#### 返回值说明

无

#### 调用示例
    
    
    #include "kernel_operator.h"
    uint16_t C1 = 2;
    uint16_t H = 4, W = 4;
    uint8_t Kh = 2, Kw = 2;
    uint16_t Cout = 16;
    uint16_t C0 = 16;
    uint8_t dilationH = 2, dilationW = 2;
    uint8_t padTop = 1, padBottom = 1, padLeft = 1, padRight = 1;
    uint8_t strideH = 1, strideW = 1;
    uint16_t coutBlocks, ho, wo, howo, howoRound;
    uint32_t featureMapA1Size, weightA1Size, featureMapA2Size, weightB2Size, dstSize, dstCO1Size;
    uint8_t padList[4] = {padLeft, padRight, padTop, padBottom};
    featureMapA2Size = howoRound * (C1 * Kh * Kw * C0);
    fmRepeat = featureMapA2Size / (16 * C0);
    
    AscendC::LocalTensor<half> featureMapA1 = inQueueFmA1.DeQue<half>();
    AscendC::LocalTensor<half> featureMapA2 = inQueueFmA2.AllocTensor<half>();
    
    AscendC::LoadData<A2, A1, half>(featureMapA2, featureMapA1, 
    { padList, H, W, 0, 0, 0, -1, -1, strideW, strideH, Kw, Kh, dilationW, dilationH, 1, 0, fmRepeat, 0, (half)(0)});
    
    inQueueFmA2.EnQue<half>(featureMapA2);
    inQueueFmA1.FreeTensor(featureMapA1);
    

**父主题：** [LoadData](atlasascendc_api_07_0238.html)
