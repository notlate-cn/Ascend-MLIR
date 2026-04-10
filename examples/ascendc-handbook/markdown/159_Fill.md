<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0237.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# Fill

# Fill

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

将特定TPosition的LocalTensor初始化为某一具体数值。

#### 函数原型
    
    
    template <typename T, typename U = PrimT<T>, typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
    __aicore__ inline void Fill(const LocalTensor<T>& dst, const InitConstValueParams<U>& initConstValueParams)
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
T | dst的数据类型。 Atlas 训练系列产品，支持的数据类型为：half Atlas 推理系列产品AI Core，支持的数据类型为：half/int16_t/uint16_t Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half/int16_t/uint16_t/bfloat16_t/float/int32_t/uint32_t Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half/int16_t/uint16_t/bfloat16_t/float/int32_t/uint32_t Atlas 200I/500 A2 推理产品，支持的数据类型为：half/int16_t/uint16_t/bfloat16_t/float/int32_t/uint32_t  
U | 初始化值的数据类型。

  * 当dst使用基础数据类型时， U和dst的数据类型T需保持一致，否则编译失败。
  * 当dst使用[TensorTrait](atlasascendc_api_07_0011.html)类型时，U和dst的数据类型T的LiteType需保持一致，否则编译失败。

最后一个模板参数仅用于上述数据类型检查，用户无需关注。  
  
表2 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
dst | 输出 | 目的操作数，结果矩阵，类型为LocalTensor。 Atlas 训练系列产品，支持的TPosition为A1/A2/B1/B2。 Atlas 推理系列产品AI Core，支持的TPosition为A1/A2/B1/B2。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的TPosition为A1/A2/B1/B2。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的TPosition为A1/A2/B1/B2。 Atlas 200I/500 A2 推理产品，支持的TPosition为A1/A2/B1/B2。 如果TPosition为A1/B1，起始地址需要满足32B对齐；如果TPosition为A2/B2，起始地址需要满足512B对齐。  
InitConstValueParams | 输入 | 初始化相关参数，类型为InitConstValueParams。 具体定义请参考${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_mm.h，${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。 参数说明请参考[表3](#ZH-CN_TOPIC_0000002552079785__table15780447181917)。 Atlas 训练系列产品，仅支持配置迭代次数（repeatTimes）和初始化值（initValue）。 Atlas 推理系列产品AI Core，仅支持配置迭代次数（repeatTimes）和初始化值（initValue）。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持配置所有参数。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持配置所有参数。 Atlas 200I/500 A2 推理产品，支持配置所有参数。

  * 仅支持配置迭代次数（repeatTimes）和初始化值（initValue）场景下，其他参数配置无效。每次迭代处理固定数据量（512字节），迭代间无间隔。
  * 支持配置所有参数场景下，支持配置迭代次数（repeatTimes）、初始化值（initValue）、每个迭代处理的数据块个数（blockNum）和迭代间间隔（dstGap）。

  
  
表3 InitConstValueParams结构体参数说明

展开

参数名称 | 含义  
---|---  
repeatTimes | 迭代次数。默认值为0。

  * 仅支持配置迭代次数（repeatTimes）和初始化值（initValue）场景下，repeatTimes∈[0, 255]。
  * 支持配置所有参数场景下，repeatTimes∈[0, 32767] 。

  
blockNum | 每次迭代初始化的数据块个数，取值范围：blockNum∈[0, 32767] 。默认值为0。

  * dst的位置为A1/B1时，每一个block（数据块）大小是32B；
  * dst的位置为A2/B2时，每一个block（数据块）大小是512B。

  
dstGap | 目的操作数前一个迭代结束地址到后一个迭代起始地址之间的距离。

  * dst的位置为A1/B1时，单位是32B；
  * dst的位置为A2/B2时，单位是512B。

取值范围：dstGap∈[0, 32767] 。默认值为0。  
initValue | 初始化的value值，支持的数据类型与dst保持一致。  
  
#### 约束说明

  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。



#### 调用示例
    
    
    #include "kernel_operator.h"
    uint32 m = 16;
    uint32 k = 16;
    TPipe pipe;
    TQue<TPosition::A1, 1> qidA1_;
    pipe.InitBuffer(qidA1_, 1, m * k * sizeof(float));
    LocalTensor<float> leftMatrix = qidA1_.template AllocTensor<float>();
    Fill(leftMatrix, {1, static_cast<uint16_t>(mLength * kLength * sizeof(Src0T) / 32), 0, 1});
    qidA1_.EnQue(leftMatrix);
    

**父主题：** [数据搬运](atlasascendc_api_07_00171.html)
