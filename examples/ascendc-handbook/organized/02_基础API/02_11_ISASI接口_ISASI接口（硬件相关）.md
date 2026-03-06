# ISASI接口（硬件相关）

> 来源: 昇腾社区官网 AscendC算子开发文档

---

## 目录

- [矢量计算ISASI](#矢量计算isasi)
- [排序ISASI](#排序isasi)
- [数据搬运ISASI](#数据搬运isasi)
- [矩阵计算ISASI](#矩阵计算isasi)
- [Conv2D_Gemm](#conv2d_gemm)
- [FixPipe](#fixpipe)
- [LoadData](#loaddata)
- [同步控制ISASI](#同步控制isasi)
- [缓存ISASI](#缓存isasi)
- [系统变量ISASI](#系统变量isasi)
- [原子操作ISASI](#原子操作isasi)
- [调试ISASI](#调试isasi)
- [Cube分组](#cube分组)

---



---

## 矢量计算ISASI


# VectorPadding(ISASI)

#### 产品支持情况

展开

产品 |  是否支持  
---|---  
Atlas A3 训练系列产品 / Atlas A3 推理系列产品  |  x  
Atlas A2 训练系列产品 / Atlas A2 推理系列产品  |  x  
Atlas 200I/500 A2 推理产品  |  x  
Atlas 推理系列产品 AI Core |  √  
Atlas 推理系列产品 Vector Core |  x  
Atlas 训练系列产品  |  x  
  
#### 功能说明

根据padMode（pad模式）与padSide（pad方向）对源操作数按照datablock进行填充操作。

假设源操作数的一个datablock有16个数，datablock[0:15]=a~p：

  * padSide==false：从datablock的左边开始填充，即datablock的起始值方向(a->p)


  * padSide==true：从datablock的右边开始填充，即datablock的结束值方向(p->a)
  * padMode==0：用邻近数作为填充值，例：aaa|abc(padSide=false)、nop|ppp(padSide=true)
  * padMode==1：用邻近datablock值对称填充，例：cba|abc(padSide=false)、nop|pon(padSide=true)
  * padMode==2：用邻近datablock值填充，偏移一个数，做对称填充，例： 
    * padSide=false：xcb|abc，xcb被填充，填充过程描述：a被丢弃，对称填充，x处填充0
    * padSide=true：nop|onx，onx被填充，填充过程描述：p被丢弃，对称填充，x处填充0



#### 函数原型

  * tensor前n个数据计算 
        
        template <typename T>
        __aicore__ inline void VectorPadding(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint8_t padMode, const bool padSide, const uint32_t count)
        

  * tensor高维切分计算 
    * mask逐bit模式 
          
          template <typename T, bool isSetMask = true>
          __aicore__ inline void VectorPadding(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint8_t padMode, const bool padSide, const uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
          

    * mask连续模式 
          
          template <typename T, bool isSetMask = true>
          __aicore__ inline void VectorPadding(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint8_t padMode, const bool padSide, const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
          




#### 参数说明

表1 模板参数说明

展开

参数名 |  描述  
---|---  
T |  操作数数据类型。 Atlas 推理系列产品 AI Core，支持的数据类型为：int16_t/uint16_t/half/int32_t/uint32_t/float  
isSetMask |  是否在接口内部设置mask。

  * true，表示在接口内部设置mask。
  * false，表示在接口外部设置mask，开发者需要使用[SetVectorMask](atlasascendc_api_07_0096.html)接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。

  
  
表2 参数说明

展开

参数名 |  输入/输出 |  描述  
---|---|---  
dst |  输出 |  目的操作数。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。  
src |  输入 |  源操作数。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。 源操作数的数据类型需要与目的操作数保持一致。  
padMode |  输入 |  padding模式，类型为uint8_t，取值范围：[0,2]。

  * 0：用邻近数作为填充值。
  * 1：用邻近datablock值对称填充。
  * 2：用邻近datablock值填充，偏移一个数，做对称填充。

  
padSide |  输入 |  padding的方向，类型为bool。

  * false：左边。
  * true：右边。

  
count |  输入 |  参与计算的元素个数。  
mask[]/mask |  输入 |  [mask](/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_0022.html#ZH-CN_TOPIC_0000002552129981__zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section4252658182)用于控制每次迭代内参与计算的元素。

  * 逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。 mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 264-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 264-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 232-1]。 例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。


  * 连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。

  
repeatTime |  输入 |  重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。 关于该参数的具体描述请参考[高维切分API](/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_0022.html)。  
repeatParams |  输入 |  控制操作数地址步长的参数。[UnaryRepeatParams](atlasascendc_api_07_0012.html)类型，包含操作数相邻迭代间相同DataBlock的地址步长，操作数同一迭代内不同DataBlock的地址步长等参数。 相邻迭代间的地址步长参数说明请参考[repeatStride](/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_0022.html#ZH-CN_TOPIC_0000002552129981__zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section139459347420)；同一迭代内DataBlock的地址步长参数说明请参考[dataBlockStride](/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_0022.html#ZH-CN_TOPIC_0000002552129981__zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section2815124173416)。  
  
#### 返回值说明

无

#### 约束说明

  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。


  * mask仅控制目的操作数中的哪些元素要写入，源操作数的读取与mask无关。
  * count表示写入目的操作数中的元素总数，源操作数的读取与count无关。



#### 调用示例

样例的srcLocal和dstLocal均为half类型。

更多样例可参考[LINK](atlasascendc_api_07_0052.html)。

  * tensor高维切分计算样例-mask连续模式 
        
        uint64_t mask = 256 / sizeof(half);
        uint8_t padMode = 0;
        bool padSide = false;
        // repeatTime = 4, 128 elements one repeat, 512 elements total
        // dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
        // dstRepStride, srcRepStride = 8, no gap between repeats
        AscendC::VectorPadding(dstLocal, srcLocal, padMode, padSide, mask, 4, { 1, 1, 8, 8 });
        

  * tensor高维切分计算样例-mask逐bit模式 
        
        uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
        uint8_t padMode = 0;
        bool padSide = false;
        // repeatTime = 4, 128 elements one repeat, 512 elements total
        // dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
        // dstRepStride, srcRepStride = 8, no gap between repeats
        AscendC::VectorPadding(dstLocal, srcLocal, padMode, padSide, mask, 4, { 1, 1, 8, 8 });
        

  * tensor前n个数据计算样例 
        
        uint8_t padMode = 0;
        bool padSide = false;
        AscendC::VectorPadding(dstLocal, srcLocal, padMode, padSide, 512);
        




结果示例如下： 
    
    
    // 以srcLocal的一个datablock的值为例，有16个数
    输入数据(srcLocal): [6.938 -8.86 -0.2263 ... 1.971 1.778]
    输出数据(dstLocal): 
    [6.938 6.938 6.938 ... 6.938 6.938]

**父主题：** [数据填充](atlasascendc_api_07_0087.html)


# BilinearInterpolation(ISASI)

#### 产品支持情况

展开

产品 |  是否支持  
---|---  
Atlas A3 训练系列产品 / Atlas A3 推理系列产品  |  √  
Atlas A2 训练系列产品 / Atlas A2 推理系列产品  |  √  
Atlas 200I/500 A2 推理产品  |  x  
Atlas 推理系列产品 AI Core |  √  
Atlas 推理系列产品 Vector Core |  x  
Atlas 训练系列产品  |  x  
  
#### 功能说明

功能分为水平迭代和垂直迭代。每个水平迭代顺序地从src0Offset读取8个偏移值，表示src0的偏移，每个偏移值指向src0的一个DataBlock的起始地址，如果repeatMode=false，从src1中取一个值，与src0中8个DataBlock中每个值进行乘操作；如果repeatMode=true，从src1中取8个值，按顺序与src0中8个DataBlock中的值进行乘操作，最后当前迭代的dst结果与前一个dst结果按DataBlock进行累加，存入目的地址，在同一个水平迭代内dst地址不变。然后进行垂直迭代，垂直迭代的dst起始地址为上一轮垂直迭代的dst起始地址加上vROffset，本轮垂直迭代占用dst空间为dst起始地址之后的8个DataBlock，每轮垂直迭代进行hRepeat次水平迭代。

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002552082099.png)

#### 函数原型

  * mask逐bit模式： 
        
        template <typename T>
        __aicore__ inline void BilinearInterpolation(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<uint32_t>& src0Offset, const LocalTensor<T>& src1, uint64_t mask[], uint8_t hRepeat, bool repeatMode, uint16_t dstBlkStride, uint16_t vROffset, uint8_t vRepeat, const LocalTensor<uint8_t> &sharedTmpBuffer)
        

  * mask连续模式： 
        
        template <typename T>
        __aicore__ inline void BilinearInterpolation(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<uint32_t>& src0Offset, const LocalTensor<T>& src1, uint64_t mask, uint8_t hRepeat, bool repeatMode, uint16_t dstBlkStride, uint16_t vROffset, uint8_t vRepeat, const LocalTensor<uint8_t> &sharedTmpBuffer)
        




#### 参数说明

表1 模板参数说明

展开

参数名 |  描述  
---|---  
T |  操作数数据类型。 Atlas A2 训练系列产品 / Atlas A2 推理系列产品 ，支持的数据类型为：half。 Atlas A3 训练系列产品 / Atlas A3 推理系列产品 ，支持的数据类型为：half。 Atlas 推理系列产品 AI Core，支持的数据类型为：half。  
  
表2 参数说明

展开

参数名 |  输入/输出 |  描述  
---|---|---  
dst |  输出 |  目的操作数。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。  
src0、src1 |  输入 |  源操作数。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。 两个源操作数的数据类型需要与目的操作数保持一致。  
src0Offset |  输入 |  源操作数。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。  
mask[]/mask |  输入 |  [mask](/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_0022.html#ZH-CN_TOPIC_0000002552129981__zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section4252658182)用于控制每次迭代内参与计算的元素。

  * 逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。 mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 264-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 264-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 232-1]。 例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。


  * 连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。

  
hRepeat |  输入 |  水平方向迭代次数，取值范围为[1, 255]。  
repeatMode |  输入 |  迭代模式：

  * false：每次迭代src0读取的8个datablock中每个值均与src1的单个数值相乘。
  * true：每次迭代src0的每个datablock分别与src1的1个数值相乘，共消耗8个block和8个elements。

  
dstBlkStride |  输入 |  单次迭代内，目的操作数不同DataBlock间地址步长，以32B为单位。  
vROffset |  输入 |  垂直迭代间，目的操作数地址偏移量，以元素为单位，取值范围为[128, 65535)，vROffset * sizeof(T)需要保证32字节对齐 。  
vRepeat |  输入 |  垂直方向迭代次数，取值范围为[1, 255]。  
sharedTmpBuffer |  输入 |  临时空间。 Atlas A2 训练系列产品 / Atlas A2 推理系列产品 ，需要保证至少分配了src0.GetSize() * 32 + src1.GetSize() * 32字节的空间。 Atlas A3 训练系列产品 / Atlas A3 推理系列产品 ，需要保证至少分配了src0.GetSize() * 32 + src1.GetSize() * 32字节的空间。 Atlas 推理系列产品 AI Core，需要保证至少分配了src0OffsetLocal.GetSize() * sizeof(uint32_t)字节的空间。  
  
#### 返回值说明

无

#### 约束说明

  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。


  * src0、src1、src0Offset之间不允许地址重叠，且两个垂直repeat的目的地址之间不允许地址重叠。



#### 调用示例

  * 接口样例-mask连续模式 
        
        AscendC::LocalTensor<half> dstLocal, src0Local, src1Local;
        AscendC::LocalTensor<uint32_t> src0OffsetLocal;
        AscendC::LocalTensor<uint8_t> tmpLocal;
        uint64_t mask = 128;        // mask连续模式
        uint8_t hRepeat = 2;        // 水平迭代2次
        bool repeatMode = false;    // 迭代模式
        uint16_t dstBlkStride = 1;  // 单次迭代内数据连续写入
        uint16_t vROffset = 128;    // 相邻迭代间数据连续写入
        uint8_t vRepeat = 2;        // 垂直迭代2次
        
        AscendC::BilinearInterpolation(dstLocal, src0Local, src0OffsetLocal, src1Local, mask, hRepeat, repeatMode,
                    dstBlkStride, vROffset, vRepeat, tmpLocal);
        

  * 接口样例-mask逐bit模式 
        
        AscendC::LocalTensor<half> dstLocal, src0Local, src1Local;
        AscendC::LocalTensor<uint32_t> src0OffsetLocal;
        AscendC::LocalTensor<uint8_t> tmpLocal;
        uint64_t mask[2] = { UINT64_MAX, UINT64_MAX }; // mask逐bit模式
        uint8_t hRepeat = 2;        // 水平迭代2次
        bool repeatMode = false;    // 迭代模式
        uint16_t dstBlkStride = 1;  // 单次迭代内数据连续写入
        uint16_t vROffset = 128;    // 相邻迭代间数据连续写入
        uint8_t vRepeat = 2;        // 垂直迭代2次
        
        AscendC::BilinearInterpolation(dstLocal, src0Local, src0OffsetLocal, src1Local, mask, hRepeat, repeatMode,
                    dstBlkStride, vROffset, vRepeat, tmpLocal);
        




结果示例如下： 
    
    
    输入数据(src0Local,half): [1,2,3,...,512]
    输入数据(src1Local,half): [2,3,4,...,17]
    输入数据(src0OffsetLocal,uint32_t): [0,32,64,...,992]
    输出数据(dstLocal,half): [389, 394, 399, 404, ...,4096]

**父主题：** [基础算术](atlasascendc_api_07_0024.html)


# GetCmpMask(ISASI)

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

此接口用于获取[Compare（结果存入寄存器）](atlasascendc_api_07_0067.html)指令的比较结果。

[Compare（结果存入寄存器）](atlasascendc_api_07_0067.html)指令会将比较后的结果写入CmpMask寄存器中，使用GetCmpMask接口可以获取到CmpMask寄存器的值从而得到Compare的结果。

#### 函数原型
    
    
    template<typename T>
    __aicore__ inline void GetCmpMask(const LocalTensor<T>& dst)
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
T | 操作数的数据类型。  
  
表2 参数说明

展开

参数名 | 输入/输出 | 描述  
---|---|---  
dst | 输出 | [Compare（结果存入寄存器）](atlasascendc_api_07_0067.html)指令的比较结果。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要16字节对齐。  
  
#### 返回值说明

无

#### 约束说明

dst的空间大小不能少于128字节。

#### 调用示例

[Compare（结果存入寄存器）](atlasascendc_api_07_0067.html)指令的结果使用uint8_t类型数据存储，因此dstLocal使用uint8_t类型。
    
    
    AscendC::LocalTensor<float> src0Local;
    AscendC::LocalTensor<float> src1Local;
    AscendC::LocalTensor<uint8_t> dstLocal;
    uint64_t mask = 256 / sizeof(float); // 256为每个迭代处理的字节数，结果为64
    AscendC::BinaryRepeatParams repeatParams = { 1, 1, 1, 8, 8, 8 };
    AscendC::Compare(src0Local, src1Local, AscendC::CMPMODE::LT, mask, repeatParams);
    AscendC::GetCmpMask(dstLocal); // mask为0x40, 比较数据类型为float，则每次迭代的32B里只有第7个float数字参与compare
    

**父主题：** [比较与选择](atlasascendc_api_07_0065.html)


# SetCmpMask(ISASI)

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

为[Select](atlasascendc_api_07_0070.html)不传入mask参数的接口设置比较寄存器。配合不同的selMode传入不同的数据。

  * 模式0（SELMODE::VSEL_CMPMASK_SPR）

SetCmpMask中传入selMask LocalTensor。



  * 模式1（SELMODE::VSEL_TENSOR_SCALAR_MODE）

SetCmpMask中传入src1 LocalTensor。



  * 模式2（SELMODE::VSEL_TENSOR_TENSOR_MODE）

SetCmpMask中传入LocalTensor，LocalTensor中存放的是selMask的地址。




#### 函数原型
    
    
    template <typename T>
    __aicore__ inline void SetCmpMask(const LocalTensor<T>& src)
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
T | 操作数的数据类型。  
  
表2 参数说明

展开

参数名 | 输入/输出 | 描述  
---|---|---  
src | 输入 | 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要16字节对齐。  
  
#### 返回值说明

无

#### 约束说明

无

#### 调用示例

  * 当selMode为模式0或模式2时：
        
        uint32_t dataSize = 256;
        uint32_t selDataSize = 8;
        TPipe pipe;
        TQue<TPosition::VECIN, 1> inQueueX;
        TQue<TPosition::VECIN, 1> inQueueY;
        TQue<TPosition::VECIN, 1> inQueueSel;
        TQue<TPosition::VECOUT, 1> outQueue;
        pipe.InitBuffer(inQueueX, 1, dataSize * sizeof(float));
        pipe.InitBuffer(inQueueY, 1, dataSize * sizeof(float));
        pipe.InitBuffer(inQueueSel, 1, selDataSize * sizeof(uint8_t));
        pipe.InitBuffer(outQueue, 1, dataSize * sizeof(float));
        AscendC::LocalTensor<float> dst = outQueue.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> sel = inQueueSel.AllocTensor<uint8_t>();
        AscendC::LocalTensor<float> src0 = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> src1 = inQueueY.AllocTensor<float>();
        uint8_t repeat = 4;
        uint32_t mask = 64;
        AscendC::BinaryRepeatParams repeatParams = { 1, 1, 1, 8, 8, 8 };
        
        // selMode为模式0（SELMODE::VSEL_CMPMASK_SPR）
        AscendC::SetCmpMask(sel);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<float>(mask);
        AscendC::Select<float, AscendC::SELMODE::VSEL_CMPMASK_SPR>(dst, src0, src1, repeat, repeatParams);
        
        // selMode为模式2（SELMODE::VSEL_TENSOR_TENSOR_MODE）
        AscendC::LocalTensor<int32_t> tempBuf;
        #if defined(ASCENDC_CPU_DEBUG) && (ASCENDC_CPU_DEBUG == 1)  // cpu调试
        tempBuf.ReinterpretCast<int64_t>().SetValue(0, reinterpret_cast<int64_t>(reinterpret_cast<__ubuf__ int64_t*>(sel.GetPhyAddr())));
        event_t eventIdSToV = static_cast<event_t>(AscendC::GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIdSToV);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIdSToV);
        #else // npu调试
        uint32_t selAddr = static_cast<uint32_t>(reinterpret_cast<int64_t>(reinterpret_cast<__ubuf__ int64_t*>(sel.GetPhyAddr())));
        AscendC::SetVectorMask<uint32_t>(32);
        AscendC::Duplicate<uint32_t, false>(tempBuf.ReinterpretCast<uint32_t>(), selAddr, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        #endif
        AscendC::SetCmpMask<int64_t>(tempBuf.ReinterpretCast<int64_t>());
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<float>(mask);
        AscendC::Select<float, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE>(dst, src0, src1, repeat, repeatParams);
        




  * 当selMode为模式1时：
        
        uint32_t dataSize = 256;
        uint32_t selDataSize = 8;
        TPipe pipe;
        TQue<TPosition::VECIN, 1> inQueueX;
        TQue<TPosition::VECIN, 1> inQueueY;
        TQue<TPosition::VECIN, 1> inQueueSel;
        TQue<TPosition::VECOUT, 1> outQueue;
        pipe.InitBuffer(inQueueX, 1, dataSize * sizeof(float));
        pipe.InitBuffer(inQueueY, 1, dataSize * sizeof(float));
        pipe.InitBuffer(inQueueSel, 1, selDataSize * sizeof(uint8_t));
        pipe.InitBuffer(outQueue, 1, dataSize * sizeof(float));
        AscendC::LocalTensor<float> dst = outQueue.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> sel = inQueueSel.AllocTensor<uint8_t>();
        AscendC::LocalTensor<float> src0 = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> tmpScalar = inQueueY.AllocTensor<float>();
        
        uint8_t repeat = 4;
        uint32_t mask = 64;
        AscendC::BinaryRepeatParams repeatParams = { 1, 1, 1, 8, 8, 8 };
        
        // selMode为模式1（SELMODE::VSEL_TENSOR_SCALAR_MODE）
        AscendC::SetVectorMask<uint32_t>(32);
        AscendC::Duplicate<float, false>(tmpScalar, static_cast<float>(1.0), MASK_PLACEHOLDER, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetCmpMask(tmpScalar);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<float>(mask);
        AscendC::Select(dst, sel, src0, repeat, repeatParams);
        




**父主题：** [比较与选择](atlasascendc_api_07_0065.html)


# GetReduceRepeatSumSpr(ISASI)

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

获取[ReduceSum](atlasascendc_api_07_0078.html#ZH-CN_TOPIC_0000002552079835__li6452183633914)接口（Tensor前n个数据计算接口，n为接口的count参数）的计算结果。

#### 函数原型
    
    
    template <typename T>
    __aicore__ inline T GetReduceRepeatSumSpr()
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
T | ReduceSum指令的数据类型，支持half、float。  
  
#### 返回值说明

ReduceSum接口（Tensor前n个数据计算接口，n为接口的count参数）的计算结果。

#### 约束说明

无。

#### 调用示例
    
    
    AscendC::LocalTensor<float> src;
    AscendC::LocalTensor<float> work;
    AscendC::LocalTensor<float> dst;
    AscendC::ReduceSum(dst, src, work, 128);
    float res = AscendC::GetReduceRepeatSumSpr<float>();
    

**父主题：** [归约计算](atlasascendc_api_07_0075.html)


# GetReduceRepeatMaxMinSpr(ISASI)

#### 产品支持情况

展开

产品 | 是否支持（仅获取最值的原型） | 是否支持（获取最值和索引的原型）  
---|---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x | √  
Atlas 200I/500 A2 推理产品 | x | x  
Atlas 推理系列产品AI Core | √ | x  
Atlas 推理系列产品Vector Core | x | x  
Atlas 训练系列产品 | x | x  
  
#### 功能说明

获取[ReduceMax](atlasascendc_api_07_0076.html)、[ReduceMin](atlasascendc_api_07_0077.html)连续场景下的最大/最小值以及相应的索引值。

#### 函数原型

  * 获取[ReduceMax](atlasascendc_api_07_0076.html)、[ReduceMin](atlasascendc_api_07_0077.html)连续场景下的最大值与最小值，以及相应的索引值。
        
        template <typename T>
        __aicore__ inline void GetReduceRepeatMaxMinSpr(T &maxMinValue, T &maxMinIndex)
        

  * 获取[ReduceMax](atlasascendc_api_07_0076.html)、[ReduceMin](atlasascendc_api_07_0077.html)连续场景下的最大值与最小值。
        
        template <typename T>
        __aicore__ inline void GetReduceRepeatMaxMinSpr(T &maxMinValue)
        




#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
T | ReduceMax/ReduceMin指令的数据类型，支持half/float。  
  
表2 参数说明

展开

参数名 | 输入/输出 | 描述  
---|---|---  
maxMinValue | 输出 | ReduceMax/ReduceMin指令的最大值/最小值。  
maxMinIndex | 输出 | ReduceMax/ReduceMin指令的最值对应的索引值。  
  
#### 返回值说明

无

#### 约束说明

  * 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，由于ReduceMax/ReduceMin的内部实现原因，直接调用GetReduceRepeatMaxMinSpr接口无法获取到准确的索引值，验证时需要使用[WholeReduceMax](atlasascendc_api_07_0079.html)/[WholeReduceMin](atlasascendc_api_07_0080.html)接口来获取准确的索引值。
  * 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，由于ReduceMax/ReduceMin的内部实现原因，直接调用GetReduceRepeatMaxMinSpr接口无法获取到准确的索引值，验证时需要使用[WholeReduceMax](atlasascendc_api_07_0079.html)/[WholeReduceMin](atlasascendc_api_07_0080.html)接口来获取准确的索引值。
  * 索引maxMinIndex数据`是按照ReduceMax/ReduceMin的数据类型进行存储的，比如ReduceMax/ReduceMin使用half类型时，maxMinIndex是按照half类型进行存储的，如果按照half格式进行读取，maxMinIndex的值是不对的，因此maxMinIndex的读取需要使用reinterpret_cast方法转换到整数类型，若输入数据类型是half，需要使用reinterpret_cast<uint16_t*>，若输入是float，需要使用reinterpret_cast<uint32_t*>。



#### 调用示例

  1. 以ReduceMax指令为例，首先执行ReduceMax指令。
         
         AscendC::LocalTensor<float> src;
         AscendC::LocalTensor<float> work;
         AscendC::LocalTensor<float> dst;
         int32_t mask = 64;
         AscendC::ReduceMax(dst, src, work, mask, 1, 8, true); // 连续场景，srcRepStride = 8，且calIndex = true
         

  2. 获取上述ReduceMax指令的最值与索引值。

针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，需要使用WholeReduceMax指令获取准确的索引值，然后再调用GetReduceRepeatMaxMinSpr指令。
         
         AscendC::LocalTensor<float> src;
         AscendC::LocalTensor<float> dst;
         int32_t mask = 64;
         AscendC::WholeReduceMax(dst, src, mask, 1, 1, 1, 8);
         float val = 0;   // 最大值
         float idx = 0;   // 最大值的索引值，与ReduceMax的结果相同，保证和WholeReduceMax的调动次序，而且要配对调用
         AscendC::GetReduceRepeatMaxMinSpr<float>(val, idx);
         

针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，需要使用WholeReduceMax指令获取准确的索引值，然后再调用GetReduceRepeatMaxMinSpr指令。
         
         AscendC::LocalTensor<float> src;
         AscendC::LocalTensor<float> dst;
         int32_t mask = 64;
         AscendC::WholeReduceMax(dst, src, mask, 1, 1, 1, 8);
         float val = 0;   // 最大值
         float idx = 0;   // 最大值的索引值，与ReduceMax的结果相同，保证和WholeReduceMax的调动次序，而且要配对调用
         AscendC::GetReduceRepeatMaxMinSpr<float>(val, idx);
         

针对Atlas 推理系列产品AI Core版本，则可在调用ReduceMax后直接调用GetReduceRepeatMaxMinSpr指令获取其最大/最小值。
         
         AscendC::LocalTensor<float> src;
         AscendC::LocalTensor<float> work;
         AscendC::LocalTensor<float> dst;
         int32_t mask = 64;
         AscendC::ReduceMax(dst, src, work, mask, 1, 8, true);
         float val = 0;   // 最大值
         AscendC::GetReduceRepeatMaxMinSpr<float>(val); // 保证和WholeReduceMax的调动次序，而且要配对调用
         




**父主题：** [归约计算](atlasascendc_api_07_0075.html)



---

## 排序ISASI


# ProposalConcat

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x  
Atlas 200I/500 A2 推理产品 | x  
Atlas 推理系列产品AI Core | √  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | √  
  
#### 功能说明

将连续元素合入Region Proposal内对应位置，每次迭代会将16个连续元素合入到16个Region Proposals的对应位置里。

**Region Proposal说明：**

目前仅支持两种数据类型：half、float。

每个Region Proposal占用连续8个half/float类型的元素，约定其格式： 
    
    
    [x1, y1, x2, y2, score, label, reserved_0, reserved_1]

对于数据类型half，每一个Region Proposal占16Bytes，Byte[15:12]是无效数据，Byte[11:0]包含6个half类型的元素，其中Byte[11:10]定义为label，Byte[9:8]定义为score，Byte[7:6]定义为y2，Byte[5:4]定义为x2，Byte[3:2]定义为y1，Byte[1:0]定义为x1。

如下图所示，总共包含16个Region Proposals。

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002521042656.png)

对于数据类型float，每一个Region Proposal占32Bytes，Byte[31:24]是无效数据，Byte[23:0]包含6个float类型的元素，其中Byte[23:20]定义为label，Byte[19:16]定义为score，Byte[15:12]定义为y2，Byte[11:8]定义为x2，Byte[7:4]定义为y1，Byte[3:0]定义为x1。

如下图所示，总共包含16个Region Proposals。

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002520882654.png)

#### 函数原型
    
    
    template <typename T>
    __aicore__ inline void ProposalConcat(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const int32_t modeNumber)
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
T | 操作数数据类型。 Atlas 训练系列产品，支持的数据类型为：half Atlas 推理系列产品AI Core，支持的数据类型为：half/float  
  
表2 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
dst | 输出 | 目的操作数。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。  
src | 输入 | 源操作数。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。 源操作数的数据类型需要与目的操作数保持一致。  
repeatTime | 输入 | 重复迭代次数，int32_t类型，每次迭代完成16个元素合入到16个Region Proposals里，下次迭代跳至相邻的下一组16个Region Proposals和下一组16个元素。取值范围：repeatTime∈[0,255]。  
modeNumber | 输入 | 合入位置参数，取值范围：modeNumber∈[0, 5]，int32_t类型，仅限于以下配置：

  * 0 – 合入x1
  * 1 – 合入y1
  * 2 – 合入x2
  * 3 – 合入y2
  * 4 – 合入score
  * 5 – 合入label

  
  
#### 返回值说明

无

#### 约束说明

  * 用户需保证dst中存储的proposal数目大于等于实际所需数目，否则会存在tensor越界错误。
  * 用户需保证src中存储的元素大于等于实际所需数目，否则会存在tensor越界错误。
  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。



#### 调用示例

  * 接口使用样例
        
        // repeatTime = 2, modeNumber = 4, 把32个数合入到32个Region Proposal中的score域中
        AscendC::ProposalConcat(dstLocal, srcLocal, 2, 4);
        

  * 完整样例
        
        #include "kernel_operator.h"
        
        class KernelVecProposal {
        public:
            __aicore__ inline KernelVecProposal() {}
            __aicore__ inline void Init(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
            {
                srcGlobal.SetGlobalBuffer((__gm__ half*)src);
                dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
                pipe.InitBuffer(inQueueSrc, 1, srcDataSize * sizeof(half));
                pipe.InitBuffer(outQueueDst, 1, dstDataSize * sizeof(half));
            }
            __aicore__ inline void Process()
            {
                CopyIn();
                Compute();
                CopyOut();
            }
        
        private:
            __aicore__ inline void CopyIn()
            {
                AscendC::LocalTensor<half> srcLocal = inQueueSrc.AllocTensor<half>();
                AscendC::DataCopy(srcLocal, srcGlobal, srcDataSize);
                inQueueSrc.EnQue(srcLocal);
            }
            __aicore__ inline void Compute()
            {
                AscendC::LocalTensor<half> srcLocal = inQueueSrc.DeQue<half>();
                AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();
                AscendC::ProposalConcat(dstLocal, srcLocal, repeat, mode); // 此处仅演示Concat指令用法，需要注意，dstLocal中非score处的数据可能是随机值
                outQueueDst.EnQue<half>(dstLocal);
                inQueueSrc.FreeTensor(srcLocal);
            }
            __aicore__ inline void CopyOut()
            {
                AscendC::LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
                AscendC::DataCopy(dstGlobal, dstLocal, dstDataSize);
                outQueueDst.FreeTensor(dstLocal);
            }
        
        private:
            AscendC::TPipe pipe;
            AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
            AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
            AscendC::GlobalTensor<half> srcGlobal, dstGlobal;
            int srcDataSize = 32;
            int dstDataSize = 256;
            int repeat = srcDataSize / 16;
            int mode = 4;
        };
        
        extern "C" __global__ __aicore__ void vec_proposal_kernel(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
        {
            KernelVecProposal op;
            op.Init(src, dstGm);
            op.Process();
        }
        
        
        示例结果 
        输入数据(src_gm):
        [ 33.3    67.56   68.5   -11.914  25.19  -72.8    11.79  -49.47   49.44
          84.4   -14.36   45.97   52.47   -5.387 -13.12  -88.9    54.    -51.62
         -20.67   59.56   35.72   -6.12  -39.4   -11.46   -7.066  30.23  -11.18
         -35.84  -40.88   60.9   -73.3    38.47 ]
        输出数据(dst_gm):
        [  0.      0.      0.      0.     33.3     0.      0.      0.      0.
           0.      0.      0.     67.56    0.      0.      0.      0.      0.
           0.      0.     68.5     0.      0.      0.      0.      0.      0.
           0.    -11.914   0.      0.      0.      0.      0.      0.      0.
          25.19    0.      0.      0.      0.      0.      0.      0.    -72.8
           0.      0.      0.      0.      0.      0.      0.     11.79    0.
           0.      0.      0.      0.      0.      0.    -49.47    0.      0.
           0.      0.      0.      0.      0.     49.44    0.      0.      0.
           0.      0.      0.      0.     84.4     0.      0.      0.      0.
           0.      0.      0.    -14.36    0.      0.      0.      0.      0.
           0.      0.     45.97    0.      0.      0.      0.      0.      0.
           0.     52.47    0.      0.      0.      0.      0.      0.      0.
          -5.387   0.      0.      0.      0.      0.      0.      0.    -13.12
           0.      0.      0.      0.      0.      0.      0.    -88.9     0.
           0.      0.      0.      0.      0.      0.     54.      0.      0.
           0.      0.      0.      0.      0.    -51.62    0.      0.      0.
           0.      0.      0.      0.    -20.67    0.      0.      0.      0.
           0.      0.      0.     59.56    0.      0.      0.      0.      0.
           0.      0.     35.72    0.      0.      0.      0.      0.      0.
           0.     -6.12    0.      0.      0.      0.      0.      0.      0.
         -39.4     0.      0.      0.      0.      0.      0.      0.    -11.46
           0.      0.      0.      0.      0.      0.      0.     -7.066   0.
           0.      0.      0.      0.      0.      0.     30.23    0.      0.
           0.      0.      0.      0.      0.    -11.18    0.      0.      0.
           0.      0.      0.      0.    -35.84    0.      0.      0.      0.
           0.      0.      0.    -40.88    0.      0.      0.      0.      0.
           0.      0.     60.9     0.      0.      0.      0.      0.      0.
           0.    -73.3     0.      0.      0.      0.      0.      0.      0.
          38.47    0.      0.      0.   ]




**父主题：** [排序组合（ISASI）](atlasascendc_api_07_0220.html)


# ProposalExtract

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x  
Atlas 200I/500 A2 推理产品 | x  
Atlas 推理系列产品AI Core | √  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | √  
  
#### 功能说明

与ProposalConcat功能相反，从Region Proposals内将相应位置的单个元素抽取后重排，每次迭代处理16个Region Proposals，抽取16个元素后连续排列。

#### 函数原型
    
    
    template <typename T>
    __aicore__ inline void ProposalExtract(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const int32_t modeNumber)
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
T | 操作数数据类型。 Atlas 训练系列产品，支持的数据类型为：half Atlas 推理系列产品AI Core，支持的数据类型为：half/float  
  
表2 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
dst | 输出 | 目的操作数。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。  
src | 输入 | 源操作数。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。 源操作数的数据类型需要与目的操作数保持一致。  
repeatTime | 输入 | 重复迭代次数，int32_t类型，每次迭代完成16个Region Proposals的元素抽取并排布到16个元素里，下次迭代跳至相邻的下一组16个Region Proposals和下一组16个元素。取值范围：repeatTime∈[0,255]。  
modeNumber | 输入 | 抽取位置参数，取值范围：modeNumber∈[0, 5]，int32_t类型，仅限于以下配置：

  * 0 – 从x1抽取
  * 1 – 从y1抽取
  * 2 – 从x2抽取
  * 3 – 从y2抽取
  * 4 – 从score抽取
  * 5 – 从label抽取

  
  
#### 返回值说明

无

#### 约束说明

  * 用户需保证src中存储的proposal数目大于等于实际所需数目，否则会存在tensor越界错误。
  * 用户需保证dst中存储的元素大于等于实际所需数目，否则会存在tensor越界错误。
  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。



#### 调用示例

  * 接口使用样例
        
        // repeatTime = 2, modeNumber = 4, 把32个Region Proposal中的score域元素抽取出来排列成32个连续元素
        AscendC::ProposalExtract(dstLocal, srcLocal, 2, 4);
        

  * 完整样例
        
        #include "kernel_operator.h"
        
        class KernelVecProposal {
        public:
            __aicore__ inline KernelVecProposal() {}
            __aicore__ inline void Init(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
            {
                srcGlobal.SetGlobalBuffer((__gm__ half*)src);
                dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
        
                pipe.InitBuffer(inQueueSrc, 1, srcDataSize * sizeof(half));
                pipe.InitBuffer(outQueueDst, 1, dstDataSize * sizeof(half));
            }
            __aicore__ inline void Process()
            {
                CopyIn();
                Compute();
                CopyOut();
            }
        
        private:
            __aicore__ inline void CopyIn()
            {
                AscendC::LocalTensor<half> srcLocal = inQueueSrc.AllocTensor<half>();
                AscendC::DataCopy(srcLocal, srcGlobal, srcDataSize);
                inQueueSrc.EnQue(srcLocal);
            }
            __aicore__ inline void Compute()
            {
                AscendC::LocalTensor<half> srcLocal = inQueueSrc.DeQue<half>();
                AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();
        
                AscendC::ProposalExtract(dstLocal, srcLocal, repeat, mode);
        
                outQueueDst.EnQue<half>(dstLocal);
                inQueueSrc.FreeTensor(srcLocal);
            }
            __aicore__ inline void CopyOut()
            {
                AscendC::LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
                AscendC::DataCopy(dstGlobal, dstLocal, dstDataSize);
                outQueueDst.FreeTensor(dstLocal);
            }
        
        private:
            AscendC::TPipe pipe;
            AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
            AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
            AscendC::GlobalTensor<half> srcGlobal, dstGlobal;
            int srcDataSize = 256;
            int dstDataSize = 32;
            int repeat = srcDataSize / 16;
            int mode = 4;
        };
        
        extern "C" __global__ __aicore__ void vec_proposal_kernel(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
        {
            KernelVecProposal op;
            op.Init(src, dstGm);
            op.Process();
        }
        
        
        示例结果 
        输入数据(src_gm):
        [  0.      0.      0.      0.     33.3     0.      0.      0.      0.
           0.      0.      0.     67.56    0.      0.      0.      0.      0.
           0.      0.     68.5     0.      0.      0.      0.      0.      0.
           0.    -11.914   0.      0.      0.      0.      0.      0.      0.
          25.19    0.      0.      0.      0.      0.      0.      0.    -72.8
           0.      0.      0.      0.      0.      0.      0.     11.79    0.
           0.      0.      0.      0.      0.      0.    -49.47    0.      0.
           0.      0.      0.      0.      0.     49.44    0.      0.      0.
           0.      0.      0.      0.     84.4     0.      0.      0.      0.
           0.      0.      0.    -14.36    0.      0.      0.      0.      0.
           0.      0.     45.97    0.      0.      0.      0.      0.      0.
           0.     52.47    0.      0.      0.      0.      0.      0.      0.
          -5.387   0.      0.      0.      0.      0.      0.      0.    -13.12
           0.      0.      0.      0.      0.      0.      0.    -88.9     0.
           0.      0.      0.      0.      0.      0.     54.      0.      0.
           0.      0.      0.      0.      0.    -51.62    0.      0.      0.
           0.      0.      0.      0.    -20.67    0.      0.      0.      0.
           0.      0.      0.     59.56    0.      0.      0.      0.      0.
           0.      0.     35.72    0.      0.      0.      0.      0.      0.
           0.     -6.12    0.      0.      0.      0.      0.      0.      0.
         -39.4     0.      0.      0.      0.      0.      0.      0.    -11.46
           0.      0.      0.      0.      0.      0.      0.     -7.066   0.
           0.      0.      0.      0.      0.      0.     30.23    0.      0.
           0.      0.      0.      0.      0.    -11.18    0.      0.      0.
           0.      0.      0.      0.    -35.84    0.      0.      0.      0.
           0.      0.      0.    -40.88    0.      0.      0.      0.      0.
           0.      0.     60.9     0.      0.      0.      0.      0.      0.
           0.    -73.3     0.      0.      0.      0.      0.      0.      0.
          38.47    0.      0.      0.   ]
        输出数据(dst_gm):
        [ 33.3    67.56   68.5   -11.914  25.19  -72.8    11.79  -49.47   49.44
          84.4   -14.36   45.97   52.47   -5.387 -13.12  -88.9    54.    -51.62
         -20.67   59.56   35.72   -6.12  -39.4   -11.46   -7.066  30.23  -11.18
         -35.84  -40.88   60.9   -73.3    38.47 ]




**父主题：** [排序组合（ISASI）](atlasascendc_api_07_0220.html)


# RpSort16

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x  
Atlas 200I/500 A2 推理产品 | x  
Atlas 推理系列产品AI Core | √  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | √  
  
#### 功能说明

根据Region Proposals中的score域对其进行排序（score大的排前面），每次排16个Region Proposals。

#### 函数原型
    
    
    template <typename T>
    __aicore__ inline void RpSort16(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t repeatTime)
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
T | 操作数数据类型。 Atlas 训练系列产品，支持的数据类型为：half Atlas 推理系列产品AI Core，支持的数据类型为：half/float  
  
表2 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
dst | 输出 | 目的操作数，存储经过排序后的Region Proposals。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。  
src | 输入 | 源操作数，存储未经过排序的Region Proposals。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。  
repeatTime | 输入 | 重复迭代次数，int32_t类型，每次排16个Region Proposals。取值范围：repeatTime∈[0,255]。  
  
#### 约束说明

  * 用户需保证src和dst中存储的Region Proposal数目大于实际所需数据，否则会存在tensor越界错误。
  * 当存在proposal[i]与proposal[j]的score值相同时，如果i>j，则proposal[j]将首先被选出来，排在前面。
  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。



#### 调用示例

  * 接口使用样例
        
        // repeatTime = 2, 对2个Region Proposal进行排序
        AscendC::RpSort16(dstLocal, dstLocal, 2);
        

  * 完整样例
        
        #include "kernel_operator.h"
        
        class KernelVecProposal {
        public:
            __aicore__ inline KernelVecProposal() {}
            __aicore__ inline void Init(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
            {
                srcGlobal.SetGlobalBuffer((__gm__ half*)src);
                dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
        
                pipe.InitBuffer(inQueueSrc, 1, srcDataSize * sizeof(half));
                pipe.InitBuffer(outQueueDst, 1, dstDataSize * sizeof(half));
            }
            __aicore__ inline void Process()
            {
                CopyIn();
                PreProcess();
                Compute();
                CopyOut();
            }
        
        private:
            __aicore__ inline void CopyIn()
            {
                AscendC::LocalTensor<half> srcLocal = inQueueSrc.AllocTensor<half>();
                AscendC::DataCopy(srcLocal, srcGlobal, srcDataSize);
                inQueueSrc.EnQue(srcLocal);
            }
            __aicore__ inline void PreProcess()
            {
                AscendC::LocalTensor<half> srcLocal = inQueueSrc.DeQue<half>();
                AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();
                AscendC::ProposalConcat(dstLocal, srcLocal, repeat, mode); // sort排序是基于score的，此处先创建一个有score数据的proposal，需要注意的是，非score处的数据可能是随机值
                outQueueDst.EnQue<half>(dstLocal);
                inQueueSrc.FreeTensor(srcLocal);
            }
            __aicore__ inline void Compute()
            {
                AscendC::LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
                AscendC::RpSort16(dstLocal, dstLocal, repeat);
                outQueueDst.EnQue<half>(dstLocal);
            }
            __aicore__ inline void CopyOut()
            {
                AscendC::LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
                AscendC::DataCopy(dstGlobal, dstLocal, dstDataSize);
                outQueueDst.FreeTensor(dstLocal);
            }
        
        private:
            AscendC::TPipe pipe;
            AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
            AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
            AscendC::GlobalTensor<half> srcGlobal, dstGlobal;
            int srcDataSize = 32;
            int dstDataSize = 256;
            int repeat = srcDataSize / 16;
            int mode = 4;
        };
        
        extern "C" __global__ __aicore__ void vec_proposal_kernel(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
        {
            KernelVecProposal op;
            op.Init(src, dstGm);
            op.Process();
        }
        
        
        示例结果
        输入数据(src_gm):
        [ -1.624 -42.3   -54.12   91.25  -99.4    36.72   67.44  -66.3   -52.53
           3.377 -62.47  -15.85  -31.47    3.143  58.47  -83.75 21.58   63.47    
           7.234  35.16  -39.72   37.8    73.06  -98.7    44.1 -77.2    67.2    
           19.62  -87.9   -14.875  15.86  -77.75]
        输出数据(dst_gm):
        [  0.      0.      0.      0.     91.25    0.      0.      0.      0.
           0.      0.      0.     67.44    0.      0.      0.      0.      0.
           0.      0.     58.47    0.      0.      0.      0.      0.      0.
           0.     36.72    0.      0.      0.      0.      0.      0.      0.
           3.377   0.      0.      0.      0.      0.      0.      0.      3.143
           0.      0.      0.      0.      0.      0.      0.     -1.624   0.
           0.      0.      0.      0.      0.      0.    -15.85    0.      0.
           0.      0.      0.      0.      0.    -31.47    0.      0.      0.
           0.      0.      0.      0.    -42.3     0.      0.      0.      0.
           0.      0.      0.    -52.53    0.      0.      0.      0.      0.
           0.      0.    -54.12    0.      0.      0.      0.      0.      0.
           0.    -62.47    0.      0.      0.      0.      0.      0.      0.
         -66.3     0.      0.      0.      0.      0.      0.      0.    -83.75
           0.      0.      0.      0.      0.      0.      0.    -99.4     0.
           0.      0.      0.      0.      0.      0.     73.06    0.      0.      
           0.      0.      0.      0.      0.     67.2     0.      0.      0.      
           0.      0.      0.      0.     63.47    0.      0.      0.      0.      
           0.      0.      0.     44.1     0.      0.      0.      0.      0.      
           0.      0.     37.8     0.      0.      0.      0.      0.      0.      
           0.     35.16    0.      0.      0.      0.      0.      0.      0.     
          21.58    0.      0.      0.      0.      0.      0.      0.     19.62    
           0.      0.      0.      0.      0.      0.      0.     15.86    0.      
           0.      0.      0.      0.      0.      0.      7.234   0.      0.      
           0.      0.      0.      0.      0.    -14.875   0.      0.      0.      
           0.      0.      0.      0.    -39.72    0.      0.      0.      0.      
           0.      0.      0.    -77.2     0.      0.      0.      0.      0.      
           0.      0.    -77.75    0.      0.      0.      0.      0.      0.      
           0.    -87.9     0.      0.      0.      0.      0.      0.      0.    
         -98.7     0.      0.      0.   ]




**父主题：** [排序组合（ISASI）](atlasascendc_api_07_0220.html)


# MrgSort4

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x  
Atlas 200I/500 A2 推理产品 | x  
Atlas 推理系列产品AI Core | √  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | √  
  
#### 功能说明

将已经排好序的最多4条Region Proposals队列，排列并合并成1条队列，结果按照score域由大到小排序。

#### 函数原型
    
    
    template <typename T>
    __aicore__ inline void MrgSort4(const LocalTensor<T>& dst, const MrgSortSrcList<T>& src, const MrgSort4Info& params)
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
T | 操作数数据类型。 Atlas 训练系列产品，支持的数据类型为：half Atlas 推理系列产品AI Core，支持的数据类型为：half/float  
  
表2 接口参数说明

展开

参数名 | 输入/输出 | 含义  
---|---|---  
dst | 输出 | 目的操作数，存储经过排序后的Region Proposals。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要保证16字节对齐（针对half数据类型），32字节对齐（针对float数据类型）。  
src | 输入 | 源操作数，4个Region Proposals队列，并且每个Region Proposal队列都已经排好序，类型为MrgSortSrcList结构体，具体定义如下：
    
    
    template <typename T> struct MrgSortSrcList {
        __aicore__ MrgSortSrcList() {}
        __aicore__ MrgSortSrcList(const LocalTensor<T>& src1In, const LocalTensor<T>& src2In, const LocalTensor<T>& src3In,
            const LocalTensor<T>& src4In)
        {
            src1 = src1In[0];
            src2 = src2In[0];
            src3 = src3In[0];
            src4 = src4In[0];
        }
        LocalTensor<T> src1; // 第一个已经排好序的Region Proposals队列
        LocalTensor<T> src2; // 第二个已经排好序的Region Proposals队列
        LocalTensor<T> src3; // 第三个已经排好序的Region Proposals队列
        LocalTensor<T> src4; // 第四个已经排好序的Region Proposals队列
    };
    

Region Proposal队列的数据类型与目的操作数保持一致。src1、src2、src3、src4类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要保证16字节对齐（针对half数据类型），32字节对齐（针对float数据类型）。  
params | 输入 | 排序所需参数，类型为MrgSort4Info结构体。 具体定义请参考${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_proposal.h，${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。 参数说明请参考[表3](#ZH-CN_TOPIC_0000002552120505__table7515358184615)。  
  
表3 MrgSort4Info参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
elementLengths | 输入 | 四个源Region Proposals队列的长度（Region Proposal数目），类型为长度为4的uint16_t数据类型的数组，理论上每个元素取值范围[0, 4095]，但不能超出UB的存储空间。  
ifExhaustedSuspension | 输入 | 某条队列耗尽后，指令是否需要停止，类型为bool，默认false。  
validBit | 输入 | 有效队列个数，取值如下：

  * 3：前两条队列有效
  * 7：前三条队列有效
  * 15：四条队列全部有效

  
repeatTimes | 输入 | 迭代次数，每一次源操作数和目的操作数跳过四个队列总长度。取值范围：repeatTimes∈[1,255]。 repeatTimes参数生效是有条件的，需要同时满足以下四个条件：

  * 四个源Region Proposals队列的长度一致
  * 四个源Region Proposals队列连续存储
  * ifExhaustedSuspension = False
  * validBit=15

  
  
#### 约束说明

  * 当存在proposal[i]与proposal[j]的score值相同时，如果i>j，则proposal[j]将首先被选出来，排在前面。
  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。


  * 不支持源操作数与目的操作数之间存在地址重叠。



#### 调用示例

  * 接口使用样例
        
        // vconcatWorkLocal为已经创建并且完成排序的4个Region Proposals，每个Region Proposal数目是16个
        struct MrgSortSrcList<half> srcList(vconcatWorkLocal[0], vconcatWorkLocal[1], vconcatWorkLocal[2], vconcatWorkLocal[3]);
        uint16_t elementLengths[4] = {16, 16, 16, 16};
        struct MrgSort4Info srcInfo(elementLengths, false, 15, 1);
        AscendC::MrgSort4(dstLocal, srcList, srcInfo);
        



  * 完整样例
        
        #include "kernel_operator.h"
        
        class KernelVecProposal {
        public:
            __aicore__ inline KernelVecProposal() {}
            __aicore__ inline void Init(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
            {
                srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(src), srcDataSize);
                dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
        
                pipe.InitBuffer(inQueueSrc, 1, srcDataSize * sizeof(half));
                pipe.InitBuffer(workQueue, 1, dstDataSize * sizeof(half));
                pipe.InitBuffer(outQueueDst, 1, dstDataSize * sizeof(half));
            }
            __aicore__ inline void Process()
            {
                CopyIn();
                Compute();
                CopyOut();
            }
        
        private:
            __aicore__ inline void CopyIn()
            {
                AscendC::LocalTensor<half> srcLocal = inQueueSrc.AllocTensor<half>();
                AscendC::DataCopy(srcLocal, srcGlobal, srcDataSize);
                inQueueSrc.EnQue(srcLocal);
            }
            __aicore__ inline void Compute()
            {
                AscendC::LocalTensor<half> srcLocal = inQueueSrc.DeQue<half>();
                AscendC::LocalTensor<half> vconcatWorkLocal = workQueue.AllocTensor<half>();
                AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();
        
                // 先构造4个region proposal然后进行合并排序
                AscendC::ProposalConcat(vconcatWorkLocal[0], srcLocal[0], repeat, mode);
                AscendC::RpSort16(vconcatWorkLocal[0], vconcatWorkLocal[0], repeat);
        
                AscendC::ProposalConcat(vconcatWorkLocal[workDataSize], srcLocal[singleDataSize], repeat, mode);
                AscendC::RpSort16(vconcatWorkLocal[workDataSize], vconcatWorkLocal[workDataSize], repeat);
        
                AscendC::ProposalConcat(vconcatWorkLocal[workDataSize * 2], srcLocal[singleDataSize * 2], repeat, mode);
                AscendC::RpSort16(vconcatWorkLocal[workDataSize * 2], vconcatWorkLocal[workDataSize * 2], repeat);
        
                AscendC::ProposalConcat(vconcatWorkLocal[workDataSize * 3], srcLocal[singleDataSize * 3], repeat, mode);
                AscendC::RpSort16(vconcatWorkLocal[workDataSize * 3], vconcatWorkLocal[workDataSize * 3], repeat);
        
                AscendC::MrgSortSrcList<half> srcList(vconcatWorkLocal[0], vconcatWorkLocal[workDataSize],
                    vconcatWorkLocal[workDataSize * 2], vconcatWorkLocal[workDataSize * 3]);
                uint16_t elementLengths[4] = {singleDataSize, singleDataSize, singleDataSize, singleDataSize};
                AscendC::MrgSort4Info srcInfo(elementLengths, false, 15, 1);
                AscendC::MrgSort4(dstLocal, srcList, srcInfo);
        
                outQueueDst.EnQue<half>(dstLocal);
                inQueueSrc.FreeTensor(srcLocal);
                workQueue.FreeTensor(vconcatWorkLocal);
            }
            __aicore__ inline void CopyOut()
            {
                AscendC::LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
                AscendC::DataCopy(dstGlobal, dstLocal, dstDataSize);
                outQueueDst.FreeTensor(dstLocal);
            }
        
        private:
            AscendC::TPipe pipe;
            AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
            AscendC::TQue<AscendC::TPosition::VECIN, 1> workQueue;
            AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
            AscendC::GlobalTensor<half> srcGlobal, dstGlobal;
        
            int srcDataSize = 64;
            uint16_t singleDataSize = srcDataSize / 4;
            int dstDataSize = 512;
            int workDataSize = dstDataSize / 4;
            int repeat = srcDataSize / 4 / 16;
            int mode = 4;
        };
        
        extern "C" __global__ __aicore__ void vec_proposal_kernel(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
        {
            KernelVecProposal op;
            op.Init(src, dstGm);
            op.Process();
        }
        
        
        示例结果
        输入数据(src_gm):
        [-38.1    82.7   -40.75  -54.62   21.67  -58.53   25.94  -79.5   -61.44
          26.7   -27.45   48.78   86.75  -18.1   -58.8    62.38   46.38  -78.94
         -87.7   -13.81  -13.25   46.94  -47.8   -50.44   34.16   20.3    80.1
         -94.1    52.4   -42.75   83.4    80.44  -66.8   -82.7   -91.44  -95.6
          66.2   -30.97  -36.53   61.66   24.92  -45.1    38.97  -34.62  -69.8
          59.1    34.22   11.695 -33.47   52.1    -4.832  46.88   56.78   71.4
          13.29  -35.78   52.44  -46.03   83.8    83.56   71.3    -9.086 -65.06
          46.25 ]
        输出数据(dst_gm):
        [  0.      0.      0.      0.     86.75    0.      0.      0.      0.
           0.      0.      0.     83.8     0.      0.      0.      0.      0.
           0.      0.     83.56    0.      0.      0.      0.      0.      0.
           0.     83.4     0.      0.      0.      0.      0.      0.      0.
          82.7     0.      0.      0.      0.      0.      0.      0.     80.44
           0.      0.      0.      0.      0.      0.      0.     80.1     0.
           0.      0.      0.      0.      0.      0.     71.4     0.      0.
           0.      0.      0.      0.      0.     71.3     0.      0.      0.
           0.      0.      0.      0.     66.2     0.      0.      0.      0.
           0.      0.      0.     62.38    0.      0.      0.      0.      0.
           0.      0.     61.66    0.      0.      0.      0.      0.      0.
           0.     59.1     0.      0.      0.      0.      0.      0.      0.
          56.78    0.      0.      0.      0.      0.      0.      0.     52.44
           0.      0.      0.      0.      0.      0.      0.     52.4     0.
           0.      0.      0.      0.      0.      0.     52.1     0.      0.
           0.      0.      0.      0.      0.     48.78    0.      0.      0.
           0.      0.      0.      0.     46.94    0.      0.      0.      0.
           0.      0.      0.     46.88    0.      0.      0.      0.      0.
           0.      0.     46.38    0.      0.      0.      0.      0.      0.
           0.     46.25    0.      0.      0.      0.      0.      0.      0.
          38.97    0.      0.      0.      0.      0.      0.      0.     34.22
           0.      0.      0.      0.      0.      0.      0.     34.16    0.
           0.      0.      0.      0.      0.      0.     26.7     0.      0.
           0.      0.      0.      0.      0.     25.94    0.      0.      0.
           0.      0.      0.      0.     24.92    0.      0.      0.      0.
           0.      0.      0.     21.67    0.      0.      0.      0.      0.
           0.      0.     20.3     0.      0.      0.      0.      0.      0.
           0.     13.29    0.      0.      0.      0.      0.      0.      0.
          11.695   0.      0.      0.      0.      0.      0.      0.     -4.832
           0.      0.      0.      0.      0.      0.      0.     -9.086   0.
           0.      0.      0.      0.      0.      0.    -13.25    0.      0.
           0.      0.      0.      0.      0.    -13.81    0.      0.      0.
           0.      0.      0.      0.    -18.1     0.      0.      0.      0.
           0.      0.      0.    -27.45    0.      0.      0.      0.      0.
           0.      0.    -30.97    0.      0.      0.      0.      0.      0.
           0.    -33.47    0.      0.      0.      0.      0.      0.      0.
         -34.62    0.      0.      0.      0.      0.      0.      0.    -35.78
           0.      0.      0.      0.      0.      0.      0.    -36.53    0.
           0.      0.      0.      0.      0.      0.    -38.1     0.      0.
           0.      0.      0.      0.      0.    -40.75    0.      0.      0.
           0.      0.      0.      0.    -42.75    0.      0.      0.      0.
           0.      0.      0.    -45.1     0.      0.      0.      0.      0.
           0.      0.    -46.03    0.      0.      0.      0.      0.      0.
           0.    -47.8     0.      0.      0.      0.      0.      0.      0.
         -50.44    0.      0.      0.      0.      0.      0.      0.    -54.62
           0.      0.      0.      0.      0.      0.      0.    -58.53    0.
           0.      0.      0.      0.      0.      0.    -58.8     0.      0.
           0.      0.      0.      0.      0.    -61.44    0.      0.      0.
           0.      0.      0.      0.    -65.06    0.      0.      0.      0.
           0.      0.      0.    -66.8     0.      0.      0.      0.      0.
           0.      0.    -69.8     0.      0.      0.      0.      0.      0.
           0.    -78.94    0.      0.      0.      0.      0.      0.      0.
         -79.5     0.      0.      0.      0.      0.      0.      0.    -82.7
           0.      0.      0.      0.      0.      0.      0.    -87.7     0.
           0.      0.      0.      0.      0.      0.    -91.44    0.      0.
           0.      0.      0.      0.      0.    -94.1     0.      0.      0.
           0.      0.      0.      0.    -95.6     0.      0.      0.   ]




**父主题：** [排序组合（ISASI）](atlasascendc_api_07_0220.html)


# Sort32

#### 产品支持情况

展开

产品 |  是否支持  
---|---  
Atlas A3 训练系列产品 / Atlas A3 推理系列产品  |  √  
Atlas A2 训练系列产品 / Atlas A2 推理系列产品  |  √  
Atlas 200I/500 A2 推理产品  |  √  
Atlas 推理系列产品 AI Core |  x  
Atlas 推理系列产品 Vector Core |  x  
Atlas 训练系列产品  |  x  
  
#### 功能说明

排序函数，一次迭代可以完成32个数的排序，数据需要按如下描述结构进行保存：

score和index分别存储在src0和src1中，按score进行排序（score大的排前面），排序好的score与其对应的index一起以（score, index）的结构存储在dst中。不论score为half还是float类型，dst中的（score, index）结构总是占据8Bytes空间。

如下所示：

  * 当score为float，index为uint32_t类型时，计算结果中index存储在高4Bytes，score存储在低4Bytes。 

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002521042288.png)

  * 当score为half，index为uint32_t类型时，计算结果中index存储在高4Bytes，score存储在低2Bytes， 中间的2Bytes保留。 

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002520882290.png)




#### 函数原型
    
    
    template <typename T>
    __aicore__ inline void Sort32(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<uint32_t>& src1, const int32_t repeatTime)
    

#### 参数说明

表1 模板参数说明

展开

参数名 |  描述  
---|---  
T |  操作数数据类型。 Atlas A3 训练系列产品 / Atlas A3 推理系列产品 ，支持的数据类型为：half/float Atlas A2 训练系列产品 / Atlas A2 推理系列产品 ，支持的数据类型为：half/float Atlas 200I/500 A2 推理产品 ，支持的数据类型为：half/float  
  
表2 参数说明

展开

参数名称 |  输入/输出 |  含义  
---|---|---  
dst |  输出 |  目的操作数。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。  
src0 |  输入 |  源操作数。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。 此源操作数的数据类型需要与目的操作数保持一致。  
src1 |  输入 |  源操作数。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。 此源操作数固定为uint32_t数据类型。  
repeatTime |  输入 |  重复迭代次数，int32_t类型，每次迭代完成32个元素的排序，下次迭代src0和src1各跳过32个elements，dst跳过32*8 Byte空间。取值范围：repeatTime∈[0,255]。  
  
#### 返回值说明

无

#### 约束说明

  * 当存在score[i]与score[j]相同时，如果i>j，则score[j]将首先被选出来，排在前面。
  * 每次迭代内的数据会进行排序，不同迭代间的数据不会进行排序。
  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。



#### 调用示例

  * 接口使用样例 
        
        AscendC::LocalTensor<float> srcLocal0 = inQueueSrc0.DeQue<float>();
        AscendC::LocalTensor<uint32_t> srcLocal1 = inQueueSrc1.DeQue<uint32_t>();
        AscendC::LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();
        // repeatTime = 4, 对128个数分成4组进行排序，每次完成1组32个数的排序
        AscendC::Sort32<float>(dstLocal, srcLocal0, srcLocal1, 4);
        outQueueDst.EnQue<float>(dstLocal);
        inQueueSrc0.FreeTensor(srcLocal0);
        inQueueSrc1.FreeTensor(srcLocal1);
        




**父主题：** [排序组合（ISASI）](atlasascendc_api_07_0220.html)


# MrgSort

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | x  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

将已经排好序的最多4条队列，合并排列成1条队列，结果按照score域由大到小排序。

MrgSort指令处理的数据一般是经过Sort32指令处理后的数据，也就是Sort32指令的输出，队列的结构如下所示：

  * 数据类型为float，每个结构占据8Bytes。

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002552081265.png)

  * 数据类型为half，每个结构也占据8Bytes，中间有2Bytes保留。

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002521041288.png)




#### 函数原型
    
    
    template <typename T>
    __aicore__ inline void MrgSort(const LocalTensor<T>& dst, const MrgSortSrcList<T>& src, const MrgSort4Info& params)
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
T | Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half/float Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half/float Atlas 200I/500 A2 推理产品，支持的数据类型为：half/float  
  
表2 接口参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
dst | 输出 | 目的操作数，存储经过排序后的数据。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。  
src | 输入 | 源操作数，4个队列，并且每个队列都已经排好序，类型为MrgSortSrcList结构体，定义如下：
    
    
    template <typename T> struct MrgSortSrcList {
        __aicore__ MrgSortSrcList() {}
        __aicore__ MrgSortSrcList(const LocalTensor<T>& src1In, const LocalTensor<T>& src2In, const LocalTensor<T>& src3In,
            const LocalTensor<T>& src4In)
        {
            src1 = src1In[0];
            src2 = src2In[0];
            src3 = src3In[0];
            src4 = src4In[0];
        }
        LocalTensor<T> src1; // 第一个已经排好序的队列
        LocalTensor<T> src2; // 第二个已经排好序的队列
        LocalTensor<T> src3; // 第三个已经排好序的队列
        LocalTensor<T> src4; // 第四个已经排好序的队列
    };
    

源操作数的数据类型与目的操作数保持一致。src1、src2、src3、src4类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。LocalTensor的起始地址需要8字节对齐。  
params | 输入 | 排序所需参数，类型为MrgSort4Info结构体。 具体定义请参考${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_proposal.h，${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。 参数说明请参考[表3](#ZH-CN_TOPIC_0000002552079899__table7515358184615)。  
  
表3 MrgSort4Info参数说明

展开

参数名称 | 含义  
---|---  
elementLengths | 四个源队列的长度（8Bytes结构的数目），类型为长度为4的uint16_t数据类型的数组，理论上每个元素取值范围[0, 4095]，但不能超出UB的存储空间。  
ifExhaustedSuspension | 某条队列耗尽后，指令是否需要停止，类型为bool，默认false。  
validBit | 有效队列个数，取值如下：

  * 3：前两条队列有效
  * 7：前三条队列有效
  * 15：四条队列全部有效

  
repeatTimes | 迭代次数，每一次源操作数和目的操作数跳过四个队列总长度。取值范围：repeatTimes∈[1,255]。 repeatTimes参数生效是有条件的，需要同时满足以下四个条件：

  * src包含四条队列并且validBit=15
  * 四个源队列的长度一致
  * 四个源队列连续存储
  * ifExhaustedSuspension = False

  
  
#### 返回值说明

无

#### 约束说明

  * 当存在score[i]与score[j]相同时，如果i>j，则score[j]将首先被选出来，排在前面。
  * 每次迭代内的数据会进行排序，不同迭代间的数据不会进行排序。
  * 需要注意此函数排序的队列非region proposal结构。
  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。



#### 调用示例

  * 接口使用样例
        
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
        pipe.InitBuffer(outQueueDst, 1, dstDataSize * sizeof(float));
        AscendC::LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();
        // 对8个已排好序的队列进行合并排序，repeatTimes = 2，数据连续存放
        // 每个队列包含32个(score,index)的8Bytes结构
        // 最后输出对score域的256个数完成排序后的结果
        AscendC::MrgSort4Info params;
        params.elementLengths[0] = 32;
        params.elementLengths[1] = 32;
        params.elementLengths[2] = 32;
        params.elementLengths[3] = 32;
        params.ifExhaustedSuspension = false;
        params.validBit = 0b1111;
        params.repeatTimes = 2;
        
        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = workLocal[0];
        srcList.src2 = workLocal[64]; // workLocal为float类型，每个队列占据256Bytes空间
        srcList.src3 = workLocal[128];
        srcList.src4 = workLocal[192];
        
        AscendC::MrgSort<float>(dstLocal, srcList, params);
        outQueueDst.EnQue<float>(dstLocal);
        outQueueDst.FreeTensor(dstLocal);
        




**父主题：** [排序组合（ISASI）](atlasascendc_api_07_0220.html)


# GetMrgSortResult

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | √  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

获取MrgSort已经处理过的队列里的Region Proposal个数，并依次存储在四个出参中。

本接口和MrgSort相关指令的配合关系如下：

  * 配合[MrgSort4指令](atlasascendc_api_07_0230.html)使用，获取MrgSort4指令处理过的队列里的Region Proposal个数。使用时，需要将MrgSort4中的MrgSort4Info.ifExhaustedSuspension参数配置为true，该配置模式下某条队列耗尽后，MrgSort4指令即停止。

以上说明适用于如下型号：

Atlas 推理系列产品AI Core

  * 配合[MrgSort指令](atlasascendc_api_07_0232.html)使用，获取MrgSort指令处理过的队列里的Region Proposal个数。使用时，需要将MrgSort中的MrgSort4Info.ifExhaustedSuspension参数配置为true，该配置模式下某条队列耗尽后，MrgSort指令即停止。

以上说明适用于如下型号：

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品

Atlas 200I/500 A2 推理产品




#### 函数原型
    
    
    __aicore__ inline void GetMrgSortResult(uint16_t &mrgSortList1, uint16_t &mrgSortList2, uint16_t &mrgSortList3, uint16_t &mrgSortList4)
    

#### 参数说明

表1 参数说明

展开

参数名 | 输入/输出 | 描述  
---|---|---  
mrgSortList1 | 输出 | 类型为uint16_t，表示MrgSort第一个队列里已经处理过的Region Proposal个数。  
mrgSortList2 | 输出 | 类型为uint16_t，表示MrgSort第二个队列里已经处理过的Region Proposal个数。  
mrgSortList3 | 输出 | 类型为uint16_t，表示MrgSort第三个队列里已经处理过的Region Proposal个数。  
mrgSortList4 | 输出 | 类型为uint16_t，表示MrgSort第四个队列里已经处理过的Region Proposal个数。  
  
#### 返回值说明

无

#### 约束说明

无

#### 调用示例

  * 配合[MrgSort指令](atlasascendc_api_07_0232.html)使用示例。
        
        AscendC::LocalTensor<float> dstLocal;
        AscendC::LocalTensor<float> workLocal;
        AscendC::LocalTensor<float> src0Local;
        AscendC::LocalTensor<uint32_t> src1Local;
        
        AscendC::Sort32(workLocal, src0Local, src1Local, 1);
        
        uint16_t elementLengths[4] = { 0 };
        uint32_t sortedNum[4] = { 0 };
        elementLengths[0] = 32;
        elementLengths[1] = 32;
        elementLengths[2] = 32;
        elementLengths[3] = 32;
        uint16_t validBit = 0b1111;
        
        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = workLocal[0];
        srcList.src2 = workLocal[32 * 1 * 2];
        srcList.src3 = workLocal[32 * 2 * 2];
        srcList.src4 = workLocal[32 * 3 * 2];
        
        AscendC::MrgSort4Info mrgSortInfo(elementLengths, true, validBit, 1);
        AscendC::MrgSort(dstLocal, srcList, mrgSortInfo);
        
        uint16_t mrgRes1 = 0;
        uint16_t mrgRes2 = 0;
        uint16_t mrgRes3 = 0;
        uint16_t mrgRes4 = 0;
        AscendC::GetMrgSortResult(mrgRes1, mrgRes2, mrgRes3, mrgRes4);
        

  * 配合[MrgSort4指令](atlasascendc_api_07_0230.html)使用示例。
        
        AscendC::LocalTensor<float> dstLocal;
        AscendC::LocalTensor<float> workLocal;
        AscendC::LocalTensor<float> src0Local;
        
        AscendC::RpSort16(workLocal, src0Local, 1);
        
        uint16_t elementLengths[4] = { 0 };
        uint32_t sortedNum[4] = { 0 };
        elementLengths[0] = 32;
        elementLengths[1] = 32;
        elementLengths[2] = 32;
        elementLengths[3] = 32;
        uint16_t validBit = 0b1111;
        
        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = workLocal[0];
        srcList.src2 = workLocal[32 * 1 * 2];
        srcList.src3 = workLocal[32 * 2 * 2];
        srcList.src4 = workLocal[32 * 3 * 2];
        
        AscendC::MrgSort4Info mrgSortInfo(elementLengths, true, validBit, 1);
        AscendC::MrgSort4(dstLocal, srcList, mrgSortInfo);
        
        uint16_t mrgRes1 = 0;
        uint16_t mrgRes2 = 0;
        uint16_t mrgRes3 = 0;
        uint16_t mrgRes4 = 0;
        AscendC::GetMrgSortResult(mrgRes1, mrgRes2, mrgRes3, mrgRes4);
        




**父主题：** [排序组合（ISASI）](atlasascendc_api_07_0220.html)



---

## 数据搬运ISASI


# Gatherb(ISASI)

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | x  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

给定一个输入的张量和一个地址偏移张量，本接口根据偏移地址按照DataBlock的粒度将输入张量收集到结果张量中。

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002552081123.png)

#### 函数原型
    
    
    template <typename T>
    __aicore__ inline void Gatherb(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<uint32_t>& offset, const uint8_t repeatTime, const GatherRepeatParams& repeatParams)
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
T | 操作数数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：uint16_t/uint32_t Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：uint16_t/uint32_t Atlas 200I/500 A2 推理产品，支持的数据类型为：int8_t/uint8_t/int16_t/uint16_t/half/float/int32_t/uint32_t/bfloat16_t/int64_t  
  
表2 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
dst | 输出 | 目的操作数。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。  
src0 | 输入 | 源操作数。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。 源操作数的数据类型需要与目的操作数保持一致。  
offset | 输入 | 每个datablock在源操作数中对应的地址偏移。 类型为[LocalTensor](atlasascendc_api_07_0006.html)，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。 该偏移量是相对于src0的基地址而言的。每个元素值要大于等于0，单位为字节；且需要保证偏移后的地址满足32字节对齐。  
repeatTime | 输入 | 重复迭代次数，每次迭代完成8个datablock的数据收集，数据范围：repeatTime∈（0,255]。  
repeatParams | 输入 | 用于控制指令迭代的相关参数。 类型为GatherRepeatParams，具体定义可参考${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_gather.h。${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。 其中dstBlkStride、dstRepStride支持用户配置，参数说明参考[表3](atlasascendc_api_07_0234.html#ZH-CN_TOPIC_0000002552119775__table2166248155314)。  
  
表3 GatherRepeatParams结构体参数说明

展开

参数名称 | 含义  
---|---  
dstBlkStride | 单次迭代内，矢量目的操作数不同datablock间的地址步长。  
dstRepStride | 相邻迭代间，矢量目的操作数相同datablock间的地址步长。  
blockNumber | 预留参数。为后续的功能做保留，开发者暂时无需关注，使用默认值即可。  
src0BlkStride  
src1BlkStride  
src0RepStride  
src1RepStride  
repeatStrideMode  
strideSizeMode  
  
#### 约束说明

无

#### 调用示例
    
    
    #include "kernel_operator.h"
    AscendC::TPipe tpipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> vecIn;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> vecOffset;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> vecOut;
    
    uint32_t bufferLen = 0;
    
    uint32_t len = 128;
    bufferLen = len;
    tpipe.InitBuffer(vecIn, 2, bufferLen * sizeof(uint16_t));
    tpipe.InitBuffer(vecOffset, 2, 8 * sizeof(uint32_t));
    tpipe.InitBuffer(vecOut, 2, bufferLen * sizeof(uint16_t));
    
    auto x_buf = vecIn.AllocTensor<uint16_t>();
    auto offset_buf = vecOffset.AllocTensor<uint32_t>();
    AscendC::DataCopy(x_buf, x_gm[index * bufferLen], bufferLen);
    AscendC::DataCopy(offset_buf, offset_gm[0], 8);
    vecIn.EnQue(x_buf);
    vecOffset.EnQue(offset_buf);
    
    auto y_buf = vecOut.DeQue<uint16_t>();
    AscendC::DataCopy(y_gm[index * bufferLen], y_buf, bufferLen);
    vecOut.FreeTensor(y_buf);
    
    auto x_buf = vecIn.DeQue<uint16_t>();
    auto offset_buf = vecOffset.DeQue<uint32_t>();
    auto y_buf = vecOut.AllocTensor<uint16_t>();
    AscendC::GatherRepeatParams params{1, 8};
    uint8_t repeatTime = bufferLen * sizeof(uint16_t) / 256;
    AscendC::Gatherb<uint16_t>(y_buf, x_buf, offset_buf, repeatTime, params);
    vecIn.FreeTensor(x_buf);
    vecOffset.FreeTensor(offset_buf);
    vecOut.EnQue(y_buf);
    

结果示例： 
    
    
    输入数据(offsetLocal): [224 192 160 128 96 64 32 0]
    输入数据(srcLocal): [0 1 2 3 4 5 6 7 ... 120 121 122 123 124 125 126 127]
    输出数据(dstGlobal):[
    112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 
    96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111
    ... 
    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    ]

**父主题：** [离散与聚合](atlasascendc_api_07_0091.html)


# Scatter(ISASI)

#### 产品支持情况

展开

产品 |  是否支持  
---|---  
Atlas A3 训练系列产品 / Atlas A3 推理系列产品  |  x  
Atlas A2 训练系列产品 / Atlas A2 推理系列产品  |  x  
Atlas 200I/500 A2 推理产品  |  √  
Atlas 推理系列产品 AI Core |  √  
Atlas 推理系列产品 Vector Core |  x  
Atlas 训练系列产品  |  x  
  
说明

该API不支持 Atlas A2 训练系列产品 / Atlas A2 推理系列产品 、 Atlas A3 训练系列产品 / Atlas A3 推理系列产品 ，如果需要在上述AI处理器实现数据离散功能，建议参考[Scatter兼容样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/3_libraries/0_scatter_kernellaunch)进行适配。

#### 功能说明

给定一个连续的输入张量和一个目的地址偏移张量，Scatter指令根据偏移地址生成新的结果张量后将输入张量分散到结果张量中。

将源操作数src中的元素按照指定的位置（由dst_offset和base_addr共同作用）分散到目的操作数dst中。

#### 函数原型

  * tensor前n个数据计算 
        
        template <typename T>
        __aicore__ inline void Scatter(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<uint32_t>& dstOffset, const uint32_t dstBaseAddr, const uint32_t count)
        

  * tensor高维切分计算 
    * mask逐bit模式 
          
          template <typename T>
          __aicore__ inline void Scatter(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<uint32_t>& dstOffset, const uint32_t dstBaseAddr, const uint64_t mask[], const uint8_t repeatTime, const uint8_t srcRepStride)
          

    * mask连续模式 
          
          template <typename T>
          __aicore__ inline void Scatter(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<uint32_t>& dstOffset, const uint32_t dstBaseAddr, const uint64_t mask, const uint8_t repeatTime, const uint8_t srcRepStride)
          




#### 参数说明

表1 模板参数说明

展开

参数名 |  描述  
---|---  
T |  操作数数据类型。 Atlas 200I/500 A2 推理产品 ，支持的数据类型为：uint8_t/int8_t/uint16_t/int16_t/half/uint32_t/int32_t/float Atlas 推理系列产品 AI Core，支持的数据类型为：uint16_t/uint32_t/float/half  
  
表2 参数说明

展开

参数名称 |  输入/输出 |  含义  
---|---|---  
dst |  输出 |  目的操作数，类型为LocalTensor。LocalTensor的起始地址需要32字节对齐。  
src |  输入 |  源操作数，类型为LocalTensor。数据类型需与dst保持一致。  
dstOffset |  输入 |  用于存储源操作数的每个元素在dst中对应的地址偏移。偏移基于dst的基地址dstBaseAddr计算，以字节为单位，取值应保证按dst数据类型位宽对齐，否则会导致非预期行为。 针对以下型号，地址偏移的取值范围不超出uint32_t的范围即可。 Atlas 推理系列产品 AI Core 针对以下型号，地址偏移的取值范围如下：当操作数为8位时，取值范围为[0, 216-1]；当操作数为16位时，取值范围为[0, 217-1]，当操作数为32位或者64位时，不超过uint32_t的范围即可。超出取值范围可能导致非预期输出。 Atlas 200I/500 A2 推理产品   
dstBaseAddr |  输入 |  dst的起始偏移地址，单位是字节。取值应保证按dst数据类型位宽对齐，否则会导致非预期行为。  
count |  输入 |  执行处理的数据个数。  
mask/mask[] |  输入 |  [mask](/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_0022.html#ZH-CN_TOPIC_0000002552129981__zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section4252658182)用于控制每次迭代内参与计算的元素。

  * 连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为8位或16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。


  * 逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。参数类型为长度为2的uint64_t类型数组。 例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。 参数取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为8位或16位时，mask[0]、mask[1]∈[0, 264-1]并且不同时为0；当操作数为32位时，mask[1]为0，mask[0]∈(0, 264-1]；当操作数为64位时，mask[1]为0，mask[0]∈(0, 232-1]。

  
repeatTime |  输入 |  指令迭代次数，每次迭代完成8个datablock的数据收集，数据范围：repeatTime∈[0,255]。 特别地，针对以下型号： 

  * Atlas 200I/500 A2 推理产品 

操作数为**8位** 时，每次迭代完成**4个datablock** （32Bytes）的数据收集。  
srcRepStride |  输入 |  相邻迭代间的地址步长，单位是datablock。  
  
#### 约束说明

  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。
  * 操作数地址重叠约束请参考[通用地址重叠约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section668772811100)。


  * dstOffset中的偏移地址不能有相同值，如果存在2个或者多个偏移重复的情况，行为是不可预期的。



#### 调用示例
    
    
    #include "kernel_operator.h"
    AscendC::TPipe m_pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queCalc;
    AscendC::GlobalTensor<T> m_valueGlobal;
    uint32_t m_concatRepeatTimes;
    uint32_t m_sortRepeatTimes;
    uint32_t m_extractRepeatTimes;
    uint32_t m_elementCount;
    AscendC::GlobalTensor<uint32_t> m_dstOffsetGlobal;
    AscendC::GlobalTensor<T> m_srcGlobal;
    AscendC::GlobalTensor<T> m_dstGlobal;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> m_queIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> m_queOut;
    
    AscendC::LocalTensor<T> srcLocal = m_queIn.AllocTensor<T>();
    AscendC::DataCopy(srcLocal, m_srcGlobal, m_elementCount);
    m_queIn.EnQue(srcLocal);
    AscendC::LocalTensor<uint32_t> dstOffsetLocal = m_queIn.AllocTensor<uint32_t>();
    AscendC::DataCopy(dstOffsetLocal, m_dstOffsetGlobal, m_elementCount);
    m_queIn.EnQue(dstOffsetLocal);
    
    AscendC::LocalTensor<T> srcLocal = m_queIn.DeQue<T>();
    AscendC::LocalTensor<uint32_t> dstOffsetLocal = m_queIn.DeQue<uint32_t>();
    AscendC::LocalTensor<T> dstLocal = m_queOut.AllocTensor<T>();
    dstLocal.SetSize(m_elementCount);
    AscendC::Scatter(dstLocal, srcLocal, dstOffsetLocal, (uint32_t)0, m_elementCount);
    m_queIn.FreeTensor(srcLocal);
    m_queIn.FreeTensor(dstOffsetLocal);
    m_queOut.EnQue(dstLocal);
    
    AscendC::LocalTensor<T> dstLocal = m_queOut.DeQue<T>();
    AscendC::DataCopy(m_dstGlobal, dstLocal, m_elementCount);
    m_queOut.FreeTensor(dstLocal)
    

结果示例： 
    
    
    输入数据dstOffsetLocal:
    [254 252 250 ... 4 2 0]
    输入数据srcLocal（128个half类型数据）: 
    [0 1 2 ... 125 126 127]
    输出数据dstGlobal:
    [127 126 125 ... 2 1 0]

**父主题：** [离散与聚合](atlasascendc_api_07_0091.html)


# DataCopyPad(ISASI)

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | x  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

该接口提供数据非对齐搬运的功能，其中从Global Memory搬运数据至Local Memory时，可以根据开发者的需要自行填充数据。

#### 函数原型

  * dataCopyParams为[DataCopyExtParams](#ZH-CN_TOPIC_0000002521040358__table10572141063919)类型，相比于[DataCopyParams](#ZH-CN_TOPIC_0000002521040358__table9182515919)类型，支持的操作数步长等参数取值范围更大
    * 通路：Global Memory->Local Memory
          
          template <typename T>
          __aicore__ inline void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyExtParams& dataCopyParams, const DataCopyPadExtParams<T>& padParams)
          

    * 通路：Local Memory->Global Memory
          
          template <typename T>
          __aicore__ inline void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyExtParams& dataCopyParams)
          

    * 通路：Local Memory->Local Memory，实际搬运过程是VECIN/VECOUT->GM->TSCM
          
          template <typename T>
          __aicore__ inline void DataCopyPad(const LocalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyExtParams& dataCopyParams, const Nd2NzParams& nd2nzParams)
          

  * dataCopyParams为[DataCopyParams](#ZH-CN_TOPIC_0000002521040358__table9182515919)类型
    * 通路：Global Memory->Local Memory
          
          template<typename T>
          __aicore__ inline void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyParams& dataCopyParams, const DataCopyPadParams& padParams)
          

    * 通路：Local Memory->Global Memory
          
          template<typename T>
          __aicore__ inline void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src,const DataCopyParams& dataCopyParams)
          

    * 通路：Local Memory->Local Memory，实际搬运过程是VECIN/VECOUT->GM->TSCM
          
          template<typename T>
          __aicore__ inline void DataCopyPad(const LocalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& dataCopyParams, const Nd2NzParams& nd2nzParams)
          




不同产品型号对函数原型的支持存在差异，请参考下表中的支持度信息，选择产品型号支持的函数原型进行开发。

表1 不同产品型号对函数原型的支持度

展开

产品型号 | 支持的数据传输通路 | 是否支持设置数据搬运模式mode（搬运模式包括单次搬运对齐和整块数据搬运对齐）  
---|---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | GM->VECIN/VECOUT、VECIN/VECOUT->GM、VECIN/VECOUT->TSCM | 否  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | GM->VECIN/VECOUT、VECIN/VECOUT->GM、VECIN/VECOUT->TSCM | 否  
Atlas 200I/500 A2 推理产品 | GM->VECIN/VECOUT、VECIN/VECOUT->GM | 否  
  
#### 参数说明

表2 模板参数说明

展开

参数名 | 描述  
---|---  
T | 操作数以及paddingValue（待填充数据值）的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half/bfloat16_t/int16_t/uint16_t/float/int32_t/uint32_t/int8_t/uint8_t/int64_t/uint64_t/double Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half/bfloat16_t/int16_t/uint16_t/float/int32_t/uint32_t/int8_t/uint8_t/int64_t/uint64_t/double Atlas 200I/500 A2 推理产品，支持的数据类型为：int8_t/uint8_t/half/bfloat16_t/int16_t/uint16_t/float/int32_t/uint32_t  
  
表3 接口参数说明

展开

参数名 | 输入/输出 | 描述  
---|---|---  
dst | 输出 | 目的操作数，类型为LocalTensor或GlobalTensor。 LocalTensor的起始地址需要保证32字节对齐。 GlobalTensor的起始地址无地址对齐约束。  
src | 输入 | 源操作数，类型为LocalTensor或GlobalTensor。 LocalTensor的起始地址需要保证32字节对齐。 GlobalTensor的起始地址无地址对齐约束。  
dataCopyParams | 输入 | 搬运参数。

  * DataCopyExtParams类型，具体参数说明请参考[表4](#ZH-CN_TOPIC_0000002521040358__table10572141063919)。
  * DataCopyParams类型，具体参数说明请参考[表5](#ZH-CN_TOPIC_0000002521040358__table9182515919)。

  
padParams | 输入 | 从Global Memory搬运数据至Local Memory时，可以根据开发者需要，在搬运数据左边或右边填充数据。padParams是用于控制数据填充过程的参数。

  * DataCopyPadExtParams类型，具体参数请参考[表6](#ZH-CN_TOPIC_0000002521040358__table844881954715)。****
  * DataCopyPadParams类型，具体参数请参考[表7](#ZH-CN_TOPIC_0000002521040358__table990103991413)。

  
nd2nzParams | 输入 | 从VECIN/VECOUT->TSCM进行数据搬运时，可以进行ND到NZ的数据格式转换。nd2nzParams是用于控制数据格式转换的参数，Nd2NzParams类型，具体参数请参考[表3](atlasascendc_api_07_00127.html#ZH-CN_TOPIC_0000002520880540__table844881954715)。 **注意：Nd2NzParams的ndNum仅支持设置为1** 。  
  
下文表格中列出的结构体参数定义请参考${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_data_copy.h，${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。

表4 DataCopyExtParams结构体参数定义

展开

参数名称 | 含义  
---|---  
blockCount | 指定该指令包含的连续传输数据块个数，数据类型为uint16_t，取值范围：blockCount∈[1, 4095]。  
blockLen | 指定该指令每个连续传输数据块长度，**该指令支持非对齐搬运** ，**每个连续传输数据块长度单位为字节** 。数据类型为uint32_t，取值范围：blockLen∈[1, 2097151]。  
srcStride | 源操作数，相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔）。 **如果源操作数的逻辑位置为VECIN/VECOUT，则单位为dataBlock(32字节)。如果源操作数的逻辑位置为GM，则单位为字节** 。 数据类型为uint32_t，srcStride不要超出该数据类型的取值范围。  
dstStride | 目的操作数，相邻连续数据块间的间隔（前面一个数据块的尾与后面数据块的头的间隔）。 **如果目的操作数的逻辑位置为VECIN/VECOUT，则单位为dataBlock(32字节)，如果目的操作数的逻辑位置为GM，则单位为字节** 。 数据类型为uint32_t，dstStride不要超出该数据类型的取值范围。  
rsv | 保留字段。  
  
表5 DataCopyParams结构体参数定义

展开

参数名称 | 含义  
---|---  
blockCount | 指定该指令包含的连续传输数据块个数，数据类型为uint16_t，取值范围：blockCount∈[1, 4095]。  
blockLen | 指定该指令每个连续传输数据块长度，**该指令支持非对齐搬运** ，**每个连续传输数据块长度单位为字节** 。数据类型为uint16_t，blockLen不要超出该数据类型的取值范围。  
srcStride | 源操作数，相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔），**如果源操作数的逻辑位置为VECIN/VECOUT，则单位为dataBlock(32字节)。如果源操作数的逻辑位置为GM，则单位为字节** 。数据类型为uint16_t，srcStride不要超出该数据类型的取值范围。  
dstStride | 目的操作数，相邻连续数据块间的间隔（前面一个数据块的尾与后面数据块的头的间隔），**如果目的操作数的逻辑位置为VECIN/VECOUT，则单位为dataBlock(32字节)，如果目的操作数的逻辑位置为GM，则单位为****字节** 。数据类型为uint16_t，dstStride不要超出该数据类型的取值范围。  
  
表6 DataCopyPadExtParams结构体参数定义

展开

参数名称 | 含义  
---|---  
isPad | 是否需要填充用户自定义的数据，取值范围：true，false。 true：填充padding value。 false：表示用户不需要指定填充值，会默认填充随机值。  
leftPadding | 连续搬运数据块左侧需要补充的数据范围，单位为元素个数。 **leftPadding、rightPadding所占的字节数均不能超过32字节。**  
rightPadValue | 连续搬运数据块右侧需要补充的数据范围，单位为元素个数。 **leftPadding、rightPadding所占的字节数均不能超过32字节。**  
padValue | 左右两侧需要填充的数据值，需要保证在数据占用字节范围内。 数据类型和源操作数保持一致，T数据类型。 **当数据类型长度为64位时，该参数只能设置为0。**  
  
表7 DataCopyPadParams结构体参数定义

展开

参数名称 | 含义  
---|---  
isPad | 是否需要填充用户自定义的数据，取值范围：true，false。 true：填充padding value。 false：表示用户不需要指定填充值，会默认填充随机值。  
leftPadding | 连续搬运数据块左侧需要补充的数据范围，单位为元素个数。 **leftPadding、rightPadding所占的字节数均不能超过32字节。**  
rightPadding | 连续搬运数据块右侧需要补充的数据范围，单位为元素个数。 **leftPadding、rightPadding所占的字节数均不能超过32字节。**  
paddingValue | 左右两侧需要填充的数据值，需要保证在数据占用字节范围内。 uint64_t数据类型，要求源操作数为uint64_t数据类型，且该参数只能设置为0。  
  
下面分别给出如下场景的配置示例：

  * [GM->VECIN/VECOUT](#ZH-CN_TOPIC_0000002521040358__li73127579197)
  * [VECIN/VECOUT->GM](#ZH-CN_TOPIC_0000002521040358__li1526352412213)
  * [VECIN/VECOUT->TSCM](#ZH-CN_TOPIC_0000002521040358__li1475016332217)
  * **GM** ->**VECIN/VECOUT**
    * 参数解释
      * 当blockLen+leftPadding+rightPadding满足32字节对齐时，若isPad为false，左右两侧填充的数据值会默认为随机值；否则为paddingValue。
      * 当blockLen+leftPadding+rightPadding不满足32字节对齐时，框架会填充一些假数据dummy，保证左右填充的数据和blockLen、假数据为32字节对齐。若leftPadding、rightPadding都为0：dummy会默认填充待搬运数据块的第一个元素值；若leftPadding/rightPadding不为0：isPad为false，左右两侧填充的数据值和dummy值均为随机值；否则为paddingValue。
    * 配置示例1：
      * blockLen为64，每个连续传输数据块包含64字节；srcStride为1，因为源操作数的逻辑位置为GM，srcStride的单位为字节，也就是说源操作数相邻数据块之间间隔1字节；dstStride为1，因为目的操作数的逻辑位置为VECIN/VECOUT，dstStride的单位为DataBlock数量（每DataBlock为32字节），也就是说目的操作数相邻数据块之间间隔1个dataBlock。
      * blockLen+leftPadding+rightPadding满足32字节对齐，isPad为false，左右两侧填充的数据值会默认为随机值；否则为paddingValue。此处示例中，leftPadding、rightPadding均为0，则不填充。
      * blockLen+leftPadding+rightPadding不满足32字节对齐时，框架会填充一些假数据dummy，保证左右填充的数据和blockLen、假数据为32字节对齐。leftPadding/rightPadding不为0：若isPad为false，左右两侧填充的数据值和dummy值均为随机值；否则为paddingValue。

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002552082615.png)

    * 配置示例2：
      * blockLen为47，每个连续传输数据块包含47字节；srcStride为1，表示源操作数相邻数据块之间间隔1字节；dstStride为1，表示目的操作数相邻数据块之间间隔1个dataBlock。
      * blockLen+leftPadding+rightPadding不满足32字节对齐，leftPadding、rightPadding均为0：dummy会默认填充待搬运数据块的第一个元素值。
      * blockLen+leftPadding+rightPadding不满足32字节对齐，leftPadding/rightPadding不为0：若isPad为false，左右两侧填充的数据值和dummy值均为随机值；否则为paddingValue。

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002552122607.png)




  * **VECIN/VECOUT** ->**GM**
    * 当每个连续传输数据块长度blockLen为32字节对齐时，下图呈现了需要传入的DataCopyParams示例，blockLen为64，每个连续传输数据块包含64字节；srcStride为1，因为源操作数的逻辑位置为VECIN/VECOUT，srcStride的单位为dataBlock(32字节)，也就是说源操作数相邻数据块之间间隔1个dataBlock；dstStride为1，因为目的操作数的逻辑位置为GM，dstStride的单位为字节，也就是说目的操作数相邻数据块之间间隔1字节。

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002552082611.png)

    * 当每个连续传输数据块长度blockLen不满足32字节对齐，由于Unified Buffer要求32字节对齐，框架在搬出时会自动补充一些假数据来保证对齐，但在当搬到GM时会自动将填充的假数据丢弃掉。下图呈现了该场景下需要传入的DataCopyParams示例和假数据补齐的原理。blockLen为47，每个连续传输数据块包含47字节，不满足32字节对齐；srcStride为1，表示源操作数相邻数据块之间间隔1个dataBlock；dstStride为1，表示目的操作数相邻数据块之间间隔1字节。框架在搬出时会自动补充17字节的假数据来保证对齐，搬到GM时再自动将填充的假数据丢弃掉。

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002521042632.png)

  * **VECIN/VECOUT****- >TSCM**

**注意：** 内部实现涉及AIC和AIV之间的通信，实际搬运路径为VECIN/VECOUT->GM->TSCM，**发送通信消息会有开销，性能会受到影响** 。

如[图1 VECIN/VECOUT->TSCM搬运示意图](#ZH-CN_TOPIC_0000002521040358__fig9329040132719)所示，展示了从VECIN/VECOUT搬运到GM，再搬运到TSCM的过程：示例中数据类型为half，单个datablock（32字节）含有16个half元素，源操作数中的A1~A6、B1~B6、C1~C6为需要进行搬运的数据。

    * 从VECIN/VECOUT->GM的搬运，数据存储格式没有发生转变，依然是ND。
      * **blockCount** 为需要搬运的连续传输数据块个数，设置为3；
      * **blockLen** 为一个连续传输数据块的大小（单位为字节），设置为6 * 32 = 192；
      * **srcStride** 为源操作数相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔），源操作数逻辑位置为VECIN/VECOUT，其单位为datablock，两个连续传输数据块（A1~A6、B1~B6）中间相隔1个A7，因此srcStride设置为1；
      * **dstStride** 为目的操作数，相邻连续数据块间的间隔（前面一个数据块的尾与后面数据块的头的间隔），目的操作数逻辑位置为GM，其单位为字节，两个连续传输数据块（A1~A6、B1~B6）中间相隔2个空白的datablock，因此dstStride设置为64字节。
    * 从GM->TSCM的搬运，数据存储格式由ND转换为NZ。
      * **ndNum** 固定为1，即A1~A6、B1~B6、C1~C6视作一整个ndMatrix；
      * **nValue** 为ndMatrix的行数，即为3行；
      * **dValue** 为ndMatrix中一行包含的元素个数，即为6 * 16 = 96个元素；
      * **srcNdMatrixStride** 为相邻ndMatrix之间的距离，因为仅涉及1个ndMatrix，所以可填为0；
      * **srcDValue** 表明ndMatrix的第x行和第x+1行所相隔的元素个数，如A1~B1的距离，即为8个datablock，8 * 16 = 128个元素；
      * **dstNzC0Stride** 为src同一行的相邻datablock在NZ矩阵中相隔datablock数，如A1~A2的距离，即为7个datablock （A1 + 空白 + B1 + 空白 + C1 + 空白 * 2）；
      * **dstNzNStride** 为src中ndMatrix的相邻行在NZ矩阵中相隔多少个datablock，如A1~B1的距离，即为2个datablock（A1 + 空白）；
      * **dstNzMatrixStride** 为相邻NZ矩阵之间的元素个数，因为仅涉及1个NZ矩阵，所以可以填为1。

**图1** VECIN/VECOUT->TSCM搬运示意图  


![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002552122605.png)




#### 返回值说明

无

#### 约束说明

  * leftPadding、rightPadding的字节数均不能超过32字节。



#### 调用示例

本示例实现了GM->VECIN->GM的非对齐搬运过程。
    
    
    #include "kernel_operator.h"
    
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<half> srcGlobal;
    AscendC::GlobalTensor<half> dstGlobal;
    AscendC::DataCopyPadExtParams<half> padParams;
    AscendC::DataCopyExtParams copyParams;
    half scalar = 0;
    AscendC::LocalTensor<half> srcLocal = inQueueSrc.AllocTensor<half>();
    AscendC::DataCopyExtParams copyParams{1, 20 * sizeof(half), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
    AscendC::DataCopyPadExtParams<half> padParams{true, 0, 2, 0};
    AscendC::DataCopyPad(srcLocal, srcGlobal, copyParams, padParams); // 从GM->VECIN搬运40字节
    inQueueSrc.EnQue<half>(srcLocal);
    
    AscendC::LocalTensor<half> srcLocal = inQueueSrc.DeQue<half>();
    AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();
    AscendC::Adds(dstLocal, srcLocal, scalar, 20);
    outQueueDst.EnQue(dstLocal);
    inQueueSrc.FreeTensor(srcLocal);
    
    AscendC::LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
    AscendC::DataCopyExtParams copyParams{1, 20 * sizeof(half), 0, 0, 0};
    AscendC::DataCopyPad(dstGlobal, dstLocal, copyParams); // 从VECIN->GM搬运40字节
    outQueueDst.FreeTensor(dstLocal);
    

结果示例： 
    
    
    输入数据src0Global: [1 2 3 ... 32]
    输出数据dstGlobal:[1 2 3 ... 20]

**父主题：** [Memory数据搬运](atlasascendc_api_07_0100.html)


# SetPadValue(ISASI)

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

设置DataCopyPad需要填充的数值。支持的通路如下：

  * GM->VECIN/GM->VECOUT



#### 函数原型
    
    
    template <typename T, TPosition pos = TPosition::MAX>
    __aicore__ inline void SetPadValue(T paddingValue)
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 输入/输出 | 描述  
---|---|---  
T | 输入 | 填充值的数据类型，与DataCopyPad接口搬运的数据类型一致。  
pos | 输入 | 用于指定DataCopyPad接口搬运过程中从GM搬运数据到哪一个目的地址，目的地址通过逻辑位置来表达。默认值为TPosition::MAX，等效于TPosition::VECIN或TPosition::VECOUT。 支持的取值为：

  * TPosition::VECIN、TPosition::VECOUT、TPosition::MAX

  
  
表2 参数说明

展开

参数名 | 输入/输出 | 描述  
---|---|---  
paddingValue | 输入 | DataCopyPad接口填充的数值，数据与DataCopyPad接口搬运的数据类型一致。  
  
#### 返回值说明

无

#### 约束说明

无

#### 调用示例
    
    
    #include "kernel_operator.h"
    
    template <typename T>
    class SetPadValueTest {
    public:
        __aicore__ inline SetPadValueTest() {}
        __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, uint32_t n1, uint32_t n2)
        {
            m_n1 = n1;
            m_n2 = n2;
            m_n2Align = n2 % 32 == 0 ? n2 : (n2 / 32 + 1) * 32;
            m_srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
            m_dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
    
            m_pipe.InitBuffer(m_queInSrc, 1, m_n1 * m_n2Align * sizeof(T));
        }
        __aicore__ inline void Process()
        {
            CopyIn();
            Compute();
            CopyOut();
        }
    private:
        __aicore__ inline void CopyIn()
        {
            AscendC::LocalTensor<T> srcLocal = m_queInSrc.AllocTensor<T>();
            AscendC::DataCopyExtParams dataCopyExtParams;
            AscendC::DataCopyPadExtParams<T> padParams;
    
            dataCopyExtParams.blockCount = m_n1;
            dataCopyExtParams.blockLen = m_n2 * sizeof(T);
            dataCopyExtParams.srcStride = 0;
            dataCopyExtParams.dstStride = 0;
    
            padParams.isPad = false;
            padParams.leftPadding = 0;
            padParams.rightPadding = 1;
    
            AscendC::SetPadValue((T)37);
            AscendC::DataCopyPad(srcLocal, m_srcGlobal, dataCopyExtParams, padParams);
            m_queInSrc.EnQue(srcLocal);
        }
        __aicore__ inline void Compute()
        {
            ;
        }
        __aicore__ inline void CopyOut()
        {
            AscendC::LocalTensor<T> dstLocal = m_queInSrc.DeQue<T>();
            AscendC::DataCopy(m_dstGlobal, dstLocal, m_n1 * m_n2Align);
            m_queInSrc.FreeTensor(dstLocal);
        }
    private:
        AscendC::TPipe m_pipe;
        uint32_t m_n1;
        uint32_t m_n2;
        uint32_t m_n2Align;
        AscendC::GlobalTensor<T> m_srcGlobal;
        AscendC::GlobalTensor<T> m_dstGlobal;
        AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queInSrc;
    };
    
    template <typename T>
    __global__ __aicore__ void testSetPadValue(GM_ADDR dstGm, GM_ADDR srcGm, uint32_t n1, uint32_t n2)
    {
        SetPadValueTest<T> op;
        op.Init(dstGm, srcGm, n1, n2);
        op.Process();
    }
    
    
    
    输入数据（srcGm, shape = [32, 31]）：[[1, 1, 1, ..., 1], [1, 1, 1, ..., 1], ... , [1, 1, 1, ..., 1]]
    输出数据（dstGm, shape = [32, 32]）：[[1, 1, 1, ..., 1, 37], [1, 1, 1, ..., 1, 37], ... , [1, 1, 1, ..., 1, 37]]
    
    
    // 对于不支持使用立即数进行赋值和初始化的数据类型，如下是一个输入类型bfloat16_t的示例：
    AscendC::SetPadValue(m_srcGlobal.GetValue(0));
    AscendC::DataCopyPad(srcLocal, m_srcGlobal, dataCopyExtParams, padParams);
    
    输入数据（srcGm, shape = [32, 31]）：[[1, 2, 3, ..., 31], [1, 2, 3, ..., 31], ... , [1, 2, 3, ..., 31]]
    输出数据（dstGm, shape = [32, 32]）：[[1, 2, 3, ..., 31, 1], [1, 2, 3, ..., 31, 1], ... , [1, 2, 3, ..., 31, 1]]
    

**父主题：** [Memory数据搬运](atlasascendc_api_07_0100.html)



---

## 矩阵计算ISASI


# Mmad

#### 产品支持情况

展开

产品 |  是否支持（ 不传入bias的原型 ） |  是否支持（ 传入bias的原型 ）  
---|---|---  
Atlas A3 训练系列产品 / Atlas A3 推理系列产品  |  √ |  √  
Atlas A2 训练系列产品 / Atlas A2 推理系列产品  |  √ |  √  
Atlas 200I/500 A2 推理产品  |  √ |  √  
Atlas 推理系列产品 AI Core |  √ |  x  
Atlas 推理系列产品 Vector Core |  x |  x  
Atlas 训练系列产品  |  √ |  x  
  
#### 功能说明

完成矩阵乘加（C += A * B）操作。矩阵ABC分别为A2/B2/CO1中的数据。

  * ABC矩阵的数据排布格式分别为ZZ，ZN，NZ。数据排布格式详解请参考[数据排布格式](/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_0099.html)。 

下图中每个小方格代表一个分形矩阵，Z字形的黑色线条代表数据的排列顺序，起始点是左上角，终点是右下角。

矩阵A：每个分形矩阵内部是行主序，分形矩阵之间是行主序。简称小Z大Z格式。分形shape为16 x (32B/sizeof(AType))，大小为512Byte。

矩阵B：每个分形矩阵内部是列主序，分形矩阵之间是行主序。简称小N大Z格式。分形shape为 (32B/sizeof(BType)) x 16，大小为512Byte。

矩阵C：每个分形矩阵内部是行主序，分形矩阵之间是列主序。简称小Z大N格式。分形shape为16 x 16，大小为256个元素。

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002521042416.jpg)

以下是一个简单的例子，假设分形矩阵的大小是2x2（并不符合真实情况，仅作为示例），矩阵ABC的大小都是4x4。

展开

0 |  1 |  2 |  3  
---|---|---|---  
4 |  5 |  6 |  7  
8 |  9 |  10 |  11  
12 |  13 |  14 |  15  
  
矩阵A的排列顺序：0，1，4，5，2，3，6，7，8，9，12，13，10，11，14，15。

矩阵B的排列顺序：0，4，1，5，2，6，3，7，8，12，9，13，10，14，11，15。

矩阵C的排列顺序：0，1，4，5，8，9，12，13，2，3，6，7，10，11，14，15。




#### 函数原型

  * 不传入bias 
        
        template <typename T, typename U, typename S>
        __aicore__ inline void Mmad(const LocalTensor<T>& dst, const LocalTensor<U>& fm, const LocalTensor<S>& filter, const MmadParams& mmadParams)
        

  * 传入bias 
        
        template <typename T, typename U, typename S, typename V>
        __aicore__ inline void Mmad(const LocalTensor<T>& dst, const LocalTensor<U>& fm, const LocalTensor<S>& filter, const LocalTensor<V>& bias, const MmadParams& mmadParams)
        




#### 参数说明

表1 模板参数说明

展开

参数名 |  描述  
---|---  
T |  目的操作数的数据类型。  
U |  左矩阵的数据类型。  
S |  右矩阵的数据类型。  
V |  Bias矩阵的数据类型。  
  
表2 参数说明

展开

参数名称 |  输入/输出 |  含义  
---|---|---  
dst |  输出 |  目的操作数，结果矩阵，类型为LocalTensor，支持的TPosition为CO1。 LocalTensor的起始地址需要256个元素对齐。  
fm |  输入 |  源操作数，左矩阵a，类型为LocalTensor，支持的TPosition为A2。 LocalTensor的起始地址需要512字节对齐。  
filter |  输入 |  源操作数，右矩阵b，类型为LocalTensor，支持的TPosition为B2。 LocalTensor的起始地址需要512字节对齐。  
bias |  输入 |  源操作数，bias矩阵，类型为LocalTensor，支持的TPosition为C2、CO1。 LocalTensor的起始地址需要128字节对齐。  
mmadParams |  输入 |  矩阵乘相关参数，该参数类型的具体定义请参考${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_mm.h，${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。 MmadParams参数说明请参考[表3](#ZH-CN_TOPIC_0000002521040190__table15780447181917)。  
  
表3 MmadParams结构体内参数说明

展开

参数名称 |  含义  
---|---  
m |  左矩阵Height，取值范围：m∈[0, 4095] 。默认值为0。  
n |  右矩阵Width，取值范围：n∈[0, 4095] 。默认值为0。  
k |  左矩阵Width、右矩阵Height，取值范围：k∈[0, 4095] 。默认值为0。  
cmatrixInitVal |  配置C矩阵初始值是否为0。默认值true。

  * true：C矩阵初始值为0；
  * false：C矩阵初始值通过cmatrixSource参数进行配置。

  
cmatrixSource |  配置C矩阵初始值是否来源于C2（存放Bias的硬件缓存区）。默认值为false。

  * false：来源于CO1；


  * true：来源于C2。

Atlas 训练系列产品 ，仅支持配置为false。 Atlas 推理系列产品 AI Core，仅支持配置为false。 Atlas A2 训练系列产品 / Atlas A2 推理系列产品 ，支持配置为true/false。 Atlas A3 训练系列产品 / Atlas A3 推理系列产品 ，支持配置为true/false。 Atlas 200I/500 A2 推理产品 ，支持配置为true/false。 注意：带bias输入的接口配置该参数无效，会根据bias输入的位置来判断C矩阵初始值是否来源于CO1还是C2。  
isBias |  **该参数废弃，新开发内容不要使用该参数。** 如果需要累加初始矩阵，请使用带bias的接口来实现；也可以通过cmatrixInitVal和cmatrixSource参数配置C矩阵的初始值来源来实现。推荐使用带bias的接口，相比于配置cmatrixInitVal和cmatrixSource参数更加简单方便。 配置是否需要累加初始矩阵，默认值为false，取值说明如下：

  * false：矩阵乘，无需累加初始矩阵，C = A * B。
  * true：矩阵乘加，需要累加初始矩阵，C += A * B。

  
unitFlag |  unitFlag是一种Mmad指令和Fixpipe指令细粒度的并行，使能该功能后，硬件每计算完一个分形，计算结果就会被搬出，该功能不适用于在L0C Buffer累加的场景。取值说明如下： 0：保留值； 2：使能unitFlag，硬件执行完指令之后，不会关闭unitFlag功能； 3：使能unitFlag，硬件执行完指令之后，会将unitFlag功能关闭。 使能该功能时，Mmad指令的unitFlag在最后1个分形设置为3、其余分形计算设置为2即可。 该参数仅支持如下型号： Atlas A2 训练系列产品 / Atlas A2 推理系列产品  Atlas A3 训练系列产品 / Atlas A3 推理系列产品   
fmOffset |  预留参数。为后续的功能做保留，开发者暂时无需关注，使用默认值即可。  
enSsparse  
enWinogradA  
enWinogradB  
kDirectionAlign |  设置是否需要对齐，默认值为false。 Atlas 训练系列产品 ，仅支持配置为false。 Atlas 推理系列产品 AI Core，仅支持配置为false。 Atlas A2 训练系列产品 / Atlas A2 推理系列产品 ，仅支持配置为false。 Atlas A3 训练系列产品 / Atlas A3 推理系列产品 ，仅支持配置为false。 Atlas 200I/500 A2 推理产品 ，仅支持配置为false。  
  
表4 dst、fm、filter支持的精度类型组合（ Atlas 训练系列产品 ）

展开

**左矩阵****fm type** |  **右矩阵****filter type** |  **结果矩阵****dst type**  
---|---|---  
uint8_t |  uint8_t |  uint32_t  
int8_t |  int8_t |  int32_t  
uint8_t |  int8_t |  int32_t  
half |  half |  half 说明 该精度类型组合，精度无法达到双千分之一，且后续处理器版本不支持该类型转换，建议直接使用half输入float输出。 双千分之一是指每个实际数据和真值数据之间的误差不超过千分之一，误差超过千分之一的数据总和不超过总数据数的千分之一。  
half |  half |  float  
  
表5 dst、fm、filter支持的精度类型组合（ Atlas 推理系列产品 AI Core ）

展开

**左矩阵****fm type** |  **右矩阵****filter type** |  **结果矩阵****dst type**  
---|---|---  
int8_t |  int8_t |  int32_t  
uint8_t |  int8_t |  int32_t  
uint8_t |  uint8_t |  int32_t  
half |  half |  half 说明 该精度类型组合，精度无法达到双千分之一，且后续处理器版本不支持该类型转换，建议直接使用half输入float输出。 双千分之一是指每个实际数据和真值数据之间的误差不超过千分之一，误差超过千分之一的数据总和不超过总数据数的千分之一。  
half |  half |  float  
int4b_t |  int4b_t |  int32_t  
  
表6 dst、fm、filter支持的精度类型组合（ Atlas 200I/500 A2 推理产品 ）（ Atlas A2 训练系列产品 / Atlas A2 推理系列产品 ）（ Atlas A3 训练系列产品 / Atlas A3 推理系列产品 ）

展开

**左矩阵****fm type** |  **右矩阵****filter type** |  **结果矩阵****dst type**  
---|---|---  
int8_t |  int8_t |  int32_t  
half |  half |  float  
float |  float |  float  
bfloat16_t |  bfloat16_t |  float  
int4b_t |  int4b_t |  int32_t  
  
表7 dst、fm、filter、bias支持的精度类型组合（ Atlas 200I/500 A2 推理产品 ）（ Atlas A2 训练系列产品 / Atlas A2 推理系列产品 ）（ Atlas A3 训练系列产品 / Atlas A3 推理系列产品 ）

展开

**左矩阵****fm type** |  **右矩阵****filter type** |  **bias type** |  **结果矩阵****dst type**  
---|---|---|---  
int8_t |  int8_t |  int32_t |  int32_t  
half |  half |  float |  float  
float |  float |  float |  float  
bfloat16_t |  bfloat16_t |  float |  float  
  
#### 约束说明

  * dst只支持位于CO1，fm只支持位于A2，filter只支持位于B2。
  * 当M、K、N中的任意一个值为0时，该指令不会被执行。
  * 当M = 1时，会默认开启GEMV（General Matrix-Vector Multiplication）功能。在这种情况下，Mmad API从L0A Buffer读取数据时，会以ND格式进行读取，而不会将其视为ZZ格式。所以此时左矩阵需要直接按照ND格式进行排布。
  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。
  * 通过一个具体的示例来介绍无效数据与有效数据的排布方式。 

数据为half类型，当M=30，K=70，N=40的时候，A2中有2x5个16x16矩阵，B2中有5x3个16x16矩阵，CO1中有2x3个16x16矩阵。在这种场景下M、K和N都不是16的倍数，A2中右下角的矩阵实际有效的数据只有14x6个，但是也需要占一个16x16矩阵的空间，其他无效数据在计算中会被忽略。一个16x16分形的数据块中，无效数据与有效数据排布的方式示意如下：

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002552082399.png)




#### 调用示例

不含矩阵乘偏置的样例请参考[Mmad样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/20_mmad_kernellaunch/MmadInvocation)。

包含矩阵乘偏置的样例请参考[包含矩阵乘偏置的Mmad样例](https://gitee.com/ascend/samples/blob/master/operator/ascendc/0_introduction/20_mmad_kernellaunch/MmadBiasInvocation/)。

**父主题：** [矩阵计算](atlasascendc_api_07_00172.html)


# MmadWithSparse

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

完成矩阵乘加操作，传入的左矩阵A为稀疏矩阵， 右矩阵B为稠密矩阵 。对于矩阵A，在MmadWithSparse计算时完成稠密化；对于矩阵B，在计算执行前的输入数据准备时自行完成稠密化（按照下文中介绍的稠密算法进行稠密化），所以输入本接口的B矩阵为稠密矩阵。B稠密矩阵需要通过调用[LoadDataWithSparse](atlasascendc_api_07_0244.html)载入，同时加载索引矩阵，索引矩阵在矩阵B稠密化的过程中生成，再用于A矩阵的稠密化。

#### 函数原型
    
    
    template <typename T = int32_t, typename U = int8_t, typename Std::enable_if<Std::is_same<PrimT<T>, int32_t>::value, bool>::type = true, typename Std::enable_if<Std::is_same<PrimT<U>, int8_t>::value, bool>::type = true>
    __aicore__ inline void MmadWithSparse(const LocalTensor<T>& dst, const LocalTensor<U>& fm, const LocalTensor<U>& filter, const MmadParams& mmadParams)
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
T | dst的数据类型。  
U | fm、filter的数据类型。

  * 当dst、fm、filter为基础数据类型时， T必须为int32_t类型，U必须为int8_t类型，否则编译失败。


  * 当dst、fm、filter为[TensorTrait](atlasascendc_api_07_0011.html)类型时，T的LiteType必须为int32_t类型，U的LiteType必须为int8_t类型，否则编译失败。

最后两个模板参数仅用于上述数据类型检查，用户无需关注。  
  
表2 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
dst | 输出 | 目的操作数，结果矩阵，类型为LocalTensor，支持的TPosition为CO1。 LocalTensor的起始地址需要256个元素（1024字节）对齐。  
fm | 输入 | 源操作数，左矩阵A，类型为LocalTensor，支持的TPosition为A2。 LocalTensor的起始地址需要512字节对齐。  
filter | 输入 | 源操作数，右矩阵B，类型为LocalTensor，支持的TPosition为B2。 LocalTensor的起始地址需要512字节对齐。  
mmadParams | 输入 | 矩阵乘相关参数，类型为MmadParams。 具体定义请参考${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_mm.h，${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。 参数说明请参考[表3](atlasascendc_api_07_0249.html#ZH-CN_TOPIC_0000002521040190__table15780447181917)。  
  
#### 约束说明

  * 原始稀疏矩阵B每4个元素中应保证最多2个非零元素，如果存在3个或更多非零元素，则仅使用前2个非零元素。
  * 当M、K、N中的任意一个值为0时，该指令不会被执行。


  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。



#### 稠密算法说明

假设原始稀疏矩阵B的每4个元素中至少有2个零，稠密化后的矩阵B是一个在每4个元素中过滤掉2个零的稠密矩阵。矩阵B稠密化的过程中生成索引矩阵，过程如下：对于稀疏矩阵B中的每4个元素，将在index矩阵中生成2个2位索引，并按照以下规则进行编码。索引必须在{0, 1, 2}范围内。

  * 第一个索引用于指示前3个元素中第1个非零元素的相对位置。
  * 第二个索引用于指示第2个非零元素在后3个元素中的相对位置。



具体可参考下表。其中，“-”表示算法不关心该位置上的值，因为其会被过滤。

展开

示例 | ele0 | ele1 | ele2 | ele3 | Index_a[i] | Index_b[i]  
---|---|---|---|---|---|---  
Two non-zero elements | 0 | 0 | X | Y | 2’b10 | 2’b10  
0 | X | 0 | Y | 2’b01 | 2’b10  
X | 0 | 0 | Y | 2’b00 | 2’b10  
0 | X | Y | - | 2’b01 | 2’b01  
X | 0 | Y | - | 2’b00 | 2’b01  
X | Y | - | - | 2’b00 | 2’b00  
One non-zero element | 0 | 0 | 0 | X | 2’b00 | 2’b10  
0 | 0 | X | 0 | 2’b10 | 2’b00  
0 | X | 0 | 0 | 2’b01 | 2’b00  
X | 0 | 0 | 0 | 2’b00 | 2’b00  
All zero | 0 | 0 | 0 | 0 | 2’b00 | 2’b00  
  
该索引矩阵用于A矩阵的稠密化，根据索引矩阵从MatrixA中的4个元素中选择2个元素参与计算，如下图所示：

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002521042688.png)

#### 调用示例
    
    
    #include "kernel_operator.h"
    int srcOffset = 0;
    int dstOffset = 0;
    AscendC::LocalTensor<int8_t> a1Local = inQueueA1.DeQue<int8_t>();
    AscendC::LocalTensor<int8_t> a2Local = inQueueA2.AllocTensor<int8_t>();
    
    AscendC::LoadData2DParams loadDataParams;
    loadDataParams.repeatTimes = kBlocks * mBlocks;
    loadDataParams.srcStride = 1;
    loadDataParams.ifTranspose = false;
    
    AscendC::LoadData(a2Local, a1Local, loadDataParams);
    
    inQueueA2.EnQue<int8_t>(a2Local);
    inQueueA1.FreeTensor(a1Local);
    
    AscendC::LocalTensor<int8_t> b2Local = inQueueB2.AllocTensor<int8_t>();
    
    // transform nz to zn
    AscendC::LoadData2DParams loadDataParams;
    loadDataParams.repeatTimes = kBlocks * nBlocks / 2;
    loadDataParams.srcStride = 0;
    loadDataParams.ifTranspose = false;
    
    AscendC::LoadDataWithSparse(b2Local, b1Local, idxb1Local, loadDataParams);
    
    inQueueB2.EnQue<int8_t>(b2Local);
    
    AscendC::LocalTensor<int8_t> b2Local = inQueueB2.DeQue<int8_t>();
    AscendC::LocalTensor<int32_t> c1Local = outQueueCO1.AllocTensor<int32_t>();
    
    uint32 m = 16;
    uint32 k = 64;
    uint32 n = 16;
    AscendC::MmadWithSparse(c1Local, a2Local, b2Local, { m, n, k, false, 0, false, false, false });
    

**父主题：** [矩阵计算](atlasascendc_api_07_00172.html)


# SetHF32Mode

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | x  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

用于设置Mmad计算是否开启HF32模式，开启该模式后L0A/L0B中的FP32数据将在参与Mmad计算之前被舍入为HF32。

#### 函数原型
    
    
    __aicore__ inline void SetHF32Mode(HF32Mode mode)
    

#### 参数说明

表1 参数说明

展开

参数名 | 输入/输出 | 描述  
---|---|---  
mode | 输入 | Mmad HF32模式控制入参，HF32Mode枚举类型。支持如下两种取值：

  * ENABLE：L0A/L0B中的FP32数据将在矩阵乘法之前被舍入为HF32。
  * DISABLE：将执行常规的FP32矩阵乘法。

  
  
#### 返回值说明

无

#### 约束说明

无

#### 调用示例
    
    
    AscendC::SetHF32Mode(HF32Mode::ENABLE); // 控制mmad计算时是否使用HF32精度进行计算。
    

**父主题：** [矩阵计算](atlasascendc_api_07_00172.html)


# SetHF32TransMode

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | x  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

设置HF32模式取整的具体方式，需要先使用[SetHF32Mode](atlasascendc_api_07_0258.html)开启HF32取整模式。

#### 函数原型
    
    
    __aicore__ inline void SetHF32TransMode(HF32TransMode mode)
    

#### 参数说明

表1 参数说明

展开

参数名 | 输入/输出 | 描述  
---|---|---  
mode | 输入 | Mmad HF32取整模式控制入参，HF32TransMode类型。支持如下两种取值：

  * NEAREST_ZERO：则FP32将以向零靠近的方式四舍五入为HF32。
  * NEAREST_EVEN：则FP32将以最接近偶数的方式四舍五入为HF32。

  
  
#### 返回值说明

无

#### 约束说明

无

#### 调用示例
    
    
    AscendC::SetHF32TransMode(HF32TransMode::NEAREST_ZERO);  
    

**父主题：** [矩阵计算](atlasascendc_api_07_00172.html)


# SetMMRowMajor

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | x  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

设置Mmad计算时优先通过N方向，CUBE将首先通过N方向，然后通过M方向生成结果。

#### 函数原型
    
    
    __aicore__ inline void SetMMRowMajor()
    

#### 参数说明

无

#### 返回值说明

无

#### 约束说明

无

#### 调用示例
    
    
    AscendC::SetMMRowMajor();// 设置Mmad优先计算输出矩阵的N方向，再计算M方向。
    

**父主题：** [矩阵计算](atlasascendc_api_07_00172.html)


# SetMMColumnMajor

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | x  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

设置Mmad计算时优先通过M方向，CUBE将首先通过M方向，然后通过N方向产生结果。

#### 函数原型
    
    
    __aicore__ inline void SetMMColumnMajor()
    

#### 参数说明

无

#### 返回值说明

无

#### 约束说明

无

#### 调用示例
    
    
    AscendC::SetMMColumnMajor(); // 设置Mmad优先计算输出矩阵的M方向，再计算N方向。
    

**父主题：** [矩阵计算](atlasascendc_api_07_00172.html)



---

## Conv2D_Gemm


# Conv2D（废弃）

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x  
Atlas 200I/500 A2 推理产品 | x  
Atlas 推理系列产品AI Core | √  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | √  
  
#### 功能说明

**该接口废弃，并将在后续版本移除，请不要使用该接口。**

计算给定输入张量和权重张量的2-D卷积，输出结果张量。Conv2d卷积层多用于图像识别，使用过滤器提取图像中的特征。

#### 函数原型
    
    
    template <typename T, typename U>
    __aicore__ inline void Conv2D(const LocalTensor<T>& dst, const LocalTensor<U>& featureMap, const LocalTensor<U>& weight, Conv2dParams& conv2dParams, Conv2dTilling& tilling)
    

入参中的tiling结构需要通过如下切分方案计算接口来获取：
    
    
    template <typename T>
    __aicore__ inline Conv2dTilling GetConv2dTiling(Conv2dParams& conv2dParams)
    

#### 参数说明

表1 接口参数说明

展开

**参数名称** | **类型** | **说明**  
---|---|---  
dst | 输出 | 目的操作数。 Atlas 训练系列产品，支持的TPosition为：CO1，CO2 Atlas 推理系列产品AI Core，支持的TPosition为：CO1，CO2 结果中有效张量格式为[Cout/16, Ho, Wo, 16]，大小为Cout * Ho * Wo，Ho与Wo可以根据其他数据计算得出。 Ho = floor((H + pad_top + pad_bottom - dilation_h * (Kh - 1) - 1) / stride_h + 1) Wo = floor((W + pad_left + pad_right - dilation_w * (Kw - 1) - 1) / stride_w + 1) 由于硬件要求Ho*Wo需为16倍数，在申请dst Tensor时，shape应向上16对齐，实际申请shape大小应为Cout * round_howo。 round_howo = ceil(Ho * Wo /16) * 16。  
featureMap | 输入 | 输入张量，Tensor的TPosition为A1。 输入张量“feature_map”的形状，格式是[C1, H, W, C0]。 C1*C0为输入的channel数，要求如下：

  * 当feature_map的数据类型为half时，C0=16。
  * 当feature_map的数据类型为int8_t时，C0=32。
  * C1取值范围：[1,4], 输入的channel的范围：[16，32，64，128]。

H为高，取值范围：[1,40]。 W为宽，取值范围：[1,40]。  
weight | 输入 | 卷积核（权重）张量，Tensor的TPosition为B1。 卷积核张量“weight”的形状，格式是[C1, Kh, Kw, Cout, C0]。 C1*C0为输入的channel数，对于C0要求如下：

  * 当feature_map的数据类型为half时，C0=16。
  * 当feature_map的数据类型为int8_t时，C0=32。
  * C1取值范围：[1,4]。
  * kernel_shape输入的channel数需与fm_shape输入的channel数保持一致。

Cout为卷积核数目，取值范围：[16，32，64，128]， Cout必须为16的倍数。 Kh为卷积核高；值的范围：[1,5]。 Kw表示卷积核宽；值的范围：[1,5]。  
conv2dParams | 输入 | 输入矩阵形状等状态参数，类型为Conv2dParams。结构体具体定义为：
    
    
    struct Conv2dParams {
        uint32_t imgShape[CONV2D_IMG_SIZE];       // [H, W]
        uint32_t kernelShapeIn[CONV2D_KERNEL_SIZE]; // [Kh, Kw]
        uint32_t stride[CONV2D_STRIDE];          // [stride_h, stride_w]
        uint32_t cin;                            // cin = C0 * C1;
        uint32_t cout;
        uint32_t padList[CONV2D_PAD];       // [pad_left, pad_right, pad_top, pad_bottom]
        uint32_t dilation[CONV2D_DILATION]; // [dilation_h, dilation_w]
        uint32_t initY;
        uint32_t partialSum;
    };
      
  
tilling | 输入 | 分形控制参数，类型为Conv2dTilling。结构体具体定义为：
    
    
    struct Conv2dTilling {
        const uint32_t blockSize = 16; // # M block size is always 16
        LoopMode loopMode = LoopMode::MODE_NM;
    
        uint32_t c0Size = 32;
        uint32_t dTypeSize = 1;
    
        uint32_t strideH = 0;
        uint32_t strideW = 0;
        uint32_t dilationH = 0;
        uint32_t dilationW = 0;
        uint32_t hi = 0;
        uint32_t wi = 0;
        uint32_t ho = 0;
        uint32_t wo = 0;
    
        uint32_t height = 0;
        uint32_t width = 0;
    
        uint32_t howo = 0;
    
        uint32_t mNum = 0;
        uint32_t nNum = 0;
        uint32_t kNum = 0;
    
        uint32_t mBlockNum = 0;
        uint32_t kBlockNum = 0;
        uint32_t nBlockNum = 0;
    
        uint32_t roundM = 0;
        uint32_t roundN = 0;
        uint32_t roundK = 0;
    
        uint32_t mTileBlock = 0;
        uint32_t nTileBlock = 0;
        uint32_t kTileBlock = 0;
    
        uint32_t mIterNum = 0;
        uint32_t nIterNum = 0;
        uint32_t kIterNum = 0;
    
        uint32_t mTileNums = 0;
    
        bool mHasTail = false;
        bool nHasTail = false;
        bool kHasTail = false;
    
        uint32_t kTailBlock = 0;
        uint32_t mTailBlock = 0;
        uint32_t nTailBlock = 0;
    
        uint32_t mTailNums = 0;
    };
      
  
表2 Conv2DParams结构体内参数说明：

展开

**参数名称** | **类型** | **说明**  
---|---|---  
imgShape | vector<int> | 输入张量“feature_map”的形状，格式是[ H, W]。

  * H为高，取值范围：[1,40]。
  * W为宽，取值范围：[1,40]。

  
kernelShape | vector<int> | 卷积核张量“weight”的形状，格式是[Kh, Kw]。

  * Kh为高，取值范围：[1,5]。
  * Kw为宽，取值范围：[1,5]。

  
stride | vector<int> | 卷积步长，格式是[stride_h, stride_w]。

  * stride_h表示步长高， 值的范围：[1,4]。
  * stride_w表示步长宽， 值的范围：[1,4]。

  
cin | int | 分形排布参数，Cin = C1 * C0，Cin为输入的channel数，C1取值范围：[1,4]。

  * 当feature_map的数据类型为float时，C0=8。输入的channel的范围：[8，16，24，32]。
  * 当feature_map的数据类型为half时，C0=16。输入的channel的范围：[16，32，48，64]。
  * 当feature_map的数据类型为int8_t时，C0=32。输入的channel的范围：[32，64，96，128]。

  
cout | int | Cout为卷积核数目，取值范围：[16，32，64，128]， Cout必须为16的倍数。  
padList | vector<int> | padding行数/列数，格式是[pad_left, pad_right, pad_top, pad_bottom]。

  * pad_left为feature_map左侧pad列数，范围[0,4]。pad_right为feature_map右侧pad列数，范围[0,4]。
  * pad_top为feature_map顶部pad行数，范围[0,4]。
  * pad_bottom为feature_map底部pad行数，范围[0,4]。

  
dilation | vector<int> | 空洞卷积参数，格式[dilation_h, dilation_w]。

  * dilation_h为空洞高，范围：[1,4]。
  * dilation_w为空洞宽，范围：[1,4]。

膨胀后卷积核宽为dilation_w * (Kw - 1) + 1，高为dilation_h * (Kh - 1) + 1。  
initY | uint32_t | 表示dst是否需要初始化。

  * 取值0：不使用bias，L0C需要初始化，dst初始矩阵保存有之前结果，新计算结果会累加前一次conv2d计算结果。
  * 取值1：不使用bias，L0C不需要初始化，dst初始矩阵中数据无意义，计算结果直接覆盖dst中的数据。

  
partialSum | uint32_t | 当dst参数所在的TPosition为CO2时，通过该参数控制计算结果是否搬出。

  * 取值0：搬出计算结果
  * 取值1：不搬出计算结果，可以进行后续计算

  
  
表3 Conv2dTilling结构体内参数说明

展开

**参数名称** | **类型** | **说明**  
---|---|---  
blockSize | uint32_t | 固定值，恒为16，一个维度内存放的元素个数。  
loopMode | LoopMode | 遍历模式，结构体具体定义为：
    
    
    enum class LoopMode {
        MODE_NM = 0,
        MODE_MN = 1,
        MODE_KM = 2,
        MODE_KN = 3
    };
      
  
c0Size | uint32_t | 一个block的字节长度，范围[16或者32]。  
dtypeSize | uint32_t | 传入的数据类型的字节长度，范围[1, 2]。  
strideH | uint32_t | 卷积步长-高，范围:[1,4]。  
strideW | uint32_t | 卷积步长-宽，范围:[1,4]。  
dilationH | uint32_t | 空洞卷积参数-高，范围：[1,4]。  
dilationW | uint32_t | 空洞卷积参数-宽，范围：[1,4]。  
hi | uint32_t | feature_map形状-高，范围：[1,40]。  
wi | uint32_t | feature_map形状-宽，范围：[1,40]。  
ho | uint32_t | feature_map形状-高，范围：[1,40]。  
wo | uint32_t | feature_map形状-宽，范围：[1,40]。  
height | uint32_t | weight形状-高，[1,5]。  
width | uint32_t | weight形状-宽，[1,5]。  
howo | uint32_t | feature_map形状大小，为ho * wo。  
mNum | uint32_t | M轴等效数据长度参数值，范围：[1,4096]。  
nNum | uint32_t | N轴等效数据长度参数值，范围：[1,4096]。  
kNum | uint32_t | K轴等效数据长度参数值，范围：[1,4096]。  
roundM | uint32_t | M轴等效数据长度参数值且以blockSize为倍数向上取整，范围：[1,4096]。  
roundN | uint32_t | N轴等效数据长度参数值且以blockSize为倍数向上取整，范围：[1,4096]。  
roundK | uint32_t | K轴等效数据长度参数值且以c0Size为倍数向上取整，范围：[1,4096]。  
mBlockNum | uint32_t | M轴Block个数，mBlockNum = mNum / blockSize，范围：[1,4096]。  
nBlockNum | uint32_t | N轴Block个数，nBlockNum = nNum / blockSize，范围：[1,4096]。  
kBlockNum | uint32_t | K轴Block个数，kBlockNum = kNum / blockSize，范围：[1,4096]。  
mIterNum | uint32_t | 遍历M轴维度数量，范围：[1,4096]。  
nIterNum | uint32_t | 遍历N轴维度数量，范围：[1,4096]。  
kIterNum | uint32_t | 遍历K轴维度数量，范围：[1,4096]。  
mTileBlock | uint32_t | M轴切分块个数，范围：[1,4096]。  
nTileBlock | uint32_t | N轴切分块个数，范围：[1,4096]。  
kTileBlock | uint32_t | K轴切分块个数，范围：[1,4096]。  
kTailBlock | uint32_t | K轴尾块个数，范围：[1,4096]。  
mTailBlock | uint32_t | M轴尾块个数，范围：[1,4096]。  
nTailBlock | uint32_t | N轴尾块个数，范围：[1,4096]。  
kHasTail | bool | K轴是否存在尾块。  
mHasTail | bool | M轴是否存在尾块。  
nHasTail | bool | N轴是否存在尾块。  
mTileNums | uint32_t | M轴切分块个数的长度，范围：[1,4096]。  
mTailNums | uint32_t | M轴尾块个数的长度，范围：[1,4096]。  
  
表4 imgShape、kernelShape和dst的数据类型组合

展开

feature_map.dtype | weight.dtype | dst.dtype  
---|---|---  
int8_t | int8_t | int32_t  
half | half | float  
half | half | half  
  
#### 约束说明

  * 该接口当前不支持W=Kw并且H>Kh的场景，其将产生不可预期的结果。
  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。



**父主题：** [矩阵计算](atlasascendc_api_07_00172.html)


# Gemm（废弃）

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x  
Atlas 200I/500 A2 推理产品 | x  
Atlas 推理系列产品AI Core | √  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | √  
  
#### 功能说明

**该接口废弃，并将在后续版本移除，请不要使用该接口。**

根据输入的切分规则，将给定的两个输入张量做矩阵乘，输出至结果张量。将A和B两个输入矩阵乘法在一起，得到一个输出矩阵C。

#### 函数原型

  * 功能接口：
        
        template <typename T, typename U, typename S>
        __aicore__ inline void Gemm(const LocalTensor<T>& dst, const LocalTensor<U>& src0, const LocalTensor<S>& src1, const uint32_t m, const uint32_t k, const uint32_t n, GemmTiling tilling, bool partialsum = true, int32_t initValue = 0)
        



  * 切分方案计算接口：
        
        template <typename T>
        __aicore__ inline GemmTiling GetGemmTiling(uint32_t m, uint32_t k, uint32_t n)
        




#### 参数说明

表1 接口参数说明

展开

**参数名称** | **类型** | **说明**  
---|---|---  
dst | 输出 | 目的操作数。 Atlas 训练系列产品，支持的TPosition为：CO1，CO2 Atlas 推理系列产品AI Core，支持的TPosition为：CO1，CO2  
src0 | 输入 | 源操作数，TPosition为A1。  
src1 | 输入 | 源操作数，TPosition为B1。  
m | 输入 | 左矩阵Src0Local有效Height，范围：[1, 4096]。 注意：m可以不是16的倍数。  
k | 输入 | 左矩阵Src0Local有效Width、右矩阵Src1Local有效Height。

  * 当输入张量Src0Local的数据类型为float时，范围：[1, 8192]
  * 当输入张量Src0Local的数据类型为half时，范围：[1, 16384]
  * 当输入张量Src0Local的数据类型为int8_t时，范围：[1, 32768]

注意：k可以不是16的倍数。  
n | 输入 | 右矩阵Src1Local有效Width，范围：[1, 4096]。 注意：n可以不是16的倍数。  
tilling | 输入 | 切分规则，类型为GemmTiling，结构体具体定义为：
    
    
    struct GemmTiling {
        const uint32_t blockSize = 16;
        LoopMode loopMode = LoopMode::MODE_NM;
        uint32_t mNum = 0;
        uint32_t nNum = 0;
        uint32_t kNum = 0;
        uint32_t roundM = 0;
        uint32_t roundN = 0;
        uint32_t roundK = 0;
        uint32_t c0Size = 32;
        uint32_t dtypeSize = 1;
        uint32_t mBlockNum = 0;
        uint32_t nBlockNum = 0;
        uint32_t kBlockNum = 0;
        uint32_t mIterNum = 0;
        uint32_t nIterNum = 0;
        uint32_t kIterNum = 0;
        uint32_t mTileBlock = 0;
        uint32_t nTileBlock = 0;
        uint32_t kTileBlock = 0;
        uint32_t kTailBlock = 0;
        uint32_t mTailBlock = 0;
        uint32_t nTailBlock = 0;
        bool kHasTail = false;
        bool mHasTail = false;
        bool nHasTail = false;
        bool kHasTailEle = false;
        uint32_t kTailEle = 0;
    };
    

参数说明请参考[表3](#ZH-CN_TOPIC_0000002552120257__table946018393169)。  
partialsum | 输入 | 当dst参数所在的TPosition为CO2时，通过该参数控制计算结果是否搬出。

  * 取值0：搬出计算结果
  * 取值1：不搬出计算结果，可以进行后续计算

  
initValue | 输入 | 表示dst是否需要初始化。

  * 取值0: dst需要初始化，dst初始矩阵保存有之前结果，新计算结果会累加前一次conv2d计算结果。
  * 取值1: dst不需要初始化，dst初始矩阵中数据无意义，计算结果直接覆盖dst中的数据。

  
  
表2 feature_map、weight和dst的数据类型组合

展开

src0.dtype | src1.dtype | dst.dtype  
---|---|---  
int8_t | int8_t | int32_t  
half | half | float  
half | half | half  
  
表3 GemmTiling结构内参数说明

展开

**参数名称** | **类型** | **说明**  
---|---|---  
blockSize | uint32_t | 固定值，恒为16，一个维度内存放的元素个数。  
loopMode | LoopMode | 遍历模式，结构体具体定义为：
    
    
    enum class LoopMode {
        MODE_NM = 0,
        MODE_MN = 1,
        MODE_KM = 2,
        MODE_KN = 3
    };
      
  
mNum | uint32_t | M轴等效数据长度参数值，范围：[1, 4096]。  
nNum | uint32_t | N轴等效数据长度参数值，范围：[1, 4096]。  
kNum | uint32_t | K轴等效数据长度参数值。

  * 当输入张量Src0Local的数据类型为float时，范围：[1, 8192]
  * 当输入张量Src0Local的数据类型为half时，范围：[1, 16384]
  * 当输入张量Src0Local的数据类型为int8_t时，范围：[1, 32768]

  
roundM | uint32_t | M轴等效数据长度参数值且以blockSize为倍数向上取整，范围：[1, 4096]  
roundN | uint32_t | N轴等效数据长度参数值且以blockSize为倍数向上取整，范围：[1, 4096]  
roundK | uint32_t | K轴等效数据长度参数值且以c0Size为倍数向上取整。

  * 当输入张量Src0Local的数据类型为float时，范围：[1, 8192]
  * 当输入张量Src0Local的数据类型为half时，范围：[1, 16384]
  * 当输入张量Src0Local的数据类型为int8_t时，范围：[1, 32768]

  
c0Size | uint32_t | 一个block的字节长度，范围：[16或者32]。  
dtypeSize | uint32_t | 传入的数据类型的字节长度，范围：[1, 2]。  
mBlockNum | uint32_t | M轴Block个数，mBlockNum = mNum / blockSize。  
nBlockNum | uint32_t | N轴Block个数，nBlockNum = nNum / blockSize。  
kBlockNum | uint32_t | K轴Block个数，kBlockNum = kNum / blockSize。  
mIterNum | uint32_t | 遍历维度数量，范围：[1, 4096]。  
nIterNum | uint32_t | 遍历维度数量，范围：[1, 4096]。  
kIterNum | uint32_t | 遍历维度数量，范围：[1, 4096]。  
mTileBlock | uint32_t | M轴切分块个数，范围：[1, 4096]。  
nTileBlock | uint32_t | N轴切分块个数，范围：[1, 4096]。  
kTileBlock | uint32_t | K轴切分块个数，范围：[1, 4096]。  
kTailBlock | uint32_t | K轴尾块个数，范围：[1, 4096]。  
mTailBlock | uint32_t | M轴尾块个数，范围：[1, 4096]。  
nTailBlock | uint32_t | N轴尾块个数，范围：[1, 4096]。  
kHasTail | bool | K轴是否存在尾块。  
mHasTail | bool | M轴是否存在尾块。  
nHasTail | bool | N轴是否存在尾块。  
kHasTailEle | bool | 是否存在尾块元素。  
kTailEle | uint32_t | K轴尾块元素，范围：[1, 4096]。  
  
#### 约束说明

  * 参数m，k，n可以不是16对齐，但因硬件原因，操作数dst，Src0Local和Src1Local的shape需满足对齐要求，即m方向，n方向要求向上16对齐，k方向根据操作数数据类型按16或32向上对齐。
  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。



**父主题：** [矩阵计算](atlasascendc_api_07_00172.html)



---

## FixPipe


# SetFixpipePreQuantFlag

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | x  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

[DataCopy](atlasascendc_api_07_00130.html)（CO1->GM、CO1->A1）过程中进行随路量化时，通过调用该接口设置量化流程中标量量化参数。

#### 函数原型
    
    
    template<template T>
    __aicore__ inline void SetFixpipePreQuantFlag(uint64_t config)
    

#### 参数说明

表1 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
config | 输入 | 量化过程中使用到的标量量化参数。  
  
#### 返回值说明

无

#### 约束说明

无

#### 调用示例

完整示例可参考[完整示例](atlasascendc_api_07_00130.html#ZH-CN_TOPIC_0000002521039734__li178441955134010)。
    
    
    float tmp = (float)0.5;
    // 将float的tmp转换成uint64_t的deqScalar
    uint64_t deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp)); 
    AscendC::SetFixpipePreQuantFlag(deqScalar);  // 设置量化参数
    AscendC::PipeBarrier<PIPE_FIX>();
    

**父主题：** [数据搬运](atlasascendc_api_07_00171.html)


# SetFixPipeConfig

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | x  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

[DataCopy](atlasascendc_api_07_00130.html)（CO1->GM、CO1->A1）过程中进行随路量化时，通过调用该接口设置量化流程中tensor量化参数。

#### 函数原型
    
    
    template <typename T>
    __aicore__ inline void SetFixPipeConfig(const LocalTensor<T>& reluPre, const LocalTensor<T>& quantPre, bool isUnitFlag = false)
    template <typename T, bool setRelu = false>
    __aicore__ inline void SetFixPipeConfig(const LocalTensor<T>& preData, bool isUnitFlag = false)
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
T | 操作数的数据类型。  
setRelu | 针对设置一个tensor的情况，当setRelu为true时，设置reluPre；反之设置quantPre。setRelu当前仅支持设置为false。  
  
表2 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
reluPre | 输入 | 源操作数，relu操作时参与计算的tensor，类型为LocalTensor，支持的TPosition为C2PIPE2GM。 reluPre为预留参数，暂未启用，为后续的功能扩展做保留，传入一个空LocalTensor即可。  
quantPre | 输入 | 源操作数，quant tensor，量化操作时参与计算的tensor，类型为LocalTensor，支持的TPosition为C2PIPE2GM。  
isUnitFlag | 输入 | UnitFlag配置项，默认值为false。

  * false：关闭UnitFlag配置。
  * true：打开UnitFlag配置。

  
preData | 输入 | 支持设置一个Tensor，通过开关控制是relu Tensor还是quant Tensor，支持的TPosition为C2PIPE2GM。当前仅支持传入quant Tensor。  
  
#### 约束说明

quantPre和reluPre必须是Fixpipe Buffer上的Tensor。

#### 返回值说明

无

#### 调用示例

完整示例可参考[完整示例](atlasascendc_api_07_00130.html#ZH-CN_TOPIC_0000002521039734__li178441955134010)。
    
    
    __aicore__inline void SetFPC(const LocalTensor <int32_t>& reluPreTensor, const LocalTensor <int32_t>& quantPreTensor)
    {
     
        AscendC::LocalTensor<uint64_t> workA1 = inQueueDeqA1.AllocTensor<uint64_t>();
        uint16_t deqSize = 128; // deq tensor的size
        AscendC::DataCopy(workA1, deqGlobal, deqSize); // deqGlobal为量化系数的gm地址
        AscendC::LocalTensor<uint64_t> deqFB = inQueueDeqFB.AllocTensor<uint64_t>(); // deq tensor在Fix上的地址
        uint16_t fbufBurstLen = deqSize / 128;  // l1->fix, burst_len unit is 128Bytes
        AscendC::DataCopyParams dataCopyParams(1, fbufBurstLen, 0, 0);
        AscendC::DataCopy(deqFB, workA1, dataCopyParams); 通过DataCopy搬入C2PIPE2GM。
        AscendC::SetFixPipeConfig(deqFB); // 设置量化tensor
        AscendC::PipeBarrier<PIPE_FIX>();
    }
    

**父主题：** [数据搬运](atlasascendc_api_07_00171.html)


# SetFixpipeNz2ndFlag

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | x  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

[DataCopy](atlasascendc_api_07_00130.html)（CO1->GM、CO1->A1）过程中进行随路格式转换（NZ格式转换为ND格式）时，通过调用该接口设置格式转换的相关配置。

#### 函数原型
    
    
    __aicore__ inline void SetFixpipeNz2ndFlag(uint16_t ndNum, uint16_t srcNdStride, uint16_t dstNdStride)
    

#### 参数说明

表1 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
ndNum | 输入 | nd的数量，类型是uint16_t，取值范围：ndNum∈[1, 65535]。  
srcNdStride | 输入 | 以分形大小为单位的源步长，源相邻nz矩阵的偏移（头与头）。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，srcNdStride∈[1, 512]，单位：fractal_size 1024B。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，srcNdStride∈[1, 512]，单位：fractal_size 1024B。 Atlas 200I/500 A2 推理产品，srcNdStride∈[1, 512]，单位：fractal_size 1024B。  
dstNdStride | 输入 | 目的相邻nd矩阵的偏移（头与头）。单位为元素。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，dstNdStride∈[1, 65535]。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，dstNdStride∈[1, 65535]。 Atlas 200I/500 A2 推理产品，dstNdStride∈[1, 65535]。  
  
#### 返回值说明

无

#### 约束说明

无

#### 调用示例

完整示例可参考[完整示例](atlasascendc_api_07_00130.html#ZH-CN_TOPIC_0000002521039734__li178441955134010)。
    
    
    uint16_t ndNum = 2;
    uint16_t srcNdStride = 2;
    uint16_t dstNdStride = 1;
    AscendC::SetFixpipeNz2ndFlag(ndNum, srcNdStride, dstNdStride); // 设置FIX搬运NZ格式到ND格式转换的参数
    

**父主题：** [数据搬运](atlasascendc_api_07_00171.html)


# SetFixpipePreQuantFlag

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | x  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

[DataCopy](atlasascendc_api_07_00130.html)（CO1->GM、CO1->A1）过程中进行随路量化时，通过调用该接口设置量化流程中标量量化参数。

#### 函数原型
    
    
    template<template T>
    __aicore__ inline void SetFixpipePreQuantFlag(uint64_t config)
    

#### 参数说明

表1 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
config | 输入 | 量化过程中使用到的标量量化参数。  
  
#### 返回值说明

无

#### 约束说明

无

#### 调用示例

完整示例可参考[完整示例](atlasascendc_api_07_00130.html#ZH-CN_TOPIC_0000002521039734__li178441955134010)。
    
    
    float tmp = (float)0.5;
    // 将float的tmp转换成uint64_t的deqScalar
    uint64_t deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp)); 
    AscendC::SetFixpipePreQuantFlag(deqScalar);  // 设置量化参数
    AscendC::PipeBarrier<PIPE_FIX>();
    

**父主题：** [数据搬运](atlasascendc_api_07_00171.html)


# SetFixPipeClipRelu

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | x  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

[DataCopy](atlasascendc_api_07_00130.html)（CO1->GM）过程中进行随路量化后，通过调用该接口设置ClipRelu操作的最大值。

ClipRelu计算公式为min(clipReluMaxVal，srcData)，clipReluMaxVal为通过该接口设置的最大值，srcData为源数据。

#### 函数原型
    
    
    __aicore__ inline void SetFixPipeClipRelu(uint64_t config)
    

#### 参数说明

表1 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
config | 输入 | clipReluMaxVal，ClipRelu操作中的最大值。clipReluMaxVal只占用0-15bit，必须大于0，不能为INF/NAN。  
  
#### 约束说明

使能Relu的情况下，先进行Relu操作，之后再进行ClipRelu。

#### 返回值说明

无

#### 调用示例

完整示例可参考[完整示例](atlasascendc_api_07_00130.html#ZH-CN_TOPIC_0000002521039734__li152921471718)。
    
    
    // 使能Relu的情况下，先进行Relu操作，之后再进行ClipRelu。value 1, half类型转换成uint64_t类型
    uint64_t clipReluMaxVal = 0x3c00;
    SetFixPipeClipRelu(clipReluMaxVal);
    

**父主题：** [数据搬运](atlasascendc_api_07_00171.html)


# SetFixPipeAddr

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | x  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

[DataCopy](atlasascendc_api_07_00130.html)（CO1->GM）过程中进行随路量化后，通过调用该接口设置Elementwise操作时LocalTensor的地址。

#### 函数原型
    
    
    template <typename T>
    __aicore__ inline void SetFixPipeAddr(const LocalTensor<T>& eleWiseData, uint16_t c0ChStride)
    

#### 参数说明

表1 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
eleWiseData | 输入 | L1 Buffer上的源操作数。类型为LocalTensor。 支持的TPosition为A1/B1/C1。起始地址需要保证32字节对齐，仅支持half数据类型。  
c0ChStride | 输入 | 在L1 Buffer上的C0 channel stride，单位是C0_SIZE（32B）。 eleWiseData沿N方向以C0为单位切分得到的数据块称为C0 channel，两块C0 channel的间隔称之为C0 channel stride。  
  
#### 约束说明

无

#### 返回值说明

无

#### 调用示例

完整示例可参考[完整示例](atlasascendc_api_07_00130.html#ZH-CN_TOPIC_0000002521039734__li152921471718)。
    
    
    __aicore__inline void SetEleSrcPara(const LocalTensor <half>& eleWiseData, uint16_t c0ChStride)
    {
        AscendC::SetFixPipeAddr(eleWiseData, c0ChStride);
    }
    

**父主题：** [数据搬运](atlasascendc_api_07_00171.html)



---

## LoadData


# Fill

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

将Global Memory上的数据初始化为指定值。该接口可用于对workspace地址或输出数据进行清零。

#### 函数原型
    
    
    template <typename T>
    __aicore__ inline void Fill(GlobalTensor<T>& gmWorkspaceAddr, const uint64_t size, const T value)
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 含义  
---|---  
T | 操作数的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：uint16_t、int16_t、half、uint32_t、int32_t、float。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：uint16_t、int16_t、half、uint32_t、int32_t、float。 Atlas 推理系列产品AI Core，支持的数据类型为：uint16_t、int16_t、half、uint32_t、int32_t、float。  
  
表2 接口参数说明

展开

参数名 | 输入/输出 | 含义  
---|---|---  
gmWorkspaceAddr | 输入 | gmWorkspaceAddr为用户定义的全局Global空间，是需要被初始化的空间，类型为GlobalTensor。GlobalTensor数据结构的定义请参考[GlobalTensor](atlasascendc_api_07_0007.html)。  
size | 输入 | 需要初始化的空间大小，单位为元素个数。  
value | 输入 | 初始化的值，数据类型与gmWorkspaceAddr保持一致。  
  
#### 返回值说明

无

#### 约束说明

  * 单核调用此接口时，如果后续操作涉及Unified Buffer的使用，则需要在调用接口后，设置MTE2流水等待MTE3流水（[MTE3_MTE2](atlasascendc_api_07_0270.html#ZH-CN_TOPIC_0000002552079879__section622mcpsimp)）的同步。
  * 当多个核调用此接口对Global Memory进行初始化时，所有核对Global Memory的初始化未必会同时结束，也可能存在核之间读后写、写后读以及写后写等数据依赖问题。这种使用场景下，可以在本接口后调用[SyncAll](atlasascendc_api_07_0204.html)接口保证多核间同步正确。
  * 该接口仅支持在程序内存分配[InitBuffer](atlasascendc_api_07_0110.html)接口前使用。



#### 调用示例

本调用示例使用8个核，每个核用当前blockIdx的值初始化zGm上的65536个数，每个核的核内计算为x和y两组全1的65536个half类型数据相加，计算结果累加到zGm。此样例中8个核的blockIdx分别为0到7，输入x和y均为全1数据，则最终zGm输出数据为2到9。
    
    
    #include "kernel_operator.h"
    
    constexpr int32_t INIT_SIZE = 65536;
    
    zGm.SetGlobalBuffer((__gm__ float*)z + INIT_SIZE * AscendC::GetBlockIdx(), INIT_SIZE);
    AscendC::Fill(zGm, INIT_SIZE, (float)(AscendC::GetBlockIdx()));
    

结果示例如下：
    
    
    输入数据(x):
    [1. 1. 1. 1. 1. ... 1.]
    输入数据(y):
    [1. 1. 1. 1. 1. ... 1.]
    输出数据(z):
    [2. 2. 2. 2. 2. ... 2.
    3. 3. 3. 3. 3. ... 3.
    4. 4. 4. 4. 4. ... 4.
    5. 5. 5. 5. 5. ... 5.
    6. 6. 6. 6. 6. ... 6.
    7. 7. 7. 7. 7. ... 7.
    8. 8. 8. 8. 8. ... 8.
    9. 9. 9. 9. 9. ... 9.]
    

**父主题：** [张量变换](atlasascendc_api_07_0864.html)


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


# LoadDataWithTranspose

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | x  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

该接口实现带转置的2D格式数据从A1/B1到A2/B2的加载。

下面通过示例来讲解接口功能和关键参数：下文图中一个N形或者一个Z形代表一个分形。

  * 对于uint8_t/int8_t数据类型，每次迭代处理32*32*1B数据，可处理2个分形（一个分形512B），每次迭代中，源操作数中2个连续的16*32分形将被合并为1个32*32的方块矩阵，基于方块矩阵做转置，转置后分裂为2个16*32分形，根据目的操作数分形间隔等参数可以有不同的排布。

如下图示例：
    * 共需要处理3072B的数据，每次迭代处理32*32*1B数据，需要3次迭代可以完成，repeatTime = 3；
    * srcStride = 1，表示相邻迭代间，源操作数前一个方块矩阵与后一个方块矩阵起始地址的间隔为1（单位：32*32*1B），这里的单位实际上是拼接后的方块矩阵的大小；
    * dstGap = 1，表示相邻迭代间，目的操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址的间隔为1（单位：512B）；
    * dstFracGap = 0，表示每个迭代内目的操作数前一个分形的结束地址与后一个分形起始地址的间隔为0（单位：512B）。

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002552120729.png)

如下图示例：

    * repeatTime和srcStride的解释和上图示例一致。
    * dstGap = 0，表示相邻迭代间，目的操作数前一个迭代第一个分形的结束地址和下一个迭代第一个分形起始地址无间隔。
    * dstFracGap = 2，表示每个迭代内目的操作数前一个分形的结束地址与后一个分形起始地址的间隔为2（单位：512B）。

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002552080731.png)



  * 对于half/bfloat16_t数据类型，每次迭代处理16*16*2B数据，可处理1个分形（一个分形512B），每次迭代中，源操作数中1个16*16分形将被转置。
    * 共需要处理1536B的数据，每次迭代处理16*16*2B数据，需要3次迭代可以完成，repeatTime = 3；
    * srcStride = 1，表示相邻迭代间，源操作数前一个方块矩阵与后一个方块矩阵起始地址的间隔为1 （单位：16*16*2B）；
    * dstGap = 0，表示相邻迭代间，目的操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址无间隔；
    * 该场景下，因为其分形即为方块矩阵，每个迭代处理一个分形，不存在迭代内分形的间隔，该参数设置无效。

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002520880752.png)



  * 对于float/int32_t/uint32_t数据类型，每次迭代处理16*16*4B数据，可处理2个分形（一个分形512B），每次迭代中，源操作数2个连续的16*8分形将被合并为1个16*16的方块矩阵，基于方块矩阵做转置，转置后分裂为2个16*8分形，根据目的操作数分形间隔等参数可以有不同的排布。

如下图示例：
    * 共需要处理3072B的数据，每次迭代处理16*16*4B数据，需要3次迭代可以完成，repeatTime = 3；
    * srcStride = 1，表示相邻迭代间，源操作数前一个方块矩阵与后一个方块矩阵起始地址的间隔为1（单位：16*16*4B），这里的单位实际上是拼接后的方块矩阵的大小；
    * dstGap = 1，表示相邻迭代间，目的操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址的间隔为1（单位：512B）；
    * dstFracGap = 0，表示每个迭代内目的操作数前一个分形结束地址与后一个分形起始地址的间隔为0（单位：512B）。

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002521040742.png)

如下图示例：

    * repeatTime和srcStride的解释和上图示例一致。
    * dstGap = 0，表示相邻迭代间，目的操作数前一个迭代第一个分形的结束地址和下一个迭代第一个分形起始地址无间隔。
    * dstFracGap = 2，表示每个迭代内目的操作数前一个分形结束地址与后一个分形起始地址的间隔为2（单位：512B）。

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002552080733.png)

  * 对于int4b_t数据类型，每次迭代处理64*64*0.5B数据，可处理4个分形（一个分形512B），每次迭代中，源操作数中4个连续的16*64分形将被合并为1个64*64的方块矩阵，基于方块矩阵做转置，转置后分裂为4个16*64分形，根据目的操作数分形间隔等参数可以有不同的排布。

int4b_t数据类型需要两个数拼成一个int8_t或uint8_t的数，拼凑的规则如下：

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002521040746.png)

如下图示例：
    * 共需要处理6144B的数据，每次迭代处理64*64*0.5B数据，需要3次迭代可以完成，repeatTime = 3；
    * srcStride = 1，表示相邻迭代间，源操作数前一个方块矩阵与后一个方块矩阵起始地址的间隔为1（单位：64*64*0.5B），这里的单位实际上是拼接后的方块矩阵的大小；
    * dstGap = 1，表示相邻迭代间，目的操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址的间隔为1（单位：512B）；
    * dstFracGap = 0，表示每个迭代内目的操作数前一个分形的结束地址与后一个分形起始地址的间隔为0（单位：512B）。

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002552120721.png)

如下图示例：

    * repeatTime和srcStride的解释和上图示例一致。
    * dstGap = 0，表示相邻迭代间，目的操作数前一个迭代第一个分形的结束地址和下一个迭代第一个分形起始地址无间隔。
    * dstFracGap = 2，表示每个迭代内目的操作数前一个分形的结束地址与后一个分形起始地址的间隔为2（单位：512B）。

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002552080729.png)




#### 函数原型
    
    
    template <typename T>
    __aicore__ inline void LoadDataWithTranspose(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LoadData2dTransposeParams& loadDataParams)
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
T | Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：int4b_t/int8_t/uint8_t/half/bfloat16_t/float/int32_t/uint32_t。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：int4b_t/int8_t/uint8_t/half/bfloat16_t/float/int32_t/uint32_t。 Atlas 200I/500 A2 推理产品，支持的数据类型为：int4b_t/uint8_t/int8_t/uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/float。 其中int4b_t数据类型仅在LocalTensor的TPosition为B2时支持。  
  
表2 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
dst | 输出 | 目的操作数，结果矩阵，类型为LocalTensor。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的TPosition为A2/B2。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的TPosition为A2/B2。 Atlas 200I/500 A2 推理产品，支持的TPosition为A2/B2。 LocalTensor的起始地址需要保证512字节对齐。 数据类型和src的数据类型保持一致。  
src | 输入 | 源操作数，类型为LocalTensor。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的TPosition为A1/B1。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的TPosition为A1/B1。 Atlas 200I/500 A2 推理产品，支持的TPosition为A1/B1。 LocalTensor的起始地址需要保证32字节对齐。 数据类型和dst的数据类型保持一致。  
loadDataParams | 输入 | LoadDataWithTranspose相关参数，类型为LoadData2dTransposeParams。 具体定义请参考${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_mm.h，${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。 参数说明请参考[表3](#ZH-CN_TOPIC_0000002520879646__table13526111319538)。  
  
表3 LoadData2dTransposeParams结构体内参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
startIndex | 输入 | 方块矩阵ID，搬运起始位置为源操作数中第几个方块矩阵（0 为源操作数中第1个方块矩阵）。取值范围：startIndex∈[0, 65535] 。默认为0。 例如，源操作数中有20个大小为16*8*4B的分形（数据类型为float），startIndex=1表示搬运起始位置为第2个方块矩阵，即将第3和第4个分形从源操作数中转置到目的操作数中（第1、2个分形组成第1个方块矩阵，第3、4个分形组成第2个方块矩阵）。  
repeatTimes | 输入 | 迭代次数。 对于uint8_t/int8_t数据类型，每次迭代处理32*32*1B数据； 对于half/bfloat16_t数据类型，每次迭代处理16*16*2B数据； 对于float/int32_t/uint32_t数据类型，每次迭代处理16*16*4B数据。 对于int4b_t数据类型，每次迭代处理16*64*0.5B数据。 取值范围：repeatTimes∈[0, 255]。默认为0。  
srcStride | 输入 | 相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔。这里的单位实际上是拼接后的方块矩阵的大小。 对于uint8_t/int8_t数据类型，单位是32*32*1B； 对于half/bfloat16_t数据类型，单位是16*16*2B； 对于float/int32_t/uint32_t数据类型，单位是16*16*4B。 对于int4b_t数据类型，每次迭代处理16*64*0.5B数据。 取值范围：srcStride∈[0, 65535]。默认为0。  
dstGap | 输入 | 相邻迭代间，目的操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址的间隔，单位：512B。取值范围：dstGap∈[0, 65535]。默认为0。  
dstFracGap | 输入 | 每个迭代内目的操作数转置前一个分形结束地址与后一个分形起始地址的间隔，单位为512B，仅在数据类型为float/int32_t/uint32_t/uint8_t/int8_t/int4b_t时有效。取值范围：dstFracGap∈[0, 65535]。默认为0。  
addrMode | 输入 | 控制地址更新方式，默认为false：

  * true：递减，每次迭代在前一个地址的基础上减去srcStride。
  * false：递增，每次迭代在前一个地址的基础上加上srcStride。

  
  
#### 约束说明

  * repeat=0表示不执行搬运操作。
  * 开发者需要保证目的操作数转置后的分形没有重叠。
  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。



#### 调用示例

  * 示例1：该示例输入a矩阵为int8_t类型，shape为[16,32]，输入b矩阵为int8_t类型，shape为[32,64]，输出c的类型为int32_t。a矩阵从A1->A2不转置，b矩阵从B1->B2转置，之后进行Mmad计算和Fixpipe计算。
        
        #include "kernel_operator.h"
        
        template <typename dst_T, typename fmap_T, typename weight_T, typename dstCO1_T> class KernelMatmul {
        public:
            __aicore__ inline KernelMatmul()
            {
                aSize = m * k;
                bSize = k * n;
                cSize = m * n;
                nBlocks = n / 16;
            }
            __aicore__ inline void Init(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *c)
            {
                aGM.SetGlobalBuffer((__gm__ fmap_T *)a);
                bGM.SetGlobalBuffer((__gm__ weight_T *)b);
                cGM.SetGlobalBuffer((__gm__ dstCO1_T *)c);
                pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(fmap_T));
                pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(fmap_T));
                pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(weight_T));
                pipe.InitBuffer(inQueueB2, 2, bSize * sizeof(weight_T));
                pipe.InitBuffer(outQueueCO1, 1, cSize * sizeof(dstCO1_T));
            }
            __aicore__ inline void Process()
            {
                CopyIn();
                SplitA();
                SplitB();
                Compute();
                CopyOut();
            }
        
        private:
            __aicore__ inline void CopyIn()
            {
                AscendC::LocalTensor<fmap_T> a1Local = inQueueA1.AllocTensor<fmap_T>();
                AscendC::LocalTensor<weight_T> b1Local = inQueueB1.AllocTensor<weight_T>();
        
                AscendC::Nd2NzParams dataCopyA1Params;
                dataCopyA1Params.ndNum = 1;
                dataCopyA1Params.nValue = m;
                dataCopyA1Params.dValue = k;
                dataCopyA1Params.srcNdMatrixStride = 0;
                dataCopyA1Params.srcDValue = k;
                dataCopyA1Params.dstNzC0Stride = m;
                dataCopyA1Params.dstNzNStride = 1;
                dataCopyA1Params.dstNzMatrixStride = 0;
                AscendC::DataCopy(a1Local, aGM, dataCopyA1Params);
        
                AscendC::Nd2NzParams dataCopyB1Params;
                dataCopyB1Params.ndNum = 1;
                dataCopyB1Params.nValue = k;
                dataCopyB1Params.dValue = n;
                dataCopyB1Params.srcNdMatrixStride = 0;
                dataCopyB1Params.srcDValue = n;
                dataCopyB1Params.dstNzC0Stride = k;
                dataCopyB1Params.dstNzNStride = 1;
                dataCopyB1Params.dstNzMatrixStride = 0;
                AscendC::DataCopy(b1Local, bGM, dataCopyB1Params);
        
                inQueueA1.EnQue(a1Local);
                inQueueB1.EnQue(b1Local);
            }
            __aicore__ inline void SplitA()
            {
                AscendC::LocalTensor<fmap_T> a1Local = inQueueA1.DeQue<fmap_T>();
                AscendC::LocalTensor<fmap_T> a2Local = inQueueA2.AllocTensor<fmap_T>();
        
                AscendC::LoadData2DParams loadL0AParams;
                loadL0AParams.repeatTimes = aSize * sizeof(fmap_T) / 512;
                loadL0AParams.srcStride = 1;
                loadL0AParams.ifTranspose = false;
                AscendC::LoadData(a2Local, a1Local, loadL0AParams);
        
                inQueueA2.EnQue<fmap_T>(a2Local);
                inQueueA1.FreeTensor(a1Local);
            }
            __aicore__ inline void SplitB()
            {
                AscendC::LocalTensor<weight_T> b1Local = inQueueB1.DeQue<weight_T>();
                AscendC::LocalTensor<weight_T> b2Local = inQueueB2.AllocTensor<weight_T>();
        
                AscendC::LoadData2dTransposeParams loadDataParams;
                loadDataParams.startIndex = 0;
                nBlockSize = 32;
                loadDataParams.repeatTimes = n / nBlockSize;
                loadDataParams.srcStride = 1;
                loadDataParams.dstGap = 1;
                loadDataParams.dstFracGap = 0;
                AscendC::LoadDataWithTranspose(b2Local, b1Local, loadDataParams);
        
                inQueueB1.FreeTensor(b1Local);
                inQueueB2.EnQue<weight_T>(b2Local);
            }
            __aicore__ inline void Compute()
            {
                AscendC::LocalTensor<fmap_T> a2Local = inQueueA2.DeQue<fmap_T>();
                AscendC::LocalTensor<weight_T> b2Local = inQueueB2.DeQue<weight_T>();
                AscendC::LocalTensor<dstCO1_T> c1Local = outQueueCO1.AllocTensor<dstCO1_T>();
        
                AscendC::MmadParams mmadParams;
                mmadParams.m = m;
                mmadParams.n = n;
                mmadParams.k = k;
                AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);
        
                outQueueCO1.EnQue<dstCO1_T>(c1Local);
                inQueueA2.FreeTensor(a2Local);
                inQueueB2.FreeTensor(b2Local);
            }
            __aicore__ inline void CopyOut()
            {
                AscendC::LocalTensor<dstCO1_T> c1Local = outQueueCO1.DeQue<dstCO1_T>();
                AscendC::FixpipeParamsV220 fixpipeParams;
                fixpipeParams.nSize = n;
                fixpipeParams.mSize = m;
                fixpipeParams.srcStride = m;
                fixpipeParams.dstStride = n;
        
                fixpipeParams.ndNum = 1;
                fixpipeParams.srcNdStride = 0;
                fixpipeParams.dstNdStride = 0;
                AscendC::Fixpipe(cGM, c1Local, fixpipeParams);
                outQueueCO1.FreeTensor(c1Local);
            }
        
        private:
            AscendC::TPipe pipe;
        
            AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
            AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
            AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
            AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
            // dst queue
            AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;
        
            AscendC::GlobalTensor<fmap_T> aGM;
            AscendC::GlobalTensor<weight_T> bGM;
            AscendC::GlobalTensor<dst_T> cGM;
        
            uint16_t m = 16, k = 32, n = 64;
            uint8_t nBlockSize = 16;
            uint16_t c0Size = 16;
            uint16_t aSize, bSize, cSize, nBlocks;
        };
        
        extern "C" __global__ __aicore__ void cube_matmul_loaddata_operator_int8_t(__gm__ uint8_t *a, __gm__ uint8_t *b,
            __gm__ uint8_t *c)
        {
            KernelMatmul<dst_type, fmap_type, weight_type, dstCO1_type> op;
            op.Init(a, b, c);
            op.Process();
        }
        

  * 示例2：该示例输入a矩阵为half类型，shape为[16,32]，输入b矩阵为half类型，shape为[32,32]，输出c的类型为float。a矩阵从A1->A2不转置，b矩阵从B1->B2转置，之后进行Mmad计算和Fixpipe计算。
        
        #include "kernel_operator.h"
        
        template <typename dst_T, typename fmap_T, typename weight_T, typename dstCO1_T> class KernelMatmul {
        public:
            __aicore__ inline KernelMatmul()
            {
                aSize = m * k;
                bSize = k * n;
                cSize = m * n;
                nBlocks = n / 16;
            }
            __aicore__ inline void Init(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *c)
            {
                aGM.SetGlobalBuffer((__gm__ fmap_T *)a);
                bGM.SetGlobalBuffer((__gm__ weight_T *)b);
                cGM.SetGlobalBuffer((__gm__ dstCO1_T *)c);
                pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(fmap_T));
                pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(fmap_T));
                pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(weight_T));
                pipe.InitBuffer(inQueueB2, 2, bSize * sizeof(weight_T));
                pipe.InitBuffer(outQueueCO1, 1, cSize * sizeof(dstCO1_T));
            }
            __aicore__ inline void Process()
            {
                CopyIn();
                SplitA();
                SplitB();
                Compute();
                CopyOut();
            }
        
        private:
            __aicore__ inline void CopyIn()
            {
                AscendC::LocalTensor<fmap_T> a1Local = inQueueA1.AllocTensor<fmap_T>();
                AscendC::LocalTensor<weight_T> b1Local = inQueueB1.AllocTensor<weight_T>();
        
                AscendC::Nd2NzParams dataCopyA1Params;
                dataCopyA1Params.ndNum = 1;
                dataCopyA1Params.nValue = m;
                dataCopyA1Params.dValue = k;
                dataCopyA1Params.srcNdMatrixStride = 0;
                dataCopyA1Params.srcDValue = k;
                dataCopyA1Params.dstNzC0Stride = m;
                dataCopyA1Params.dstNzNStride = 1;
                dataCopyA1Params.dstNzMatrixStride = 0;
                AscendC::DataCopy(a1Local, aGM, dataCopyA1Params);
        
                AscendC::Nd2NzParams dataCopyB1Params;
                dataCopyB1Params.ndNum = 1;
                dataCopyB1Params.nValue = k;
                dataCopyB1Params.dValue = n;
                dataCopyB1Params.srcNdMatrixStride = 0;
                dataCopyB1Params.srcDValue = n;
                dataCopyB1Params.dstNzC0Stride = k;
                dataCopyB1Params.dstNzNStride = 1;
                dataCopyB1Params.dstNzMatrixStride = 0;
                AscendC::DataCopy(b1Local, bGM, dataCopyB1Params);
        
                inQueueA1.EnQue(a1Local);
                inQueueB1.EnQue(b1Local);
            }
            __aicore__ inline void SplitA()
            {
                AscendC::LocalTensor<fmap_T> a1Local = inQueueA1.DeQue<fmap_T>();
                AscendC::LocalTensor<fmap_T> a2Local = inQueueA2.AllocTensor<fmap_T>();
        
                AscendC::LoadData2DParams loadL0AParams;
                loadL0AParams.repeatTimes = aSize * sizeof(fmap_T) / 512;
                loadL0AParams.srcStride = 1;
                loadL0AParams.ifTranspose = false;
                AscendC::LoadData(a2Local, a1Local, loadL0AParams);
        
                inQueueA2.EnQue<fmap_T>(a2Local);
                inQueueA1.FreeTensor(a1Local);
            }
            __aicore__ inline void SplitB()
            {
                AscendC::LocalTensor<weight_T> b1Local = inQueueB1.DeQue<weight_T>();
                AscendC::LocalTensor<weight_T> b2Local = inQueueB2.AllocTensor<weight_T>();
        
                AscendC::LoadData2dTransposeParams loadDataParams;
                loadDataParams.startIndex = 0;
                nBlockSize = 16;
                loadDataParams.repeatTimes = k / nBlockSize;
                loadDataParams.srcStride = 1;
                loadDataParams.dstGap = 1;
                for (int i = 0; i < (n / nBlockSize); ++i) {
                    AscendC::LoadDataWithTranspose(b2Local[i * 16 * nBlockSize], b1Local[i * k * nBlockSize], loadDataParams);
                }
        
                inQueueB1.FreeTensor(b1Local);
                inQueueB2.EnQue<weight_T>(b2Local);
            }
            __aicore__ inline void Compute()
            {
                AscendC::LocalTensor<fmap_T> a2Local = inQueueA2.DeQue<fmap_T>();
                AscendC::LocalTensor<weight_T> b2Local = inQueueB2.DeQue<weight_T>();
                AscendC::LocalTensor<dstCO1_T> c1Local = outQueueCO1.AllocTensor<dstCO1_T>();
        
                AscendC::MmadParams mmadParams;
                mmadParams.m = m;
                mmadParams.n = n;
                mmadParams.k = k;
                AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);
        
                outQueueCO1.EnQue<dstCO1_T>(c1Local);
                inQueueA2.FreeTensor(a2Local);
                inQueueB2.FreeTensor(b2Local);
            }
            __aicore__ inline void CopyOut()
            {
                AscendC::LocalTensor<dstCO1_T> c1Local = outQueueCO1.DeQue<dstCO1_T>();
                AscendC::FixpipeParamsV220 fixpipeParams;
                fixpipeParams.nSize = n;
                fixpipeParams.mSize = m;
                fixpipeParams.srcStride = m;
                fixpipeParams.dstStride = n;
        
                fixpipeParams.ndNum = 1;
                fixpipeParams.srcNdStride = 0;
                fixpipeParams.dstNdStride = 0;
                AscendC::Fixpipe(cGM, c1Local, fixpipeParams);
                outQueueCO1.FreeTensor(c1Local);
            }
        
        private:
            AscendC::TPipe pipe;
        
            AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
            AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
            AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
            AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
            // dst queue
            AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;
        
            AscendC::GlobalTensor<fmap_T> aGM;
            AscendC::GlobalTensor<weight_T> bGM;
            AscendC::GlobalTensor<dst_T> cGM;
        
            uint16_t m = 16, k = 32, n = 32;
            uint8_t nBlockSize = 16;
            uint16_t c0Size = 16;
            uint16_t aSize, bSize, cSize, nBlocks;
        };
        
        extern "C" __global__ __aicore__ void cube_matmul_loaddata_operator_half(__gm__ uint8_t *a, __gm__ uint8_t *b,
            __gm__ uint8_t *c)
        {
            KernelMatmul<dst_type, fmap_type, weight_type, dstCO1_type> op;
            op.Init(a, b, c);
            op.Process();
        }
        

  * 示例3：该示例输入a矩阵为float类型，shape为[16,16]，输入b矩阵为float类型，shape为[16,32]，输出c的类型为float。a矩阵从A1->A2不转置，b矩阵从B1->B2转置，之后进行Mmad计算和Fixpipe计算。
        
        #include "kernel_operator.h"
        
        template <typename dst_T, typename fmap_T, typename weight_T, typename dstCO1_T> class KernelMatmul {
        public:
            __aicore__ inline KernelMatmul()
            {
                aSize = m * k;
                bSize = k * n;
                cSize = m * n;
                nBlocks = n / 16;
            }
            __aicore__ inline void Init(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *c)
            {
                aGM.SetGlobalBuffer((__gm__ fmap_T *)a);
                bGM.SetGlobalBuffer((__gm__ weight_T *)b);
                cGM.SetGlobalBuffer((__gm__ dstCO1_T *)c);
                pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(fmap_T));
                pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(fmap_T));
                pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(weight_T));
                pipe.InitBuffer(inQueueB2, 2, bSize * sizeof(weight_T));
                pipe.InitBuffer(outQueueCO1, 1, cSize * sizeof(dstCO1_T));
            }
            __aicore__ inline void Process()
            {
                CopyIn();
                SplitA();
                SplitB();
                Compute();
                CopyOut();
            }
        
        private:
            __aicore__ inline void CopyIn()
            {
                AscendC::LocalTensor<fmap_T> a1Local = inQueueA1.AllocTensor<fmap_T>();
                AscendC::LocalTensor<weight_T> b1Local = inQueueB1.AllocTensor<weight_T>();
        
                AscendC::Nd2NzParams dataCopyA1Params;
                dataCopyA1Params.ndNum = 1;
                dataCopyA1Params.nValue = m;
                dataCopyA1Params.dValue = k;
                dataCopyA1Params.srcNdMatrixStride = 0;
                dataCopyA1Params.srcDValue = k;
                dataCopyA1Params.dstNzC0Stride = m;
                dataCopyA1Params.dstNzNStride = 1;
                dataCopyA1Params.dstNzMatrixStride = 0;
                AscendC::DataCopy(a1Local, aGM, dataCopyA1Params);
        
                AscendC::Nd2NzParams dataCopyB1Params;
                dataCopyB1Params.ndNum = 1;
                dataCopyB1Params.nValue = k;
                dataCopyB1Params.dValue = n;
                dataCopyB1Params.srcNdMatrixStride = 0;
                dataCopyB1Params.srcDValue = n;
                dataCopyB1Params.dstNzC0Stride = k;
                dataCopyB1Params.dstNzNStride = 1;
                dataCopyB1Params.dstNzMatrixStride = 0;
                AscendC::DataCopy(b1Local, bGM, dataCopyB1Params);
        
                inQueueA1.EnQue(a1Local);
                inQueueB1.EnQue(b1Local);
            }
            __aicore__ inline void SplitA()
            {
                AscendC::LocalTensor<fmap_T> a1Local = inQueueA1.DeQue<fmap_T>();
                AscendC::LocalTensor<fmap_T> a2Local = inQueueA2.AllocTensor<fmap_T>();
        
                AscendC::LoadData2DParams loadL0AParams;
                loadL0AParams.repeatTimes = aSize * sizeof(fmap_T) / 512;
                loadL0AParams.srcStride = 1;
                loadL0AParams.ifTranspose = false;
                AscendC::LoadData(a2Local, a1Local, loadL0AParams);
        
                inQueueA2.EnQue<fmap_T>(a2Local);
                inQueueA1.FreeTensor(a1Local);
            }
            __aicore__ inline void SplitB()
            {
                AscendC::LocalTensor<weight_T> b1Local = inQueueB1.DeQue<weight_T>();
                AscendC::LocalTensor<weight_T> b2Local = inQueueB2.AllocTensor<weight_T>();
        
                AscendC::LoadData2dTransposeParams loadDataParams;
                loadDataParams.startIndex = 0;
                nBlockSize = 16;
                loadDataParams.repeatTimes = n / nBlockSize;
                loadDataParams.srcStride = 1;
                loadDataParams.dstGap = 0;
                loadDataParams.dstFracGap = n / nBlockSize - 1;
                AscendC::LoadDataWithTranspose(b2Local, b1Local, loadDataParams);
        
                inQueueB1.FreeTensor(b1Local);
                inQueueB2.EnQue<weight_T>(b2Local);
            }
            __aicore__ inline void Compute()
            {
                AscendC::LocalTensor<fmap_T> a2Local = inQueueA2.DeQue<fmap_T>();
                AscendC::LocalTensor<weight_T> b2Local = inQueueB2.DeQue<weight_T>();
                AscendC::LocalTensor<dstCO1_T> c1Local = outQueueCO1.AllocTensor<dstCO1_T>();
        
                AscendC::MmadParams mmadParams;
                mmadParams.m = m;
                mmadParams.n = n;
                mmadParams.k = k;
                AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);
        
                outQueueCO1.EnQue<dstCO1_T>(c1Local);
                inQueueA2.FreeTensor(a2Local);
                inQueueB2.FreeTensor(b2Local);
            }
            __aicore__ inline void CopyOut()
            {
                AscendC::LocalTensor<dstCO1_T> c1Local = outQueueCO1.DeQue<dstCO1_T>();
                AscendC::FixpipeParamsV220 fixpipeParams;
                fixpipeParams.nSize = n;
                fixpipeParams.mSize = m;
                fixpipeParams.srcStride = m;
                fixpipeParams.dstStride = n;
        
                fixpipeParams.ndNum = 1;
                fixpipeParams.srcNdStride = 0;
                fixpipeParams.dstNdStride = 0;
                AscendC::Fixpipe(cGM, c1Local, fixpipeParams);
                outQueueCO1.FreeTensor(c1Local);
            }
        
        private:
            AscendC::TPipe pipe;
        
            AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
            AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
            AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
            AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
            // dst queue
            AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;
        
            AscendC::GlobalTensor<fmap_T> aGM;
            AscendC::GlobalTensor<weight_T> bGM;
            AscendC::GlobalTensor<dst_T> cGM;
        
            uint16_t m = 16, k = 16, n = 32;
            uint8_t nBlockSize = 16;
            uint16_t c0Size = 16;
            uint16_t aSize, bSize, cSize, nBlocks;
        };
        
        extern "C" __global__ __aicore__ void cube_matmul_loaddata_operator_float(__gm__ uint8_t *a, __gm__ uint8_t *b,
            __gm__ uint8_t *c)
        {
            KernelMatmul<dst_type, fmap_type, weight_type, dstCO1_type> op;
            op.Init(a, b, c);
            op.Process();
        }
        




**父主题：** [数据搬运](atlasascendc_api_07_00171.html)


# SetAippFunctions

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

设置图片预处理（AIPP，AI core pre-process）相关参数。和[LoadImageToLocal](atlasascendc_api_07_0241.html)接口配合使用。设置后，调用[LoadImageToLocal](atlasascendc_api_07_0241.html)接口可在搬运过程中完成图像预处理操作：包括数据填充，通道交换，单行读取、数据类型转换、通道填充、色域转换。调用SetAippFunctions接口时需传入源图片在Global Memory上的矩阵、源图片的图片格式。

  * **数据填充** ：在图片HW方向上padding。分为如下几种模式： 
    * 模式0：常量填充模式，padding区域各位置填充为常数，支持设置每个通道填充的常数。该模式下仅支持左右padding，不支持上下padding。 

**图1** 常量填充模式（图片中间的绿色区域表示原始数据，其他为padding数据）   


![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002521041436.png)

    * 模式1：行列填充模式，padding区域各位置填充行/列上最邻近源图片位置的数据。 

**图2** 行列填充模式（图片中间的绿色区域表示原始数据，其他为padding数据）   


![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002552081401.png)

    * 模式2：块填充模式，按照padding的宽高，从源图片拷贝数据块进行padding区域填充。 

**图3** 块填充模式（图片中间的绿色区域表示原始数据，其他为padding数据）   


![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002520881438.png)

    * 模式3：镜像块填充模式，按照padding的宽高，从源图片拷贝数据块的镜像进行padding区域填充。 

**图4** 镜像块填充模式（图片中间的绿色区域表示原始数据，其他为padding数据）   


![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002552081411.png)



  * **通道交换** ：将图片通道进行交换。 

对于RGB888格式，支持交换R和B通道。

对于YUV420SP格式，支持交换U和V通道。

对于XRGB8888格式，支持X通道后移（XRGB->RGBX）、支持交换R和B通道。



  * **单行读取** ：源图片中仅读取一行。 

说明

调用数据搬运接口时，开启单行读取后设置的目的图片高度参数无效，如[LoadImageToLocal](atlasascendc_api_07_0241.html)接口的loadImageToLocalParams.vertSize。



  * **数据类型转换** ：转换像素的数据类型，支持uint8_t转换为int8_t或half。当uint8_t转换成int8_t的时候，输出数据范围限制在[-128， 127]。 
        
        // 例1：实现uint8_t ->int8_t 的类型转换，同时实现零均值化：设置每个通道mean值为该通道所有数据的平均值（min和var值无效，不用设置）。
        output[i][j][k] = input[i][j][k] - mean[k]
        // 例2：实现uint8_t -> fp16 的类型转换，同时实现归一化：设置每个通道mean值为该通道所有数据的平均值，min值为该通道所有数据零均值化后的最小值，var值为该通道所有数据的最大值减最小值的倒数。
        uint8_t -> fp16:  output[i][j][k] = (input[i][j][k] - mean[k] - min[k]) * var[k]
        

说明

转换后的数据类型是由模板参数U配置，U为uint8_t时数据类型转换功能不生效。

调用数据搬运接口时，目的Tensor的数据类型需要与本接口输出数据类型保持一致，如[LoadImageToLocal](atlasascendc_api_07_0241.html)的dstLocal参数的数据类型。



  * **通道填充** ：在图片通道方向上padding。默认为模式0。 

模式0：将通道padding至32Bytes。即输出数据类型为uint8_t/int8_t时，padding至32通道；输出数据类型为fp16时，padding至16通道。

模式1：将通道padding至4通道。



  * **色域转换** ：RGB格式转换为YUV格式，或YUV模式转换为RGB格式。 

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_formulaimage_0000002521041424.png)

![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_formulaimage_0000002521041408.png)




#### 函数原型

  * 输入图片格式为YUV400、RGB888、XRGB8888 
        
        template<typename T, typename U>
        __aicore__ inline void SetAippFunctions(const GlobalTensor<T>& src0, AippInputFormat format, AippParams<U> config)
        

  * 输入图片格式为YUV420 Semi-Planar 
        
        template<typename T, typename U>
        __aicore__ inline void SetAippFunctions(const GlobalTensor<T>& src0, const GlobalTensor<T>& src1, AippInputFormat format, AippParams<U> config)
        




#### 参数说明

表1 模板参数说明

展开

参数名称 |  含义  
---|---  
T |  输入的数据类型，需要与format中设置的数据类型保持一致。  
U |  输出的数据类型，需要在搬运接口配置同样的数据类型，如[LoadImageToLocal](atlasascendc_api_07_0241.html)的dstLocal参数数据类型。

  * 如果不使能数据类型转换功能，需要与输入类型保持一致；
  * 如果使能数据类型转换功能，需要与期望转换后的类型保持一致。

  
  
表2 参数说明

展开

参数名称 |  输入/输出 |  含义  
---|---|---  
src0 |  输入 |  源图片在Global Memory上的矩阵。 源图片格式为YUV420SP时，表示Y维度在Global Memory上的矩阵。  
src1 |  输入 |  源图片格式为YUV420SP时，表示UV维度在Global Memory上的矩阵。 源图片格式为其他格式时，该参数无效。  
format |  输入 |  源图片的图片格式。AippInputFormat为枚举类型，取值为： AippInputFormat::YUV420SP_U8：图片格式为YUV420 Semi-Planar，数据类型为uint8_t AippInputFormat::XRGB8888_U8：图片格式为XRGB8888，数据类型为uint8_t AippInputFormat::RGB888_U8：图片格式为RGB888，数据类型为uint8_t AippInputFormat::YUV400_U8：图片格式为YUV400，数据类型为uint8_t
    
    
    enum class AippInputFormat : uint8_t {
        YUV420SP_U8 = 0,
        XRGB8888_U8 = 1,
        RGB888_U8 = 4,
        YUV400_U8 = 9,
    };  
  
config |  输入 |  图片预处理的相关参数，类型为AippParams，结构体具体定义为：
    
    
    template <typename T>
    struct AippParams {
        AippPaddingParams<T> paddingParams;
        AippSwapParams swapParams;
        AippSingleLineParams singleLineParams;
        AippDataTypeConvParams dtcParams;
        AippChannelPaddingParams<T> cPaddingParams;
        AippColorSpaceConvParams cscParams;
    };
    

AippParams结构体内各子结构体定义如下：

  * 数据填充功能相关参数，说明见[表3](#ZH-CN_TOPIC_0000002520879926__table8955841508)。 
        
        template <typename T>
        struct AippPaddingParams {
            uint32_t paddingMode;
            T paddingValueCh0;
            T paddingValueCh1;
            T paddingValueCh2;
            T paddingValueCh3;
        };
        

  * 通道交换功能相关参数，说明见[表4](#ZH-CN_TOPIC_0000002520879926__table679014222918)。 
        
        struct AippSwapParams {
            bool isSwapRB;
            bool isSwapUV;
            bool isSwapAX;
        };
        

  * 单行读取功能相关参数，说明见[表5](#ZH-CN_TOPIC_0000002520879926__table193501032193419)。 
        
        struct AippSingleLineParams {
            bool isSingleLineCopy;
        };
        



  * 数据类型转换功能相关参数，说明见[表6](#ZH-CN_TOPIC_0000002520879926__table14611192613519)。 
        
        struct AippDataTypeConvParams {
            uint8_t dtcMeanCh0{ 0 };
            uint8_t dtcMeanCh1{ 0 };
            uint8_t dtcMeanCh2{ 0 };
            half dtcMinCh0{ 0 };
            half dtcMinCh1{ 0 };
            half dtcMinCh2{ 0 };
            half dtcVarCh0{ 1.0 };
            half dtcVarCh1{ 1.0 };
            half dtcVarCh2{ 1.0 };
            uint32_t dtcRoundMode{ 0 };
        };
        



  * 通道填充功能相关参数，说明见[表7](#ZH-CN_TOPIC_0000002520879926__table163681812917)。 
        
        template <typename T>
        struct AippChannelPaddingParams {
            uint32_t cPaddingMode;
            T cPaddingValue;
        };
        

  * 色域转换功能相关参数，说明见[表8](#ZH-CN_TOPIC_0000002520879926__table7858175271018)。 
        
        struct AippColorSpaceConvParams {
            bool isEnableCsc;
            int16_t cscMatrixR0C0;
            int16_t cscMatrixR0C1;
            int16_t cscMatrixR0C2;
            int16_t cscMatrixR1C0;
            int16_t cscMatrixR1C1;
            int16_t cscMatrixR1C2;
            int16_t cscMatrixR2C0;
            int16_t cscMatrixR2C1;
            int16_t cscMatrixR2C2;
            uint8_t cscBiasIn0;
            uint8_t cscBiasIn1;
            uint8_t cscBiasIn2;
            uint8_t cscBiasOut0;
            uint8_t cscBiasOut1;
            uint8_t cscBiasOut2;
        };
        


  
  
表3 AippPaddingParams结构体内参数说明

展开

参数名称 |  输入/输出 |  含义  
---|---|---  
paddingMode |  输入 |  padding的模式，取值范围[0, 3]，默认值为0。 0：常数填充模式，此模式仅支持左右填充。 1：行列拷贝模式。 2：块拷贝模式。 3：镜像块拷贝模式。  
paddingValueCh0 |  输入 |  padding区域中channel0填充的数据，仅常数填充模式有效，数据类型为T，默认值为0。  
paddingValueCh1 |  输入 |  padding区域中channel1填充的数据，仅常数填充模式有效，数据类型为T，默认值为0。  
paddingValueCh2 |  输入 |  padding区域中channel2填充的数据，仅常数填充模式有效，数据类型为T，默认值为0。  
paddingValueCh3 |  输入 |  padding区域中channel3填充的数据，仅常数填充模式有效，数据类型为T，默认值为0。  
  
表4 AippSwapParams结构体内参数说明

展开

参数名称 |  输入/输出 |  含义  
---|---|---  
isSwapRB |  输入 |  对于RGB888、XRGB8888格式，是否交换R和B通道。默认值为false。  
isSwapUV |  输入 |  对于YUV420SP格式，是否交换U和V通道。默认值为false。  
isSwapAX |  输入 |  对于XRGB8888格式，是否将X通道后移，即XRGB->RGBX。默认值为false。  
  
表5 AippSingleLineParams结构体内参数说明

展开

参数名称 |  输入/输出 |  含义  
---|---|---  
isSingleLineCopy |  输入 |  是否开启单行读取模式。开启后，仅从源图片读取一行。默认值为false。  
  
表6 AippDataTypeConvParams结构体内参数说明

展开

参数名称 |  输入/输出 |  含义  
---|---|---  
dtcMeanCh0 |  输入 |  计算公式内的mean值，channel0，数据类型为uint8_t，默认值为0。  
dtcMeanCh1 |  输入 |  计算公式内的mean值，channel1，数据类型为uint8_t，默认值为0。  
dtcMeanCh2 |  输入 |  计算公式内的mean值，channel2，数据类型为uint8_t，默认值为0。  
dtcMinCh0 |  输入 |  计算公式内的min值，channel0，数据类型为half，默认值为0。 Atlas 200I/500 A2 推理产品 不支持配置该参数。  
dtcMinCh1 |  输入 |  计算公式内的min值，channel1，数据类型为half，默认值为0。 Atlas 200I/500 A2 推理产品 不支持配置该参数。  
dtcMinCh2 |  输入 |  计算公式内的min值，channel2，数据类型为half，默认值为0。 Atlas 200I/500 A2 推理产品 不支持配置该参数。  
dtcVarCh0 |  输入 |  计算公式内的var值，channel0，数据类型为half，默认值为1.0。  
dtcVarCh1 |  输入 |  计算公式内的var值，channel1，数据类型为half，默认值为1.0。  
dtcVarCh2 |  输入 |  计算公式内的var值，channel2，数据类型为half，默认值为1.0。  
dtcRoundMode |  输入 |  控制dtc做数据类型转换的模式，数据类型为uint32_t，默认值为0。 0：四舍五入到最接近的整数值（C语言round）。 1：四舍五入到最接近的偶数（C语言rint）。 仅 Atlas 200I/500 A2 推理产品 支持配置该参数。  
  
表7 AippChannelPaddingParams结构体内参数说明

展开

参数名称 |  输入/输出 |  含义  
---|---|---  
cPaddingMode |  输入 |  channel padding的类型，取值范围为[0, 1]，默认值为0。 0：填充到32B。即输出数据类型U为uint8_t/int8_t时填充到32通道，为half时填充到16通道。 1：填充到4通道。  
cPaddingValue |  输入 |  channel padding填充的值，数据类型为T，默认值为0。  
  
表8 AippColorSpaceConvParams结构体内参数说明

展开

参数名称 |  输入/输出 |  含义  
---|---|---  
isEnableCsc |  输入 |  是否开启色域转换功能，默认值为false。  
cscMatrixR0C0 |  输入 |  色域转换矩阵cscMatrix[0][0]。  
cscMatrixR0C1 |  输入 |  色域转换矩阵cscMatrix[0][1]。  
cscMatrixR0C2 |  输入 |  色域转换矩阵cscMatrix[0][2]。  
cscMatrixR1C0 |  输入 |  色域转换矩阵cscMatrix[1][0]。  
cscMatrixR1C1 |  输入 |  色域转换矩阵cscMatrix[1][1]。  
cscMatrixR1C2 |  输入 |  色域转换矩阵cscMatrix[1][2]。  
cscMatrixR2C0 |  输入 |  色域转换矩阵cscMatrix[2][0]。  
cscMatrixR2C1 |  输入 |  色域转换矩阵cscMatrix[2][1]。  
cscMatrixR2C2 |  输入 |  色域转换矩阵cscMatrix[2][2]。  
cscBiasIn0 |  输入 |  RGB转YUV偏置cscBiasIn[0]。YUV转RGB时无效。  
cscBiasIn1 |  输入 |  RGB转YUV偏置cscBiasIn[1]。YUV转RGB时无效。  
cscBiasIn2 |  输入 |  RGB转YUV偏置cscBiasIn[2]。YUV转RGB时无效。  
cscBiasOut0 |  输入 |  YUV转RGB偏置cscBiasOut0[0]。RGB转YUV时无效。  
cscBiasOut1 |  输入 |  YUV转RGB偏置cscBiasOut1[1]。RGB转YUV时无效。  
cscBiasOut2 |  输入 |  YUV转RGB偏置cscBiasOut2[2]。RGB转YUV时无效。  
  
#### 约束说明

  * src0、src1在Global Memory上的地址对齐要求如下： 

展开

图片格式 |  src0 |  src1  
---|---|---  
YUV420SP |  必须2Bytes对齐 |  必须2Bytes对齐  
XRGB8888 |  必须4Bytes对齐 |  -  
RGB888 |  无对齐要求 |  -  
YUV400 |  无对齐要求 |  -  
  


  * 对于XRGB输入格式的数据，芯片在处理的时候会默认丢弃掉第四个通道的数据输出RGB格式的数据，所以如果是X在channel0的场景下，为了达成上述目的，X通道后移的功能必须使能，将输入的通道转换为RGBX；反之如果是X在channel3的场景下，X通道后移的功能必须不使能以输出RGB格式的数据。



#### 返回值说明

无

#### 调用示例

  * 该调用示例支持的运行平台为 Atlas 推理系列产品 AI Core，示例图片格式为YUV420SP。 
        
        uint16_t horizSize = 32, vertSize = 32, horizStartPos = 0, vertStartPos = 0, srcHorizSize = 32, srcVertSize = 32, leftPadSize = 0, rightPadSize = 0;
        uint32_t dstHorizSize = 32, dstVertSize = 32, cSize = 32;
        uint8_t topPadSize = 0, botPadSize = 0;
        uint32_t gmSrc0Size = 0, gmSrc1Size = 0, dstSize = 0;
        AscendC::AippInputFormat inputFormat = AscendC::AippInputFormat::YUV420SP_U8;
        uint32_t cPadMode = 0;
        int8_t cPaddingValue = 0;
        
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
        AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueUB;
        AscendC::LocalTensor<int8_t> featureMapA1 = inQueueA1.AllocTensor<int8_t>();
        uint64_t fm_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fmGlobal.GetPhyAddr()));
                // aipp config
        AscendC::AippParams<int8_t> aippConfig;
        aippConfig.cPaddingParams.cPaddingMode = cPadMode;
        aippConfig.cPaddingParams.cPaddingValue = cPaddingValue;
        // fmGlobal为整张输入图片，src1参数处填入图片UV维度的起始地址
        AscendC::SetAippFunctions(fmGlobal, fmGlobal[gmSrc0Size], inputFormat, aippConfig);
        AscendC::LoadImageToLocal(featureMapA1, { horizSize, vertSize, horizStartPos, vertStartPos, srcHorizSize, topPadSize, botPadSize, leftPadSize, rightPadSize });
        




**父主题：** [数据搬运](atlasascendc_api_07_00171.html)


# LoadImageToLocal

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | √  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

将图像数据从Global Memory搬运到Local Memory。 搬运过程中可以完成图像预处理操作：包括图像翻转，改变图像尺寸（抠图，裁边，缩放，伸展），以及色域转换，类型转换等。图像预处理的相关参数通过[SetAippFunctions](atlasascendc_api_07_0240.html)进行配置。

#### 函数原型
    
    
    template <typename T>
    __aicore__ inline void LoadImageToLocal(const LocalTensor<T>& dst, const LoadImageToLocalParams& loadDataParams)
    

#### 参数说明

表1 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
dst | 输出 | 目的操作数，类型为LocalTensor。 LocalTensor的起始地址需要保证32字节对齐。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型：int8_t、half；支持的TPosition为A1、B1。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型：int8_t、half；支持的TPosition为A1、B1。 Atlas 200I/500 A2 推理产品 支持数据类型为:uint8_t、int8_t、half；支持的TPosition为A1、B1。 Atlas 推理系列产品AI Core，支持的数据类型为：uint8_t、int8_t、half；支持的TPosition为A1、B1。  
loadDataParams | 输入 | LoadData参数结构体，类型为LoadImageToLocalParams。 具体定义请参考${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_mm.h，${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。 参数说明参考[表2](#ZH-CN_TOPIC_0000002520880228__table8955841508)。  
  
表2 LoadImageToLocalParams结构体内参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
horizSize | 输入 | 从源图中加载图片的水平宽度，单位为像素，取值范围：horizSize∈[2, 4095] 。  
vertSize | 输入 | 从源图中加载图片的垂直高度，单位为像素，取值范围：vertSize∈[2, 4095]。  
horizStartPos | 输入 | 加载图片在源图片上的水平起始地址，单位为像素，取值范围：horizStartPos∈[0, 4095] 。默认为0。 注意：当输入图片为YUV420SP、XRGB8888， RGB888和YUV400格式时，该参数需要是偶数。  
vertStartPos | 输入 | 加载图片在源图片上的垂直起始地址，单位为像素，取值范围：vertStartPos∈[0, 4095] 。默认为0。 注意：当输入图片为YUV420SP格式时，该参数需要是偶数。  
srcHorizSize | 输入 | 源图像水平宽度 ，单位为像素，取值范围：srcHorizSize∈[2, 4095] 。 注意：当输入图片为YUV420SP格式时，该参数需要是偶数。  
topPadSize | 输入 | 目的图像顶部填充的像素数 ，取值范围：topPadSize∈[0, 32] ，默认为0。进行数据填充时使用，需要先调用[SetAippFunctions](atlasascendc_api_07_0240.html)通过[AippPaddingParams](atlasascendc_api_07_0240.html#ZH-CN_TOPIC_0000002520879926__table8955841508)配置填充的数值，再通过topPadSize、botPadSize、leftPadSize、rightPadSize配置填充的大小范围。  
botPadSize | 输入 | 目的图像底部填充的像素数，取值范围：botPadSize∈[0, 32] ，默认为0。  
leftPadSize | 输入 | 目的图像左边填充的像素数，取值范围：leftPadSize∈[0, 32] ，默认为0。  
rightPadSize | 输入 | 目的图像右边填充的像素数，取值范围：rightPadSize∈[0, 32] ，默认为0。  
sid | 输入 | 预留参数。为后续的功能做保留，开发者暂时无需关注，使用默认值即可。  
  
#### 约束说明

  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。
  * 加载到dst的图片的大小加padding的大小必须小于等于所在存储空间的大小。
  * 当通过[SetAippFunctions](atlasascendc_api_07_0240.html)配置padding模式为块填充模式或者镜像块填充模式时，因为padding的数据来自于抠出的图片，左右padding的长度（leftPadSize、rightPadSize）必须小于或等于抠图的水平长度（horizSize），上下padding的长度（topPadSize、botPadSize）必须小于或等于抠图的垂直的长度（vertSize）。



#### 返回值说明

无

#### 调用示例

该调用示例支持的运行平台为Atlas 推理系列产品AI Core，示例图片格式为YUV420SP。
    
    
    uint16_t horizSize = 32, vertSize = 32, horizStartPos = 0, vertStartPos = 0, srcHorizSize = 32, srcVertSize = 32, leftPadSize = 0, rightPadSize = 0;
    uint32_t dstHorizSize = 32, dstVertSize = 32, cSize = 32;
    uint8_t topPadSize = 0, botPadSize = 0;
    uint32_t gmSrc0Size = 0, gmSrc1Size = 0, dstSize = 0;
    AscendC::AippInputFormat inputFormat = AscendC::AippInputFormat::YUV420SP_U8;
    uint32_t cPadMode = 0;
    int8_t cPaddingValue = 0;
    
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueUB;
    AscendC::LocalTensor<int8_t> featureMapA1 = inQueueA1.AllocTensor<int8_t>();
    uint64_t fm_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fmGlobal.GetPhyAddr()));
    // aipp config
    AscendC::AippParams<int8_t> aippConfig;
    aippConfig.cPaddingParams.cPaddingMode = cPadMode;
    aippConfig.cPaddingParams.cPaddingValue = cPaddingValue;
    // fmGlobal为整张输入图片，src1参数处填入图片UV维度的起始地址
    AscendC::SetAippFunctions(fmGlobal, fmGlobal[gmSrc0Size], inputFormat, aippConfig);
    AscendC::LoadImageToLocal(featureMapA1, { horizSize, vertSize, horizStartPos, vertStartPos, srcHorizSize, topPadSize, botPadSize, leftPadSize, rightPadSize });
    

**父主题：** [数据搬运](atlasascendc_api_07_00171.html)


# LoadUnzipIndex

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x  
Atlas 200I/500 A2 推理产品 | x  
Atlas 推理系列产品AI Core | √  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

加载GM上的压缩索引表到内部寄存器。

索引表为LoadDataUnzip压缩信息，例如压缩长度等，以获取压缩后的数据。

索引表由压缩工具根据对应的权重数据离线生成。一个LoadUnzipIndex指令可以加载多个索引表，而每个LoadDataUnzip指令只能消耗一个索引表。因此，索引表之间的顺序应该由用户来确定，以确保其与压缩数据的对应性。

#### 函数原型
    
    
    template <typename T = int8_t, typename Std::enable_if<Std::is_same<PrimT<T>, int8_t>::value, bool>::type = true> 
    __aicore__ inline void LoadUnzipIndex(const GlobalTensor<T>& src, uint32_t numOfIndexTabEntry)
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
T | src的数据类型。

  * 当src使用基础数据类型时， 其数据类型必须为uint8_t，否则编译失败。
  * 当src使用[TensorTrait](atlasascendc_api_07_0011.html)类型时， src数据类型T的LiteType必须为int8_t，否则编译失败。

最后一个模板参数仅用于上述数据类型检查，用户无需关注。  
  
表2 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
src | 输入 | 源操作数，索引表地址，类型为GlobalTensor。 src地址必须2字节对齐。src长度必须是512字节的整数倍，最大为32KB。  
numOfIndexTabEntry | 输入 | 输入数据，表示加载的索引表个数。索引表个数必须大于0。  
  
#### 返回值说明

无

#### 约束说明

  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。
  * LoadUnzipIndex必须在任何LoadDataUnzip指令之前执行。
  * LoadUnzipIndex加载的索引表个数必须大于或等于LoadDataUnzip指令执行的次数。



#### 调用示例

该调用示例支持的运行平台为Atlas 推理系列产品AI Core。详细用例请参考[LoadDataUnzip](atlasascendc_api_07_0243.html)。
    
    
    indexGlobal.SetGlobalBuffer((__gm__ int8_t*)indexGm);
    AscendC::LoadUnzipIndex(indexGlobal, numOfIndexTabEntry);
    

**父主题：** [数据搬运](atlasascendc_api_07_00171.html)


# LoadDataUnzip

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x  
Atlas 200I/500 A2 推理产品 | x  
Atlas 推理系列产品AI Core | √  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

将GM上的数据解压并搬运到A1/B1/B2上。执行该API前需要执行[LoadUnzipIndex](atlasascendc_api_07_0242.html)加载压缩索引表。

#### 函数原型
    
    
    template <typename T>
    __aicore__ inline void LoadDataUnzip(const LocalTensor<T>& dst, const GlobalTensor<T>& src)
    

#### 参数说明

表1 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
dst | 输出 | 目的操作数，类型为LocalTensor，支持的TPosition为A1/B1/B2。 LocalTensor的起始地址需要保证：TPosition为A1/B1时，32字节对齐；TPosition为B2时，512B对齐。 支持的数据类型为：int8_t。  
src | 输入 | 源操作数，类型为GlobalTensor。数据类型需要与dst保持一致。  
  
#### 约束说明

  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。



#### 返回值说明

无

#### 调用示例

该调用示例支持的运行平台为Atlas 推理系列产品AI Core。
    
    
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueUB;
    
    AscendC::GlobalTensor<int8_t> weGlobal;
    AscendC::GlobalTensor<int8_t> dstGlobal;
    AscendC::GlobalTensor<int8_t> indexGlobal;
    
    pipe.InitBuffer(inQueueB1, 1, dstLen * sizeof(int8_t));
    pipe.InitBuffer(outQueueUB, 1, dstLen * sizeof(int8_t));
    uint32_t srcLen = 896, dstLen = 1024, numOfIndexTabEntry = 1;
    AscendC::LocalTensor<int8_t> weightB1 = inQueueB1.AllocTensor<int8_t>();
    AscendC::LoadUnzipIndex(indexGlobal, numOfIndexTabEntry);
    AscendC::LoadDataUnzip(weightB1, weGlobal);
    inQueueB1.EnQue(weightB1);
    inQueueB1.FreeTensor(weightB1);
    

**父主题：** [数据搬运](atlasascendc_api_07_00171.html)


# LoadDataWithSparse

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

用于搬运存放在B1里的512B的稠密权重矩阵到B2里，同时读取128B的索引矩阵用于稠密矩阵的稀疏化。索引矩阵的数据类型为int2，需要拼成int8的数据类型，再传入接口。

索引矩阵在一个int8的地址中的排布是逆序排布的，例如：索引矩阵1 2 0 1 0 2 1 0，在地址中的排布为1 0 2 1 0 1 2 0，其中1 0 2 1（对应索引矩阵前四位1 2 0 1）为一个int8，0 1 2 0（对应索引矩阵后四位0 2 1 0）为一个int8。

索引矩阵的功能说明参考[MmadWithSparse](atlasascendc_api_07_0250.html)。

#### 函数原型
    
    
    template <typename T = int8_t, typename U = uint8_t, typename Std::enable_if<Std::is_same<PrimT<T>, int8_t>::value, bool>::type = true, typename Std::enable_if<Std::is_same<PrimT<U>, uint8_t>::value, bool>::type = true>
    __aicore__ inline void LoadDataWithSparse(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<U>& idx, const LoadData2dParams& loadDataParam)
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
T | dst、src的数据类型。  
U | idx的数据类型。

  * 当dst、src、idx为基础数据类型时，T和U必须为uint8_t类型，否则编译失败。


  * 当dst、src、idx为[TensorTrait](atlasascendc_api_07_0011.html)类型时，T和U的LiteType必须为int8_t类型，否则编译失败。

最后两个模板参数仅用于上述数据类型检查，用户无需关注。  
  
表2 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
dst | 输出 | 目的操作数，类型为LocalTensor，支持的TPosition为B2，LocalTensor的起始地址需要512字节对齐。 支持的数据类型为int8_t。 数据连续排列顺序要求为小N大Z格式。  
src | 输入 | 源操作数，类型为LocalTensor，支持的TPosition为B1，LocalTensor的起始地址需要32字节对齐。 支持的数据类型为int8_t。  
idx | 输入 | 源操作数，类型为LocalTensor，支持的TPosition为B1，LocalTensor的起始地址需要32字节对齐。 支持的数据类型为int8_t。  
loadDataParam | 输入 | LoadData参数结构体，LoadData2DParams类型，详细说明参考[LoadData2DParams结构体内参数说明](atlasascendc_api_07_00169.html#ZH-CN_TOPIC_0000002520880136__table8955841508)。  
  
#### 约束说明

  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。
  * repeat=0表示不执行。
  * 每次迭代中的startIndex不能小于零。
  * 不支持转置功能。



#### 返回值说明

无

#### 调用示例

详细用例请参考[MmadWithSparse](atlasascendc_api_07_0250.html)。

**父主题：** [数据搬运](atlasascendc_api_07_00171.html)


# SetFmatrix

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | √  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

用于调用[Load3Dv1/Load3Dv2](atlasascendc_api_07_0238.html)时设置FeatureMap的属性描述。Load3Dv1/Load3Dv2的模板参数isSetFMatrix设置为false时，表示Load3Dv1/Load3Dv2传入的FeatureMap的属性（包括l1H、l1W、padList，参数介绍参考[表4 LoadData3DParamsV1结构体内参数说明](atlasascendc_api_07_00170.html#ZH-CN_TOPIC_0000002552079741__table679014222918)、[表5 LoadData3DParamsV2结构体内参数说明](atlasascendc_api_07_00170.html#ZH-CN_TOPIC_0000002552079741__table193501032193419)）描述不生效，开发者需要通过该接口进行设置。

#### 函数原型
    
    
    __aicore__ inline void SetFmatrix(uint16_t l1H, uint16_t l1W, const uint8_t padList[4], const FmatrixMode& fmatrixMode)
    

#### 参数说明

表1 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
l1H | 输入 | 源操作数height，取值范围：l1H∈[1, 32767]。  
l1W | 输入 | 源操作数width，取值范围：l1W∈[1, 32767] 。  
padList | 输入 | padding列表 [padding_left, padding_right, padding_top, padding_bottom]，每个元素取值范围：[0,255]。默认为{0, 0, 0, 0}。  
fmatrixMode | 输入 | 用于控制LoadData指令从left还是right寄存器获取信息。FmatrixMode类型，定义如下。当前只支持FMATRIX_LEFT，左右矩阵均使用该配置。
    
    
    enum class FmatrixMode : uint8_t {
        FMATRIX_LEFT = 0,
        FMATRIX_RIGHT = 1,
    }; 
      
  
#### 约束说明

  * 该接口需要配合load3Dv1/load3Dv2接口一起使用，需要在load3Dv1/load3Dv2接口之前调用。
  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。



#### 调用示例
    
    
    AscendC::TPipe pipe;
    
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueFmA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueFmA2;
    // weight queue
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueWeB1;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueWeB2;
    pipe.InitBuffer(inQueueFmA1, 1, featureMapA1Size * sizeof(fmap_T));
    pipe.InitBuffer(inQueueFmA2, 1, featureMapA2Size * sizeof(fmap_T));
    pipe.InitBuffer(inQueueWeB1, 1, weightA1Size * sizeof(weight_T));
    pipe.InitBuffer(inQueueWeB2, 1, weightB2Size * sizeof(weight_T));
    pipe.InitBuffer(outQueueCO1, 1, dstCO1Size * sizeof(dstCO1_T));
    
    AscendC::LocalTensor<fmap_T> featureMapA1 = inQueueFmA1.DeQue<fmap_T>();
    AscendC::LocalTensor<weight_T> weightB1 = inQueueWeB1.DeQue<weight_T>();
    AscendC::LocalTensor<fmap_T> featureMapA2 = inQueueFmA2.AllocTensor<fmap_T>();
    AscendC::LocalTensor<weight_T> weightB2 = inQueueWeB2.AllocTensor<weight_T>();
    uint16_t channelSize = 32;
    uint16_t H = 4, W = 4;
    uint8_t Kh = 2, Kw = 2;
    uint16_t Cout = 16;
    uint16_t C0, C1;
    uint8_t dilationH = 2, dilationW = 2;
    
    uint8_t padList[PAD_SIZE] = {0, 0, 0, 0};
    AscendC::SetFmatrix(H, W, padList, FmatrixMode::FMATRIX_LEFT);
    AscendC::SetLoadDataPaddingValue(0);
    AscendC::SetLoadDataRepeat({0, 1, 0});
    AscendC::SetLoadDataBoundary((uint32_t)0);
    static constexpr AscendC::IsResetLoad3dConfig LOAD3D_CONFIG = {false,false};
    AscendC::LoadData<fmap_T, LOAD3D_CONFIG>(featureMapA2, featureMapA1,
        { padList, H, W, channelSize, k, howoRound, 0, 0, 1, 1, Kw, Kh, dilationW, dilationH, false, false, 0 });
    AscendC::LoadData(weightB2, weightB1, { 0, weRepeat, 1, 0, 0, false, 0 });
    
    inQueueFmA2.EnQue<fmap_T>(featureMapA2);
    inQueueWeB2.EnQue<weight_T>(weightB2);
    inQueueFmA1.FreeTensor(featureMapA1);
    inQueueWeB1.FreeTensor(weightB1);
    

**父主题：** [数据搬运](atlasascendc_api_07_00171.html)


# SetLoadDataBoundary

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

设置[Load3D](atlasascendc_api_07_0238.html)时A1/B1边界值。

如果Load3D指令在处理源操作数时，源操作数在A1/B1上的地址超出设置的边界，则会从A1/B1起始地址开始读取数据。

#### 函数原型
    
    
    __aicore__ inline void SetLoadDataBoundary(uint32_t boundaryValue)
    

#### 参数说明

表1 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
boundaryValue | 输入 | 边界值。 Load3Dv1指令：单位是32字节。 Load3Dv2指令：单位是字节。  
  
#### 约束说明

  * 用于Load3Dv1时，boundaryValue的最小值是16（单位：32字节）；用于Load3Dv2时，boundaryValue的最小值是1024（单位：字节）。
  * 如果使用SetLoadDataBoundary接口设置了边界值，配合Load3D指令使用时，Load3D指令的A1/B1初始地址要在设置的边界内。
  * 如果boundaryValue设置为0，则表示无边界，可使用整个A1/B1。
  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。



#### 调用示例

参考[调用示例](atlasascendc_api_07_0245.html#ZH-CN_TOPIC_0000002552079805__section642mcpsimp)。

**父主题：** [数据搬运](atlasascendc_api_07_00171.html)


# SetLoadDataRepeat

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | x  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

用于设置[Load3Dv2接口](atlasascendc_api_07_00170.html#ZH-CN_TOPIC_0000002552079741__li83241850104315)的repeat参数。设置repeat参数后，可以通过调用一次Load3Dv2接口完成多个迭代的数据搬运。

#### 函数原型
    
    
    __aicore__ inline void SetLoadDataRepeat(const LoadDataRepeatParam& repeatParams)
    

#### 参数说明

表1 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
repeatParams | 输入 | 设置Load3Dv2接口的repeat参数，类型为LoadDataRepeatParam。 具体定义请参考${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_mm.h，${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。 参数说明请参考[表2](#ZH-CN_TOPIC_0000002552119613__table15780447181917)。  
  
表2 LoadDataRepeatParam结构体参数说明

展开

参数名称 | 含义  
---|---  
repeatTime | height/width方向上的迭代次数，取值范围：repeatTime ∈[0, 255] 。默认值为1。  
repeatStride | height/width方向上的前一个迭代与后一个迭代起始地址的距离，取值范围：n∈[0, 65535]，默认值为0。

  * repeatMode为0，repeatStride的单位为16个元素。
  * repeatMode为1，repeatStride的单位和具体型号有关。下文中的data_type指Load3Dv2中源操作数的数据类型。Atlas A2 训练系列产品/Atlas A2 推理系列产品，repeatStride的单位为32/sizeof(data_type)个元素 。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，repeatStride的单位为32/sizeof(data_type)个元素 。 Atlas 200I/500 A2 推理产品，repeatStride的单位为64/sizeof(data_type)个元素。

  
repeatMode | 控制repeat迭代的方向，取值范围：k∈[0, 1] 。默认值为0。 0：迭代沿height方向； 1：迭代沿width方向。  
  
#### 调用示例

参考[调用示例](atlasascendc_api_07_0245.html#ZH-CN_TOPIC_0000002552079805__section642mcpsimp)

**父主题：** [数据搬运](atlasascendc_api_07_00171.html)


# SetLoadDataPaddingValue

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | √  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

用于调用[Load3Dv1接口](atlasascendc_api_07_00170.html#ZH-CN_TOPIC_0000002552079741__li1135744819417)/[Load3Dv2接口](atlasascendc_api_07_00170.html#ZH-CN_TOPIC_0000002552079741__li83241850104315)时设置Pad填充的数值。Load3Dv1/Load3Dv2的模板参数isSetPadding设置为true时，用户需要通过本接口设置Pad填充的数值，设置为false时，本接口设置的填充值不生效。

#### 函数原型
    
    
    template <typename T>
    __aicore__ inline void SetLoadDataPaddingValue(const T padValue)
    

#### 参数说明

表1 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
padValue | 输入 | Pad填充值的数值。 Atlas 推理系列产品AI Core，支持的数据类型为：int8_t/uint8_t/half/int16_t/uint16_t Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型：int8_t/uint8_t/half/int16_t/uint16_t/bfloat16_t/int32_t/uint32_t/float Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型：int8_t/uint8_t/half/int16_t/uint16_t/bfloat16_t/int32_t/uint32_t/float Atlas 200I/500 A2 推理产品， 支持数据类型：int8_t/uint8_t/half/int16_t/uint16_t/bfloat16_t/int32_t/uint32_t/float  
  
#### 约束说明

无

#### 调用示例

参考[调用示例](atlasascendc_api_07_0245.html#ZH-CN_TOPIC_0000002552079805__section642mcpsimp)

**父主题：** [数据搬运](atlasascendc_api_07_00171.html)



---

## 同步控制ISASI


> 文档 SetFlag/WaitFlag(ISASI) 未找到


# PipeBarrier(ISASI)

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

阻塞相同流水，具有数据依赖的相同流水之间需要插入此同步。

#### 函数原型
    
    
    template <pipe_t pipe>
    __aicore__ inline void PipeBarrier()
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
pipe | 模板参数，表示阻塞的流水类别。 支持的流水参考[硬件流水类型](atlasascendc_api_07_0179.html#ZH-CN_TOPIC_0000002521039978__section1272612276459)。 如果不关注流水类别，希望阻塞所有流水，可以传入PIPE_ALL。  
  
#### 返回值说明

无

#### 约束说明

Scalar流水之间的同步由硬件自动保证，调用PipeBarrier<PIPE_S>()会引发硬件错误。

#### 调用示例

如下示例，Mul指令的输入dst0Local是Add指令的输出，两个矢量运算指令产生依赖，需要插入PipeBarrier保证两条指令的执行顺序。

注：仅作为示例参考，开启自动同步（Kernel直调算子工程和自定义算子开发工程已默认开启）的情况下，编译器自动插入PIPE_V同步，无需开发者手动插入。

**图1** Mul指令和Add指令是串行关系，必须等待Add指令执行完成后，才能执行Mul指令。  


![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002521040848.png)
    
    
    AscendC::LocalTensor<half> src0Local;
    AscendC::LocalTensor<half> src1Local;
    AscendC::LocalTensor<half> src2Local;
    AscendC::LocalTensor<half> dst0Local;
    AscendC::LocalTensor<half> dst1Local;
    
    AscendC::Add(dst0Local, src0Local, src1Local, 512);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(dst1Local, dst0Local, src2Local, 512);
    

**父主题：** [核内同步](atlasascendc_api_07_00013.html)


# DataSyncBarrier(ISASI)

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | x  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

用于阻塞后续的指令执行，直到所有之前的内存访问指令（需要等待的内存位置可通过参数控制）执行结束。

#### 函数原型
    
    
    template <MemDsbT arg0>
    __aicore__ inline void DataSyncBarrier()
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
arg0 | 模板参数，表示需要等待的内存位置，类型为MemDsbT，可取值为：

  * ALL，等待所有内存访问指令。
  * DDR，等待GM访问指令。
  * UB，等待UB访问指令。
  * SEQ，等待SEQ访问指令。

  
  
#### 返回值说明

无

#### 约束说明

无

#### 调用示例
    
    
    AscendC::Mmad(...);
    AscendC::DataSyncBarrier<MemDsbT::ALL>();
    AscendC::Fixpipe(...);
    

**父主题：** [核内同步](atlasascendc_api_07_00013.html)


# CrossCoreSetFlag(ISASI)

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

面向分离模式的核间同步控制接口。

该接口和[CrossCoreWaitFlag](atlasascendc_api_07_0274.html)接口配合使用。使用时需传入核间同步的标记ID(flagId)，每个ID对应一个用于控制同步的计数器。

同步控制分为以下几种模式，如[图1](#ZH-CN_TOPIC_0000002552119769__fig37581010773)所示：

  * 模式0：AI Core核间的同步控制。对于AIC场景，同步所有的AIC核，直到所有的AIC核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的指令才会执行；对于AIV场景，同步所有的AIV核，直到所有的AIV核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的指令才会执行。
  * 模式1：AI Core内部，AIV核之间的同步控制。如果两个AIV核都运行了CrossCoreSetFlag，CrossCoreWaitFlag后续的指令才会执行。
  * 模式2：AI Core内部，AIC与AIV之间的同步控制。在AIC核执行CrossCoreSetFlag之后， 两个AIV上CrossCoreWaitFlag后续的指令才会继续执行；两个AIV都执行CrossCoreSetFlag后，AIC上CrossCoreWaitFlag后续的指令才能执行。



**图1** 同步控制模式示意图  


![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002552081119.png)

#### 函数原型
    
    
    template <uint8_t modeId, pipe_t pipe>
    __aicore__ inline void CrossCoreSetFlag(uint16_t flagId)
    

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


  * 因为[Matmul高阶API](atlasascendc_api_07_0612.html)内部实现中使用了本接口进行核间同步控制，所以不建议开发者同时使用该接口和Matmul高阶API，否则会有flagID冲突的风险。
  * 同一flagId的计数器最多设置15次。



#### 调用示例
    
    
    // 使用模式0的方式同步所有的AIV核
    if (g_coreType == AscendC::AIV) {
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(0x8);
        AscendC::CrossCoreWaitFlag(0x8);
    }
    
    // 使用模式1的方式同步当前AICore内的所有AIV子核
    if (g_coreType == AscendC::AIV) {
        AscendC::CrossCoreSetFlag<0x1, PIPE_MTE3>(0x8);
        AscendC::CrossCoreWaitFlag(0x8);
    }
    
    // 注意：如果调用高阶API,无需开发者处理AIC和AIV的同步
    // 以Matmul为例：AIC侧做完Matmul计算后通知AIV进行后处理
    if (g_coreType == AscendC::AIC) {
        // Matmul处理
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);
    }
    
    // AIV侧等待AIC Set消息, 进行Vector后处理
    if (g_coreType == AscendC::AIV) {
        AscendC::CrossCoreWaitFlag(0x8);
        // Vector后处理
    }
    

**父主题：** [核间同步](atlasascendc_api_07_0268.html)


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



---

## 缓存ISASI


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


# GetICachePreloadStatus(ISASI)

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

获取ICACHE的PreLoad的状态。

#### 函数原型
    
    
    __aicore__ inline int64_t GetICachePreloadStatus()
    

#### 参数说明

无

#### 返回值说明

int64_t类型，0表示空闲，1表示忙。

#### 约束说明

无

#### 调用示例
    
    
    // 获取预取状态，0-空闲，1-忙
    int64_t cachePreloadStatus = AscendC::GetICachePreloadStatus();
    

**父主题：** [缓存控制](atlasascendc_api_07_0275.html)



---

## 系统变量ISASI


# GetProgramCounter(ISASI)

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

获取程序计数器的指针，程序计数器用于记录当前程序执行的位置。

#### 函数原型
    
    
    __aicore__ inline int64_t GetProgramCounter()
    

#### 参数说明

无

#### 返回值说明

返回int64_t类型的程序计数器指针。

#### 约束说明

无

#### 调用示例
    
    
    int64_t pc = AscendC::GetProgramCounter(); // 获取程序计数器的指针pc
    

**父主题：** [系统变量访问](atlasascendc_api_07_0183.html)


# GetSubBlockNum(ISASI)

#### 产品支持情况

展开

产品 |  是否支持  
---|---  
Atlas A3 训练系列产品 / Atlas A3 推理系列产品  |  √  
Atlas A2 训练系列产品 / Atlas A2 推理系列产品  |  √  
Atlas 200I/500 A2 推理产品  |  x  
Atlas 推理系列产品 AI Core |  x  
Atlas 推理系列产品 Vector Core |  x  
Atlas 训练系列产品  |  x  
  
#### 功能说明

[分离模式](/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html#ZH-CN_TOPIC_0000002552129949__li188191010204418)下，获取一个AI Core上Cube Core（AIC）或者Vector Core（AIV）的数量。

#### 函数原型
    
    
    __aicore__ inline int64_t GetSubBlockNum()
    

#### 参数说明

无

#### 返回值说明

不同Kernel类型下（通过[设置Kernel类型](atlasascendc_api_07_0218.html)设置），在AIC和AIV上调用该接口的返回值如下：

表1 返回值列表

展开

Kernel类型 |  KERNEL_TYPE_AIV_ONLY |  KERNEL_TYPE_AIC_ONLY |  KERNEL_TYPE_MIX_AIC_1_2 |  KERNEL_TYPE_MIX_AIC_1_1 |  KERNEL_TYPE_MIX_AIC_1_0 |  KERNEL_TYPE_MIX_AIV_1_0  
---|---|---|---|---|---|---  
AIV |  1 |  - |  2 |  1 |  - |  1  
AIC |  - |  1 |  1 |  1 |  1 |  -  
  
#### 约束说明

无

#### 调用示例
    
    
    int64_t subBlockNum = AscendC::GetSubBlockNum();
    

**父主题：** [系统变量访问](atlasascendc_api_07_0183.html)


# GetSubBlockIdx(ISASI)

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

获取AI Core上Vector核的ID。

#### 函数原型
    
    
    __aicore__ inline int64_t GetSubBlockIdx()
    

#### 参数说明

无

#### 返回值说明

返回Vector核ID。

#### 约束说明

无

#### 调用示例
    
    
    int64_t subBlockID = AscendC::GetSubBlockIdx();
    

**父主题：** [系统变量访问](atlasascendc_api_07_0183.html)


# GetSystemCycle(ISASI)

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

获取当前系统cycle数，若换算成时间需要按照50MHz的频率，时间单位为us，换算公式为：time = (cycle数/50) us 。

#### 函数原型
    
    
    __aicore__ inline int64_t GetSystemCycle()
    

#### 参数说明

无

#### 返回值说明

返回系统cycle数。

#### 约束说明

该接口是PIPE_S流水，若需要测试其他流水的指令时间，需要在调用该接口前通过[PipeBarrier](atlasascendc_api_07_0271.html)插入对应流水的同步，具体请参考[调用示例](#ZH-CN_TOPIC_0000002520880324__li126441923175612)。

#### 调用示例

  * 如下示例通过GetSystemCycle获取系统cycle数，并换算成时间（单位：us）。
        
        #include "kernel_operator.h"
        
        __aicore__ inline void InitTilingParam(int32_t& totalSize, int32_t& loopSize)
        {
            int64_t systemCycleBefore = AscendC::GetSystemCycle(); // 调用GetBlockNum指令前的cycle数
            loopSize = totalSize / AscendC::GetBlockNum();
            int64_t systemCycleAfter = AscendC::GetSystemCycle(); // 调用GetBlockNum指令后的cycle数
            int64_t GetBlockNumCycle = systemCycleAfter - systemCycleBefore; // 执行GetBlockNum指令所用的cycle数
            int64_t CycleToTimeBase = 50; // cycle数转换成时间的基准单位，固定为50
            int64_t GetBlockNumTime = GetBlockNumCycle/CycleToTimeBase; // 执行GetBlockNum指令所用时间，单位为us
        };
        

  * 如下示例为获取矢量计算Add指令时间的关键代码片段，在调用GetSystemCycle之前，插入了PIPE_ALL同步，可以保证相关指令执行完后再获取cycle数。
        
        PipeBarrier<PIPE_ALL>();
        int64_t systemCycleBefore = AscendC::GetSystemCycle(); // 调用Add指令前的cycle数
        AscendC::Add(dstLocal, src0Local, src1Local, 512);
        PipeBarrier<PIPE_ALL>();
        int64_t systemCycleAfter = AscendC::GetSystemCycle(); // 调用Add指令后的cycle数
        int64_t GetBlockNumCycle = systemCycleAfter - systemCycleBefore; // 执行Add指令所用的cycle数
        int64_t CycleToTimeBase = 50; // cycle数转换成时间的基准单位，固定为50
        int64_t GetBlockNumTime = GetBlockNumCycle/CycleToTimeBase; // 执行Add指令所用时间，单位为us
        




**父主题：** [系统变量访问](atlasascendc_api_07_0183.html)



---

## 原子操作ISASI


# SetAtomicMax(ISASI)

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

原子操作函数，设置后续从VECOUT传输到GM的数据是否执行原子比较：将待拷贝的内容和GM已有内容进行比较，将最大值写入GM。

可通过设置模板参数来设定不同的数据类型。

#### 函数原型
    
    
    template <typename T>
    __aicore__ inline void SetAtomicMax() 
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
T | 设定不同的数据类型。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持int8_t/int16_t/half/bfloat16_t/int32_t/float Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持int8_t/int16_t/half/bfloat16_t/int32_t/float  
  
#### 返回值说明

无

#### 约束说明

  * 使用完后，建议通过[DisableDmaAtomic](atlasascendc_api_07_0212.html)关闭原子最大操作，以免影响后续相关功能。
  * 对于Atlas A2 训练系列产品/Atlas A2 推理系列产品，目前无法对bfloat16_t类型设置inf/nan模式。



#### 调用示例
    
    
    #include "kernel_operator.h"
    
    uint32_t size = 256;
    AscendC::LocalTensor<half> dst0Local = queueDst0.DeQue<half>();
    AscendC::LocalTensor<half> dst1Local = queueDst1.DeQue<half>();
    AscendC::DataCopy(dstGlobal, dst1Local, size);
    AscendC::PipeBarrier<PIPE_MTE3>();
    AscendC::SetAtomicMax<half>();
    AscendC::DataCopy(dstGlobal, dst0Local, size);
    queueDst0.FreeTensor(dst0Local);
    queueDst1.FreeTensor(dst1Local);
    AscendC::DisableDmaAtomic();
    
    每个核的输入数据为: 
    Src0: [1,1,1,1,1,...,1] // 256个1
    Src1: [2,2,2,2,2,...,2] // 256个2
    最终输出数据: [2,2,2,2,2,...,2] // 256个2
    

**父主题：** [原子操作](atlasascendc_api_07_0209.html)


# SetAtomicMin(ISASI)

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

原子操作函数，设置后续从VECOUT传输到GM的数据是否执行原子比较，将待拷贝的内容和GM已有内容进行比较，将最小值写入GM。

可通过设置模板参数来设定不同的数据类型。

#### 函数原型
    
    
    template <typename T>
    __aicore__ inline void SetAtomicMin() 
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
T | 设定不同的数据类型。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持int8_t/int16_t/half/bfloat16_t/int32_t/float。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持int8_t/int16_t/half/bfloat16_t/int32_t/float。  
  
#### 返回值说明

无

#### 约束说明

使用完后，建议通过[DisableDmaAtomic](atlasascendc_api_07_0212.html)关闭原子最小操作，以免影响后续相关指令功能。

#### 调用示例
    
    
    #include "kernel_operator.h"
    
    uint32_t size = 256;
    AscendC::LocalTensor<half> dst0Local = queueDst0.DeQue<half>();
    AscendC::LocalTensor<half> dst1Local = queueDst1.DeQue<half>();
    AscendC::DataCopy(dstGlobal, dst1Local, size);
    AscendC::PipeBarrier<PIPE_MTE3>();
    AscendC::SetAtomicMin<half>();
    AscendC::DataCopy(dstGlobal, dst0Local, size);
    queueDst0.FreeTensor(dst0Local);
    queueDst1.FreeTensor(dst1Local);
    AscendC::DisableDmaAtomic();
    
    每个核的输入数据为: 
    Src0: [1,1,1,1,1,...,1] // 256个1
    Src1: [2,2,2,2,2,...,2] // 256个2
    最终输出数据: [1,1,1,1,1,...,1] // 256个1
    

**父主题：** [原子操作](atlasascendc_api_07_0209.html)


# SetStoreAtomicConfig(ISASI)

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | x  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

设置原子操作使能位与原子操作类型。

#### 函数原型
    
    
    template <AtomicDtype type, AtomicOp op>
    __aicore__ inline void SetStoreAtomicConfig()
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 输入/输出 | 描述  
---|---|---  
type | 输入 | 原子操作使能位，AtomicDtype枚举类的定义如下：
    
    
    enum class AtomicDtype {
        ATOMIC_NONE = 0,  // 无原子操作
        ATOMIC_F32,       // 使能原子操作，进行原子操作的数据类型为float
        ATOMIC_F16,       // 使能原子操作，进行原子操作的数据类型为half
        ATOMIC_S16,       // 使能原子操作，进行原子操作的数据类型为int16_t
        ATOMIC_S32,       // 使能原子操作，进行原子操作的数据类型为int32_t
        ATOMIC_S8,        // 使能原子操作，进行原子操作的数据类型为int8_t
        ATOMIC_BF16       // 使能原子操作，进行原子操作的数据类型为bfloat16_t
    }; 
      
  
op | 输入 | 原子操作类型，仅当使能原子操作时有效（即“type”为非“ATOMIC_NONE”的场景），当前仅支持求和操作。
    
    
    enum class AtomicOp {
        ATOMIC_SUM = 0   // 求和操作
    };
      
  
#### 返回值说明

无

#### 约束说明

无

#### 调用示例
    
    
    // 设置原子操作为求和操作，支持的数据类型为half
    AscendC::SetStoreAtomicConfig<AscendC::AtomicDtype::ATOMIC_F16, AscendC::AtomicOp::ATOMIC_SUM>();
    

**父主题：** [原子操作](atlasascendc_api_07_0209.html)


# GetStoreAtomicConfig(ISASI)

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | x  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

获取原子操作使能位与原子操作类型的值，详细说明见[表1](atlasascendc_api_07_0286.html#ZH-CN_TOPIC_0000002520879680__zh-cn_topic_0235751031_table33761356)。

#### 函数原型
    
    
    __aicore__ inline void GetStoreAtomicConfig(uint16_t& atomicType, uint16_t& atomicOp)
    

#### 参数说明

表1 参数说明

展开

参数名 | 输入/输出 | 描述  
---|---|---  
atomicType | 输出 | 原子操作使能位。 0：无原子操作 1：使能原子操作，进行原子操作的数据类型为float 2：使能原子操作，进行原子操作的数据类型为half 3：使能原子操作，进行原子操作的数据类型为int16_t 4：使能原子操作，进行原子操作的数据类型为int32_t 5：使能原子操作，进行原子操作的数据类型为int8_t 6：使能原子操作，进行原子操作的数据类型为bfloat16_t  
atomicOp | 输出 | 原子操作类型。 0：求和操作  
  
#### 返回值说明

无

#### 约束说明

此接口需要与[SetStoreAtomicConfig(ISASI)](atlasascendc_api_07_0286.html)配合使用，用以获取原子操作使能位与原子操作类型的值。

#### 调用示例
    
    
    AscendC::SetStoreAtomicConfig<AscendC::AtomicDtype::ATOMIC_F16, AscendC::AtomicOp::ATOMIC_SUM>();
    uint16_t type = 0;       // 原子操作使能位
    uint16_t op = 0;         // 原子操作类型
    AscendC::GetStoreAtomicConfig(type, op);
    

**父主题：** [原子操作](atlasascendc_api_07_0209.html)



---

## 调试ISASI


# CheckLocalMemoryIA(ISASI)

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | √  
Atlas 推理系列产品AI Core | √  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

check设定范围内的UB读写行为，如果有设定范围的读写行为则会出现EXCEPTION报错，无设定范围的读写行为则不会报错。

#### 函数原型
    
    
    __aicore__ inline void CheckLocalMemoryIA(const CheckLocalMemoryIAParam& checkParams)
    

#### 参数说明

表1 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
checkParams | 输入 | 用于配置对UB访问的检查行为，类型为CheckLocalMemoryIAParam。 具体定义请参考${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_mm.h，${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。 参数说明请参考[表2](#ZH-CN_TOPIC_0000002552080191__table15780447181917)。  
  
表2 CheckLocalMemoryIAParam结构体内参数说明

展开

参数名称 | 含义  
---|---  
enableBit | 配置的异常寄存器，取值范围：enableBit∈[0,3]，默认为0。

  * 0：异常寄存器0。
  * 1：异常寄存器1。
  * 2：异常寄存器2。
  * 3：异常寄存器3。

  
startAddr | Check的起始地址，32B对齐，取值范围：startAddr∈[0, 65535]，默认值为0。比如，可通过LocalTensor.GetPhyAddr()/32来获取startAddr。  
endAddr | Check的结束地址，32B对齐，取值范围：endAddr∈[0, 65535] 。默认值为0。  
isScalarRead | Check标量读访问。

  * false：不开启，默认为false。
  * true：开启。

  
isScalarWrite | Check标量写访问。

  * false：不开启，默认为false。
  * true：开启。

  
isVectorRead | Check矢量读访问。

  * false：不开启，默认为false。
  * true：开启。

  
isVectorWrite | Check矢量写访问。

  * false：不开启，默认为false。
  * true：开启。

  
isMteRead | Check Mte读访问。

  * false：不开启，默认为false。
  * true：开启。

  
isMteWrite | Check Mte写访问。

  * false：不开启，默认为false。
  * true：开启。

  
isEnable | 是否使能enableBit参数配置的异常寄存器。

  * false：不使能，默认为false。
  * true：使能。

  
reserved | 预留参数。为后续的功能做保留，开发者暂时无需关注，使用默认值即可。  
  
#### 约束说明

  * startAddr/endAddr的单位是32B，check的范围不包含startAddr，包含endAddr，即(startAddr，endAddr]。
  * 每次调用完该接口需要进行复位（配置isEnable为false进行复位）；
  * 操作数地址对齐要求请参见[通用地址对齐约束](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__section796754519912)。



#### 调用示例

该示例check矢量写访问是否在设定的(startAddr, endAddr]范围内。当前示例check到矢量写在设定的范围内，结果会报错（ACL_ERROR_RT_VECTOR_CORE_EXCEPTION）。
    
    
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc0, inQueueSrc1;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst
    pipe.InitBuffer(inQueueSrc0, 1, 512 * sizeof(half));
    pipe.InitBuffer(inQueueSrc1, 1, 512 * sizeof(half));
    pipe.InitBuffer(outQueueDst, 1, 512 * sizeof(half));
    AscendC::LocalTensor<half> src0Local = inQueueSrc0.DeQue<half>();
    AscendC::LocalTensor<half> src1Local = inQueueSrc1.DeQue<half>();
    AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();
    AscendC::CheckLocalMemoryIA({ 0, (uint32_t)(dstLocal.GetPhyAddr() / 32),(uint32_t)((dstLocal.GetPhyAddr() + 512 * sizeof(half)) / 32), false, false, false, true, false, false,
    true });
    

**父主题：** [异常检测](atlasascendc_api_07_00178.html)



---

## Cube分组


# CubeResGroupHandle使用说明

CubeResGroupHandle用于在分离模式下对AI Core计算资源分组。分组后，开发者可以对不同的分组指定不同的计算任务。一个AI Core分组可包含多个AIV和AIC，AIV和AIC之间采取Client和Server架构进行任务处理。AIV为Client，每一个Cube计算任务为一个消息，AIV发送消息至消息队列，AIC作为Server，遍历消息队列的消息，根据消息类型及内容执行对应的计算任务。一个CubeResGroupHandle中可以有一个或多个AIC，同一个AIC只能属于一个CubeResGroupHandle，AIV无此限制，即同一个AIV可以属于多个CubeResGroupHandle。

如下图所示，CubeResGroupHandle1中有2个AIC，10个AIV，AIC为Block0和Block1。其中Block0与Queue0、Queue1、Queue2、Queue3、Queue4进行通信，Block1与Queue 5、Queue 6、Queue 7、Queue 8、Queue9进行通信。每一个消息队列对应一个AIV，消息队列的深度固定为4，即一次性最多可以容纳4个消息。CubeResGroupHandle2的消息队列个数为12，表明有12个AIV。CubeResGroupHandle的消息处理顺序如CubeResGroupHandle2中黑色箭头所示。

**图1** 基于CubeResGroupHandle的AI Core计算资源分组通信示意图   


![](/doc_center/source/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/figure/zh-cn_image_0000002552080857.png)

基于CubeResGroupHandle实现AI Core计算资源分组步骤如下：

  1. 创建AIC上所需要的计算对象类型。
  2. 创建通信区域描述[KfcWorkspace](atlasascendc_api_07_0307.html)，用于记录通信消息Msg的地址分配。
  3. 自定义消息结构体，用于通信。
  4. 自定义回调计算结构体，根据实际业务场景实现Init函数和Call函数。
  5. 创建CubeResGroupHandle。
  6. 绑定AIV到CubeResGroupHandle。
  7. 收发消息。
  8. AIV退出消息队列。



下文仅提供示例代码片段，更多完整样例请参考[CubeGroup样例](https://gitee.com/ascend/samples/blob/master/operator/ascendc/2_features/12_cube_group/CubeGroupCustom)。

  1. 创建AIC上所需要的计算对象类型。

用户根据实际需求，自定义AIC所需要的计算对象类型，或者高阶API已提供的Matmul类型。例如，创建Matmul类型如下，其中A_TYPE、B_TYPE、 C_TYPE、BIAS_TYPE、CFG_NORM等含义请参考[Matmul模板参数](atlasascendc_api_07_0615.html)。
         
         // A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM根据实际需求场景构造
         using MatmulApiType = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, C_TYPE, CFG_NORM>;
         

  2. 创建KfcWorkspace。

使用[KfcWorkspace](atlasascendc_api_07_0307.html)管理不同CubeResGrouphandle的消息通信区的划分。 
         
         // 创建KfcWorkspace对象前，需要对该workspaceGM清零
         KfcWorkspace desc(workspaceGM);
         

  3. 自定义消息结构体。

用户需要自行构造消息结构体[CubeMsgBody](#ZH-CN_TOPIC_0000002552119691__table189051237164018)，用于AIV向AIC发送通信消息。构造的CubeMsgBody必须64字节对齐，该结构体最前面需要定义2字节的CubeGroupMsgHead，使消息收发机制正常运行，CubeGroupMsgHead结构定义请参考[表2](#ZH-CN_TOPIC_0000002552119691__table77221554135216)。除2字节的CubeGroupMsgHead外，其余参数根据业务需求自行构造。 

表1 CubeMsgBody消息结构体

展开

参数名称 |  含义  
---|---  
CubeMsgBody |  用户自定义的消息结构体。结构体名称可自定义，结构体大小需要64字节对齐。
         
         // 这里提供64B对齐的结构体示例，用户实际使用时，除CubeGroupMsgHead外，其他参数个数及参数类型可自行构造
         struct CubeMsgBody {
            CubeGroupMsgHead head;  // 2B，需放在结构体最前面, 自定义的CubeMsgBody中，CubeGroupMsgHead的变量名需设置为head，否则会编译报错。
            uint8_t funcID;
            uint8_t skipCnt;
            uint32_t value;
            bool isTransA;
            bool isTransB;
            bool isAtomic;
            bool isLast;                 
            int32_t tailM;              
            int32_t tailN;
            int32_t tailK;               
            uint64_t aAddr;
            uint64_t bAddr;
            uint64_t cAddr;
            uint64_t aGap;
            uint64_t bGap;
         }
           
  
表2 CubeGroupMsgHead结构体参数定义

展开

参数名称 |  含义  
---|---  
msgState |  表明该位置的消息状态。参数取值如下：
     * CubeMsgState::FREE：表明该位置还未填写消息，可执行[AllocMessage](atlasascendc_api_07_0293.html)。
     * CubeMsgState::VALID：表明该位置已经含有AIV发送的消息，待AIC接收执行。
     * CubeMsgState::QUIT：表明该位置的消息为通知AIC有AIV将退出流程。
     * CubeMsgState::FAKE：表明该位置的消息为假消息。在消息合并场景，被跳过处理任务的AIV需要发送假消息，消息合并场景请参考[PostFakeMsg](atlasascendc_api_07_0295.html)中的介绍。  
aivID |  发送消息的AIV的序号。  
  
  4. 自定义回调计算结构体，根据实际业务场景实现Init函数和Call函数。
         
         template<class MatmulApiCfg, class CubeMsgBody>
         struct NormalCallbackFuncs {
             __aicore__ inline static void Call(MatmulApiCfg &mm, __gm__ CubeMsgBody *rcvMsg, CubeResGroupHandle<CubeMsgBody> &handle){
               // 用户自行实现逻辑
             };
         
             __aicore__ inline static void Init(NormalCallbackFuncs<MatmulApiCfg, CubeMsgBody> &foo, MatmulApiCfg &mm, GM_ADDR tilingGM){
                // 用户自行实现逻辑
             };
            
         };
         

计算逻辑结构体的模板参数请参考[表3](#ZH-CN_TOPIC_0000002552119691__table18865397406)。

表3 模板参数说明

展开

参数 |  说明  
---|---  
MatmulApiCfg |  用户自定义的AIC上计算所需要对象的数据类型，参考[步骤1](#ZH-CN_TOPIC_0000002552119691__li27691150733)，该模板参数必须填入。  
CubeMsgBody |  [用户自定义的消息结构体](#ZH-CN_TOPIC_0000002552119691__table189051237164018)，该模板参数必须填入。  
  
用户自定义回调计算结构体中需要包含固定的Init函数和Call函数，函数原型如下所示。其中，Init函数的参数说明请参考[表4](#ZH-CN_TOPIC_0000002552119691__zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_table111938719446)，Call函数的参数说明请参考[表5](#ZH-CN_TOPIC_0000002552119691__table9997952179)。
         
         // 该函数的参数和名称为固定格式，函数实现根据业务逻辑自行实现。
         __aicore__ inline static void Init(MyCallbackFunc<MatmulApiCfg, CubeMsgBody> &myCallBack, MatmulApiCfg &mm, GM_ADDR tilingGM){
              // 用户自行实现内部逻辑
         }
         

表4 Init函数参数说明

展开

参数 |  输入/输出 |  说明  
---|---|---  
myCallBack |  输入 |  用户自定义的带[模板参数](#ZH-CN_TOPIC_0000002552119691__table18865397406)的回调计算结构体。  
mm |  输入 |  AIC上计算对象，多为Matmul对象。  
tilingGM |  输入 |  用户传入的tiling指针。  
           
         // 该函数的参数和名称为固定格式，函数实现根据业务逻辑自行实现。
         __aicore__ inline static void Call(MatmulApiCfg &mm, __gm__ CubeMsgBody *rcvMsg, CubeResGroupHandle<CubeMsgBody> &handle){
                 // 用户自行实现内部逻辑
         }
         

表5 Call函数参数说明

展开

参数 |  输入/输出 |  说明  
---|---|---  
mm |  输入 |  AIC上计算对象，多为Matmul对象。  
rcvMsg |  输入 |  用户自定义的消息结构体指针。  
handle |  输入 |  分组管理Handle，用户调用其接口进行收发消息，释放消息等。  
  
某算子的回调计算结构体的代码示例如下。 
         
         // 用户自定义的回调计算逻辑
         template<class MatmulApiCfg, typename CubeMsgBody>
         struct MyCallbackFunc
         {
             template<int32_t funcId>
             __aicore__ inline static typename IsEqual<funcId, 0>::Type CubeGroupCallBack(MatmulApiCfg &mm, __gm__ CubeMsgBody *rcvMsg, CubeResGroupHandle<CubeMsgBody> &handle)
             {
                 GlobalTensor<int64_t> msgGlobal;
                 msgGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *> (rcvMsg) + sizeof(int64_t));
                 DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT> (msgGlobal);
                 using SrcAT = typename MatmulApiCfg::AType::T;
                 auto skipNum = 0;
                 for (int i = 0; i < skipNum + 1; ++i)
                 {
                     auto tmpId = handle.FreeMessage(rcvMsg + i); // msgPtr process is complete
                 }
                 handle.SetSkipMsg(skipNum);
             }
             template<int32_t funcId>
             __aicore__ inline static typename IsEqual<funcId, 1>::Type CubeGroupCallBack(MatmulApiCfg &mm, __gm__ CubeMsgBody *rcvMsg, CubeResGroupHandle<CubeMsgBody> &handle)
             {
                 GlobalTensor<int64_t> msgGlobal;
                 msgGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *> (rcvMsg) + sizeof(int64_t));
                 DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT> (msgGlobal);
                 using SrcAT = typename MatmulApiCfg::AType::T;
                 LocalTensor<SrcAT> tensor_temp;
                 auto skipNum = 3;
                 auto tmpId = handle.FreeMessage(rcvMsg, CubeMsgState::VALID);
                 for (int i = 1; i < skipNum + 1; ++i)
                 {
                     auto tmpId = handle.FreeMessage(rcvMsg + i, CubeMsgState::FAKE);
                 }
                 handle.SetSkipMsg(skipNum); // notify the cube not to process
             }
             __aicore__ inline static void Call(MatmulApiCfg &mm, __gm__ CubeMsgBody *rcvMsg, CubeResGroupHandle<CubeMsgBody> &handle)
             {
                 if (rcvMsg->funcId == 0)
                 {
                     CubeGroupCallBack<0> (mm, rcvMsg, handle);
                 }
                 else if(rcvMsg->funcId == 1)
                 {
                     CubeGroupCallBack<1> (mm, rcvMsg, handle);
                 }
             }
             __aicore__ inline static void Init(MyCallbackFunc<MatmulApiCfg, CubeMsgBody> &foo, MatmulApiCfg &mm, GM_ADDR tilingGM)
             {
                 auto tempTilingGM = (__gm__ uint32_t*)tilingGM;
                 auto tempTiling = (uint32_t*)&(foo.tiling);
                 for (int i = 0; i < sizeof(TCubeTiling) / sizeof(int32_t); ++i, ++tempTilingGM, ++tempTiling)
                 {
                     *tempTiling = *tempTilingGM;
                 }
                 mm.SetSubBlockIdx(0);
                 mm.Init(&foo.tiling, GetTPipePtr());
             }
             TCubeTiling tiling;
         };
         

  5. 创建CubeResGroupHandle。

用户使用[CreateCubeResGroup](atlasascendc_api_07_0300.html)接口创建一个或多个CubeResGroupHandle。 
         
         /* 
          * groupID为用户自定义的CreateCubeResGroup的groupID
          * MatmulApiType为定义好的AIC上计算对象的类型
          * MyCallbackFunc为定义好的自定义回调计算结构体
          * CubeMsgBody为自定义消息结构体
          * desc为用户初始化好的通信区域描述
          * groupID为1，blockStart为0，blockSize为12，msgQueueSize为48，tilingGm为指针，存储了用户在AIC上所需要的tiling信息
         */
         auto handle =  AscendC::CreateCubeResGroup<groupID, MatmulApiType, MyCallbackFunc, CubeMsgBody>(desc, 0, 12, 48, tilingGM);
         

  6. 绑定AIV到CubeResGroupHandle。

绑定AIV和消息队列序号。注意：消息队列序号queIdx小于该CubeGroupHandle的消息队列总数，每个AIV需要传入不同的queIdx。handle为[步骤5](#ZH-CN_TOPIC_0000002552119691__li355132105919)中CreateCubeResGroup创建的CubeResGroupHandle对象。 
         
         handle.AssignQueue(queIdx);
         

  7. AIV发消息。

用户调用[AllocMessage](atlasascendc_api_07_0293.html), [PostMessage](atlasascendc_api_07_0294.html)等接口进行消息的收发。其中，调用AllocMessage获取消息结构体指针，通过PostMessage发送消息，在消息合并场景调用[PostFakeMessage](atlasascendc_api_07_0295.html)发送假消息，示例如下。 
         
         CubeGroupMsgHead head = {CubeMsgState::VALID, (uint8_t)queIdx};
         CubeMsgBody aCubeMsgBody {head, 0, 0, 0, false, false, false, false, 0, 0, 0, 0, 0, 0, 0, 0};
         CubeMsgBody bCubeMsgBody {head, 1, 0, 0, false, false, false, false, 0, 0, 0, 0, 0, 0, 0, 0};
         auto offset = 0;
         if (GetBlockIdx() == 0)
         {
             auto msgPtr = handle.template AllocMessage(); // alloc for queue space
             offset = handle.template PostMessage(msgPtr, bCubeMsgBody); // post true msgPtr
             bool waitState = handle.template Wait<true> (offset); // wait until the msgPtr is processed
         }
         else if (GetBlockIdx() < 4)
         {
             auto msgPtr = handle.AllocMessage();
             offset = handle.PostFakeMsg(msgPtr); // post fake msgPtr
             bool waitState = handle.template Wait<true> (offset); // wait until the msgPtr is processed
         }
         else
         {
             auto msgPtr = handle.template AllocMessage();
             offset = handle.template PostMessage(msgPtr, aCubeMsgBody);
             bool waitState = handle.template Wait<true> (offset); // wait until the msgPtr is processed
         }
         

  8. AIV退出消息队列。

调用AllocMessage获取消息结构体指针后，通过SendQuitMsg发送当前消息队列退出。 
         
         auto msgPtr = handle.AllocMessage();        // 获取消息空间指针msgPtr
         handle.SetQuit(msgPtr);              // 发送退出消息
         




**父主题：** [CubeResGroupHandle](atlasascendc_api_07_0289.html)


# GroupBarrier使用说明

当同一个[CubeResGroupHandle](atlasascendc_api_07_0289.html)中的两个AIV任务之间存在依赖关系时，可以使用GroupBarrier控制同步。假设一组AIV A做完任务x以后，另外一组AIV B才可以开始后续业务，称AIV A组为Arrive组，AIV B组为Wait组。

基于GroupBarrier的组同步使用步骤如下：

  1. 创建GroupBarrier。
  2. 被等待的AIV调用Arrive，需要等待的AIV调用Wait。



下文仅提供示例代码片段，更多完整样例请参考[GroupBarrier样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/2_features/16_group_barrier)。

  1. 创建GroupBarrier。
         
         constexpr int32_t ARRIVE_NUM = 2; // Arrive组的AIV个数
         constexpr int32_t WAIT_NUM = 6; // Wait组的AIV个数
         AscendC::GroupBarrier<AscendC::PipeMode::MTE3_MODE> barA(workspace, ARRIVE_NUM, WAIT_NUM);  // 创建GroupBarrier，用户自行管理并对这部分workspace清零
         

  2. 被等待的AIV调用Arrive，需要等待的AIV调用Wait。
         
         auto id = AscendC::GetBlockIdx();
         if (id > 0 && id < ARRIVE_NUM) {
           //各种Vector计算逻辑，用户自行实现
           barA.Arrive(id);
         } else(id >= ARRIVE_NUM && id < ARRIVE_NUM + WAIT_NUM){
           barA.Wait(id - ARRIVE_NUM);
           // 各种Vector计算逻辑，用户自行实现
         }
         




**父主题：** [GroupBarrier](atlasascendc_api_07_0301.html)

