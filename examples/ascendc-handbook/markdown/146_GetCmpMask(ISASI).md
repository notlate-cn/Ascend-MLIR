<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0223.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# GetCmpMask(ISASI)

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
