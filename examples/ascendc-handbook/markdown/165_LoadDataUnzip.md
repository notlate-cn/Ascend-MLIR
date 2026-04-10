<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0243.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# LoadDataUnzip

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
