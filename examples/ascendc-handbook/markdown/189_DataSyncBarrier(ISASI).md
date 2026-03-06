<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0272.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# DataSyncBarrier(ISASI)

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
