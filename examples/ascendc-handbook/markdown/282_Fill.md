<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0891.html -->
<!-- 下载时间: 2026-03-04 11:39:03 -->

# Fill

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
