<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0247.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# SetLoadDataRepeat

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
