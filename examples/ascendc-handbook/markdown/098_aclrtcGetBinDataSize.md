<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_00158.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# aclrtcGetBinDataSize

# aclrtcGetBinDataSize

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

获取编译的二进制数据大小。用于在[aclrtcGetBinData](atlasascendc_api_07_00157.html)获取二进制数据时分配对应大小的内存空间。

#### 函数原型
    
    
    aclError aclrtcGetBinDataSize(aclrtcProg prog, size_t *binDataSizeRet)
    

#### 参数说明

表1 接口参数说明

展开

参数名 | 输入/输出 | 描述  
---|---|---  
prog | 输入 | 运行时编译程序的句柄。  
binDataSizeRet | 输出 | 编译得到的Device侧二进制数据大小。  
  
#### 返回值说明

aclError为int类型变量，详细说明请参考[RTC错误码](atlasascendc_api_07_00161.html)。

#### 约束说明

无

#### 调用示例
    
    
    aclrtcProg prog;
    size_t binDataSizeRet = 0;
    aclError result = aclrtcGetBinDataSize(prog, &binDataSizeRet);
    

**父主题：** [RTC](atlasascendc_api_07_00162.html)
