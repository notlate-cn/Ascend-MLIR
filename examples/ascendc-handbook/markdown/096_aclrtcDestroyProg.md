<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_00156.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# aclrtcDestroyProg

# aclrtcDestroyProg

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

销毁编译程序的实例。

#### 函数原型
    
    
    aclError aclrtcDestroyProg(aclrtcProg *prog)
    

#### 参数说明

表1 接口参数说明

展开

参数名 | 输入/输出 | 描述  
---|---|---  
prog | 输入 | 运行时编译程序的句柄。  
  
#### 返回值说明

aclError为int类型变量，详细说明请参考[RTC错误码](atlasascendc_api_07_00161.html)。

#### 约束说明

无

#### 调用示例
    
    
    aclrtcProg prog;
    aclError result = aclrtcDestroyProg(&prog);
    

**父主题：** [RTC](atlasascendc_api_07_00162.html)
