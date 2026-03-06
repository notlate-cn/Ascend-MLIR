<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_00154.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# aclrtcCompileProg

# aclrtcCompileProg

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

编译接口，编译指定的程序。

#### 函数原型
    
    
    aclError aclrtcCompileProg(aclrtcProg prog, int numOptions, const char **options)
    

#### 参数说明

表1 接口参数说明

展开

参数名 |  输入/输出 |  描述  
---|---|---  
prog |  输入 |  运行时编译程序的句柄。  
numOptions |  输入 |  编译选项数量。  
options |  输入 |  编译选项数组，保存具体的编译选项（默认添加-std=c++17）。 支持的编译选项可以参考[毕昇编译器编译选项](/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/BishengCompiler/atlas_bisheng_10_0010.html)。  
  
#### 返回值说明

aclError为int类型变量，详细说明请参考[RTC错误码](atlasascendc_api_07_00161.html)。

#### 约束说明

无

#### 调用示例
    
    
    aclrtcProg prog;
    const char *options[] = {"--npu-arch=dav-2201"};
    int numOptions = sizeof(options) / sizeof(options[0]);
    aclError result = aclrtcCompileProg(prog, numOptions, options);
    

**父主题：** [RTC](atlasascendc_api_07_00162.html)
