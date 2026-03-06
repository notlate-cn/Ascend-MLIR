<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0172.html -->
<!-- 下载时间: 2026-03-04 11:39:02 -->

# GetUserWorkspace

# GetUserWorkspace

#### 产品支持情况

展开

产品 |  是否支持  
---|---  
Atlas A3 训练系列产品 / Atlas A3 推理系列产品  |  √  
Atlas A2 训练系列产品 / Atlas A2 推理系列产品  |  √  
Atlas 200I/500 A2 推理产品  |  x  
Atlas 推理系列产品 AI Core |  √  
Atlas 推理系列产品 Vector Core |  x  
Atlas 训练系列产品  |  √  
  
#### 功能说明

获取用户使用的workspace指针。workspace的具体介绍请参考[如何使用workspace](/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_0092.html)。Kernel直调开发方式下，如果未开启[HAVE_WORKSPACE](/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_0056.html#ZH-CN_TOPIC_0000002552089991__table481718169817)编译选项，框架不会自动设置系统workspace。如果使用了[Matmul Kernel侧接口](atlasascendc_api_07_0613.html)等需要系统workspace的高阶API，kernel侧需要通过[SetSysWorkSpace](atlasascendc_api_07_0171.html)设置系统workspace，此时用户workspace需要通过该接口获取。

#### 函数原型
    
    
    __aicore__ inline GM_ADDR GetUserWorkspace(GM_ADDR workspace)
    

#### 参数说明

表1 接口参数说明

展开

参数名称 |  输入/输出 |  描述  
---|---|---  
workspace |  输入 |  传入workspace的指针，包括系统workspace和用户使用的workspace。  
  
#### 约束说明

无

#### 返回值说明

用户使用workspace指针。

**父主题：** [workspace](atlasascendc_api_07_0169.html)
