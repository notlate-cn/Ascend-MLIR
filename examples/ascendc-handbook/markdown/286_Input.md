<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0946.html -->
<!-- 下载时间: 2026-03-04 11:39:03 -->

# Input

# Input

#### 功能说明

注册算子输入，调用该接口后会返回一个OpParamDef结构，后续可通过该结构配置算子输入信息。

#### 函数原型
    
    
    OpParamDef &Input(const char *name)
    

#### 参数说明

展开

参数 | 输入/输出 | 说明  
---|---|---  
name | 输入 | 算子输入名称。  
  
#### 返回值说明

算子参数定义，OpParamDef实例，具体请参考[OpParamDef](atlasascendc_api_07_0957.html)。

#### 约束说明

参数注册的顺序需要和算子kernel入口函数一致。

**父主题：** [OpDef](atlasascendc_api_07_0946.html)
