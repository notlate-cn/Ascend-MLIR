<!-- 原始URL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0945.html -->
<!-- 下载时间: 2026-03-04 11:39:03 -->

# 原型注册接口（OP_ADD）

# 原型注册接口（OP_ADD）

#### 功能说明

注册算子的原型定义，从而确保算子能够被框架正确识别、编译和执行****。

算子原型主要描述了算子的输入输出、属性等信息以及算子在AI处理器上相关实现信息，并关联[tiling实现](/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_0064.html)等函数。算子原型通过自定义的算子类来承载，该算子类继承自[OpDef类](atlasascendc_api_07_0946.html)。完成算子的原型定义等操作后，需要调用[OP_ADD](atlasascendc_api_07_0945.html)接口，传入算子类型（自定义算子类的类名），进行算子原型注册。详细内容请参考[算子原型定义](/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_0062.html)。

#### 函数原型
    
    
    OP_ADD(opType)
    

#### 参数说明

展开

参数 |  输入/输出 |  说明  
---|---|---  
opType |  输入 |  算子类型名称  
  
#### 返回值说明

无

#### 约束说明

无

**父主题：** [原型注册与管理](atlasascendc_api_07_0944.html)
