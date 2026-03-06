# 语言扩展层 C API

> 来源: 昇腾社区官网 AscendC算子开发文档

---

## 目录

- [简介](#简介)
- [模板参数](#模板参数)
- [构造函数](#构造函数)
- [随路量化](#随路量化)
- [Async](#async)
- [DEVICE_IMPL_OP_OPTILING](#device_impl_op_optiling)
- [ASCENDC_TPL_SEL_PARAM](#ascendc_tpl_sel_param)

---



---

## 简介


# 简介

ContextBuilder类提供一系列的API接口，支持手动构造[TilingContext](/document/detail/zh/CANNCommunityEdition/900beta1/API/basicdataapi/atlasopapi_07_00223.html)类来验证Tiling函数以及KernelContext类用于TilingParse函数的验证。

#### 调用示例
    
    
    // 构造KernelContext
    auto kernelContextHolder = context_ascendc::ContextBuilder()
        .Inputs(...)
        .Outputs(...)
        .BuildKernelRunContext();
    gert::KernelContext* tilingParseContext = kernelContextHolder->GetContext<gert::KernelContext>();
    
    // 构造TilingContext
    auto tilingContextHolder = context_ascendc::ContextBuilder()
        .SetOpNameType(...,...)
        .NodeIoNum(...)
        .IrInstanceNum(...)
        .AddInputTd(...)
        .AddOutputTd(...)
        .AddAttr(...)
        .BuildTilingContext(...);
    gert::TilingContext* tilingContext = tilingContextHolder->GetContext<gert::TilingContext>();
    

**父主题：** [ContextBuilder](atlasascendc_api_07_1007.html)



---

## 模板参数


# 模板参数定义

#### 功能说明

通过以下函数原型进行模板参数ASCENDC_TPL_ARGS_DECL和模板参数组合ASCENDC_TPL_ARGS_SEL（即可使用的模板）的定义。详细内容请参考[Tiling模板编程](/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_00025.html)。

#### 函数原型
    
    
    // ParamStruct是存放用户设置的模板参数ASCENDC_TPL_ARGS_DECL和模板参数组合ASCENDC_TPL_ARGS_SEL的结构体，用作后续的Tilingkey与模板参数之间的编解码，用户无需关注
    struct ParamStruct {
        const char* name;
        uint32_t paramType;
        uint8_t bitWidth;
        std::vector<uint64_t> vals;
        const char* macroType;
        ParamStruct(const char* inName, uint32_t inParamType, uint8_t inBitWidth, std::vector<uint64_t> inVals,
            const char* inMacroType):
            name(inName), paramType(inParamType), bitWidth(inBitWidth), vals(std::move(inVals)),
            macroType(inMacroType) {}
    };
    using TilingDeclareParams = std::vector<ParamStruct>;
    using TilingSelectParams = std::vector<std::vector<ParamStruct>>;
    
    // 模板参数定义相关接口
    #define ASCENDC_TPL_DTYPE_DECL(x, ...) ParamStruct{#x, ASCENDC_TPL_DTYPE, ASCENDC_TPL_8_BW, {__VA_ARGS__}, "DECL"}
    #define ASCENDC_TPL_DATATYPE_DECL(x, ...) ParamStruct{#x, ASCENDC_TPL_DTYPE, ASCENDC_TPL_8_BW, {__VA_ARGS__}, "DECL"}
    #define ASCENDC_TPL_FORMAT_DECL(x, ...) ParamStruct{#x, ASCENDC_TPL_FORMAT, ASCENDC_TPL_8_BW, {__VA_ARGS__}, "DECL"}
    #define ASCENDC_TPL_UINT_DECL(x, bw, ...) ParamStruct{#x, ASCENDC_TPL_UINT, bw, {__VA_ARGS__}, "DECL"}
    #define ASCENDC_TPL_BOOL_DECL(x, ...) ParamStruct{#x, ASCENDC_TPL_BOOL, ASCENDC_TPL_1_BW, {__VA_ARGS__}, "DECL"}
    #define ASCENDC_TPL_KERNEL_TYPE_DECL(x, ...) ParamStruct{#x, ASCENDC_TPL_SHARED_KERNEL_TYPE, ASCENDC_TPL_8_BW, {__VA_ARGS__}, "DECL"}
    
    #define ASCENDC_TPL_DTYPE_SEL(x, ...) ParamStruct{#x, ASCENDC_TPL_DTYPE, ASCENDC_TPL_8_BW, {__VA_ARGS__}, "SEL"}
    #define ASCENDC_TPL_DATATYPE_SEL(x, ...) ParamStruct{#x, ASCENDC_TPL_DTYPE, ASCENDC_TPL_8_BW, {__VA_ARGS__}, "SEL"}
    #define ASCENDC_TPL_FORMAT_SEL(x, ...) ParamStruct{#x, ASCENDC_TPL_FORMAT, ASCENDC_TPL_8_BW, {__VA_ARGS__}, "SEL"}
    #define ASCENDC_TPL_UINT_SEL(x, ...) ParamStruct{#x, ASCENDC_TPL_UINT, 0, {__VA_ARGS__}, "SEL"}
    #define ASCENDC_TPL_BOOL_SEL(x, ...) ParamStruct{#x, ASCENDC_TPL_BOOL, ASCENDC_TPL_1_BW, {__VA_ARGS__}, "SEL"}
    #define ASCENDC_TPL_KERNEL_TYPE_SEL(...) ParamStruct{"kernel_type", ASCENDC_TPL_KERNEL_TYPE, ASCENDC_TPL_8_BW, {__VA_ARGS__}, "SEL"}
    #define ASCENDC_TPL_DETERMINISTIC_SEL(...) ParamStruct{"deterministic", ASCENDC_TPL_DETERMINISTIC, ASCENDC_TPL_1_BW, {__VA_ARGS__}, "SEL"}
    #define ASCENDC_TPL_SHARED_KERNEL_TYPE_SEL(x, ...) ParamStruct{#x, ASCENDC_TPL_SHARED_KERNEL_TYPE, ASCENDC_TPL_8_BW, {__VA_ARGS__}, "SEL"}
    
    #define ASCENDC_TPL_ARGS_DECL(x, ...) static TilingDeclareParams g_tilingDeclareParams{ __VA_ARGS__ }
    #define ASCENDC_TPL_ARGS_SEL(...) { __VA_ARGS__}
    #define ASCENDC_TPL_SEL(...) static TilingSelectParams g_tilingSelectParams{ __VA_ARGS__ }
    

#### 参数说明

表1 Tiling模板参数定义说明

展开

宏 |  功能描述 |  参数解释  
---|---|---  
ASCENDC_TPL_ARGS_DECL(args0, ...) |  用于定义算子的模板参数。 | 

  * args0：表示算子Optype。
  * args1-argsn：后续为若干个DTYPE、FORMAT、UINT、BOOL、KERNEL_TYPE的模板参数定义，分别通过ASCENDC_TPL_DTYPE_DECL、ASCENDC_TPL_DATATYPE_DECL、ASCENDC_TPL_FORMAT_DECL、ASCENDC_TPL_UINT_DECL、ASCENDC_TPL_BOOL_DECL，ASCENDC_TPL_KERNEL_TYPE_DECL进行定义。

  
ASCENDC_TPL_DTYPE_DECL(args0, ...) |  自定义DataType类型的模板参数定义。 | 

  * args0：参数名。
  * args1-argsn：后续若干个参数为穷举的自定义DataType枚举值。

  
ASCENDC_TPL_DATATYPE_DECL(args0, ...) |  原生DataType类型的模板参数定义。 | 

  * args0：参数名。
  * args1-argsn：存在两种情况，后续若干个参数为穷举的原生DataType选项；或者为对应的输入参数的索引值（使用ASCENDC_TPL_INPUT(x)进行指定，其中x为对应数值）或对应输出参数的索引值（使用ASCENDC_TPL_OUTPUT(x)进行指定，其中x为对应数值），注意：存在多个时，仅第一个生效。
  * 支持设置的原生DataType取值如下，数据类型的具体介绍请参考[C_DataType](/document/detail/zh/CANNCommunityEdition/900beta1/API/basicdataapi/atlasopapi_07_00719.html)。 
        
        C_DT_FLOAT
        C_DT_FLOAT16
        C_DT_INT8
        C_DT_INT32
        C_DT_UINT8
        C_DT_INT16
        C_DT_UINT16
        C_DT_UINT32
        C_DT_INT64
        C_DT_UINT64
        C_DT_DOUBLE
        C_DT_BOOL
        C_DT_COMPLEX64
        C_DT_BF16
        C_DT_INT4
        C_DT_UINT1
        C_DT_INT2
        C_DT_COMPLEX32
        C_DT_HIFLOAT8
        C_DT_FLOAT8_E5M2
        C_DT_FLOAT8_E4M3FN
        C_DT_FLOAT4_E2M1
        C_DT_FLOAT4_E1M2


  
ASCENDC_TPL_FORMAT_DECL(args0, ...) |  支持两种模式： 1\. 均为自定义Format类型的模板参数定义。 2\. 均为原生Format类型的模板参数定义。 | 

  * args0：参数名。
  * args1-argsn：存在两种模式 
    * 1\. 后续若干个参数为穷举的自定义Format枚举值。
    * 2\. 该模式存在两种情况：后续若干个参数为穷举的原生Format选项；或者对应的输入参数的索引值（使用ASCENDC_TPL_INPUT(x)进行指定，其中x为对应数值）或对应输出参数的索引值（使用ASCENDC_TPL_OUTPUT(x)进行指定，其中x为对应数值），注意：存在多个时，仅第一个生效。
  * 支持设置的原生Format选项如下，数据格式的具体介绍请参考[C_Format](/document/detail/zh/CANNCommunityEdition/900beta1/API/basicdataapi/atlasopapi_07_00720.html)。 
        
        C_FORMAT_NCHW
        C_FORMAT_NHWC
        C_FORMAT_ND
        C_FORMAT_NC1HWC0
        C_FORMAT_FRACTAL_Z
        C_FORMAT_NC1C0HWPAD
        C_FORMAT_NHWC1C0
        C_FORMAT_FSR_NCHW
        C_FORMAT_FRACTAL_DECONV
        C_FORMAT_C1HWNC0
        C_FORMAT_FRACTAL_DECONV_TRANSPOSE
        C_FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS
        C_FORMAT_NC1HWC0_C04
        C_FORMAT_FRACTAL_Z_C04
        C_FORMAT_CHWN
        C_FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS
        C_FORMAT_HWCN
        C_FORMAT_NC1KHKWHWC0
        C_FORMAT_BN_WEIGHT
        C_FORMAT_FILTER_HWCK
        C_FORMAT_HASHTABLE_LOOKUP_LOOKUPS
        C_FORMAT_HASHTABLE_LOOKUP_KEYS
        C_FORMAT_HASHTABLE_LOOKUP_VALUE
        C_FORMAT_HASHTABLE_LOOKUP_OUTPUT
        C_FORMAT_HASHTABLE_LOOKUP_HITS
        C_FORMAT_C1HWNCoC0
        C_FORMAT_MD
        C_FORMAT_NDHWC
        C_FORMAT_FRACTAL_ZZ
        C_FORMAT_FRACTAL_NZ
        C_FORMAT_NCDHW
        C_FORMAT_DHWCN
        C_FORMAT_NDC1HWC0
        C_FORMAT_FRACTAL_Z_3D
        C_FORMAT_CN
        C_FORMAT_NC
        C_FORMAT_DHWNC
        C_FORMAT_FRACTAL_Z_3D_TRANSPOSE
        C_FORMAT_FRACTAL_ZN_LSTM
        C_FORMAT_FRACTAL_Z_G
        C_FORMAT_RESERVED
        C_FORMAT_ALL
        C_FORMAT_NULL
        C_FORMAT_ND_RNN_BIAS
        C_FORMAT_FRACTAL_ZN_RNN
        C_FORMAT_NYUV
        C_FORMAT_NYUV_A
        C_FORMAT_NCL
        C_FORMAT_FRACTAL_Z_WINO
        C_FORMAT_C1HWC0
        C_FORMAT_FRACTAL_NZ_C0_16
        C_FORMAT_FRACTAL_NZ_C0_32
        C_FORMAT_FRACTAL_NZ_C0_2
        C_FORMAT_FRACTAL_NZ_C0_4
        C_FORMAT_FRACTAL_NZ_C0_8


  
ASCENDC_TPL_UINT_DECL(args0, args1, args2, ...) |  自定义UINT类型（无符号整形）的模板参数定义。 | 

  * args0：参数名。
  * args1：最大位宽，模板参数的个数不能超过最大位宽。
  * args2：参数定义的模式。支持以下三种模式： 
    * ASCENDC_TPL_UI_RANGE：范围模式，设置该模式，后续紧跟着第一个值表示范围个数，第一个值后面的每两个数值为一组分别表示该范围的起、终位置；注意定义的范围个数要和后续的组数保持一致。 **举例：** ASCENDC_TPL_UINT_DECL(args0, args1,ASCENDC_TPL_UI_RANGE,2,0,2,3,5)表示2组参数，这2组参数范围为{0, 2}，{3, 5}，因此该参数定义的UINT参数合法值为{0, 1, 2, 3, 4, 5}。
    * ASCENDC_TPL_UI_LIST：穷举模式，设置该模式，则表示后续将穷举出所有的参数值。 **举例：** ASCENDC_TPL_UINT_DECL(args0, args1,ASCENDC_TPL_UI_LIST,10,12,13,9,8,7,6)表示1组穷举参数，[10, 12, 13, 9, 8, 7, 6]为穷举值，因此该参数定义的UINT参数合法值为{10, 12, 13, 9, 8, 7, 6}。
    * ASCENDC_TPL_UI_MIX：混合模式，设置该模式，则表示前n个数值为范围模式的参数定义，后m个数值为穷举模式的参数定义。 **举例** ： ASCENDC_TPL_UINT_DECL(args0, args1,ASCENDC_TPL_UI_MIX,2,0,2,3, 5, 10, 12, 13, 9, 8)表示2组穷举参数，这2组范围为{0, 2}, {3, 5}，[10, 12, 13, 9, 8]为穷举值，因此该参数定义的UINT参数合法值为{0, 1, 2, 3, 4, 5, 10, 12, 13, 9, 8}。
  * args3-argsn：对应不同范围模式的参数数值。

  
ASCENDC_TPL_BOOL_DECL(args0, ...) |  自定义bool类型的模板参数定义。 |  args0：参数名。 args1-args2：取值范围0，1。  
ASCENDC_TPL_KERNEL_TYPE_DECL(args0, ...) |  定义算子模板参数的kernel类型 |  args0：参数名 args1-argsn：后续为若干kernel类型。 当前支持的Kernel类型如下：

  * ASCENDC_TPL_AIV_ONLY // 算子执行时仅启动AI Core上的Vector核
  * ASCENDC_TPL_AIC_ONLY // 算子执行时仅启动AI Core上的Cube核
  * ASCENDC_TPL_MIX_AIV_1_0 // AIC、AIV混合场景下，算子执行时仅会启动AI Core上的Vector核
  * ASCENDC_TPL_MIX_AIC_1_0 // AIC、AIV混合场景下，算子执行时仅会启动AI Core上的Cube核
  * ASCENDC_TPL_MIX_AIC_1_1 // AIC、AIV混合场景下，算子执行时会同时启动AI Core上的Cube核和Vector核，比例为1：1
  * ASCENDC_TPL_MIX_AIC_1_2 // AIC、AIV混合场景下，算子执行时会同时启动AI Core上的Cube核和Vector核，比例为1：2
  * ASCENDC_TPL_AICORE // 算子执行时仅会启动AI Core
  * ASCENDC_TPL_VECTORCORE // 该参数为预留参数，当前版本暂不支持
  * ASCENDC_TPL_MIX_AICORE // 该参数为预留参数，当前版本暂不支持
  * ASCENDC_TPL_MIX_VECTOR_CORE // 算子执行时会同时启动AI Core和Vector Core

本接口只允许与ASCENDC_TPL_SHARED_KERNEL_TYPE_SEL(args0, ...)配合使用。  
  
表2 Tiling模板参数组合定义

展开

宏 |  功能描述 |  参数解释  
---|---|---  
ASCENDC_TPL_SEL(...) |  算子的模板参数整体组合。 |  包含多个算子的模板参数组合。  
ASCENDC_TPL_ARGS_SEL(...) |  算子的模板参数组合。 |  一个算子的模板参数组合。  
ASCENDC_TPL_KERNEL_TYPE_SEL(args0) |  用于设置算子模板参数组合的Kernel类型，但该参数并不能作为核函数的模板参数传入。 |  args0：该模板参数组合下，算子的Kernel类型。如不选择将走自动推导流程，ASCENDC_TPL_SEL下的所有算子对于是否选择Kernel类型需要保持一致。 当前支持的Kernel类型如下：

  * ASCENDC_TPL_AIV_ONLY // 算子执行时仅启动AI Core上的Vector核
  * ASCENDC_TPL_AIC_ONLY // 算子执行时仅启动AI Core上的Cube核
  * ASCENDC_TPL_MIX_AIV_1_0 // AIC、AIV混合场景下，算子执行时仅会启动AI Core上的Vector核
  * ASCENDC_TPL_MIX_AIC_1_0 // AIC、AIV混合场景下，算子执行时仅会启动AI Core上的Cube核
  * ASCENDC_TPL_MIX_AIC_1_1 // AIC、AIV混合场景下，算子执行时会同时启动AI Core上的Cube核和Vector核，比例为1：1
  * ASCENDC_TPL_MIX_AIC_1_2 // AIC、AIV混合场景下，算子执行时会同时启动AI Core上的Cube核和Vector核，比例为1：2
  * ASCENDC_TPL_AICORE // 算子执行时仅会启动AI Core
  * ASCENDC_TPL_VECTORCORE // 该参数为预留参数，当前版本暂不支持
  * ASCENDC_TPL_MIX_AICORE // 该参数为预留参数，当前版本暂不支持
  * ASCENDC_TPL_MIX_VECTOR_CORE // 算子执行时会同时启动AI Core和Vector Core 通过本接口配置Kernel类型，Kernel类型的取值范围同KERNEL_TASK_TYPE_DEFAULT接口一致，详见[设置Kernel类型](atlasascendc_api_07_0218.html)。

  
ASCENDC_TPL_DTYPE_SEL(args0, ...) |  自定义DataType类型的模板参数组合。 | 

  * args0：表示参数名。
  * args1-argsn ：后续若干个参数为ASCENDC_TPL_DTYPE_DECL中定义的参数范围子集。

  
ASCENDC_TPL_DATATYPE_SEL(args0, ...) |  原生DataType类型的模板参数组合 | 

  * args0：表示参数名。
  * args1-argsn ：后续若干个参数为ASCENDC_TPL_DATATYPE_DECL中定义的参数选项范围的子集。

  
ASCENDC_TPL_FORMAT_SEL(args0, ...) |  Format类型的模板参数组合。 | 

  * args0：表示参数名。
  * args1-argsn：后续若干个参数为ASCENDC_TPL_FORMAT_DECL中定义的参数选项范围子集。

  
ASCENDC_TPL_UINT_SEL(args0, args1, args2, ...) |  UINT类型的模板参数组合。 | 

  * args0：表示参数名。
  * args1：参数定义的模式。支持如下取值： 
    * ASCENDC_TPL_UI_RANGE：范围模式。
    * ASCENDC_TPL_UI_LIST：穷举模式。
    * ASCENDC_TPL_UI_MIX：混合模式。
  * args2-argsn：后续若干个参数为ASCENDC_TPL_UINT_DECL中定义的参数范围子集。

模式和参数的配置方式参考ASCENDC_TPL_UINT_DECL(args0, args1, args2, ...)。  
ASCENDC_TPL_BOOL_SEL(args0, ...) |  bool类型的模板参数组合。 |  args0：表示参数名。 args1-args2 ：后续若干个参数为ASCENDC_TPL_BOOL_DECL定义的参数范围子集。  
ASCENDC_TPL_DETERMINISTIC_SEL(args0) |  该组模板参数组合用于配置是否使能确定性计算。 |  args0: 表示参数名， 可选值范围[true, false, 1, 0]，其中[true/1]表示该组模板参数组合使能确定性计算，[false/0]表示不使能确定性计算。需要注意，该值不作为算子的模板参数入参，在使能该值编译时，会添加"-DDETERMINISTIC_MODE=1", 同时会生成以"_deterministic"结尾的json与.o文件，例如："AddCustomTemplate_816f04e052850554f4b3cacb35f8e8c6_deterministic.json"/"AddCustomTemplate_816f04e052850554f4b3cacb35f8e8c6_deterministic.o"。 备注：若通过ASCENDC_TPL_DETERMINISTIC_SEL(true)接口编译出了确定性计算的版本，在算子调用时，通常需要打开确定性计算的的开关，例如通过aclnn单算子调用时，需要使用aclrtCtxSetSysParamOpt接口进行相关配置。 该参数仅支持如下型号：

  * Atlas A3 训练系列产品 / Atlas A3 推理系列产品 
  * Atlas A2 训练系列产品 / Atlas A2 推理系列产品 

  
ASCENDC_TPL_SHARED_KERNEL_TYPE_SEL(args0, ...) |  设置算子模板参数组合的Kernel类型，该参数可以作为核函数的模板参数传入。 |  args0: 参数名 args1-argsn: 该模板参数组合下，算子的Kernel类型，后续参数为若干Kernel类型。该接口不能与ASCENDC_TPL_KERNEL_TYPE_SEL接口同时使用。 若同时使用KERNEL_TASK_TYPE_DEFAULT(value)接口，本接口优先级更高。  
  
#### 返回值说明

无。

#### 约束说明

对模板参数定义的取值进行修改或新增后，需要重新编译自定义算子包，不能再继续使用之前的算子二进制。

**父主题：** [Tiling模板编程](atlasascendc_api_07_00184.html)


# 模板参数定义

#### 功能说明

通过以下函数原型进行模板参数ASCENDC_TPL_ARGS_DECL和模板参数组合ASCENDC_TPL_ARGS_SEL（即可使用的模板）的定义。详细内容请参考[Tiling模板编程](/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_00025.html)。

#### 函数原型
    
    
    // ParamStruct是存放用户设置的模板参数ASCENDC_TPL_ARGS_DECL和模板参数组合ASCENDC_TPL_ARGS_SEL的结构体，用作后续的Tilingkey与模板参数之间的编解码，用户无需关注
    struct ParamStruct {
        const char* name;
        uint32_t paramType;
        uint8_t bitWidth;
        std::vector<uint64_t> vals;
        const char* macroType;
        ParamStruct(const char* inName, uint32_t inParamType, uint8_t inBitWidth, std::vector<uint64_t> inVals,
            const char* inMacroType):
            name(inName), paramType(inParamType), bitWidth(inBitWidth), vals(std::move(inVals)),
            macroType(inMacroType) {}
    };
    using TilingDeclareParams = std::vector<ParamStruct>;
    using TilingSelectParams = std::vector<std::vector<ParamStruct>>;
    
    // 模板参数定义相关接口
    #define ASCENDC_TPL_DTYPE_DECL(x, ...) ParamStruct{#x, ASCENDC_TPL_DTYPE, ASCENDC_TPL_8_BW, {__VA_ARGS__}, "DECL"}
    #define ASCENDC_TPL_DATATYPE_DECL(x, ...) ParamStruct{#x, ASCENDC_TPL_DTYPE, ASCENDC_TPL_8_BW, {__VA_ARGS__}, "DECL"}
    #define ASCENDC_TPL_FORMAT_DECL(x, ...) ParamStruct{#x, ASCENDC_TPL_FORMAT, ASCENDC_TPL_8_BW, {__VA_ARGS__}, "DECL"}
    #define ASCENDC_TPL_UINT_DECL(x, bw, ...) ParamStruct{#x, ASCENDC_TPL_UINT, bw, {__VA_ARGS__}, "DECL"}
    #define ASCENDC_TPL_BOOL_DECL(x, ...) ParamStruct{#x, ASCENDC_TPL_BOOL, ASCENDC_TPL_1_BW, {__VA_ARGS__}, "DECL"}
    #define ASCENDC_TPL_KERNEL_TYPE_DECL(x, ...) ParamStruct{#x, ASCENDC_TPL_SHARED_KERNEL_TYPE, ASCENDC_TPL_8_BW, {__VA_ARGS__}, "DECL"}
    
    #define ASCENDC_TPL_DTYPE_SEL(x, ...) ParamStruct{#x, ASCENDC_TPL_DTYPE, ASCENDC_TPL_8_BW, {__VA_ARGS__}, "SEL"}
    #define ASCENDC_TPL_DATATYPE_SEL(x, ...) ParamStruct{#x, ASCENDC_TPL_DTYPE, ASCENDC_TPL_8_BW, {__VA_ARGS__}, "SEL"}
    #define ASCENDC_TPL_FORMAT_SEL(x, ...) ParamStruct{#x, ASCENDC_TPL_FORMAT, ASCENDC_TPL_8_BW, {__VA_ARGS__}, "SEL"}
    #define ASCENDC_TPL_UINT_SEL(x, ...) ParamStruct{#x, ASCENDC_TPL_UINT, 0, {__VA_ARGS__}, "SEL"}
    #define ASCENDC_TPL_BOOL_SEL(x, ...) ParamStruct{#x, ASCENDC_TPL_BOOL, ASCENDC_TPL_1_BW, {__VA_ARGS__}, "SEL"}
    #define ASCENDC_TPL_KERNEL_TYPE_SEL(...) ParamStruct{"kernel_type", ASCENDC_TPL_KERNEL_TYPE, ASCENDC_TPL_8_BW, {__VA_ARGS__}, "SEL"}
    #define ASCENDC_TPL_DETERMINISTIC_SEL(...) ParamStruct{"deterministic", ASCENDC_TPL_DETERMINISTIC, ASCENDC_TPL_1_BW, {__VA_ARGS__}, "SEL"}
    #define ASCENDC_TPL_SHARED_KERNEL_TYPE_SEL(x, ...) ParamStruct{#x, ASCENDC_TPL_SHARED_KERNEL_TYPE, ASCENDC_TPL_8_BW, {__VA_ARGS__}, "SEL"}
    
    #define ASCENDC_TPL_ARGS_DECL(x, ...) static TilingDeclareParams g_tilingDeclareParams{ __VA_ARGS__ }
    #define ASCENDC_TPL_ARGS_SEL(...) { __VA_ARGS__}
    #define ASCENDC_TPL_SEL(...) static TilingSelectParams g_tilingSelectParams{ __VA_ARGS__ }
    

#### 参数说明

表1 Tiling模板参数定义说明

展开

宏 |  功能描述 |  参数解释  
---|---|---  
ASCENDC_TPL_ARGS_DECL(args0, ...) |  用于定义算子的模板参数。 | 

  * args0：表示算子Optype。
  * args1-argsn：后续为若干个DTYPE、FORMAT、UINT、BOOL、KERNEL_TYPE的模板参数定义，分别通过ASCENDC_TPL_DTYPE_DECL、ASCENDC_TPL_DATATYPE_DECL、ASCENDC_TPL_FORMAT_DECL、ASCENDC_TPL_UINT_DECL、ASCENDC_TPL_BOOL_DECL，ASCENDC_TPL_KERNEL_TYPE_DECL进行定义。

  
ASCENDC_TPL_DTYPE_DECL(args0, ...) |  自定义DataType类型的模板参数定义。 | 

  * args0：参数名。
  * args1-argsn：后续若干个参数为穷举的自定义DataType枚举值。

  
ASCENDC_TPL_DATATYPE_DECL(args0, ...) |  原生DataType类型的模板参数定义。 | 

  * args0：参数名。
  * args1-argsn：存在两种情况，后续若干个参数为穷举的原生DataType选项；或者为对应的输入参数的索引值（使用ASCENDC_TPL_INPUT(x)进行指定，其中x为对应数值）或对应输出参数的索引值（使用ASCENDC_TPL_OUTPUT(x)进行指定，其中x为对应数值），注意：存在多个时，仅第一个生效。
  * 支持设置的原生DataType取值如下，数据类型的具体介绍请参考[C_DataType](/document/detail/zh/CANNCommunityEdition/900beta1/API/basicdataapi/atlasopapi_07_00719.html)。 
        
        C_DT_FLOAT
        C_DT_FLOAT16
        C_DT_INT8
        C_DT_INT32
        C_DT_UINT8
        C_DT_INT16
        C_DT_UINT16
        C_DT_UINT32
        C_DT_INT64
        C_DT_UINT64
        C_DT_DOUBLE
        C_DT_BOOL
        C_DT_COMPLEX64
        C_DT_BF16
        C_DT_INT4
        C_DT_UINT1
        C_DT_INT2
        C_DT_COMPLEX32
        C_DT_HIFLOAT8
        C_DT_FLOAT8_E5M2
        C_DT_FLOAT8_E4M3FN
        C_DT_FLOAT4_E2M1
        C_DT_FLOAT4_E1M2


  
ASCENDC_TPL_FORMAT_DECL(args0, ...) |  支持两种模式： 1\. 均为自定义Format类型的模板参数定义。 2\. 均为原生Format类型的模板参数定义。 | 

  * args0：参数名。
  * args1-argsn：存在两种模式 
    * 1\. 后续若干个参数为穷举的自定义Format枚举值。
    * 2\. 该模式存在两种情况：后续若干个参数为穷举的原生Format选项；或者对应的输入参数的索引值（使用ASCENDC_TPL_INPUT(x)进行指定，其中x为对应数值）或对应输出参数的索引值（使用ASCENDC_TPL_OUTPUT(x)进行指定，其中x为对应数值），注意：存在多个时，仅第一个生效。
  * 支持设置的原生Format选项如下，数据格式的具体介绍请参考[C_Format](/document/detail/zh/CANNCommunityEdition/900beta1/API/basicdataapi/atlasopapi_07_00720.html)。 
        
        C_FORMAT_NCHW
        C_FORMAT_NHWC
        C_FORMAT_ND
        C_FORMAT_NC1HWC0
        C_FORMAT_FRACTAL_Z
        C_FORMAT_NC1C0HWPAD
        C_FORMAT_NHWC1C0
        C_FORMAT_FSR_NCHW
        C_FORMAT_FRACTAL_DECONV
        C_FORMAT_C1HWNC0
        C_FORMAT_FRACTAL_DECONV_TRANSPOSE
        C_FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS
        C_FORMAT_NC1HWC0_C04
        C_FORMAT_FRACTAL_Z_C04
        C_FORMAT_CHWN
        C_FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS
        C_FORMAT_HWCN
        C_FORMAT_NC1KHKWHWC0
        C_FORMAT_BN_WEIGHT
        C_FORMAT_FILTER_HWCK
        C_FORMAT_HASHTABLE_LOOKUP_LOOKUPS
        C_FORMAT_HASHTABLE_LOOKUP_KEYS
        C_FORMAT_HASHTABLE_LOOKUP_VALUE
        C_FORMAT_HASHTABLE_LOOKUP_OUTPUT
        C_FORMAT_HASHTABLE_LOOKUP_HITS
        C_FORMAT_C1HWNCoC0
        C_FORMAT_MD
        C_FORMAT_NDHWC
        C_FORMAT_FRACTAL_ZZ
        C_FORMAT_FRACTAL_NZ
        C_FORMAT_NCDHW
        C_FORMAT_DHWCN
        C_FORMAT_NDC1HWC0
        C_FORMAT_FRACTAL_Z_3D
        C_FORMAT_CN
        C_FORMAT_NC
        C_FORMAT_DHWNC
        C_FORMAT_FRACTAL_Z_3D_TRANSPOSE
        C_FORMAT_FRACTAL_ZN_LSTM
        C_FORMAT_FRACTAL_Z_G
        C_FORMAT_RESERVED
        C_FORMAT_ALL
        C_FORMAT_NULL
        C_FORMAT_ND_RNN_BIAS
        C_FORMAT_FRACTAL_ZN_RNN
        C_FORMAT_NYUV
        C_FORMAT_NYUV_A
        C_FORMAT_NCL
        C_FORMAT_FRACTAL_Z_WINO
        C_FORMAT_C1HWC0
        C_FORMAT_FRACTAL_NZ_C0_16
        C_FORMAT_FRACTAL_NZ_C0_32
        C_FORMAT_FRACTAL_NZ_C0_2
        C_FORMAT_FRACTAL_NZ_C0_4
        C_FORMAT_FRACTAL_NZ_C0_8


  
ASCENDC_TPL_UINT_DECL(args0, args1, args2, ...) |  自定义UINT类型（无符号整形）的模板参数定义。 | 

  * args0：参数名。
  * args1：最大位宽，模板参数的个数不能超过最大位宽。
  * args2：参数定义的模式。支持以下三种模式： 
    * ASCENDC_TPL_UI_RANGE：范围模式，设置该模式，后续紧跟着第一个值表示范围个数，第一个值后面的每两个数值为一组分别表示该范围的起、终位置；注意定义的范围个数要和后续的组数保持一致。 **举例：** ASCENDC_TPL_UINT_DECL(args0, args1,ASCENDC_TPL_UI_RANGE,2,0,2,3,5)表示2组参数，这2组参数范围为{0, 2}，{3, 5}，因此该参数定义的UINT参数合法值为{0, 1, 2, 3, 4, 5}。
    * ASCENDC_TPL_UI_LIST：穷举模式，设置该模式，则表示后续将穷举出所有的参数值。 **举例：** ASCENDC_TPL_UINT_DECL(args0, args1,ASCENDC_TPL_UI_LIST,10,12,13,9,8,7,6)表示1组穷举参数，[10, 12, 13, 9, 8, 7, 6]为穷举值，因此该参数定义的UINT参数合法值为{10, 12, 13, 9, 8, 7, 6}。
    * ASCENDC_TPL_UI_MIX：混合模式，设置该模式，则表示前n个数值为范围模式的参数定义，后m个数值为穷举模式的参数定义。 **举例** ： ASCENDC_TPL_UINT_DECL(args0, args1,ASCENDC_TPL_UI_MIX,2,0,2,3, 5, 10, 12, 13, 9, 8)表示2组穷举参数，这2组范围为{0, 2}, {3, 5}，[10, 12, 13, 9, 8]为穷举值，因此该参数定义的UINT参数合法值为{0, 1, 2, 3, 4, 5, 10, 12, 13, 9, 8}。
  * args3-argsn：对应不同范围模式的参数数值。

  
ASCENDC_TPL_BOOL_DECL(args0, ...) |  自定义bool类型的模板参数定义。 |  args0：参数名。 args1-args2：取值范围0，1。  
ASCENDC_TPL_KERNEL_TYPE_DECL(args0, ...) |  定义算子模板参数的kernel类型 |  args0：参数名 args1-argsn：后续为若干kernel类型。 当前支持的Kernel类型如下：

  * ASCENDC_TPL_AIV_ONLY // 算子执行时仅启动AI Core上的Vector核
  * ASCENDC_TPL_AIC_ONLY // 算子执行时仅启动AI Core上的Cube核
  * ASCENDC_TPL_MIX_AIV_1_0 // AIC、AIV混合场景下，算子执行时仅会启动AI Core上的Vector核
  * ASCENDC_TPL_MIX_AIC_1_0 // AIC、AIV混合场景下，算子执行时仅会启动AI Core上的Cube核
  * ASCENDC_TPL_MIX_AIC_1_1 // AIC、AIV混合场景下，算子执行时会同时启动AI Core上的Cube核和Vector核，比例为1：1
  * ASCENDC_TPL_MIX_AIC_1_2 // AIC、AIV混合场景下，算子执行时会同时启动AI Core上的Cube核和Vector核，比例为1：2
  * ASCENDC_TPL_AICORE // 算子执行时仅会启动AI Core
  * ASCENDC_TPL_VECTORCORE // 该参数为预留参数，当前版本暂不支持
  * ASCENDC_TPL_MIX_AICORE // 该参数为预留参数，当前版本暂不支持
  * ASCENDC_TPL_MIX_VECTOR_CORE // 算子执行时会同时启动AI Core和Vector Core

本接口只允许与ASCENDC_TPL_SHARED_KERNEL_TYPE_SEL(args0, ...)配合使用。  
  
表2 Tiling模板参数组合定义

展开

宏 |  功能描述 |  参数解释  
---|---|---  
ASCENDC_TPL_SEL(...) |  算子的模板参数整体组合。 |  包含多个算子的模板参数组合。  
ASCENDC_TPL_ARGS_SEL(...) |  算子的模板参数组合。 |  一个算子的模板参数组合。  
ASCENDC_TPL_KERNEL_TYPE_SEL(args0) |  用于设置算子模板参数组合的Kernel类型，但该参数并不能作为核函数的模板参数传入。 |  args0：该模板参数组合下，算子的Kernel类型。如不选择将走自动推导流程，ASCENDC_TPL_SEL下的所有算子对于是否选择Kernel类型需要保持一致。 当前支持的Kernel类型如下：

  * ASCENDC_TPL_AIV_ONLY // 算子执行时仅启动AI Core上的Vector核
  * ASCENDC_TPL_AIC_ONLY // 算子执行时仅启动AI Core上的Cube核
  * ASCENDC_TPL_MIX_AIV_1_0 // AIC、AIV混合场景下，算子执行时仅会启动AI Core上的Vector核
  * ASCENDC_TPL_MIX_AIC_1_0 // AIC、AIV混合场景下，算子执行时仅会启动AI Core上的Cube核
  * ASCENDC_TPL_MIX_AIC_1_1 // AIC、AIV混合场景下，算子执行时会同时启动AI Core上的Cube核和Vector核，比例为1：1
  * ASCENDC_TPL_MIX_AIC_1_2 // AIC、AIV混合场景下，算子执行时会同时启动AI Core上的Cube核和Vector核，比例为1：2
  * ASCENDC_TPL_AICORE // 算子执行时仅会启动AI Core
  * ASCENDC_TPL_VECTORCORE // 该参数为预留参数，当前版本暂不支持
  * ASCENDC_TPL_MIX_AICORE // 该参数为预留参数，当前版本暂不支持
  * ASCENDC_TPL_MIX_VECTOR_CORE // 算子执行时会同时启动AI Core和Vector Core 通过本接口配置Kernel类型，Kernel类型的取值范围同KERNEL_TASK_TYPE_DEFAULT接口一致，详见[设置Kernel类型](atlasascendc_api_07_0218.html)。

  
ASCENDC_TPL_DTYPE_SEL(args0, ...) |  自定义DataType类型的模板参数组合。 | 

  * args0：表示参数名。
  * args1-argsn ：后续若干个参数为ASCENDC_TPL_DTYPE_DECL中定义的参数范围子集。

  
ASCENDC_TPL_DATATYPE_SEL(args0, ...) |  原生DataType类型的模板参数组合 | 

  * args0：表示参数名。
  * args1-argsn ：后续若干个参数为ASCENDC_TPL_DATATYPE_DECL中定义的参数选项范围的子集。

  
ASCENDC_TPL_FORMAT_SEL(args0, ...) |  Format类型的模板参数组合。 | 

  * args0：表示参数名。
  * args1-argsn：后续若干个参数为ASCENDC_TPL_FORMAT_DECL中定义的参数选项范围子集。

  
ASCENDC_TPL_UINT_SEL(args0, args1, args2, ...) |  UINT类型的模板参数组合。 | 

  * args0：表示参数名。
  * args1：参数定义的模式。支持如下取值： 
    * ASCENDC_TPL_UI_RANGE：范围模式。
    * ASCENDC_TPL_UI_LIST：穷举模式。
    * ASCENDC_TPL_UI_MIX：混合模式。
  * args2-argsn：后续若干个参数为ASCENDC_TPL_UINT_DECL中定义的参数范围子集。

模式和参数的配置方式参考ASCENDC_TPL_UINT_DECL(args0, args1, args2, ...)。  
ASCENDC_TPL_BOOL_SEL(args0, ...) |  bool类型的模板参数组合。 |  args0：表示参数名。 args1-args2 ：后续若干个参数为ASCENDC_TPL_BOOL_DECL定义的参数范围子集。  
ASCENDC_TPL_DETERMINISTIC_SEL(args0) |  该组模板参数组合用于配置是否使能确定性计算。 |  args0: 表示参数名， 可选值范围[true, false, 1, 0]，其中[true/1]表示该组模板参数组合使能确定性计算，[false/0]表示不使能确定性计算。需要注意，该值不作为算子的模板参数入参，在使能该值编译时，会添加"-DDETERMINISTIC_MODE=1", 同时会生成以"_deterministic"结尾的json与.o文件，例如："AddCustomTemplate_816f04e052850554f4b3cacb35f8e8c6_deterministic.json"/"AddCustomTemplate_816f04e052850554f4b3cacb35f8e8c6_deterministic.o"。 备注：若通过ASCENDC_TPL_DETERMINISTIC_SEL(true)接口编译出了确定性计算的版本，在算子调用时，通常需要打开确定性计算的的开关，例如通过aclnn单算子调用时，需要使用aclrtCtxSetSysParamOpt接口进行相关配置。 该参数仅支持如下型号：

  * Atlas A3 训练系列产品 / Atlas A3 推理系列产品 
  * Atlas A2 训练系列产品 / Atlas A2 推理系列产品 

  
ASCENDC_TPL_SHARED_KERNEL_TYPE_SEL(args0, ...) |  设置算子模板参数组合的Kernel类型，该参数可以作为核函数的模板参数传入。 |  args0: 参数名 args1-argsn: 该模板参数组合下，算子的Kernel类型，后续参数为若干Kernel类型。该接口不能与ASCENDC_TPL_KERNEL_TYPE_SEL接口同时使用。 若同时使用KERNEL_TASK_TYPE_DEFAULT(value)接口，本接口优先级更高。  
  
#### 返回值说明

无。

#### 约束说明

对模板参数定义的取值进行修改或新增后，需要重新编译自定义算子包，不能再继续使用之前的算子二进制。

**父主题：** [Tiling模板编程](atlasascendc_api_07_00184.html)



---

## 构造函数


# 构造函数与析构函数

#### 产品支持情况

展开

产品 | 是否支持  
---|---  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √  
Atlas 200I/500 A2 推理产品 | x  
Atlas 推理系列产品AI Core | x  
Atlas 推理系列产品Vector Core | x  
Atlas 训练系列产品 | x  
  
#### 功能说明

用于创建KfcWorkspace对象，Kfc全称为kernel function call，表示核间通信调用。

#### 函数原型
    
    
    class KfcWorkspace;
    __aicore__ inline KfcWorkspace(GM_ADDR workspace)
    __aicore__ inline ~KfcWorkspace()
    

#### 参数说明

表1 KfcWorkspace构造函数参数说明

展开

参数 | 输入/输出 | 说明  
---|---|---  
workspace | 输入 | Global Memory上的消息空间地址，用户需保证地址对齐和清零。  
  
#### 返回值说明

KfcWorkspace对象实例。

#### 约束说明

不能和[REGIST_MATMUL_OBJ](atlasascendc_api_07_0628.html)接口同时使用。使用资源管理API时，用户自主管理AIC和AIV的核间通信，REGIST_MATMUL_OBJ内部是由框架管理AIC和AIV的核间通信，同时使用可能会导致通信消息错误等异常。

#### 调用示例
    
    
    AscendC::KfcWorkspace desc(workspaceGM);
    

**父主题：** [KfcWorkspace](atlasascendc_api_07_0307.html)



---

## 随路量化


# 随路量化激活搬运

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

支持在数据搬运过程中进行量化和Relu激活等操作，同时支持Local Memory到Global Memory通路NZ到ND格式的转换。

#### 函数原型

  * Local Memory -> Global Memory，支持量化和Relu激活等操作，同时支持NZ到ND格式的转换
        
        template <typename T, typename U>
        __aicore__ inline void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<U>& src, const DataCopyCO12DstParams& intriParams)
        

  * Local Memory -> Local Memory，支持量化和Relu激活等操作
        
        template <typename T, typename U>
        __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const DataCopyCO12DstParams& intriParams)
        




说明

各原型支持的具体数据通路和数据类型，请参考[支持的通路和数据类型](#ZH-CN_TOPIC_0000002521039734__section2614932173011)。

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
T | 目的操作数的数据类型。支持的数据类型请参考[支持的通路和数据类型](#ZH-CN_TOPIC_0000002521039734__section2614932173011)。  
U | 源操作数的数据类型。支持的数据类型请参考[支持的通路和数据类型](#ZH-CN_TOPIC_0000002521039734__section2614932173011)。  
  
表2 参数说明

展开

参数名称 | 输入/输出 | 含义  
---|---|---  
dst | 输出 | 目的操作数，类型为LocalTensor或GlobalTensor。  
src | 输入 | 源操作数，类型为LocalTensor。  
intriParams | 输入 | 搬运参数，类型为[DataCopyCO12DstParams](#ZH-CN_TOPIC_0000002521039734__table35908519282)。 具体定义请参考${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_data_copy.h，${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。  
  
表3 DataCopyCO12DstParams结构体参数定义（C0取值：一般情况下，C0 = 16；使能channelSplit（channel切分）时，C0 = 8）

展开

参数名称 | 含义  
---|---  
nSize | src横向方向的size大小。

  * 不使能NZ2ND功能，必须为C0的倍数，此时连续传输数据块的个数为nSize / C0。
  * 使能NZ2ND功能，不受限制。

  
mSize | src纵向方向的size大小。

  * 不使能NZ2ND功能，连续传输数据块的大小为mSize * C0个元素的长度。


  * 使能NZ2ND功能，NZ/ND矩阵的大小为mSize * nSize。

  
dstStride | 

  * 不使能NZ2ND功能dst相邻连续数据片段间隔（前面一个数据块的头与后面数据块的头的间隔），取值不为0。单位为DataBlock（32字节）。


  * 使能NZ2ND功能dst同一ND矩阵的相邻行的偏移（头与头），取值不为0， 单位为元素。

  
srcStride | 

  * 不使能NZ2ND功能src相邻连续数据片段间隔（前面一个数据块的头与后面数据块的头的间隔），必须为16的倍数。取值范围：srcStride∈[0, 65535]， 单位：C0_Size(C0 * sizeof(U)，U为src的数据类型)。
  * 使能NZ2ND功能src同一NZ矩阵的相邻Z排布的偏移（头与头），必须为16的倍数，取值范围：srcStride∈[0, 65535]，单位C0_size。

  
quantPre | 用于控制量化模式，QuantMode_t类型，具体定义如下。默认值为QuantMode_t::NoQuant，即不使能量化功能。 配置为scalar量化时，需要调用[SetFixpipePreQuantFlag](atlasascendc_api_07_0254.html)接口来设置scalar量化参数；配置为tensor量化时，需要调用[SetFixPipeConfig](atlasascendc_api_07_0252.html)来设置tensor量化参数。
    
    
    enum QuantMode_t
    {
        NoQuant,      // 不使能量化功能
        F322F16,      // float量化成half, scalar量化
        F322BF16,     // float量化成bfloat16_t, scalar量化
        DEQF16,       // int32_t量化成half, scalar量化
        VDEQF16,      // int32_t量化成half，tensor量化
        QF322B8_PRE,  // float量化成int8_t/uint8_t，scalar量化
        VQF322B8_PRE, // float量化成int8_t/uint8_t，tensor量化
        REQ8,         // int32_t量化成int8_t/uint8_t，scalar量化
        VREQ8,        // int32_t量化成int8_t/uint8_t，tensor量化
    };
      
  
reluPre | 用于配置relu操作的模式，类型为uint8_t，取值如下：

  * 0：不使能relu
  * 1：Normal relu

  
channelSplit | 类型为bool，配置是否使能channel切分，对于float类型的dst生效。

  * false：不使能
  * true：使能

  
nz2ndEn | 类型为bool，配置是否使能NZ2ND的格式转换，仅在CO1 -> GM通路生效。 如果要使能NZ2ND的功能需要同步调用[SetFixpipeNz2ndFlag](atlasascendc_api_07_0253.html)来设置格式转换的相关配置信息。

  * false：不使能
  * true：使能

  
clipReluPre | 用于配置是否使能ClipRelu操作，参数类型为uint8_t，取值如下：0，不使能ClipRelu；1，使能ClipRelu，此时需要调用[SetFixPipeClipRelu](atlasascendc_api_07_0255.html)来设置clipRelu的最大值。

  * 该操作在随路量化后进行，quantPre配置后才能使用，当前支持的量化模式有F322F16/DEQF16/VDEQF16/QF322B8_PRE/VQF322B8_PRE/REQ8/VREQ8。
  * 该参数仅在Atlas 200I/500 A2 推理产品支持。

  
eltWiseOp | 用于配置是否使能Elementwise操作及操作模式。Elementwise操作是指进行随路量化后，可以逐个元素加/减一个LocalTensor，大小为mSize * nSize，具体LocalTensor地址相关参数需要调用[SetFixPipeAddr](atlasascendc_api_07_0256.html)来设置。 eltWiseOp参数类型为uint8_t，取值如下：

  * 0：不使能Elementwise
  * 1：Elementwise Addition
  * 2：Elementwise Subtraction

该参数仅在Atlas 200I/500 A2 推理产品支持。  
sid | 预留参数，为后续的功能做保留，开发者暂时无需关注。  
  
#### 返回值说明

无

#### 约束说明

无

#### 支持的通路和数据类型

下文的数据通路均通过逻辑位置[TPosition](atlasascendc_api_07_0174.html#ZH-CN_TOPIC_0000002520880532__table5376122715308)来表达，并注明了对应的物理通路。TPosition与物理内存的映射关系见[表1](atlasascendc_api_07_0004.html#ZH-CN_TOPIC_0000002552120513__table07372185712)。

表4 Local Memory -> Global Memory具体通路和支持的数据类型

展开

支持型号 | 数据通路 | 源操作数的数据类型 | 目的操作数的数据类型  
---|---|---|---  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | CO1 -> GM（L0C Buffer -> GM） | float | uint8_t、int8_t、half、bfloat16_t、float  
int32_t | uint8_t、int8_t、half、int16_t、int32_t  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | CO1 -> GM（L0C Buffer -> GM） | float | uint8_t、int8_t、half、bfloat16_t、float  
int32_t | uint8_t、int8_t、half、int16_t、int32_t  
Atlas 200I/500 A2 推理产品 | CO1 -> GM（L0C Buffer -> GM） | float | uint8_t、int8_t、half、bfloat16_t、float  
int32_t | uint8_t、int8_t、half、int16_t、int32_t  
  
表5 Local Memory -> Local Memory具体通路和支持的数据类型

展开

支持型号 | 数据通路 | 源操作数的数据类型 | 目的操作数的数据类型  
---|---|---|---  
Atlas A2 训练系列产品/Atlas A2 推理系列产品 | CO1 -> A1（L0C Buffer -> L1 Buffer） | float | uint8_t、int8_t、half、bfloat16_t  
int32_t | uint8_t、int8_t、half、int16_t  
Atlas A3 训练系列产品/Atlas A3 推理系列产品 | CO1 -> A1（L0C Buffer -> L1 Buffer） | float | uint8_t、int8_t、half、bfloat16_t  
int32_t | uint8_t、int8_t、half、int16_t  
  
#### 调用示例

  * 随路格式转换数据搬运，通路：CO1->A1、CO1->GM

示例：Mmad含有矩阵乘偏置，左矩阵和右矩阵的数据类型为int8_t，结果矩阵的数据类型为int32_t。量化模式DEQF16，scalar量化参数为0.5，将Mmad计算出的结果由int32_t量化成half并搬出。
        
        #ifdef ASCENDC_CPU_DEBUG
        #include "tikicpulib.h"
        #endif
        #include "kernel_operator.h"
        #include "../../instrs/common_utils/register_utils.h"
        SET_G_CORE_TYPE_IS_AIC
        template <typename dst_T, typename fmap_T, typename weight_T, typename dstCO1_T> class KernelCubeDataCopy{
        public:
            __aicore__ inline KernelCubeDataCopy(uint16_t CoutIn, uint8_t dilationHIn, uint8_t dilationWIn, QuantMode_t deqModeIn)
            {
                // ceiling of 16
                Cout = CoutIn;
                dilationH = dilationHIn;
                dilationW = dilationWIn;
                C0 = 32 / sizeof(fmap_T);
                C1 = channelSize / C0;
                coutBlocks = (Cout + 16 - 1) / 16;
                ho = H - dilationH * (Kh - 1);
                wo = W - dilationW * (Kw - 1);
                howo = ho * wo;
                howoRound = ((howo + 16 - 1) / 16) * 16;
                featureMapA1Size = C1 * H * W * C0;      // shape: [C1, H, W, C0]
                weightA1Size = C1 * Kh * Kw * Cout * C0; // shape: [C1, Kh, Kw, Cout, C0]
                featureMapA2Size = howoRound * (C1 * Kh * Kw * C0);
                weightB2Size = (C1 * Kh * Kw * C0) * coutBlocks * 16;
                m = howo;
                k = C1 * Kh * Kw * C0;
                n = Cout;
                biasSize = Cout;                  // shape: [Cout]
                dstSize = coutBlocks * howo * 16; // shape: [coutBlocks, howo, 16]
                dstCO1Size = coutBlocks * howoRound * 16;
                fmRepeat = featureMapA2Size / (16 * C0);
                weRepeat = weightB2Size / (16 * C0);
                deqMode = deqModeIn;
            }
            __aicore__ inline void Init(__gm__ uint8_t* fmGm, __gm__ uint8_t* weGm, __gm__ uint8_t* biasGm, __gm__ uint8_t* deqGm, __gm__ uint8_t* dstGm)
            {
                fmGlobal.SetGlobalBuffer((__gm__ fmap_T*)fmGm);
                weGlobal.SetGlobalBuffer((__gm__ weight_T*)weGm);
                biasGlobal.SetGlobalBuffer((__gm__ dstCO1_T*)biasGm);
                deqGlobal.SetGlobalBuffer((__gm__ uint64_t*)deqGm);
                dstGlobal.SetGlobalBuffer((__gm__ dst_T*)dstGm);
                pipe.InitBuffer(inQueueFmA1, 1, featureMapA1Size * sizeof(fmap_T));
                pipe.InitBuffer(inQueueFmA2, 1, featureMapA2Size * sizeof(fmap_T));
                pipe.InitBuffer(inQueueWeB1, 1, weightA1Size * sizeof(weight_T));
                pipe.InitBuffer(inQueueWeB2, 1, weightB2Size * sizeof(weight_T));
                pipe.InitBuffer(inQueueBiasA1, 1, biasSize * sizeof(dstCO1_T));
                pipe.InitBuffer(inQueueDeqA1, 1, dstCO1Size * sizeof(uint64_t));
                pipe.InitBuffer(inQueueDeqFB, 1, dstCO1Size * sizeof(uint64_t));
                pipe.InitBuffer(outQueueCO1, 1, dstCO1Size * sizeof(dstCO1_T));
                pipe.InitBuffer(outQueueA1, 1, dstCO1Size * sizeof(dst_T));
             }
            __aicore__ inline void Process()
            {
                CopyIn();
                Split();
                Compute();
                CopyOut();
            }
        private:
            __aicore__ inline void CopyIn()
            {
                AscendC::LocalTensor<fmap_T> featureMapA1 = inQueueFmA1.AllocTensor<fmap_T>();
                AscendC::LocalTensor<weight_T> weightB1 = inQueueWeB1.AllocTensor<weight_T>();
                AscendC::LocalTensor<dstCO1_T> biasA1 = inQueueBiasA1.AllocTensor<dstCO1_T>();
                AscendC::DataCopy(featureMapA1, fmGlobal, { 1, static_cast<uint16_t>(featureMapA1Size * sizeof(fmap_T) / 32), 0, 0 });
                AscendC::DataCopy(weightB1, weGlobal, { 1, static_cast<uint16_t>(weightA1Size * sizeof(weight_T) / 32), 0, 0 });
                AscendC::DataCopy(biasA1, biasGlobal, { 1, static_cast<uint16_t>(biasSize * sizeof(dstCO1_T) / 32), 0, 0 });
                inQueueFmA1.EnQue(featureMapA1);
                inQueueWeB1.EnQue(weightB1);
                inQueueBiasA1.EnQue(biasA1);
            }
            __aicore__ inline void Split()
            {
                AscendC::LocalTensor<fmap_T> featureMapA1 = inQueueFmA1.DeQue<fmap_T>();
                AscendC::LocalTensor<weight_T> weightB1 = inQueueWeB1.DeQue<weight_T>();
                AscendC::LocalTensor<fmap_T> featureMapA2 = inQueueFmA2.AllocTensor<fmap_T>();
                AscendC::LocalTensor<weight_T> weightB2 = inQueueWeB2.AllocTensor<weight_T>();
                uint8_t padList[] = {0, 0, 0, 0};
                // load3dv2
                AscendC::LoadData(featureMapA2, featureMapA1, { padList, H, W, channelSize, k, howoRound, 0, 0, 1, 1, Kw, Kh, dilationW, dilationH, false, false, 0 });
                // load2d
                AscendC::LoadData(weightB2, weightB1, { 0, weRepeat, 1, 0, 0, false, 0 });
                inQueueFmA2.EnQue<fmap_T>(featureMapA2);
                inQueueWeB2.EnQue<weight_T>(weightB2);
                inQueueFmA1.FreeTensor(featureMapA1);
                inQueueWeB1.FreeTensor(weightB1);
            }
            __aicore__ inline void Compute()
            {
                AscendC::LocalTensor<fmap_T> featureMapA2 = inQueueFmA2.DeQue<fmap_T>();
                AscendC::LocalTensor<weight_T> weightB2 = inQueueWeB2.DeQue<weight_T>();
                AscendC::LocalTensor<dstCO1_T> dstCO1 = outQueueCO1.AllocTensor<dstCO1_T>();
                AscendC::LocalTensor<dstCO1_T> biasA1 = inQueueBiasA1.DeQue<dstCO1_T>();
                // C = A * B + bias
                // m: 左矩阵Height, k: 左矩阵Width, n: 右矩阵Width
                AscendC::Mmad(dstCO1, featureMapA2, weightB2, biasA1, { m, n, k, true, 0, false, false, false });
                outQueueCO1.EnQue<dstCO1_T>(dstCO1);
                inQueueFmA2.FreeTensor(featureMapA2);
                inQueueWeB2.FreeTensor(weightB2);
            }
            __aicore__ inline void CopyOut()
            {
                AscendC::LocalTensor<dstCO1_T> dstCO1 = outQueueCO1.DeQue<dstCO1_T>();
                AscendC::LocalTensor<dst_T> dstA1 = outQueueA1.DeQue<dst_T>();
                // 使能DEQF16量化，量化参数设置为0.5
                float tmp = (float)0.5;
                // 将float的tmp转换成uint64_t的deqScalar
                uint64_t deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp));
                bool nz2ndEn = false;
                // nz2nd不使能时，nSize必须为16的倍数
                uint16_t nSize = coutBlocks * 16;
                uint16_t mSize = m;
                // srcStride必须为16的倍数
                uint16_t srcStride = (m + 16 - 1) / 16 * 16;
                // nz2nd不使能时，dstStride为burst头到头的距离，且为32B对齐
                uint32_t dstStride = m * sizeof(dst_T) * 16 / 32;
                if (nz2ndEn) {
                    // nd矩阵的数量为1，src_nd_stride和dst_nd_stride填1
                    AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
                    // nz2nd使能时，nSize可以不为16的倍数，与Mmad的n保持一致
                    nSize = n;
                    // nz2nd使能时，dstStride表示同一nd矩阵的相邻连续行的间隔，与n保持一致
                    dstStride = nSize;
                };
                // 不使能relu与channelSplit
                AscendC::DataCopyCO12DstParams intriParams(nSize, mSize, dstStride, srcStride, deqMode, 0, false, nz2ndEn);
              
                // mov l0c to gm, deq scalar quant
                AscendC::SetFixpipePreQuantFlag(deqScalar);  // 设置量化参数
                AscendC::PipeBarrier<PIPE_FIX>();
                AscendC::DataCopy(dstGlobal, dstCO1, intriParams);
                // // mov l0c to gm, deq tensor quant
                // // 需要额外申请deq tensor的gm空间，将值搬运到workA1
                // AscendC::LocalTensor<uint64_t> workA1 = inQueueDeqA1.AllocTensor<uint64_t>();
                // // deq tensor的size
                // uint16_t deqSize = 128;
                // AscendC::DataCopy(workA1, deqGlobal, deqSize);
                // // deq tensor在fix上的地址
                // AscendC::LocalTensor<uint64_t> deqFB = inQueueDeqFB.AllocTensor<uint64_t>();
                // // l1->fix, burst_len unit is 128Bytes
                // uint16_t fbufBurstLen = deqSize / 128;
                // AscendC::DataCopyParams dataCopyParams(1, fbufBurstLen, 0, 0);
                // AscendC::DataCopy(deqFB, workA1, dataCopyParams);
                // // 设置量化tensor
                // AscendC::SetFixPipeConfig(deqFB);
                // AscendC::PipeBarrier<PIPE_FIX>();
                // AscendC::DataCopy(dstGlobal, dstCO1, intriParams);
                // inQueueDeqA1.FreeTensor(workA1);
                // inQueueDeqFB.FreeTensor(deqFB);
                // // mov l0c to l1, deq scalar quant, and then mov l1 to gm
                // AscendC::SetFixpipePreQuantFlag(deqScalar);  // 设置量化参数
                // AscendC::PipeBarrier<PIPE_FIX>();
                // AscendC::DataCopy(dstA1, dstCO1, intriParams);
                // AscendC::DataCopy(dstGlobal, dstA1, dstCO1Size);
                // // mov l0c to l1, deq tensor quant, and then mov l1 to gm
                // AscendC::LocalTensor<uint64_t> workA1 = inQueueDeqA1.AllocTensor<uint64_t>();
                // uint16_t deqSize = 128;
                // AscendC::DataCopy(workA1, deqGlobal, deqSize);
                // AscendC::LocalTensor<uint64_t> deqFB = inQueueDeqFB.AllocTensor<uint64_t>();
                // uint16_t fbufBurstLen = deqSize / 128;
                // AscendC::DataCopyParams dataCopyParams(1, fbufBurstLen, 0, 0);
                // AscendC::DataCopy(deqFB, workA1, dataCopyParams);
                // // 设置量化tensor
                // AscendC::SetFixPipeConfig(deqFB);
                // AscendC::PipeBarrier<PIPE_FIX>();
                // AscendC::DataCopy(dstA1, dstCO1, intriParams);
                // AscendC::DataCopy(dstGlobal, dstA1, dstCO1Size);
                // inQueueDeqA1.FreeTensor(workA1);
                // inQueueDeqFB.FreeTensor(deqFB);
                // outQueueCO1.FreeTensor(dstCO1);
                // outQueueA1.FreeTensor(dstA1);
            }
        private:
            AscendC::TPipe pipe;
            // feature map queue
            AscendC::TQue<AscendC::TPosition::A1, 1> inQueueFmA1;
            AscendC::TQue<AscendC::TPosition::A2, 1> inQueueFmA2;
            // weight queue
            AscendC::TQue<AscendC::TPosition::B1, 1> inQueueWeB1;
            AscendC::TQue<AscendC::TPosition::B2, 1> inQueueWeB2;
            // bias queue
            AscendC::TQue<AscendC::TPosition::A1, 1> inQueueBiasA1;
            // deq tensor queue
            AscendC::TQue<AscendC::TPosition::A1, 1> inQueueDeqA1;
            // fb dst of deq tensor
            AscendC::TQue<AscendC::TPosition::C2PIPE2GM, 1> inQueueDeqFB;
            // dst queue
            AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;
            AscendC::TQue<AscendC::TPosition::A1, 1> outQueueA1;
            AscendC::GlobalTensor<fmap_T> fmGlobal;
            AscendC::GlobalTensor<weight_T> weGlobal;
            AscendC::GlobalTensor<dst_T> dstGlobal;
            AscendC::GlobalTensor<uint64_t> deqGlobal;
            AscendC::GlobalTensor<dstCO1_T> biasGlobal;
            AscendC::GlobalTensor<half> eleWiseGlobal;
            uint16_t channelSize = 32;
            uint16_t H = 4, W = 4;
            uint8_t Kh = 2, Kw = 2;
            uint16_t Cout;
            uint16_t C0, C1;
            uint8_t dilationH, dilationW;
            uint16_t coutBlocks, ho, wo, howo, howoRound;
            uint32_t featureMapA1Size, weightA1Size, featureMapA2Size, weightB2Size, biasSize, dstSize, dstCO1Size;
            uint16_t m, k, n;
            uint8_t fmRepeat, weRepeat;
            QuantMode_t deqMode = QuantMode_t::NoQuant;
        };
        #define KERNEL_CUBE_DATACOPY(dst_type, fmap_type, weight_type, dstCO1_type, CoutIn, dilationHIn, dilationWIn, deqModeIn)  \
            extern "C" __global__ __aicore__ void cube_datacopy_kernel_##fmap_type(__gm__ uint8_t* fmGm, __gm__ uint8_t* weGm,    \
                __gm__ uint8_t* biasGm, __gm__ uint8_t* deqGm, __gm__ uint8_t* dstGm)                                             \
            {                                                                                                                     \
                if (g_coreType == AscendC::AIV) {                                                                                 \
                    return;                                                                                                       \
                }                                                                                                                 \
                KernelCubeDataCopy<dst_type, fmap_type, weight_type, dstCO1_type> op(CoutIn, dilationHIn, dilationWIn,            \
                    deqModeIn);                                                                                                   \
                op.Init(fmGm, weGm, biasGm, deqGm, dstGm);                                                                        \
                op.Process();                                                                                                     \
            }
        KERNEL_CUBE_DATACOPY(half, int8_t, int8_t, int32_t, 128, 1, 1, QuantMode_t::DEQF16);
        

  * 针对Atlas 200I/500 A2 推理产品，随路格式转换数据搬运，通路：CO1->GM。

示例：Mmad含有矩阵乘偏置，左矩阵和右矩阵的数据类型为int8_t，结果矩阵的数据类型为int32_t。量化模式DEQF16，scalar量化参数为0.5，将Mmad计算出的结果由int32_t量化成half并搬出。
        
        #ifdef ASCENDC_CPU_DEBUG
        #include "tikicpulib.h"
        #endif
        #include "kernel_operator.h"
        #include "../../instrs/common_utils/register_utils.h"
        template <typename dst_T, typename fmap_T, typename weight_T, typename dstCO1_T> class KernelCubeDataCopy{
        public:
            __aicore__ inline KernelCubeDataCopy(uint16_t CoutIn, uint8_t dilationHIn, uint8_t dilationWIn, QuantMode_t deqModeIn)
            {
                // ceiling of 16
                Cout = CoutIn;
                dilationH = dilationHIn;
                dilationW = dilationWIn;
                C0 = 32 / sizeof(fmap_T);
                C1 = channelSize / C0;
                coutBlocks = (Cout + 16 - 1) / 16;
                ho = H - dilationH * (Kh - 1);
                wo = W - dilationW * (Kw - 1);
                howo = ho * wo;
                howoRound = ((howo + 16 - 1) / 16) * 16;
                featureMapA1Size = C1 * H * W * C0;      // shape: [C1, H, W, C0]
                weightA1Size = C1 * Kh * Kw * Cout * C0; // shape: [C1, Kh, Kw, Cout, C0]
                featureMapA2Size = howoRound * (C1 * Kh * Kw * C0);
                weightB2Size = (C1 * Kh * Kw * C0) * coutBlocks * 16;
                m = howo;
                k = C1 * Kh * Kw * C0;
                n = Cout;
                biasSize = Cout;                  // shape: [Cout]
                dstSize = coutBlocks * howo * 16; // shape: [coutBlocks, howo, 16]
                dstCO1Size = coutBlocks * howoRound * 16;
                fmRepeat = featureMapA2Size / (16 * C0);
                weRepeat = weightB2Size / (16 * C0);
                deqMode = deqModeIn;
            }
            __aicore__ inline void Init(__gm__ uint8_t* fmGm, __gm__ uint8_t* weGm, __gm__ uint8_t* biasGm, __gm__ uint8_t* deqGm, __gm__ uint8_t* eleWiseGm, __gm__ uint8_t* dstGm)
            {
                fmGlobal.SetGlobalBuffer((__gm__ fmap_T*)fmGm);
                weGlobal.SetGlobalBuffer((__gm__ weight_T*)weGm);
                biasGlobal.SetGlobalBuffer((__gm__ dstCO1_T*)biasGm);
                deqGlobal.SetGlobalBuffer((__gm__ uint64_t*)deqGm);
                dstGlobal.SetGlobalBuffer((__gm__ dst_T*)dstGm);
                eleWiseGlobal.SetGlobalBuffer((__gm__ half*)eleWiseGm);
                pipe.InitBuffer(inQueueFmA1, 1, featureMapA1Size * sizeof(fmap_T));
                pipe.InitBuffer(inQueueFmA2, 1, featureMapA2Size * sizeof(fmap_T));
                pipe.InitBuffer(inQueueWeB1, 1, weightA1Size * sizeof(weight_T));
                pipe.InitBuffer(inQueueWeB2, 1, weightB2Size * sizeof(weight_T));
                pipe.InitBuffer(inQueueBiasA1, 1, biasSize * sizeof(dstCO1_T));
                pipe.InitBuffer(inQueueDeqA1, 1, dstCO1Size * sizeof(uint64_t));
                pipe.InitBuffer(inQueueDeqFB, 1, dstCO1Size * sizeof(uint64_t));
                pipe.InitBuffer(outQueueCO1, 1, dstCO1Size * sizeof(dstCO1_T));
                pipe.InitBuffer(inQueueC1, 1, dstSize * sizeof(half));
            }
            __aicore__ inline void Process()
            {
                CopyIn();
                Split();
                Compute();
                CopyOut();
            }
        private:
            __aicore__ inline void CopyIn()
            {
                AscendC::LocalTensor<fmap_T> featureMapA1 = inQueueFmA1.AllocTensor<fmap_T>();
                AscendC::LocalTensor<weight_T> weightB1 = inQueueWeB1.AllocTensor<weight_T>();
                AscendC::LocalTensor<dstCO1_T> biasA1 = inQueueBiasA1.AllocTensor<dstCO1_T>();
                AscendC::DataCopy(featureMapA1, fmGlobal, { 1, static_cast<uint16_t>(featureMapA1Size * sizeof(fmap_T) / 32), 0, 0 });
                AscendC::DataCopy(weightB1, weGlobal, { 1, static_cast<uint16_t>(weightA1Size * sizeof(weight_T) / 32), 0, 0 });
                AscendC::DataCopy(biasA1, biasGlobal, { 1, static_cast<uint16_t>(biasSize * sizeof(dstCO1_T) / 32), 0, 0 });
                inQueueFmA1.EnQue(featureMapA1);
                inQueueWeB1.EnQue(weightB1);
                inQueueBiasA1.EnQue(biasA1);
            }
            __aicore__ inline void Split()
            {
                AscendC::LocalTensor<fmap_T> featureMapA1 = inQueueFmA1.DeQue<fmap_T>();
                AscendC::LocalTensor<weight_T> weightB1 = inQueueWeB1.DeQue<weight_T>();
                AscendC::LocalTensor<fmap_T> featureMapA2 = inQueueFmA2.AllocTensor<fmap_T>();
                AscendC::LocalTensor<weight_T> weightB2 = inQueueWeB2.AllocTensor<weight_T>();
                uint8_t padList[] = {0, 0, 0, 0};
                // load3dv2
                AscendC::LoadData(featureMapA2, featureMapA1, { padList, H, W, channelSize, k, howoRound, 0, 0, 1, 1, Kw, Kh, dilationW, dilationH, false, false, 0 });
                // load2d
                AscendC::LoadData(weightB2, weightB1, { 0, weRepeat, 1, 0, 0, false, 0 });
                inQueueFmA2.EnQue<fmap_T>(featureMapA2);
                inQueueWeB2.EnQue<weight_T>(weightB2);
                inQueueFmA1.FreeTensor(featureMapA1);
                inQueueWeB1.FreeTensor(weightB1);
            }
            __aicore__ inline void Compute()
            {
                AscendC::LocalTensor<fmap_T> featureMapA2 = inQueueFmA2.DeQue<fmap_T>();
                AscendC::LocalTensor<weight_T> weightB2 = inQueueWeB2.DeQue<weight_T>();
                AscendC::LocalTensor<dstCO1_T> dstCO1 = outQueueCO1.AllocTensor<dstCO1_T>();
                AscendC::LocalTensor<dstCO1_T> biasA1 = inQueueBiasA1.DeQue<dstCO1_T>();
                // C = A * B + bias
                // m: 左矩阵Height, k: 左矩阵Width, n: 右矩阵Width
                AscendC::Mmad(dstCO1, featureMapA2, weightB2, biasA1, { m, n, k, true, 0, false, false, false });
                outQueueCO1.EnQue<dstCO1_T>(dstCO1);
                inQueueFmA2.FreeTensor(featureMapA2);
                inQueueWeB2.FreeTensor(weightB2);
            }
            __aicore__ inline void CopyOut()
            {
                AscendC::LocalTensor<dstCO1_T> dstCO1 = outQueueCO1.DeQue<dstCO1_T>();
                // 使能DEQF16量化，量化参数设置为0.5
                float tmp = (float)0.5;
                // 将float的tmp转换成uint64_t的deqScalar
                uint64_t deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp));
                bool nz2ndEn = false;
                // nz2nd不使能时，nSize必须为16的倍数
                uint16_t nSize = coutBlocks * 16;
                uint16_t mSize = m;
                // srcStride必须为16的倍数
                uint16_t srcStride = (m + 16 - 1) / 16 * 16;
                // nz2nd不使能时，dstStride为burst头到头的距离，且为32B对齐
                uint32_t dstStride = m * sizeof(dst_T) * 16 / 32;
                if (nz2ndEn) {
                    // nd矩阵的数量为1，src_nd_stride与dst_nd_stride填1
                    AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
                    // nz2nd使能时，nSize可以不为16的倍数，与Mmad的n保持一致
                    nSize = n;
                    // nz2nd使能时，dstStride表示同一nd矩阵的相邻连续行的间隔，与n保持一致
                    dstStride = nSize;
                };
                // 不使能relu与channelSplit
                AscendC::DataCopyCO12DstParams intriParams(nSize, mSize, dstStride, srcStride, deqMode, 0, false, nz2ndEn);
               
                // mov l0c to gm, deq scalar quant
                AscendC::SetFixpipePreQuantFlag(deqScalar);  // 设置量化参数
                AscendC::PipeBarrier<PIPE_FIX>();
                AscendC::DataCopy(dstGlobal, dstCO1, intriParams);
                // // mov l0c to gm, deq tensor quant
                // // 需要额外申请deq tensor的gm空间，将值搬运到workA1
                // AscendC::LocalTensor<uint64_t> workA1 = inQueueDeqA1.AllocTensor<uint64_t>();
                // // deq tensor的size
                // uint16_t deqSize = 128;
                // AscendC::DataCopy(workA1, deqGlobal, deqSize);
                // // deq tensor在fix上的地址
                // AscendC::LocalTensor<uint64_t> deqFB = inQueueDeqFB.AllocTensor<uint64_t>();
                // // l1->fix, burst_len unit is 128Bytes
                // uint16_t fbufBurstLen = deqSize / 128;
                // AscendC::DataCopyParams dataCopyParams(1, fbufBurstLen, 0, 0);
                // AscendC::DataCopy(deqFB, workA1, dataCopyParams);
                // // 设置量化tensor
                // AscendC::SetFixPipeConfig(deqFB);
                // AscendC::PipeBarrier<PIPE_FIX>();
                // // mov l0c to gm, 量化操作后使能ClipRelu操作
                // intriParams.clipReluPre = 1; 
                // // 设置clip relu的值到寄存器
                // uint64_t clipReluVal = 0x3c00; // value 1, half
                // SetFixPipeClipRelu(clipReluVal);
                // //mov l0c to gm, 量化操作后，设置 element-wise 操作，Add
                // intriParams.eltWiseOp = 1;
                // // 需要额外申请 element-wise tensor的gm空间，将值搬到eleWiseTensor
                // AscendC::LocalTensor<half> eleWiseTensor = inQueueC1.AllocTensor<half>();
                // DataCopy(eleWiseTensor, eleWiseGlobal, { 1, static_cast<uint16_t>(sizeof(half) * dst_size / 32), 0, 0 });
                // AscendC::PipeBarrier<PIPE_ALL>();
                // // 将存放element-wise tensor的地址设置到寄存器里
                // SetFixPipeAddr(eleWiseTensor, 1);
        
                // AscendC::DataCopy(dstGlobal, dstCO1, intriParams);
                // inQueueDeqA1.FreeTensor(workA1);
                // inQueueDeqFB.FreeTensor(deqFB);
                // outQueueCO1.FreeTensor(dstCO1);
                // inQueueC1.FreeTensor(eleWiseTensor);
             }
        private:
            AscendC::TPipe pipe;
            // feature map queue
            AscendC::TQue<AscendC::TPosition::A1, 1> inQueueFmA1;
            AscendC::TQue<AscendC::TPosition::A2, 1> inQueueFmA2;
            // weight queue
            AscendC::TQue<AscendC::TPosition::B1, 1> inQueueWeB1;
            AscendC::TQue<AscendC::TPosition::B2, 1> inQueueWeB2;
            // bias queue
            AscendC::TQue<AscendC::TPosition::A1, 1> inQueueBiasA1;
            // deq tensor queue
            AscendC::TQue<AscendC::TPosition::A1, 1> inQueueDeqA1;
            // fb dst of deq tensor
            AscendC::TQue<AscendC::TPosition::C2PIPE2GM, 1> inQueueDeqFB;
            // dst queue
            AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;
            // element-wise tensor
            AscendC::TQue<AscendC::TPosition::C1, 1> inQueueC1;
            AscendC::GlobalTensor<fmap_T> fmGlobal;
            AscendC::GlobalTensor<weight_T> weGlobal;
            AscendC::GlobalTensor<dst_T> dstGlobal;
            AscendC::GlobalTensor<uint64_t> deqGlobal;
            AscendC::GlobalTensor<dstCO1_T> biasGlobal;
            AscendC::GlobalTensor<half> eleWiseGlobal;
            uint16_t channelSize = 32;
            uint16_t H = 4, W = 4;
            uint8_t Kh = 2, Kw = 2;
            uint16_t Cout;
            uint16_t C0, C1;
            uint8_t dilationH, dilationW;
            uint16_t coutBlocks, ho, wo, howo, howoRound;
            uint32_t featureMapA1Size, weightA1Size, featureMapA2Size, weightB2Size, biasSize, dstSize, dstCO1Size;
            uint16_t m, k, n;
            uint8_t fmRepeat, weRepeat;
            QuantMode_t deqMode = QuantMode_t::NoQuant;
        };
        #define KERNEL_CUBE_DATACOPY(dst_type, fmap_type, weight_type, dstCO1_type, CoutIn, dilationHIn, dilationWIn, deqModeIn)  \
            extern "C" __global__ __aicore__ void cube_datacopy_kernel_##fmap_type(__gm__ uint8_t* fmGm, __gm__ uint8_t* weGm,    \
                __gm__ uint8_t* biasGm, __gm__ uint8_t* deqGm, __gm__ uint8_t* eleWiseGm, __gm__ uint8_t* dstGm)                                             \
            {                                                                                                                     \
                if (g_coreType == AscendC::AIV) {                                                                                 \
                    return;                                                                                                       \
                }                                                                                                                 \
                KernelCubeDataCopy<dst_type, fmap_type, weight_type, dstCO1_type> op(CoutIn, dilationHIn, dilationWIn,            \
                    deqModeIn);                                                                                                   \
                op.Init(fmGm, weGm, biasGm, deqGm, eleWiseGm, dstGm);                                                                        \
                op.Process();                                                                                                     \
            }
        KERNEL_CUBE_DATACOPY(half, int8_t, int8_t, int32_t, 128, 1, 1, QuantMode_t::DEQF16);
        




**父主题：** [DataCopy](atlasascendc_api_07_0101.html)



---

## Async


# Async

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

Async提供了一个统一的接口，用于在不同模式下（AIC或AIV）执行特定函数，从而避免代码中直接的硬件条件判断（如使用ASCEND_IS_AIV或ASCEND_IS_AIC）。

#### 函数原型
    
    
    template <EngineType engine, auto funPtr, class... Args>
    __aicore__ void Async(Args... args)
    

#### 参数说明

表1 模板参数说明

展开

参数名 | 描述  
---|---  
engine | 引擎模式，参数取值分别为AIC、AIV。
    
    
    enum class EngineType : int32_t {
        AIC = 1, // 仅AIC
        AIV = 2  // 仅AIV
    };
      
  
funPtr | 函数指针，指定要执行的函数，函数签名和参数类型由class... Args决定。  
class... Args | 可变参数模板，表示函数参数的类型列表，用于传递给funPtr。  
  
表2 参数说明

展开

参数名 | 输入/输出 | 描述  
---|---|---  
Args... args | 输入 | 与class... Args对应的参数列表，表示传递给funPtr的实际参数。  
  
#### 返回值说明

无

#### 约束说明

无

#### 调用示例
    
    
    extern "C" __global__ __aicore__ void baremix_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                                  GM_ADDR workspace, GM_ADDR tilingGm)
    {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        AscendC::TPipe pipe;
        TCubeTiling tiling;
        CopyTiling(&tiling, tilingGm);
        // 避免代码中直接的硬件条件判断（如使用ASCEND_IS_AIV或ASCEND_IS_AIC）
        Async<EngineType::AIC, aicOperation>(a, b, bias, c, workspace, tiling, &pipe);
        Async<EngineType::AIV, aivOperation>(c, tiling, &pipe);
    }
    __aicore__ inline void aicOperation(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace, const TCubeTiling &tiling, AscendC::TPipe *pipe) {
        MatmulLeakyKernel<half, half, float, float> matmulLeakyKernel;
        matmulLeakyKernel.Init(a, b, bias, c, workspace, tiling, pipe);
        REGIST_MATMUL_OBJ(pipe, GetSysWorkSpacePtr(), matmulLeakyKernel.matmulObj, &matmulLeakyKernel.tiling);
        matmulLeakyKernel.Process(pipe);
    }
    
    __aicore__ inline void aivOperation(GM_ADDR c, const TCubeTiling &tiling, AscendC::TPipe *pipe) {
        LeakyReluKernel<float> leakyReluKernel;
        leakyReluKernel.Init(c, tiling, pipe);
        leakyReluKernel.Process(pipe);
    }
    

**父主题：** [工具函数](atlasascendc_api_07_00044.html)



---

## DEVICE_IMPL_OP_OPTILING


# DEVICE_IMPL_OP_OPTILING

#### 功能说明

在[Tiling下沉](/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_00014.html)场景中，该宏定义用于生成Tiling下沉的注册类，再通过调用注册类的成员函数来注册需要下沉的Tiling函数。

#### 函数原型
    
    
    namespace optiling {
    using SinkTilingFunc = std::function<ge::graphStatus(gert::TilingContext *context)>;
    
    class DeviceOpImplRegisterImpl;
    // 开发者仅关注Tiling成员函数
    class DeviceOpImplRegister {
    public:
      DeviceOpImplRegister(const char *opType);
      ~DeviceOpImplRegister();
      DeviceOpImplRegister(DeviceOpImplRegister &&other) noexcept;
      DeviceOpImplRegister(const DeviceOpImplRegister &other);
      DeviceOpImplRegister &operator=(const DeviceOpImplRegister &) = delete;
      DeviceOpImplRegister &operator=(DeviceOpImplRegister &&) = delete;
      DeviceOpImplRegister &Tiling(SinkTilingFunc func);
    
    // ...
    };
    }  // namespace optiling
    
    #define DEVICE_IMPL_OP_OPTILING(optype)                                                                      \
      static optiling::DeviceOpImplRegister VAR_UNUSED g_deviceOpImplRegister##optype =                                    \
          optiling::DeviceOpImplRegister(#optype)
    #endif
    

#### 参数说明

表1 DEVICE_IMPL_OP_OPTILING参数说明

展开

参数 |  输入/输出 |  说明  
---|---|---  
optype |  输入 |  需要注册Tiling函数的OpType（算子类型）。  
  
表2 Tiling成员函数参数说明

展开

参数 |  输入/输出 |  说明  
---|---|---  
func |  输入 |  需要注册的Tiling函数，该函数接受一个[TilingContext](/document/detail/zh/CANNCommunityEdition/900beta1/API/basicdataapi/atlasopapi_07_00223.html)作为输入，以[ge::graphStatus](/document/detail/zh/CANNCommunityEdition/900beta1/API/basicdataapi/atlasopapi_07_00513.html)为返回值。  
  
#### 返回值说明

无

#### 约束说明

无

#### 调用示例
    
    
    DEVICE_IMPL_OP_OPTILING(TestOptype).Tiling(TestTilingFunc); // 将Tiling函数以及其OpType注册到Tiling下沉
    

**父主题：** [Tiling下沉](atlasascendc_api_07_00185.html)



---

## ASCENDC_TPL_SEL_PARAM


# ASCENDC_TPL_SEL_PARAM

#### 功能说明

Tiling模板编程时，开发者通过调用此接口自动生成并配置TilingKey。

使用该接口需要包含定义模板参数和模板参数组合的头文件。详细内容请参考[Tiling模板编程](/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_00025.html)。

#### 函数原型
    
    
    #define ASCENDC_TPL_SEL_PARAM(context, ...)           \
    do {                                                  \
        uint64_t key = GET_TPL_TILING_KEY({__VA_ARGS__}); \
        context->SetTilingKey(key);                       \
    } while(0)
    // context指代TilingFunc(gert::TilingContext *context)中的context
    

#### 参数说明

展开

参数 |  输入/输出 |  说明  
---|---|---  
context |  输入 |  TilingFunc注册上下文。  
... |  输入 |  可变长参数，模板参数的具体值，传入时需要与定义模板参数和模板参数组合的头文件中的模板参数顺序保持一致。  
  
#### 返回值说明

无

#### 约束说明

无

#### 调用示例
    
    
    #include "tiling_key_add_custom.h"
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        TilingDataTemplate tiling;
        uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
        ge::DataType dtype_x = context->GetInputDesc(0)->GetDataType();
        ge::DataType dtype_y = context->GetInputDesc(1)->GetDataType();
        ge::DataType dtype_z = context->GetOutputDesc(0)->GetDataType();
        uint32_t D_T_X = static_cast<int>(dtype_x), D_T_Y = static_cast<int>(dtype_y), D_T_Z = static_cast<int>(dtype_z), TILE_NUM = 1, IS_SPLIT = 0;
        if (totalLength < MIN_LENGTH_FOR_SPLIT) {
            IS_SPLIT = 0;
            TILE_NUM = 1;
        } else {
            IS_SPLIT = 1;
            TILE_NUM = DEFAULT_TILE_NUM;
        }
        context->SetBlockDim(NUM_BLOCKS);
        tiling.set_totalLength(totalLength);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        ASCENDC_TPL_SEL_PARAM(context, D_T_X, D_T_Y, D_T_Z, TILE_NUM, IS_SPLIT);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
    

**父主题：** [Tiling模板编程](atlasascendc_api_07_00184.html)

