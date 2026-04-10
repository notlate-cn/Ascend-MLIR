# Kernel Tiling接口

> 来源: 昇腾社区官网 AscendC算子开发文档

---

## 目录

- [GET_TILING_DATA](#get_tiling_data)
- [GET_TILING_DATA_WITH_STRUCT](#get_tiling_data_with_struct)
- [GET_TILING_DATA_MEMBER](#get_tiling_data_member)
- [TILING_KEY_IS](#tiling_key_is)
- [Tiling注册](#tiling注册)
- [Kernel类型](#kernel类型)

---

---

## GET_TILING_DATA

# GET_TILING_DATA_PTR_WITH_STRUCT

#### 功能说明

在使用该宏时，开发者可以通过指定结构体名称来获取相应的Tiling信息，并将其填入对应的Tiling结构体中。完成填充后，该宏将返回一个指向该Tiling结构体的指针，并使用__tiling_data_ptr__修饰符对该指针进行修饰。这种修饰方式能够确保在动静态Shape场景下代码的统一性和兼容性。

#### 函数原型
    
    GET_TILING_DATA_PTR_WITH_STRUCT(tiling_struct, dst_ptr, tiling_ptr)
    
#### 参数说明

参数 | 输入/输出 | 说明  
tiling_struct | 输入 | 指定的结构体名称。  
dst_ptr | 输出 | 返回指定的Tiling结构体指针。  
tiling_ptr | 输入 | 算子入口函数处传入的Tiling参数。  
  
  * 该宏需在算子Kernel代码处使用，并且传入的dst_ptr参数无需声明类型。
  * 动态Shape场景下，获取到的dst_ptr是指向Global Memory变量的指针；静态Shape场景下，获取到的dst_ptr是指向局部变量的指针，需确保在合理的作用域范围内使用。
  * 暂不支持Kernel直调工程。

    extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *tiling)
    {
        KernelAdd op;
    
        GET_TILING_DATA_PTR_WITH_STRUCT(AddCustomTilingData, tilingDataPtr, tiling);
    
        op.Init(x, y, z, tilingDataPtr->totalLength, tilingDataPtr->tileNum);
        op.Process();
        
    }
    
以下是错误调用的示例：
    
    __aicore__ __tiling_data_ptr__ AddCustomTilingData* foo(__gm__ uint8_t *tiling)
    {
        GET_TILING_DATA_PTR_WITH_STRUCT(AddCustomTilingData, tilingDataPtr, tiling);
        return tilingDataPtr;
    }
    
    extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *tiling)
    {
        KernelAdd op;
    
        auto tilingDataPtr = foo(tiling);  // 错误，foo函数已经执行完成，非法访问生命周期已经结束的局部变量
    
        op.Init(x, y, z, tilingDataPtr->totalLength, tilingDataPtr->tileNum);
        op.Process();
    }

---

## GET_TILING_DATA_WITH_STRUCT

# GET_TILING_DATA_WITH_STRUCT

#### 功能说明

#### 函数原型
    
    GET_TILING_DATA_WITH_STRUCT(struct_name, tiling_data, tiling_arg)
    
#### 参数说明

参数 | 输入/输出 | 说明  
struct_name | 输入 | 指定的结构体名称。  
tiling_data | 输出 | 返回指定Tiling结构体变量。  
tiling_arg | 输入 | 此参数为算子入口函数处传入的tiling参数。  
  
  * 本函数需在算子Kernel代码处使用，并且传入的tiling_data参数不需要声明类型。
  * 暂不支持Kernel直调工程。

    extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *tiling)
    {
        KernelAdd op;
        if (TILING_KEY_IS(1)) {
            GET_TILING_DATA_WITH_STRUCT(Add_Struct_Special, tilingData, tiling); // 使用算子指定注册的结构体
    	op.Init(x, y, z, tilingData.totalLengthSpecial, tilingData.tileNumSpecial);
        } else {
            GET_TILING_DATA(tilingData, tiling);   // 使用算子默认注册的结构体
    	op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
        }
        if (TILING_KEY_IS(1)) {
            op.Process();
        }  else  if (TILING_KEY_IS(2)) {
            op.Process();
        } else  if (TILING_KEY_IS(3)) {
            op.Process();
        }
    }
    
---

## GET_TILING_DATA_MEMBER

# GET_TILING_DATA_MEMBER

#### 功能说明

用于获取tiling结构体的成员变量。

#### 函数原型
    
    GET_TILING_DATA_MEMBER(struct_name, mem_name, tiling_data, tiling_arg)
    
#### 参数说明

参数 | 输入/输出 | 说明  
struct_name | 输入 | 指定的结构体名称。  
mem_name | 输入 | 指定的成员变量名称。  
tiling_data | 输出 | 返回指定Tiling结构体的成员变量。  
tiling_arg | 输入 | 此参数为算子入口函数处传入的tiling参数。  
  
  * 本函数需在算子kernel代码处使用，并且传入的tiling_data参数不需要声明类型。
  * 暂不支持Kernel直调工程。

    extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *tiling)
    {
        KernelAdd op;
        if ASCEND_IS_AIV {
            GET_TILING_DATA(tilingData, tiling);   // Vector侧使用算子默认注册的完整结构体
    	op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
            op.Process();
        } else {
            GET_TILING_DATA_MEMBER(Add_Struct, tCubeTiling, tCubeTilingVar, tiling); // Cube侧仅使用算子注册结构体的成员变量tCubeTiling
    	op.Init(x, y, z, tCubeTilingVar);
    	op.Process();
        }
    }
    
---

## TILING_KEY_IS

# TILING_KEY_IS

#### 功能说明

在核函数中判断本次执行时的tiling_key是否等于host侧运行时设置的某个key，从而标识tiling_key==key的一条kernel分支。

#### 函数原型
    
    TILING_KEY_IS(key)
    
#### 参数说明

参数 | 输入/输出 | 说明  
key | 输入 | key表示某个核函数的分支，必须是非负整数。  
  
  * TILING_KEY_IS运用于if和else if分支，不支持else分支，即用TILING_KEY_IS函数来表征N个分支，必须用N个TILING_KEY_IS(key)来分别表示。
  * 暂不支持Kernel直调工程。

    extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        if (workspace == nullptr) {
            return;
        }
        KernelAdd op;
        op.Init(x, y, z, tilingData.numBlocks, tilingData.totalLength, tilingData.tileNum);
        // 当TilingKey为1时，执行Process1；为2时，执行Process2；为3时，执行Process3
        if (TILING_KEY_IS(1)) {
            op.Process1();
        } else if (TILING_KEY_IS(2)) {
            op.Process2();
        } else if (TILING_KEY_IS(3)) {
            op.Process3();
        }
        // 其他代码逻辑
        ...
        // 此处示例当TilingKey为3时，会执行ProcessOther
        if (TILING_KEY_IS(3)) {
            op.ProcessOther();
        }
    }
    
配套的host侧tiling函数示例（伪代码）：
    
    ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        // 其他代码逻辑
        ...
        if (context->GetInputShape(0) > 10) {
            context->SetTilingKey(1);
        } else if (some condition) {
            context->SetTilingKey(2);
        } else if (some condition) {
            context->SetTilingKey(3);
        }
    }
    
---

## Tiling注册

# REGISTER_TILING_DEFAULT

#### 功能说明

用于在kernel侧注册用户使用标准C++语法自定义的默认TilingData结构体。

注册TilingData结构体用于告知框架侧用户使用标准C++语法来定义TilingData，同时告知框架TilingData结构体类型，用于框架做tiling数据解析。

#### 函数原型
    
    REGISTER_TILING_DEFAULT(TILING_STRUCT)
    
#### 参数说明

参数 | 输入/输出 | 说明  
TILING_STRUCT | 输入 | 用户注册的默认自定义TilingData结构体。  
  
  * 若TilingData结构体在命名空间内，注册时需要携带对应的命名空间作用域符。
  * 暂不支持Kernel直调工程。

    extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *tiling)
    {
        REGISTER_TILING_DEFAULT(optiling::TilingData);
        GET_TILING_DATA(tilingData, tiling);
        KernelAdd op;
        op.Init(x, y, z, tilingData.blkDim, tilingData.totalSize, tilingData.splitTile);
        op.Process();
    }
    
# REGISTER_TILING_FOR_TILINGKEY

#### 功能说明

用于在kernel侧注册与TilingKey相匹配的TilingData自定义结构体；该接口需提供一个逻辑表达式，逻辑表达式以字符串“TILING_KEY_VAR”代指实际TilingKey，表达TilingKey所满足的范围。

#### 函数原型
    
    REGISTER_TILING_FOR_TILINGKEY(EXPRESSION, TILING_STRUCT)
    
#### 参数说明

参数 | 输入/输出 | 说明  
EXPRESSION | 输入 | EXPRESSION为逻辑运算，其中用TILING_KEY_VAR指代TilingKey。  
TILING_STRUCT | 输入 | 用户注册的与TilingKey相匹配的TilingData自定义结构体。  
  
  * 使用该接口时，需确保已使用REGISTER_TILING_DEFAULT注册默认的用户自定义TilingData结构体，用于告知框架侧用户使用标准C++语法来定义TilingData。
  * EXPRESSION当前支持位运算：&、|、~、^；移位运算符：<<、>>；算术运算：+、-、*、/、%；条件运算符：==、!=、>、<、>=、<=；逻辑与&&、或||以及()。优先级同C++。
  * 若TilingData结构体在命名空间内，注册时需要携带对应的命名空间作用域符。
  * 不支持同个TilingKey指向不同TilingData结构体，会出现拦截报错。
  * 暂不支持kernel直调工程。

    extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *tiling)
    {
        REGISTER_TILING_DEFAULT(optiling::TilingData);  // 注册用户默认自定义TilingData结构体
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1", optiling::TilingDataA); // 注册TilingKey为1的TilingData结构体
        REGISTER_TILING_FOR_TILINGKEY("(TILING_KEY_VAR >= 10) && (TILING_KEY_VAR <= 15)", optiling::TilingDataB); // 注册TilingKey在[10,15]之间的TilingData结构体
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR & 0xFF", optiling::TilingDataC); // 注册TilingKey低16位为1的TilingData结构体
        if (TILING_KEY_IS(1)) {
            GET_TILING_DATA_WITH_STRUCT(optiling::TilingDataA, tilingData, tiling);
            ......
        } else if (TILING_KEY_IS(11)) {
            GET_TILING_DATA_WITH_STRUCT(optiling::TilingDataB, tilingData, tiling);
            ......
        } else if (TILING_KEY_IS(14)) {
            GET_TILING_DATA_WITH_STRUCT(optiling::TilingDataB, tilingData, tiling);
            ......
        } else if (TILING_KEY_IS(255)) {
            GET_TILING_DATA_WITH_STRUCT(optiling::TilingDataC, tilingData, tiling);
            ......
        } else {
            GET_TILING_DATA(tilingData, tiling);
            ......
        }
    }
    
使用标准C++语法注册tiling结构体：
    
    class TilingDataA{
    public:
        ...
    };
    class TilingDataB{
    public:
        ...
    };
    class TilingDataC{
    public:
        ...
    };
    
配套的host侧tiling函数示例：
    
    ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        // 其他代码逻辑
        ...
        if(condition1){
            context->SetTilingKey(1);
            optiling::TilingDataA *Addtiling = context->GetTilingData<optiling::TilingDataA>();
            ...
        } else if (condition2){
            context->SetTilingKey(11);
            optiling::TilingDataB *Addtiling = context->GetTilingData<optiling::TilingDataB >();
            ...
        } else if (condition3){
            context->SetTilingKey(14);
            optiling::TilingDataB *Addtiling = context->GetTilingData<optiling::TilingDataB >();
            ...
        } else if (condition4){
            context->SetTilingKey(255);
            optiling::TilingDataC *Addtiling = context->GetTilingData<optiling::TilingDataC >();
            ...
        }
        ...
        // 其他代码逻辑
    }
    
# REGISTER_NONE_TILING

#### 功能说明

在Kernel侧使用标准C++语法自定义的TilingData结构体时，若用户不确定需要注册哪些结构体，可使用该接口告知框架侧需使用未注册的标准C++语法来定义TilingData，并配套GET_TILING_DATA_WITH_STRUCT，GET_TILING_DATA_MEMBER，GET_TILING_DATA_PTR_WITH_STRUCT来获取对应的TilingData。

#### 函数原型
    
    REGISTER_NONE_TILING
    
#### 参数说明

无

  * 暂不支持Kernel直调工程。
  * 使用GET_TILING_DATA需提供默认注册的TilingData结构体，但本接口不注册TilingData结构体，故不支持与5.11.1-GET_TILING_DATA组合使用。
  * 不支持和REGISTER_TILING_DEFAULT或REGISTER_TILING_FOR_TILINGKEY混用，即不支持注册TilingData结构体的场景与非注册场景混合使用。

    # Tiling模板库提供方，无法预知用户实例化何种TilingData结构体
    template <class BrcDag>
    struct BroadcastBaseTilingData {
        int32_t scheMode;
        int32_t shapeLen;
        int32_t ubSplitAxis;
        int32_t ubFormer;
        int32_t ubTail;
        int64_t ubOuter;
        int64_t blockFormer;
        int64_t blockTail;
        int64_t dimProductBeforeUbInner;
        int64_t elemNum;
        int64_t blockNum;
        int64_t outputDims[BROADCAST_MAX_DIMS_NUM];
        int64_t outputStrides[BROADCAST_MAX_DIMS_NUM];
        int64_t inputDims[BrcDag::InputSize][2]; // 整块 + 尾块
        int64_t inputBrcDims[BrcDag::CopyBrcSize][BROADCAST_MAX_DIMS_NUM];
        int64_t inputVecBrcDims[BrcDag::VecBrcSize][BROADCAST_MAX_DIMS_NUM];
        int64_t inputStrides[BrcDag::InputSize][BROADCAST_MAX_DIMS_NUM];
        int64_t inputBrcStrides[BrcDag::CopyBrcSize][BROADCAST_MAX_DIMS_NUM];
        int64_t inputVecBrcStrides[BrcDag::VecBrcSize];
        char scalarData[BROADCAST_MAX_SCALAR_BYTES];
    };
    
    template <uint64_t schMode, class BrcDag> class BroadcastSch {
    public:
        __aicore__ inline explicit BroadcastSch(GM_ADDR& tmpTiling)
            : tiling(tmpTiling)
        {}
        template <class... Args>
        __aicore__ inline void Process(Args... args)
        {
            REGISTER_NONE_TILING; // 告知框架侧使用未注册的TilingData结构体
            if constexpr (schMode == 1) {
                GET_TILING_DATA_WITH_STRUCT(BroadcastBaseTilingData<BrcDag>, tilingData, tiling);
                GET_TILING_DATA_MEMBER(BroadcastBaseTilingData<BrcDag>, blockNum, blockNumVar, tiling);
                TPipe pipe;
                BroadcastNddmaSch<BrcDag, false> sch(&tilingData); // 获取Schedule
                sch.Init(&pipe, args...);
                sch.Process();
            }   else if constexpr (schMode == 202) {
                GET_TILING_DATA_PTR_WITH_STRUCT(BroadcastOneDimTilingDataAdvance, tilingDataPtr, tiling);
                BroadcastOneDimAdvanceSch<BrcDag> sch(tilingDataPtr); // 获取Schedule
                sch.Init(args...);
                sch.Process();
            }
        }
    public:
        GM_ADDR tiling;
    };
    
    #用户通过传入schMode, OpDag模板参数来实例化模板库
    using namespace AscendC;
    template <uint64_t schMode>
    __global__ __aicore__ void mul(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
    {
        if constexpr (std::is_same<DTYPE_X1, int8_t>::value) {
            // int8
            using OpDag = MulDag::MulInt8Op::OpDag;
            BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(x1, x2, y);
        } else if constexpr (std::is_same<DTYPE_X1, uint8_t>::value) {
            // uint8
            using OpDag = MulDag::MulUint8Op::OpDag;
            BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(x1, x2, y);
        }
    }
    
---

## Kernel类型

# 设置Kernel类型

#### 功能说明

用于用户自定义设置kernel类型，控制算子执行时只启动该类型的核，避免启动不需要工作的核，缩短核启动开销。

#### 函数原型

  * 设置全局默认的kernel type，对所有的tiling key生效。 

当前支持在自定义算子工程和Kernel直调工程中使用。
        
        KERNEL_TASK_TYPE_DEFAULT(value)
        
  * 设置某一个具体的tiling key对应的kernel type。 

当前仅支持在自定义算子工程中使用。
        
        KERNEL_TASK_TYPE(key, value)
        
#### 参数说明

表1 参数说明

参数 |  输入/输出 |  说明  
key |  输入 |  tiling key的key值，此参数是正数，表示某个核函数的分支。  
value |  输入 |  设置的kernel类型，可选值范围，kernel类型具体说明请参考[表2](#ZH-CN_TOPIC_0000002520880714__table76335324910)。不同硬件架构支持的参数取值不同，具体支持的参数取值请参考[kernel type取值约束](#ZH-CN_TOPIC_0000002520880714__li693212153417)。
    
    enum KernelMetaType {
        KERNEL_TYPE_AIV_ONLY,
        KERNEL_TYPE_AIC_ONLY,
        KERNEL_TYPE_MIX_AIV_1_0,
        KERNEL_TYPE_MIX_AIC_1_0,
        KERNEL_TYPE_MIX_AIC_1_1,
        KERNEL_TYPE_MIX_AIC_1_2,
        KERNEL_TYPE_AICORE,
        KERNEL_TYPE_VECTORCORE,
        KERNEL_TYPE_MIX_AICORE,
        KERNEL_TYPE_MIX_VECTOR_CORE,
        KERNEL_TYPE_MAX
    };
      
表2 kernel type取值说明

参数 |  说明  
KERNEL_TYPE_AIV_ONLY |  算子执行时仅启动AI Core上的Vector核：比如用户在host侧设置numBlocks为10，则会启动10个Vector核。  
KERNEL_TYPE_AIC_ONLY |  算子执行时仅启动AI Core上的Cube核：比如用户在host侧设置numBlocks为10，则会启动10个Cube核。  
KERNEL_TYPE_MIX_AIV_1_0 |  AIC、AIV混合场景下，使用了多核控制相关指令时，设置核函数的类型为MIX AIV:AIC 1:0（带有硬同步），算子执行时仅会启动AI Core上的Vector核，比如用户在host侧设置numBlocks为10，则会启动10个Vector核。 硬同步的概念解释如下：当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用SyncAll()函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。目前多核同步分为硬同步和软同步，硬同步是利用硬件自带的全核同步指令由硬件保证多核同步。  
KERNEL_TYPE_MIX_AIC_1_0 |  AIC、AIV混合场景下，使用了多核控制相关指令时，设置核函数的类型为MIX AIC:AIV 1:0（带有硬同步），算子执行时仅会启动AI Core上的Cube核，比如用户在host侧设置numBlocks为10，则会启动10个Cube核。  
KERNEL_TYPE_MIX_AIC_1_1 |  AIC、AIV混合场景下，设置核函数的类型为MIX AIC:AIV 1:1，算子执行时会同时启动AI Core上的Cube核和Vector核，比如用户在host侧设置numBlocks为10，则会启动10个Cube核和10个Vector核。  
KERNEL_TYPE_MIX_AIC_1_2 |  AIC、AIV混合场景下，设置核函数的类型为MIX AIC:AIV 1:2，算子执行时会同时启动AI Core上的Cube核和Vector核，比如用户在host侧设置numBlocks为10，则会启动10个Cube核和20个Vector核。  
KERNEL_TYPE_AICORE  |  算子执行时仅会启动AI Core，比如用户在host侧设置numBlocks为5，则会启动5个AI Core。  
KERNEL_TYPE_VECTORCORE |  该参数为预留参数，当前版本暂不支持。  
KERNEL_TYPE_MIX_AICORE |  该参数为预留参数，当前版本暂不支持。  
KERNEL_TYPE_MIX_VECTOR_CORE |  基于Ascend C开发的矢量计算相关的算子可以运行在Vector Core上，调用本接口传入该参数用于使能Vector Core。 使能Vector Core后，算子执行时会同时启动AI Core和Vector Core，用于并行计算。比如用户在host侧设置numBlocks为10，则会启动总数为10的AI Core和Vector Core。 需要注意的是，通过SetBlockDim设置核数时，需要大于AI Core的核数，否则不会启动VectorCore。  
  
  * kernel type取值约束 
  * **KERNEL_TASK_TYPE** 优先级高于**KERNEL_TASK_TYPE_DEFAULT** ，同时设置了全局kernel type和某一个tiling key的kernel type，该tiling key的kernel type以**KERNEL_TASK_TYPE** 设置的为准。
  * 没有设置全局默认kernel type的情况下，如果开发者只为其中的某几个tiling key设置kernel type，即部分tiling key没有设置kernel type，会导致算子kernel编译报错。
  * 当设置具体的kernel task type时，用户的算子实现需要与kernel type相匹配。比如用户设置kernel type为KERNEL_TYPE_MIX_AIC_1_2，则算子内部实现应与核配比AIC:AIV为1:2相对应；若用户设置kernel type为KERNEL_TYPE_AIC_ONLY， 则算子内部实现应该为纯cube逻辑，不应该存在vector部分的逻辑。其他的kernel type类似。
  * 当纯cube或者纯vec算子强制设定kernel type为MIX类型时，workspace的大小不能设置为0，需要设置一个大于0的值（比如16、32等）。
  * 使用Tiling模板编程时，需要通过ASCENDC_TPL_KERNEL_TYPE_SEL设置Kernel类型即可，无需再通过该接口进行设置，本接口不生效。

  * 示例一：使能VectorCore样例 
    1. 完成算子kernel侧开发时，需要通过本接口使能Vector Core，算子执行时会同时启动AI Core和Vector Core， 此时AI Core会当成Vector Core使用。示例如下： 
           
           extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
           {
               GET_TILING_DATA(tilingData, tiling);
               if (workspace == nullptr) {
                   return;
               }
               KernelAdd op;
               op.Init(x, y, z, tilingData.numBlocks, tilingData.totalLength, tilingData.tileNum);
               KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_VECTOR_CORE); // 使能VectorCore
               if (TILING_KEY_IS(1)) {
                   op.Process1();
               } else if (TILING_KEY_IS(2)) {
                   op.Process2();
               }
               // ...
           }
           
    2. 完成算子host侧Tiling开发时，设置的numBlocks代表的是AI Core和Vector Core的总数，比如用户在host侧设置numBlocks为10，则会启动总数为10的AI Core和Vector Core；为保证启动Vector Core，设置数值应大于AI Core的核数。您可以通过GetCoreNumAic接口获取AI Core的核数，GetCoreNumVector接口获取Vector Core的核数。 如下代码片段，展示了numBlocks的设置方法，此处设置为AI Core和Vector Core的总和，表示所有AI Core和Vector Core都启动。 
           
           // 配套的host侧tiling函数示例：
           ge::graphStatus TilingFunc(gert::TilingContext* context)
           {	
               // 使能VectorCore，将numBlocks置为AI Core中vector核数 + Vector Core中的vector核数
               auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
               auto totalCoreNum = ascendcPlatform.GetCoreNumAiv();
               // ASCENDXXX请替换为实际的版本型号
               if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCENDXXX) {
                  totalCoreNum = totalCoreNum + ascendcPlatform.GetCoreNumVector();
               }
               context->SetBlockDim(totalCoreNum);
           }
           
  * 示例二：设置某一个具体的tiling key对应的kernel type。如下代码为伪代码 ，不可直接运行。 
        
        extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
        {
            GET_TILING_DATA(tilingData, tiling);
            if (workspace == nullptr) {
                return;
            }
            KernelAdd op;
            op.Init(x, y, z, tilingData.numBlocks, tilingData.totalLength, tilingData.tileNum);
            KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 设置默认的kernel类型为纯AIV类型
            if (TILING_KEY_IS(1)) {
                KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIV_1_0); // 设置tiling key=1对应的kernel类型为MIX AIV 1:0
                op.Process1();
            } else if (TILING_KEY_IS(2)) {
                KERNEL_TASK_TYPE(2, KERNEL_TYPE_AIV_ONLY); // 设置tiling key=2对应的kernel类型为纯AIV类型
                op.Process2();
            }
            // ...
        }
        // 配套的host侧tiling函数示例：
        ge::graphStatus TilingFunc(gert::TilingContext* context)
        {	
            // ...
            if (context->GetInputShape(0) > 10) {
                context->SetTilingKey(1);
            } else if (some condition) {
                context->SetTilingKey(2);
            }
        }
        