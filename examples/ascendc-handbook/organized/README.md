# AscendC 算子开发文档

> 来源: 昇腾社区官网
> 文档版本: CANN Community Edition 900beta1

## 文档结构

### [基础数据结构](./01_基础数据结构_基础数据结构.md)

- LocalTensor
- GlobalTensor
- Coordinate
- Layout
- TensorTrait
- TPosition

### [基础API](./02_基础API_基础API.md)

#### [数据搬运](./02_基础API/02_01_数据搬运_数据搬运.md)

- DataCopy
- Copy

#### [矢量计算](./02_基础API/02_02_矢量计算_矢量计算.md)

#### [标量计算](./02_基础API/02_03_标量计算_标量计算.md)

- GetBitCount
- CountLeadingZero
- Cast标量
- CountBitsCntSameAsSignBit
- GetSFFValue

#### [资源管理](./02_基础API/02_04_资源管理_资源管理.md)

- TPipe
- GetTPipePtr
- TBufPool
- TQue
- TQueBind
- TBuf
- SPM_Buffer
- Workspace

#### [同步控制](./02_基础API/02_05_同步控制_同步控制.md)

- TQueSync
- IBSet
- IBWait
- SyncAll
- 核间同步
- 任务同步

#### [缓存处理](./02_基础API/02_06_缓存处理_缓存处理.md)

- DataCachePreload
- DataCacheCleanAndInvalid

#### [系统变量访问](./02_基础API/02_07_系统变量_系统变量访问.md)

- GetBlockNum
- GetBlockIdx
- GetDataBlockSizeInBytes
- GetArchVersion
- InitSocState

#### [原子操作](./02_基础API/02_08_原子操作_原子操作.md)

- SetAtomicAdd
- SetAtomicType
- DisableDmaAtomic

#### [调试接口](./02_基础API/02_09_调试接口_调试接口.md)

- DumpTensor
- printf
- assert
- DumpAccChkPoint
- PrintTimeStamp
- Trap
- CPU调测
- 性能仿真

#### [Kernel Tiling接口](./02_基础API/02_10_Kernel_Tiling_Kernel Tiling接口.md)

- GET_TILING_DATA
- GET_TILING_DATA_WITH_STRUCT
- GET_TILING_DATA_MEMBER
- TILING_KEY_IS
- Tiling注册
- Kernel类型

#### [ISASI接口（硬件相关）](./02_基础API/02_11_ISASI接口_ISASI接口（硬件相关）.md)

- 矢量计算ISASI
- 排序ISASI
- 数据搬运ISASI
- 矩阵计算ISASI
- Conv2D_Gemm
- FixPipe
- LoadData
- 同步控制ISASI
- 缓存ISASI
- 系统变量ISASI
- 原子操作ISASI
- 调试ISASI
- Cube分组

### [高阶API](./03_高阶API_高阶API.md)

#### [数学计算](./03_高阶API/03_01_数学计算_数学计算.md)

- 三角函数
- 指数对数
- 取整函数
- 其他数学

#### [量化操作](./03_高阶API/03_02_量化操作_量化操作.md)

- AscendQuant
- AscendDequant
- AscendAntiQuant

#### [归一化操作](./03_高阶API/03_03_归一化_归一化操作.md)

- LayerNorm
- RmsNorm
- BatchNorm
- DeepNorm
- GroupNorm
- Normalize
- Welford

#### [激活函数](./03_高阶API/03_04_激活函数_激活函数.md)

- SoftMax
- Gelu
- GLU系列
- 其他激活

#### [归约操作](./03_高阶API/03_05_归约操作_归约操作.md)

- Sum
- Mean
- ReduceXorSum
- 高阶归约

#### [排序操作](./03_高阶API/03_06_排序操作_排序操作.md)

- TopK
- Sort

#### [数据过滤](./03_高阶API/03_07_数据过滤_数据过滤.md)

- Select
- DropOut

#### [张量变换](./03_高阶API/03_08_张量变换_张量变换.md)

- Transpose
- TransData
- Broadcast
- Pad
- Fill
- Arange

#### [矩阵计算](./03_高阶API/03_09_矩阵计算_矩阵计算.md)

- Matmul

#### [HCCL通信](./03_高阶API/03_10_HCCL通信_HCCL通信.md)

- HCCL

#### [卷积计算](./03_高阶API/03_11_卷积计算_卷积计算.md)

- Conv3D

### [Utils API（公共辅助函数）](./04_Utils_API_Utils API（公共辅助函数）.md)

#### [C++标准库](./04_Utils_API/04_01_标准库_C++标准库.md)

- 比较函数
- 序列容器
- 类型判断
- 类型修改
- 条件编译

#### [运行时编译](./04_Utils_API/04_02_运行时编译_运行时编译.md)

- aclrtc

#### [平台信息](./04_Utils_API/04_03_平台信息_平台信息.md)

- PlatformAscendC

#### [日志输出](./04_Utils_API/04_04_日志输出_日志输出.md)

- ASC_CPU_LOG

### [语言扩展层 C API](./05_语言扩展层C_API_语言扩展层 C API.md)

- 简介
- 模板参数
- 构造函数
- 随路量化
- Async
- DEVICE_IMPL_OP_OPTILING
- ASCENDC_TPL_SEL_PARAM

### [算子原型注册](./06_算子原型注册_算子原型注册.md)

- 原型注册
- Input
- ParamType
- OpAttrDef
- SetTiling
- OpAICoreConfig
- OpMC2Def
- TilingData

