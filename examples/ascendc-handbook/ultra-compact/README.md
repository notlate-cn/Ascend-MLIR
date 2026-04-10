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


- DataCopy
- Copy



- GetBitCount
- CountLeadingZero
- Cast标量
- CountBitsCntSameAsSignBit
- GetSFFValue


- TPipe
- GetTPipePtr
- TBufPool
- TQue
- TQueBind
- TBuf
- SPM_Buffer
- Workspace


- TQueSync
- IBSet
- IBWait
- SyncAll
- 核间同步
- 任务同步


- DataCachePreload
- DataCacheCleanAndInvalid


- GetBlockNum
- GetBlockIdx
- GetDataBlockSizeInBytes
- GetArchVersion
- InitSocState


- SetAtomicAdd
- SetAtomicType
- DisableDmaAtomic


- DumpTensor
- printf
- assert
- DumpAccChkPoint
- PrintTimeStamp
- Trap
- CPU调测
- 性能仿真


- GET_TILING_DATA
- GET_TILING_DATA_WITH_STRUCT
- GET_TILING_DATA_MEMBER
- TILING_KEY_IS
- Tiling注册
- Kernel类型


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


- 三角函数
- 指数对数
- 取整函数
- 其他数学


- AscendQuant
- AscendDequant
- AscendAntiQuant


- LayerNorm
- RmsNorm
- BatchNorm
- DeepNorm
- GroupNorm
- Normalize
- Welford


- SoftMax
- Gelu
- GLU系列
- 其他激活


- Sum
- Mean
- ReduceXorSum
- 高阶归约


- TopK
- Sort


- Select
- DropOut


- Transpose
- TransData
- Broadcast
- Pad
- Fill
- Arange


- Matmul


- HCCL


- Conv3D

### [Utils API（公共辅助函数）](./04_Utils_API_Utils API（公共辅助函数）.md)


- 比较函数
- 序列容器
- 类型判断
- 类型修改
- 条件编译


- aclrtc


- PlatformAscendC


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
