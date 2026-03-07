# AscendC API 系统性分析报告

## 一、API级别体系说明

AscendC API采用**L0/L1/L2/L3**四级体系：

| 级别 | 名称 | 描述 | 典型特征 |
|------|------|------|----------|
| **L0** | Level 0 | 最底层API，直接操作硬件 | 需要mask、repeatTimes、stride等详细参数 |
| **L1** | Level 1 | 中间层API，支持切片操作 | 使用SliceInfo结构体描述数据布局 |
| **L2** | Level 2 | 高级API，简化接口 | 只需提供count/dataSize参数 |
| **L3** | Level 3 | 最高层API，自动Tiling | 自动计算tiling参数，最易使用 |

---

## 二、API分类详解

### 1. 一元数学运算

| API名称 | 函数签名 | L0版本 | L2版本 | 支持类型 |
|---------|----------|--------|--------|----------|
| **Abs** | `dst[i] = abs(src[i])` | ✓ | ✓ | half, float, int32_t |
| **Exp** | `dst[i] = exp(src[i])` | ✓ | ✓ | half, float |
| **Ln** | `dst[i] = ln(src[i])` | ✓ | ✓ | half, float |
| **Sqrt** | `dst[i] = sqrt(src[i])` | ✓ | ✓ | half, float |
| **Rsqrt** | `dst[i] = 1/sqrt(src[i])` | ✓ | ✓ | half, float |
| **Reciprocal** | `dst[i] = 1/src[i]` | ✓ | ✓ | half, float |
| **Relu** | `dst[i] = max(0, src[i])` | ✓ | ✓ | half, float |
| **Not** | `dst[i] = ~src[i]` | ✓ | ✓ | 整数类型 |
| **Neg** | `dst[i] = -src[i]` | - | ✓ | half, float |

**扩展API**:

| API名称 | 功能描述 | 说明 |
|---------|----------|------|
| **IsFinite** | 检查元素是否有限 | 检测非NaN/Inf |
| **IsInf** | 检查元素是否为无穷 | 检测Inf |
| **IsNan** | 检查元素是否为NaN | 检测NaN |

**L0函数签名示例**:
```cpp
template <typename T, bool isSetMask = true>
__aicore__ inline void Exp(const LocalTensor<T>& dst, const LocalTensor<T>& src, 
    uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
```

**L2函数签名示例**:
```cpp
template <typename T>
__aicore__ inline void Exp(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t& count);
```

---

### 2. 二元数学运算

| API名称 | 函数签名 | L0版本 | L2版本 | L3版本 | 支持类型 |
|---------|----------|--------|--------|--------|----------|
| **Add** | `dst = src0 + src1` | ✓ | ✓ | ✓ | half, float, int32_t, int16_t, int8_t, uint8_t |
| **Sub** | `dst = src0 - src1` | ✓ | ✓ | ✓ | half, float, int32_t, int16_t, int8_t, uint8_t |
| **Mul** | `dst = src0 * src1` | ✓ | ✓ | ✓ | half, float, int32_t, int16_t, int8_t, uint8_t |
| **Div** | `dst = src0 / src1` | ✓ | ✓ | ✓ | half, float |
| **Max** | `dst = max(src0, src1)` | ✓ | ✓ | - | half, float, int32_t |
| **Min** | `dst = min(src0, src1)` | ✓ | ✓ | - | half, float, int32_t |
| **And** | `dst = src0 & src1` | ✓ | ✓ | - | 整数类型 |
| **Or** | `dst = src0 \| src1` | ✓ | ✓ | - | 整数类型 |
| **MulAddDst** | `dst = src0 * src1 + dst` | ✓ | ✓ | - | half, float |
| **AddRelu** | `dst = relu(src0 + src1)` | ✓ | ✓ | - | half, float |
| **SubRelu** | `dst = relu(src0 - src1)` | ✓ | ✓ | - | half, float |
| **FusedMulAdd** | `dst = src0 * dst + src1` | ✓ | ✓ | - | half, float |
| **FusedMulAddRelu** | `dst = relu(src0 * dst + src1)` | ✓ | ✓ | - | half, float |

**扩展API**:

| API名称 | 功能描述 | 说明 |
|---------|----------|------|
| **Axpy** | `dst = src + alpha * src1` | AXPY运算 |
| **AxpyExtend** | AXPY扩展实现 | 支持更多类型 |
| **BitwiseAnd** | 按位与 | 整数类型 |
| **BitwiseOr** | 按位或 | 整数类型 |
| **BitwiseXor** | 按位异或 | 整数类型 |
| **BitwiseNot** | 按位取反 | 整数类型 |
| **Fmod** | 浮点取模 | half, float |
| **Hypot** | 直角三角形斜边 | half, float |
| **LogicalAnd** | 逻辑与 | 布尔类型 |
| **LogicalOr** | 逻辑或 | 布尔类型 |
| **LogicalXor** | 逻辑异或 | 布尔类型 |
| **Power** | 幂运算 | half, float |
| **FloorDiv** | 整除 | 整数类型 |

**L0函数签名**:
```cpp
template <typename T, bool isSetMask = true>
__aicore__ inline void Add(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
    const LocalTensor<T>& src1, uint64_t mask[], const uint8_t repeatTime,
    const BinaryRepeatParams& repeatParams);
```

**L2函数签名**:
```cpp
template <typename T>
__aicore__ inline void Add(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
    const LocalTensor<T>& src1, const int32_t& count);
```

---

### 3. 激活函数

| API名称 | 函数签名 | L2/L3版本 | 支持类型 | 特点 |
|---------|----------|-----------|----------|------|
| **Gelu** | `dst = x * Φ(x)` | ✓ | half, float | 高精度/高性能模式可选 |
| **Sigmoid** | `dst = 1/(1+e^(-x))` | ✓ | half, float | 需要临时缓冲区 |
| **Silu** | `dst = x * sigmoid(x)` | ✓ | half, float | Swish激活函数 |
| **Swish** | `dst = x * sigmoid(x)` | ✓ | half, float | 同Silu |
| **Softmax** | `dst = exp(x) / sum(exp(x))` | ✓ | half, float | 支持Flash版本 |
| **LogSoftmax** | `dst = log(softmax(x))` | ✓ | half, float | - |
| **GeGLU** | GELU门控线性单元 | ✓ | half, float | GLU变体 |
| **SwiGLU** | Swish门控线性单元 | ✓ | half, float | GLU变体 |
| **ReGLU** | ReLU门控线性单元 | ✓ | half, float | GLU变体 |

**扩展API**:

| API名称 | 功能描述 | 说明 |
|---------|----------|------|
| **GeluExtend** | Gelu扩展实现 | 支持更多配置 |
| **SigmoidExtend** | Sigmoid扩展实现 | 支持更多配置 |
| **ClipByValue** | 数值裁剪 | 限制数值范围 |

**Gelu函数签名**:
```cpp
template <typename T, bool highPrecision = false, bool highPerformance = false>
__aicore__ inline void Gelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, 
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t dataSize);
```

---

### 4. 数据搬运

| API名称 | 功能描述 | L0版本 | L1版本 | L2版本 |
|---------|----------|--------|--------|--------|
| **DataCopy** | 数据拷贝 | ✓ | ✓ | ✓ |
| **DataCopyPad** | 带填充的数据拷贝 | ✓ | - | ✓ |
| **Copy** | 向量数据拷贝 | ✓ | - | ✓ |
| **Brcb** | 广播拷贝 | ✓ | - | - |
| **Gather** | 收集操作 | ✓ | ✓ | ✓ |
| **Scatter** | 散射操作 | ✓ | ✓ | ✓ |
| **GatherMask** | 带掩码的收集 | ✓ | - | - |

**Broadcast多形态**:

| API名称 | 功能描述 |
|---------|----------|
| **Broadcast** | 主入口函数，根据类型分发 |
| **BroadcastFirstDim** | 第一维度广播 (1,B)→(A,B) |
| **BroadcastMiddleDim** | 中间维度广播 (A,1)→(A,B) |
| **BroadcastMiddleDimWithCopy** | 使用Copy的中间维度广播 |
| **BroadcastWithStride** | 带步长的广播（非连续内存） |
| **BroadcastCommon** | 通用广播实现 |
| **BroadcastWithCast** | 带类型转换的广播（uint8/int8） |
| **BroadcastInt64** | int64/uint64特化广播 |
| **BroadcastInt64LastDim** | int64最后一维广播 |

**扩展API**:

| API名称 | 功能描述 | 说明 |
|---------|----------|------|
| **GatherExtend** | Gather扩展实现 | 支持更多配置 |
| **Concat** | 张量拼接 | 沿指定轴拼接 |
| **Split** | 张量分割 | 沿指定轴分割 |
| **RemovePad** | 移除填充 | 移除padding |
| **Interleave** | 交错操作 | 交替合并两个张量 |
| **DeInterleave** | 反交错 | 分离交错数据 |

**Broadcast函数签名**:
```cpp
template <typename T>
inline __aicore__ void Broadcast(const LocalTensor<T>& dst, const LocalTensor<T>& src, 
    const uint32_t src_m, const uint32_t src_k, const uint32_t src_z,
    const uint32_t dst_m, const uint32_t dst_k, const uint32_t dst_z, 
    LocalTensor<uint8_t>& tmp_buf);
```

---

### 5. 三元运算

| API名称 | 函数签名 | L2版本 | 支持类型 | 说明 |
|---------|----------|--------|----------|------|
| **Fma** | `dst = src0 * src1 + src2` | ✓ | half, float | 融合乘加 |
| **Mad** | `dst = src0 * src1 + src2` | ✓ | half, float | 乘加 |
| **Where** | `dst = cond ? src0 : src1` | ✓ | 多种类型 | 条件选择 |
| **Select** | `dst = mask ? src0 : src1` | ✓ | half, float | 按位选择 |

**扩展API**:

| API名称 | 功能描述 | 说明 |
|---------|----------|------|
| **WhereExtend** | Where扩展实现 | 支持多维度 |

**Fma函数签名**:
```cpp
template <const FmaConfig& config = DEFAULT_FMA_CONFIG, typename T>
__aicore__ inline void Fma(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
    const LocalTensor<T>& src1, const LocalTensor<T>& src2, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const uint32_t count);
```

---

### 6. 归约运算

| API名称 | 功能描述 | L0版本 | L2版本 | 支持类型 |
|---------|----------|--------|--------|----------|
| **WholeReduceSum** | 完整求和 | ✓ | ✓ | half, float, int32_t |
| **WholeReduceMax** | 完整最大值 | ✓ | ✓ | half, float |
| **WholeReduceMin** | 完整最小值 | ✓ | ✓ | half, float |
| **BlockReduceSum** | 块内求和 | ✓ | - | half, float |
| **BlockReduceMax** | 块内最大值 | ✓ | - | half, float |
| **BlockReduceMin** | 块内最小值 | ✓ | - | half, float |
| **PairReduceSum** | 相邻元素求和 | ✓ | - | half, float |
| **ReduceSum** | 归约求和 | - | ✓ | half, float |
| **ReduceMax** | 归约最大值 | - | ✓ | half, float |
| **ReduceMin** | 归约最小值 | - | ✓ | half, float |
| **ReduceProd** | 归约乘积 | - | ✓ | half, float |
| **ReduceMean** | 归约均值 | - | ✓ | half, float |

**扩展API**:

| API名称 | 功能描述 | 说明 |
|---------|----------|------|
| **ReduceLast** | 归约运算扩展 | 支持更多配置 |

**WholeReduceSum L0签名**:
```cpp
template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const uint64_t mask[], const int32_t repeatTime, const int32_t dstRepStride, 
    const int32_t srcBlkStride, const int32_t srcRepStride);
```

---

### 7. 比较运算

| API名称 | 功能描述 | L0版本 | L2版本 | 比较模式 |
|---------|----------|--------|--------|----------|
| **Compare** | 张量比较 | ✓ | ✓ | EQ, NE, GT, GE, LT, LE |
| **CompareScalar** | 标量比较 | ✓ | ✓ | EQ, NE, GT, GE, LT, LE |

**扩展API**:

| API名称 | 功能描述 | 说明 |
|---------|----------|------|
| **CompareExtend** | 比较运算扩展 | 支持int64等类型 |
| **CompareScalarExtend** | 标量比较扩展 | 支持int64等类型 |
| **CompareScalarExtendInt32** | int32标量比较扩展 | int32特化优化 |

**比较模式枚举**:
```cpp
enum class CMPMODE {
    EQ,  // 等于
    NE,  // 不等于
    GT,  // 大于
    GE,  // 大于等于
    LT,  // 小于
    LE   // 小于等于
};
```

**Compare函数签名**:
```cpp
template <typename T, typename U>
__aicore__ inline void Compare(const LocalTensor<U>& dst, const LocalTensor<T>& src0,
    const LocalTensor<T>& src1, CMPMODE cmpMode, uint32_t count);
```

---

### 8. 类型转换

| API名称 | 功能描述 | L0版本 | L2版本 | 支持类型 |
|---------|----------|--------|--------|----------|
| **Cast** | 类型转换 | ✓ | ✓ | 多种类型互转 |
| **ShiftLeft** | 左移 | ✓ | ✓ | int16_t, int32_t |
| **ShiftRight** | 右移 | ✓ | ✓ | int16_t, int32_t |

**扩展API**:

| API名称 | 功能描述 | 说明 |
|---------|----------|------|
| **CastExtend** | 扩展类型转换 | 支持更多类型组合 |

**Cast L0签名**:
```cpp
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void Cast(const LocalTensor<T>& dst, const LocalTensor<U>& src, 
    RoundMode roundMode, uint64_t mask, const uint8_t repeatTime, 
    const UnaryRepeatParams& repeatParams);
```

**舍入模式**:
```cpp
enum class RoundMode {
    CAST_NONE,      // 不舍入
    CAST_RINT,      // 就近舍入
    CAST_ROUND,     // 向正无穷舍入
    CAST_TRUNC,     // 向零截断
    CAST_FLOOR,     // 向负无穷舍入
    CAST_CEIL       // 向正无穷舍入
};
```

---

### 9. 位运算

| API名称 | 功能描述 | L0版本 | L2版本 | 支持类型 |
|---------|----------|--------|--------|----------|
| **And** | 按位与 | ✓ | ✓ | 整数类型 |
| **Or** | 按位或 | ✓ | ✓ | 整数类型 |
| **Xor** | 按位异或 | ✓ | ✓ | 整数类型 |
| **Not** | 按位取反 | ✓ | ✓ | 整数类型 |

---

### 10. 标量运算

| API名称 | 函数签名 | L0版本 | 说明 |
|---------|----------|--------|------|
| **Adds** | `dst = src + scalar` | ✓ | 标量加法 |
| **Muls** | `dst = src * scalar` | ✓ | 标量乘法 |
| **Maxs** | `dst = max(src, scalar)` | ✓ | 标量最大值 |
| **Mins** | `dst = min(src, scalar)` | ✓ | 标量最小值 |
| **Duplicate** | `dst = scalar` | ✓ | 标量填充 |

---

## 三、关键参数结构体

### BinaryRepeatParams (二元运算参数)
```cpp
struct BinaryRepeatParams {
    uint8_t dstBlkStride;      // 目标块步幅
    uint8_t src0BlkStride;     // 源0块步幅
    uint8_t src1BlkStride;     // 源1块步幅
    uint8_t dstRepStride;      // 目标重复步幅
    uint8_t src0RepStride;     // 源0重复步幅
    uint8_t src1RepStride;     // 源1重复步幅
    uint8_t blockNumber;       // 块数量
};
```

### UnaryRepeatParams (一元运算参数)
```cpp
struct UnaryRepeatParams {
    uint8_t dstBlkStride;      // 目标块步幅
    uint8_t srcBlkStride;      // 源块步幅
    uint8_t dstRepStride;      // 目标重复步幅
    uint8_t srcRepStride;      // 源重复步幅
};
```

### DataCopyParams (数据拷贝参数)
```cpp
struct DataCopyParams {
    uint16_t blockCount;       // 块数量
    uint32_t blockLen;         // 块长度
    uint32_t srcGap;           // 源间隙
    uint32_t dstGap;           // 目标间隙
};
```

---

## 四、特殊类型支持

### int64_t / uint64_t 类型处理

由于硬件限制，int64_t类型需要特殊处理：

```cpp
// int64比较需要转换为float再处理
template <typename T>
inline __aicore__ void CompareScalarExtend(const LocalTensor<T>& dst, 
    const LocalTensor<int64_t>& src0, const int64_t src1, CMPMODE mode, 
    const uint32_t cal_cnt, LocalTensor<uint8_t>& tmp_buf);
```

### int32_t 特化处理

int32_t比较有专门的优化实现：

```cpp
template <typename OutT>
inline __aicore__ void CompareScalarExtendInt32(const LocalTensor<OutT>& dst, 
    const LocalTensor<int32_t>& src0, const int32_t scalar_src1, 
    CMPMODE mode, const uint32_t cal_cnt, LocalTensor<uint8_t>& tmp_buf);
```

---

## 五、内存位置编码

| TPosition | Value | memref拼写 | 说明 |
|-----------|-------|------------|------|
| GM | 0 | (省略) | 全局内存 |
| A1 | 1 | `1 : i32` | L1输入A |
| A2 | 2 | `2 : i32` | L0A输入A |
| B1 | 3 | `3 : i32` | L1输入B |
| B2 | 4 | `4 : i32` | L0B输入B |
| CO1 | 7 | `7 : i32` | L0C输出 |
| VECIN | 9 | `9 : i32` | UB向量输入 |
| VECOUT | 10 | `10 : i32` | UB向量输出 |
| VECCALC | 11 | `11 : i32` | UB向量计算 |

---

## 六、API使用建议

1. **优先使用L2/L3级别API**：更简单、更安全
2. **性能关键场景使用L0级别API**：需要精确控制硬件行为
3. **注意临时缓冲区管理**：许多高级API需要临时缓冲区
4. **类型转换注意舍入模式**：选择合适的RoundMode
5. **int64_t类型需要特殊处理**：使用Extend版本的API

---

## 七、文件路径汇总

### 目录1: `/Volumes/GM9/code/ge/compiler/graph/optimize/autofuse/ascendc/api`
包含37个头文件，主要是扩展API实现：
- gelu.h
- sigmoid.h
- broadcast.h
- compare.h
- where.h
- reduce.h
- cast.h
- logical.h
- duplicate.h
- datacopy.h
- gather.h
- axpy.h

### 目录2: `/Volumes/GM9/code/ge/compiler/graph/optimize/autofuse/v35/ascendc/api_regbase`
包含21个头文件，v35版本的寄存器基址API：
- broadcast.h
- cast.h
- compare.h
- where.h

### 目录3: `/Volumes/GM9/code/ge/compiler/graph/optimize/autofuse/v35/ascendc/api_cube`
包含矩阵乘法相关API：
- matmul.h
- batch_matmul.h

### 目录4: `/Volumes/GM9/cann/Ascend/20251209_newest/ascend-toolkit/8.5.0/aarch64-linux/asc/include`
CANN官方头文件，包含多个子目录：
- **basic_api/**: 基础API（向量运算、数据搬运、矩阵运算等）
- **adv_api/**: 高级API（激活函数、归一化、量化等）
- **micro_api/**: 微操作API
- **simt_api/**: SIMT编程模型API

---

## 八、PyAsc Op定义统计

### Basic目录 Op定义

| 文件 | Op数量 | 说明 |
|------|--------|------|
| OpVecUnary.td | 9 | Abs, Exp, Ln, Sqrt, Rsqrt, Reciprocal, Relu, Not, Neg |
| OpVecBinary.td | 17 | Add, Sub, Mul, Div, Max, Min, And, Or, MulAddDst, AddRelu, SubRelu, FusedMulAdd, FusedMulAddRelu, AddDeqRelu, FusedAbsSub, FusedExpSub, Prelu |
| OpVecBinaryScalar.td | 7 | Adds, Muls, Maxs, Mins, LeakyRelu, ShiftLeft, ShiftRight |
| OpVecCmpsel.td | 14 | Compare(L0/L1/L2), CompareScalar(L0/L1/L2), Select(L0/L1/L2), SelectScalar(L0/L1/L2), GetCmpMask, SetCmpMask, SelectScalarReg, SelectReg |
| OpVecReduce.td | 16 | BlockReduceSum/Max/Min(L0/L1), PairReduceSum(L0/L1), RepeatReduceSum, WholeReduceMax/Min/Sum(L0/L1), ReduceMax/Min/Sum(L0/L1/L2) |
| OpVecReduceND.td | - | 多维归约运算 |
| OpVecDuplicate.td | 3 | Duplicate(L0/L1/L2) |
| OpVecBrcb.td | 1 | BrcbL0 |
| OpVecGather.td | 4 | GatherbL0, Gather(L0/L1/L2) |
| OpVecScatter.td | 3 | Scatter(L0/L1/L2) |
| OpVecGatherMask.td | - | 带掩码的Gather |
| OpVecTranspose.td | - | 转置操作 |
| OpVecBilinearInterpolation.td | - | 双线性插值 |
| OpVecCreatevecindex.td | - | 创建向量索引 |
| OpVecMulCast.td | - | 乘法类型转换 |
| OpVecTensor.td | - | 张量操作 |
| OpVecTernaryScalar.td | - | 三元标量运算 |
| OpVecVconv.td | - | 向量转换 |
| OpVecVpadding.td | - | 向量填充 |
| **Basic扩展文件** | | |
| OpVecUnaryExt.td | 3 | IsFinite, IsInf, IsNan |
| OpVecBinaryExt.td | 11 | BitwiseAnd, BitwiseOr, BitwiseXor, Fmod, Hypot, LogicalAnd, LogicalOr, LogicalXor, FloorDiv, BitwiseNot, AxpyExtend |
| OpVecActivationExt.td | 11 | Gelu, Sigmoid, Silu, Swish, LogSoftmax, GeGLU, SwiGLU, ReGLU, ClipByValue, GeluExtend, SigmoidExtend |
| OpVecCastExt.td | 3 | CastExtend, ShiftLeft, ShiftRight |
| OpVecCompareExt.td | 3 | CompareExtend, CompareScalarExtend, CompareScalarExtendInt32 |
| OpVecDataMoveExt.td | 17 | Broadcast多形态, Gather, Scatter, Concat, Split, GatherExtend等 |
| OpVecReduceExt.td | 3 | ReduceProd, ReduceMean, ReduceLast |
| OpVecScalarExt.td | 1 | DuplicateL2WithTmpBuf |
| OpVecTernaryExt.td | 2 | WhereTernary, WhereExtend |

### Adv目录 Op定义

| 文件 | Op数量 | 说明 |
|------|--------|------|
| Adv/Math.td | 28 | Acos, Acosh, Asin, Asinh, Atan, Atanh, Ceil, Cos, Cosh, Digamma, Erf, Erfc, Floor, Frac, Lgamma, Log, Round, Sign, Sin, Sinh, Tan, Tanh, Trunc, Power, Xor, Axpy, ClampMax, ClampMin, CumSum, Exp |
| Adv/Activation.td | 2 | SimpleSoftMax, SoftMax |
| Adv/Normalization.td | 1 | RmsNorm |
| Adv/Quantization.td | 1 | Quant |
| Adv/Sort.td | 2 | Concat, Extract |
| Adv/Matmul.td | - | 矩阵乘法相关 |

### 统计汇总

| 类别 | Op数量 |
|------|--------|
| Basic目录原有 | ~77个 |
| Basic目录扩展 | ~54个 |
| Adv目录 | ~34个 |
| **总计** | **~165个** |

---

## 九、总结

本次分析系统性地遍历了4个目录中的AscendC API头文件，整理了：

1. **14个主要API类别**：涵盖一元/二元运算、激活函数、数据搬运、归约、比较、类型转换等
2. **L0/L1/L2/L3四级API体系**：从底层硬件操作到高级封装
3. **关键参数结构体**：BinaryRepeatParams、UnaryRepeatParams、DataCopyParams等
4. **扩展API**：支持int64_t等特殊类型的高级封装，已归类到对应的API分类中
5. **内存位置编码**：GM/A1/A2/B1/B2/CO1/VECIN/VECOUT/VECCALC

所有API均支持模板化设计，支持half、float、int32_t等多种数据类型，并通过不同级别的接口满足从性能优化到易用性的不同需求。
