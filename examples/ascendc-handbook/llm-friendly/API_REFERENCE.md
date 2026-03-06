# AscendC 算子开发 API 参考 (LLM精简版)

> 本文档为AscendC算子开发API的精简参考，去除冗余信息，适合作为LLM上下文使用。
> 文档版本: CANN Community Edition 900beta1

---

## 1. 基础数据结构

### LocalTensor
AI Core内部存储数据结构，支持逻辑位置: VECIN, VECOUT, VECCALC, A1, A2, B1, B2, CO1, CO2

```cpp
#include "kernel_operator.h"

template <typename T> class LocalTensor : public BaseLocalTensor<T> {
    // 构造函数
    __aicore__ inline LocalTensor<T>() {};
    
    // 常用方法
    __aicore__ inline uint64_t GetPhyAddr() const;
    __aicore__ inline uint32_t GetSize() const;
    __aicore__ inline void SetSize(const uint32_t size);
    __aicore__ inline int32_t GetPosition() const;
    
    // 数据访问
    __aicore__ inline PrimType GetValue(const uint32_t index) const;
    __aicore__ inline void SetValue(const uint32_t index, const T1 value) const;
    __aicore__ inline LocalTensor operator[](const uint32_t offset) const;
    
    // 类型转换
    template <typename CAST_T> __aicore__ inline LocalTensor<CAST_T> ReinterpretCast() const;
};
```

### GlobalTensor
Global Memory全局数据结构

```cpp
template <typename T> class GlobalTensor : public BaseGlobalTensor<T> {
    // 构造函数
    __aicore__ inline GlobalTensor<T>() {}
    
    // 初始化
    __aicore__ inline void SetGlobalBuffer(const GlobalTensor<T>& tensor, uint64_t offset);
    __aicore__ inline void SetGlobalBuffer(T* buffer, uint64_t bufferSize);
    
    // 数据访问
    __aicore__ inline T GetValue(const uint64_t offset) const;
    __aicore__ inline void SetValue(const uint64_t offset, const T value);
    __aicore__ inline uint64_t GetSize() const;
};
```

### TPosition 枚举
逻辑位置定义:
| 值 | 位置 | 说明 |
|---|---|---|
| 0 | GM | Global Memory |
| 1 | A1 | L1 input A |
| 2 | A2 | L0A input A |
| 3 | B1 | L1 input B |
| 4 | B2 | L0B input B |
| 7 | CO1 | L0C output |
| 9 | VECIN | UB vector input |
| 10 | VECOUT | UB vector output |
| 11 | VECCALC | UB vector compute |

---

## 2. 基础API

### 2.1 数据搬运

#### DataCopy
数据搬运接口，支持多种模式

```cpp
// 基础搬运
template <typename T>
__aicore__ inline void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, const DataCopyParams& params);

template <typename T>
__aicore__ inline void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, const DataCopyParams& params);

// 搬运参数
struct DataCopyParams {
    uint32_t blockCount;      // 数据块数量
    uint32_t blockLen;        // 每块数据长度
    uint32_t srcStride;       // 源数据块间隔
    uint32_t dstStride;       // 目的数据块间隔
};

// 常用参数快捷方式
DataCopyExtParams dataCopyParams;
dataCopyParams.blockCount = 1;
dataCopyParams.blockLen = len;
```

#### Copy
VECIN/VECCALC/VECOUT间搬运

```cpp
template <typename T>
__aicore__ inline void Copy(LocalTensor<T> dst, LocalTensor<T> src, const CopyParams& params);
```

### 2.2 矢量计算

#### 基础算术运算
```cpp
// 逐元素运算
template <typename T>
void Exp(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);

template <typename T>
void Ln(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);

template <typename T>
void Abs(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);

template <typename T>
void Sqrt(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);

template <typename T>
void Relu(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);

// 二元运算
template <typename T>
void Add(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, const uint32_t count);

template <typename T>
void Sub(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, const uint32_t count);

template <typename T>
void Mul(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, const uint32_t count);

template <typename T>
void Div(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, const uint32_t count);

template <typename T>
void Max(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, const uint32_t count);

template <typename T>
void Min(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, const uint32_t count);

// 标量运算
template <typename T>
void Adds(LocalTensor<T> dst, LocalTensor<T> src, T scalar, const uint32_t count);

template <typename T>
void Muls(LocalTensor<T> dst, LocalTensor<T> src, T scalar, const uint32_t count);
```

#### 精度转换
```cpp
template <typename dst_T, typename src_T>
void Cast(LocalTensor<dst_T> dst, LocalTensor<src_T> src, const uint32_t count);
// 支持: half, float, int8_t, int16_t, int32_t, bfloat16_t
```

#### 归约计算
```cpp
template <typename T>
void ReduceSum(LocalTensor<T> dst, LocalTensor<T> src, LocalTensor<T> work, const ReduceOpParams& params);

template <typename T>
void ReduceMax(LocalTensor<T> dst, LocalTensor<T> src, LocalTensor<T> work, const ReduceOpParams& params);

template <typename T>
void ReduceMin(LocalTensor<T> dst, LocalTensor<T> src, LocalTensor<T> work, const ReduceOpParams& params);
```

#### 掩码操作
```cpp
void SetMaskCount();
void SetMaskNorm();
void SetVectorMask(int32_t mask);
void ResetMask();
```

### 2.3 资源管理

#### TPipe
全局内存管理框架

```cpp
class TPipe {
    void InitBuffer(TQue& que, uint8_t num, uint32_t len);
    void InitBuffer(TBuf& buf, uint32_t len);
};
```

#### TQue 队列管理
```cpp
template <QuePosition pos, int32_t depth>
class TQue {
    template <typename T>
    LocalTensor<T> AllocTensor();
    
    template <typename T>
    void FreeTensor(LocalTensor<T>& tensor);
    
    void EnQue(LocalTensor<T>& tensor);
    LocalTensor<T> DeQue<T>();
};
```

#### TBuf 临时内存
```cpp
template <TPosition pos>
class TBuf {
    template <typename T>
    LocalTensor<T> Get();
};
```

### 2.4 同步控制

```cpp
// 核间同步
void SyncAll();  // 全核同步

// 核间通知
void WaitPreBlock();
void NotifyNextBlock();

// 栅栏
void PipeBarrier<PIPE_ALL>();
void PipeBarrier<PIPE_MTE1>();
void PipeBarrier<PIPE_MTE2>();
void PipeBarrier<PIPE_MTE3>();
void PipeBarrier<PIPE_V>();
```

### 2.5 系统接口

```cpp
uint32_t GetBlockNum();   // 获取总核数
uint32_t GetBlockIdx();   // 获取当前核ID
uint8_t GetArchVersion(); // 获取架构版本
```

### 2.6 原子操作

```cpp
void SetAtomicAdd();
void SetAtomicMax();
void SetAtomicMin();
void SetAtomicType(AtomicType type);
```

---

## 3. 高阶API

### 3.1 数学计算

```cpp
// 三角函数
void Sin(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);
void Cos(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);
void Tan(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);

// 取整函数
void Floor(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);
void Ceil(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);
void Round(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);
```

### 3.2 归一化

```cpp
// LayerNorm
template <typename T>
void LayerNorm(LocalTensor<T> dst, LocalTensor<T> mean, LocalTensor<T> variance,
               LocalTensor<T> src, LocalTensor<T> gamma, LocalTensor<T> beta,
               const LayerNormParams& params);

// RmsNorm
template <typename T>
void RmsNorm(LocalTensor<T> dst, LocalTensor<T> src, LocalTensor<T> gamma, 
             LocalTensor<T> beta, const RmsNormParams& params);
```

### 3.3 激活函数

```cpp
// SoftMax
template <typename T>
void SoftMax(LocalTensor<T> dst, LocalTensor<T> src, LocalTensor<T> work, 
             const SoftMaxParams& params);

// Gelu
template <typename T>
void Gelu(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);

// Sigmoid
template <typename T>
void Sigmoid(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);
```

### 3.4 矩阵计算

```cpp
// Matmul
template <typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class Matmul {
    void Init(TBuf<> aBuf, TBuf<> bBuf, TBuf<> cBuf);
    void SetTensorA(LocalTensor<A_TYPE> a);
    void SetTensorB(LocalTensor<B_TYPE> b);
    void IterateAll(LocalTensor<C_TYPE> c);
    void End();
};
```

---

## 4. 核心约束与注意事项

### 4.1 内存约束
- LocalTensor必须在支持的TPosition上
- 数据需要按datablock(32字节)对齐
- 注意UB/L0A/L0B/L0C/L1的容量限制

### 4.2 数据类型支持
| API | float | half | int8 | int16 | int32 | bfloat16 |
|-----|-------|------|------|-------|-------|----------|
| Add | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Mul | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Exp | ✓ | ✓ | - | - | - | ✓ |
| Cast | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 4.3 性能优化建议
- 尽量使用高阶API而非组合基础API
- 合理使用双缓冲(Double Buffer)
- 注意内存对齐和连续访问

---

## 5. 代码模板

### 5.1 基础Kernel模板

```cpp
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void my_kernel(
    GM_ADDR x, GM_ADDR y, GM_ADDR z) 
{
    // 1. 初始化Pipe和队列
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQue;
    TQue<QuePosition::VECOUT, 2> outQue;
    
    pipe.InitBuffer(inQue, 2, 1024 * sizeof(float));
    pipe.InitBuffer(outQue, 2, 1024 * sizeof(float));
    
    // 2. 获取全局Tensor
    GlobalTensor<float> gmX, gmY, gmZ;
    gmX.SetGlobalBuffer((__gm__ float*)x, totalLength);
    gmY.SetGlobalBuffer((__gm__ float*)y, totalLength);
    gmZ.SetGlobalBuffer((__gm__ float*)z, totalLength);
    
    // 3. 循环处理
    for (uint32_t i = 0; i < totalLength; i += 1024) {
        // 搬入
        LocalTensor<float> xLocal = inQue.AllocTensor<float>();
        LocalTensor<float> yLocal = inQue.AllocTensor<float>();
        DataCopy(xLocal, gmX[i], 1024);
        DataCopy(yLocal, gmY[i], 1024);
        inQue.EnQue(xLocal);
        inQue.EnQue(yLocal);
        
        // 计算
        xLocal = inQue.DeQue<float>();
        yLocal = inQue.DeQue<float>();
        LocalTensor<float> zLocal = outQue.AllocTensor<float>();
        Add(zLocal, xLocal, yLocal, 1024);
        outQue.EnQue(zLocal);
        
        // 搬出
        zLocal = outQue.DeQue<float>();
        DataCopy(gmZ[i], zLocal, 1024);
        outQue.FreeTensor(zLocal);
        inQue.FreeTensor(xLocal);
        inQue.FreeTensor(yLocal);
    }
}
```

### 5.2 Matmul模板

```cpp
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void matmul_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c) 
{
    TPipe pipe;
    TBuf<QuePosition::A1> aBuf;
    TBuf<QuePosition::B1> bBuf;
    TBuf<QuePosition::CO1> cBuf;
    
    pipe.InitBuffer(aBuf, 1024);
    pipe.InitBuffer(bBuf, 1024);
    pipe.InitBuffer(cBuf, 1024);
    
    Matmul<float, float, float, float> mm;
    mm.Init(aBuf, bBuf, cBuf);
    
    // 设置输入
    LocalTensor<float> aLocal = aBuf.Get<float>();
    LocalTensor<float> bLocal = bBuf.Get<float>();
    
    // 从GM拷贝数据...
    
    mm.SetTensorA(aLocal);
    mm.SetTensorB(bLocal);
    
    // 计算
    LocalTensor<float> cLocal = cBuf.Get<float>();
    mm.IterateAll(cLocal);
    
    // 拷贝结果到GM...
    
    mm.End();
}
```

---

## 6. 文件结构

完整的详细文档位于 `organized/` 目录:
- `01_基础数据结构_基础数据结构.md` - LocalTensor, GlobalTensor等
- `02_基础API/` - 数据搬运、矢量计算、资源管理等
- `03_高阶API/` - 数学计算、归一化、激活函数等
- `04_Utils_API/` - C++标准库、平台信息等

按需查阅具体API的详细说明。
