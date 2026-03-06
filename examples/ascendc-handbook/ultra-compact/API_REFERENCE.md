# AscendC API 参考 (超精简版)

## 1. 基础数据结构

### LocalTensor
```cpp
template <typename T> class LocalTensor {
    __aicore__ inline LocalTensor<T>() {};
    __aicore__ inline uint64_t GetPhyAddr() const;
    __aicore__ inline uint32_t GetSize() const;
    __aicore__ inline void SetSize(const uint32_t size);
    __aicore__ inline PrimType GetValue(const uint32_t index) const;
    __aicore__ inline void SetValue(const uint32_t index, const T1 value) const;
    template <typename CAST_T> __aicore__ inline LocalTensor<CAST_T> ReinterpretCast() const;
};
```
支持位置: VECIN, VECOUT, VECCALC, A1, A2, B1, B2, CO1, CO2

### GlobalTensor
```cpp
template <typename T> class GlobalTensor {
    __aicore__ inline GlobalTensor<T>() {}
    __aicore__ inline void SetGlobalBuffer(T* buffer, uint64_t bufferSize);
    __aicore__ inline T GetValue(const uint64_t offset) const;
    __aicore__ inline void SetValue(const uint64_t offset, const T value);
};
```

### TPosition
| 值 | 位置 | 说明 |
|---|---|---|
| 0 | GM | Global Memory |
| 1 | A1 | L1 input A |
| 2 | A2 | L0A input A |
| 3 | B1 | L1 input B |
| 4 | B2 | L0B input B |
| 7 | CO1 | L0C output |
| 9 | VECIN | UB input |
| 10 | VECOUT | UB output |
| 11 | VECCALC | UB compute |

## 2. 基础API

### 数据搬运
```cpp
void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, const DataCopyParams& params);
void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, const DataCopyParams& params);
```

### 矢量计算
```cpp
void Exp(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);
void Ln(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);
void Abs(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);
void Sqrt(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);
void Relu(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);
void Add(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, const uint32_t count);
void Sub(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, const uint32_t count);
void Mul(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, const uint32_t count);
void Div(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, const uint32_t count);
void Max(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, const uint32_t count);
void Min(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, const uint32_t count);
void Cast(LocalTensor<dst_T> dst, LocalTensor<src_T> src, const uint32_t count);
```

### 资源管理
```cpp
class TPipe { void InitBuffer(TQue& que, uint8_t num, uint32_t len); };
class TQue { LocalTensor<T> AllocTensor(); void FreeTensor(LocalTensor<T>& t); };
class TBuf { LocalTensor<T> Get(); };
```

### 同步控制
```cpp
void SyncAll();
void PipeBarrier<PIPE_ALL>();
void WaitPreBlock();
void NotifyNextBlock();
```

### 系统接口
```cpp
uint32_t GetBlockNum();
uint32_t GetBlockIdx();
uint8_t GetArchVersion();
```

## 3. 高阶API

### 归一化
```cpp
void LayerNorm(LocalTensor<T> dst, LocalTensor<T> mean, LocalTensor<T> variance,
               LocalTensor<T> src, LocalTensor<T> gamma, LocalTensor<T> beta, const LayerNormParams& params);
void RmsNorm(LocalTensor<T> dst, LocalTensor<T> src, LocalTensor<T> gamma, LocalTensor<T> beta, const RmsNormParams& params);
```

### 激活函数
```cpp
void SoftMax(LocalTensor<T> dst, LocalTensor<T> src, LocalTensor<T> work, const SoftMaxParams& params);
void Gelu(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);
void Sigmoid(LocalTensor<T> dst, LocalTensor<T> src, const uint32_t count);
```

### 矩阵计算
```cpp
class Matmul {
    void Init(TBuf<> aBuf, TBuf<> bBuf, TBuf<> cBuf);
    void SetTensorA(LocalTensor<A_TYPE> a);
    void SetTensorB(LocalTensor<B_TYPE> b);
    void IterateAll(LocalTensor<C_TYPE> c);
    void End();
};
```

## 4. 代码模板

```cpp
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void kernel(GM_ADDR x, GM_ADDR y) {
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> que;
    pipe.InitBuffer(que, 2, 1024 * sizeof(float));
    
    GlobalTensor<float> gmX, gmY;
    gmX.SetGlobalBuffer((__gm__ float*)x, len);
    gmY.SetGlobalBuffer((__gm__ float*)y, len);
    
    for (uint32_t i = 0; i < len; i += 1024) {
        LocalTensor<float> local = que.AllocTensor<float>();
        DataCopy(local, gmX[i], 1024);
        // 计算...
        DataCopy(gmY[i], local, 1024);
        que.FreeTensor(local);
    }
}
```
