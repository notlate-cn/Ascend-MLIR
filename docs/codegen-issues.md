# ascir-translate Codegen 问题清单

> 记录 `ascir-translate -mlir-to-ascendc` 生成 C++ 代码时遇到的问题。
> 以 `broadcast-add-reduce` 场景为例，对比手写的 `step8_kernel-adjust.cpp`。

## 构建问题

### B-1: CAPI 链接缺失 Conversion 库

**文件**: `lib/CAPI/Dialect/CMakeLists.txt`

**现象**: 链接 `libAFIRPythonCAPI.so` 时报 undefined reference，涉及 5 个 `create*Pass()` 函数。

**原因**: `AFIRPasses.cpp` 注册了所有 Conversion pass，但 CMakeLists.txt 未链接对应的库。

**修复**: 在 `LINK_LIBS` 中添加：
- `AscendCBufferPlacementConversion`
- `AscendCFoldConcatAllocConversion`
- `AscendCParallelizeConversion`
- `AscendCPrepareForEmitConversion`
- `LinalgToAscendCConversion`

---

## Codegen 问题（ascir-translate）

### C-1: 缺少 `#include` 和 `TilingData` 定义

**现象**: 生成的 `.cpp` 没有任何 `#include`，编译报 `unknown type name '__aicore__'` 等。

**期望**:
```cpp
#include "kernel_operator.h"

struct TilingData {
  int64_t TB_M;
  int64_t TB_N;
  int64_t dim_arg0_0;
  int64_t dim_arg1_1;
  // ...
};
```

**临时方案**: `test_e2e.py` 的 `ensure_kernel_preamble()` 函数在编译前自动注入。

---

### C-2: 函数参数缺少 `__gm__` 修饰符

**生成的代码**:
```cpp
void broadcast_add_reducesum(half* v1, half* v2, __gm__ TilingData* v3, half* v4)
```

**期望**: bisheng 编译器要求所有 kernel 函数的指针参数都带 `__gm__`（或使用 `GM_ADDR`）。
```cpp
void broadcast_add_reducesum(GM_ADDR input_a, GM_ADDR input_b, GM_ADDR output, ...)
```

---

### C-3: 函数签名与 Runtime Executor 约定不匹配

**生成的参数顺序**: `(input_a, input_b, tiling_ptr, output)`

**Executor 传参顺序**: `[inputs..., outputs..., workspace, tiling_fields...]`

即 executor 期望: `(input_a, input_b, output, workspace, tiling)`

**另外**: 生成代码中 tiling 是 `__gm__ TilingData*`（指针），而 executor 将 tiling 字段拆为 int64 追加到 args 末尾，对应 AscendC 的**按值传递**约定：`TilingData tiling`。

---

### C-4: 整数类型使用 `reinterpret_cast`

**生成的代码**:
```cpp
reinterpret_cast<uint64_t>(v42)  // v42 是 int32_t
```

**问题**: `reinterpret_cast` 不能在整数类型间转换，编译报错。

**修复**: 应使用 `static_cast<uint64_t>(v42)`。

---

### C-5: `Broadcast` API 调用错误

**生成的代码**:
```cpp
AscendC::Broadcast<half, half, 2>(v44, v38, uint64_t, uint64_t);
```

**SDK 实际签名**:
```cpp
template <typename T, int32_t dim, int32_t axis, bool isReuseSource = false>
void Broadcast(const LocalTensor<T>& dst, const LocalTensor<T>& src,
               const uint32_t dstShape[dim], const uint32_t srcShape[dim]);
```

**问题**:
1. 模板参数错误：第二个参数应是 `dim`（int），不是类型
2. 函数参数应是 `uint32_t[]` 数组，不是标量
3. 缺少 `axis` 模板参数

**手写版方案**: 逐行用 `Duplicate(local_src, a_val, dim_n)` 替代 Broadcast，更简单。

---

### C-6: `ReduceSum` API 不存在

**生成的代码**:
```cpp
AscendC::ReduceSum<AscendC::ReduceLayout::AR>(v50, v41);
```

**问题**: SDK 中没有 `ReduceLayout::AR`，也没有 2 参数的 `ReduceSum`。

**SDK 实际签名**:
```cpp
template <typename T>
void ReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,
               const LocalTensor<T>& sharedTmpBuffer, const int32_t count);
```

**修复**: 需要提供 tmpBuffer 和元素数量。

---

### C-7: 多余的 `Add` 指令导致结果翻倍

**生成的代码**:
```cpp
AscendC::Add(v41, v44, v45, v39);  // v41 = v44 + v45（正确）
AscendC::Add(v41, v41, v41, v39);  // v41 = v41 + v41（多余，结果翻倍！）
```

**原因**: 可能是 codegen 在降级 reduction 时引入了错误的累加指令。

---

### C-8: Buffer 在循环内反复 InitBuffer

**生成的代码**:
```cpp
for (...) {
    pipe.InitBuffer(tbuf_in, size);   // 每次迭代重新初始化！
    pipe.InitBuffer(tbuf_calc, size);
    // ...
}
```

**问题**: `InitBuffer` 应在循环外调用一次。循环内反复调用导致 CPU 仿真器报 `mem_map_calc_49bit invalid ldst addr` 错误。

**手写版**: 所有 `InitBuffer` 在循环外一次性完成，循环内只做 Alloc/DeQue/Free。

---

## 问题优先级建议

| 优先级 | 编号 | 影响 |
|--------|------|------|
| P0 | C-1 | 生成的代码完全无法编译 |
| P0 | C-2 | 生成的代码完全无法编译 |
| P0 | C-3 | 运行时参数传递错误，结果全为 0 |
| P0 | C-8 | 运行时内存错误，仿真器崩溃 |
| P1 | C-5, C-6 | API 调用不匹配，编译失败 |
| P1 | C-7 | 计算结果错误 |
| P2 | C-4 | 编译报错，但修复简单 |
| P2 | B-1 | 构建问题，已修复 |

## 当前临时方案

`test_e2e.py` 使用手写的 `step8_kernel-adjust.cpp` 跳过 codegen 问题，验证后续 runtime/executor 链路。