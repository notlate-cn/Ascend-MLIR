# AscendC 仿真框架 + AutoTuner 设计文档

**日期**：2026-03-18
**状态**：待实现
**范围**：SimValidator（独立仿真工具）+ AutoTunerPass（编译期 tiling 求解与代码生成）

---

## 1. 背景与目标

### 现状

当前每个 example（`examples/broadcast-add-reduce/` 等）有独立的 `test_e2e.py`，通过 Python runtime（`python/runtime/`）完成：bisheng 编译 → CPU 仿真执行 → numpy 对比验证。Python 原型已跑通，但：

- 每个新 example 需重写 runtime 样板代码
- 无法集成进 MLIR Pass 编译流水线（Python 无法作为 Pass 调用）
- 缺少编译期自动生成 TilingData 的能力

### 目标

1. **SimValidator**：C++ 独立仿真工具，标准化 compile + execute + verify 流程，各 example 直接接入
2. **AutoTunerPass**：MLIR Pass，编译期解析 step7_kernel.mlir，为每个 impl_graph（模板）求解最优 tiling 参数表达式，生成 Host 侧 C++ tiling func

### 不在范围内

- 模板生成（轴切分）：由上游 Transform Pass 负责，AutoTuner 不修改计算图
- 运行期在线 AutoTuning：预留接口，暂不实现
- 实际 NPU 执行：SimValidator 预留 `RealDevice` 模式接口，暂不实现

---

## 2. 整体架构

```
编译期（afir-opt）：

  [上游 Transform]
  原图 MLIR → 多个 impl_graph（不同轴切分模板）
    impl_graph_0 → step7_kernel_0.mlir   ← AutoTuner 输入层
    impl_graph_1 → step7_kernel_1.mlir
    ...

  [AutoTunerPass]（在 step7 之后，step8 codegen 之前插入）
  每个 impl_graph 独立处理：
    MlirAnalyzer  → 提取符号轴、op 结构、搬运量表达式（AffineExpr）
    HardwareProfile → SoC 硬件参数（UB size、AIV num、带宽）
    TilingSpace   → 该模板的搜索空间描述（tiling_space.json）
    Solver        → 约束剪枝 + 枚举，输出 tiling 参数（AffineExpr + 常量）
    TilingFuncEmitter → 生成 Host C++ tiling_func_i.cpp

  跨模板汇总：
    生成 get_tiling.cpp（运行期选模板 + 计算 TilingData + block_dim）

编译期产物：
  step8_kernel_i.cpp   device 侧（现有流水线不变）
  tiling_func_i.cpp    host 侧（per-impl_graph，AutoTuner 新增）
  get_tiling.cpp       host 侧（模板选择入口）

运行期（Host）：
  GetTiling(M, N, ..., selected_kernel_id*, block_dim*, tiling*)
    → 选最优模板 → 填充 TilingData + block_dim
        │
        ▼ rtKernelLaunch(kernel_{i}.bin, block_dim, tiling)
  Kernel 执行（Device）

独立工具（SimValidator）：
  step8_kernel.cpp + tiling（手动或由 tiling_func 生成）
        │ Compiler（bisheng）+ Executor（libruntime_camodel）
        ▼
  验证输出正确性（与参考实现对比）
```

---

## 3. 目录结构

```
include/Runtime/
    Compiler.h
    Executor.h            BackendMode::Simulation | RealDevice（预留）
    SimValidator.h

lib/Runtime/
    Compiler.cpp
    Executor.cpp
    SimValidator.cpp
    CMakeLists.txt

tools/sim-validator/
    sim_validator_main.cpp   独立命令行工具

include/AutoTuner/
    MlirAnalyzer.h
    HardwareProfile.h
    TilingSpace.h
    Solver.h              SolverBase（可插拔）
    TilingFuncEmitter.h
    AutoTunerPass.h

lib/AutoTuner/
    MlirAnalyzer.cpp
    HardwareProfile.cpp
    TilingSpace.cpp
    Solver.cpp            前期：约束剪枝 + 枚举
    TilingFuncEmitter.cpp
    AutoTunerPass.cpp
    CMakeLists.txt

hardware/
    Ascend910B1.json      SoC 硬件参数
    Ascend910B2.json

examples/*/
    tiling_space.json     每个 example 的搜索空间描述

python/runtime/           保持不动，不做任何修改
```

---

## 4. SimValidator 设计

对应 `python/runtime/` 的 C++ 重写，作为独立工具，不参与编译流水线。

### 4.1 类型约定

整个 Runtime 模块使用 `llvm::Expected<T>` 作为错误返回类型（与 MLIR/LLVM 约定一致），`Status` 用 `llvm::Error`。

### 4.2 Compiler

```cpp
// include/Runtime/Compiler.h
class Compiler {
public:
  struct Config {
    std::string soc_version = "Ascend910B1";
    std::string arch        = "dav-c220-vec";  // dav-c220-cube 用于 cube kernel
    int         opt_level   = 3;
    bool        verbose     = false;
  };

  explicit Compiler(const Config& config = {});

  // src_file → .o → .bin ELF，返回 binary 路径
  llvm::Expected<std::string> Compile(const std::string& src_file,
                                      const std::string& output_dir,
                                      const std::string& kernel_name);
};
```

内部流程与 `python/runtime/compiler.py` 一致：
1. 查找 `${ASCEND_HOME_PATH}/compiler/ccec_compiler/bin/bisheng`
2. 调用 `bisheng -c -x cce` 生成 `.o`
3. 调用 `ld.lld -m aicorelinux` 链接生成 `.bin`
4. 环境变量（`ASCEND_HOME_PATH`、`SOC_VERSION` 等）在初始化时设置

`arch` 参数透传给 `--cce-aicore-arch`，支持 vec/cube 两种 kernel。

### 4.3 Executor

```cpp
// include/Runtime/Executor.h
enum class BackendMode {
  Simulation,   // libruntime_camodel.so（CPU 仿真）
  RealDevice,   // 实际 NPU（预留，暂不实现）
};

struct NDArray {
  void*                 data;
  std::vector<int64_t>  shape;
  DType                 dtype;   // F16 / F32 / INT32 等
  size_t                nbytes() const;
};

struct RunArgs {
  std::vector<NDArray>  inputs;
  std::vector<NDArray>  outputs;       // 预分配，执行后填充
  std::vector<uint8_t>  tiling;        // pack 好的 TilingData bytes
  int                   block_dim = 1;
  size_t                workspace_size = 8192;  // 默认 8KB workspace
};

class Executor {
public:
  explicit Executor(BackendMode mode = BackendMode::Simulation);
  llvm::Error Initialize(int device_id = 0);

  llvm::Error Run(const std::string& binary_path,
                  const std::string& function_name,
                  RunArgs& args);

private:
  // Simulation：封装 rtMalloc/rtMemcpy/rtKernelLaunch
  // RealDevice：预留，接口相同，内部行为待实现
  BackendMode mode_;
};
```

内部行为与 `python/runtime/executor.py` 一致：
- `rtMalloc` 分配 `size + 512`，对齐到 512 字节边界，保存原始地址用于 `rtFree`
- `memcpy_h2d`：256 字节分块（已在 Python 原型中验证安全）
- `memcpy_d2h`：4 字节逐元素（安全依赖于 `rtMalloc` 的 `+512` 超额分配，确保读取越界 4 字节时不访问非法地址）
- `rtSetDevice` 返回值忽略（仿真模式下正常，RealDevice 模式需严格检查）
- args 布局：`[input_addrs..., output_addrs..., workspace_addr, tiling_words...]`

### 4.4 SimValidator

```cpp
// include/Runtime/SimValidator.h
class SimValidator {
public:
  struct Result {
    bool        passed;
    double      max_abs_diff;
    double      mean_abs_diff;
    std::string error_msg;
  };

  // compile + execute + compare 一体
  Result Validate(
      const std::string&          kernel_src,
      const std::string&          kernel_name,
      RunArgs&                    args,         // 含输入数据、tiling、block_dim
      const std::vector<NDArray>& expected,     // 参考输出
      double                      atol = 1.0,
      double                      rtol = 1e-2);
};
```

### 4.5 命令行工具

`sim-validator` 接受两种 tiling 输入方式：

```
# 方式 A：直接指定 tiling 参数（工具内部 pack 为 bytes）
sim-validator \
  --kernel    examples/broadcast-add-reduce/step8_kernel.cpp \
  --name      broadcast_add_reducesum \
  --tiling-params "TB_M=16,TB_N=32,dim_arg0_0=32,dim_arg1_1=32,dim_arg0_1=32,dim_arg1_0=32" \
  --tiling-layout int64,int64,int64,int64,int64,int64 \
  --inputs    input_a.npy,input_b.npy \
  --expected  expected_output.npy \
  --block-dim 2

# 方式 B：使用预打包的 tiling binary
sim-validator \
  --kernel    examples/broadcast-add-reduce/step8_kernel.cpp \
  --name      broadcast_add_reducesum \
  --tiling    tiling.bin \
  --inputs    input_a.npy,input_b.npy \
  --expected  expected_output.npy \
  --block-dim 2
```

方式 A 消除了对预生成 `tiling.bin` 文件的依赖，使 SimValidator 在 AutoTuner 未完成时也可独立运行。

---

## 5. AutoTuner 设计

### 5.1 职责边界

- **输入**：一组 impl_graph 的 **step7_kernel.mlir**（含 `emitasc.py_struct` TilingData 类型、`ascendc.*` compute ops、`ascendc.get_block_idx` 多核调度）
- **输出**：per-graph 的 `tiling_func_i.cpp` + 跨 graph 的 `get_tiling.cpp`
- **不做**：轴切分、计算图修改、运行期搜索（预留接口）

选择 step7 的原因：step7 的 `emitasc.py_struct` 类型直接携带完整的 TilingData struct 字段名列表（如 `["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]`），字段语义清晰（TilingParam vs ShapeDim）；func 参数仍保留 `memref<?xf16>` 动态维度，shape 符号可直接提取；`ascendc.data_copy_l2` 的 count 操作数可直接用于搬运量估算；`emitasc.member` 调用序列揭示了 tiling 参数的使用方式。

### 5.2 MlirAnalyzer

从 step7 IR 提取符号信息。step7 的 `emitasc.py_struct` 类型直接携带完整字段名列表，是提取 TilingData schema 的权威来源。

```cpp
// include/AutoTuner/MlirAnalyzer.h

// TilingData struct 的字段描述
// 直接从 step7 IR 的 emitasc.py_struct 类型提取
struct TilingField {
  std::string name;         // 如 "TB_M"、"dim_arg0_0"
  enum class Role {
    TilingParam,   // 切分参数（TB_M、TB_N 等），值由 Solver 求解
    ShapeDim,      // 原始 shape 维度（dim_arg0_0=M 等），值直接透传
  } role;
  int32_t field_index;      // 在 struct 中的顺序位置（0-based）
  // ShapeDim 字段额外记录：对应 func 参数的第几个 arg、第几个维度
  int32_t arg_index  = -1;  // func arg 索引
  int32_t dim_index  = -1;  // memref 维度索引
};

struct KernelAnalysis {
  std::string kernel_name;
  mlir::MLIRContext* ctx;

  // TilingData struct 字段列表（按 struct 字段顺序排列）
  // 从 emitasc.py_struct<"TilingData", [types...], [names...]> 提取
  // 示例（broadcast-add-reduce）：
  //   [TB_M(TilingParam,0), TB_N(TilingParam,1),
  //    dim_arg0_0(ShapeDim,2,arg=0,dim=0), dim_arg1_1(ShapeDim,3,arg=1,dim=1)]
  std::vector<TilingField> tiling_fields;

  // 符号 shape 维度（AffineDimExpr），从 func 参数的 dynamic memref dim 提取
  // 如 %arg0: memref<?xf16> → shape_dims[0] = d0（对应 M）
  //    %arg1: memref<?x?xf16> → shape_dims[1]=d1（M），shape_dims[2]=d2（N）
  std::vector<mlir::AffineExpr> shape_dims;

  // 搬运量表达式（per core per iteration）
  // 变量：shape_dims（AffineDimExpr）+ tiling params（AffineSymbolExpr，用名称索引）
  // 前期：只建模 MTE2/MTE3（memory bound），VEC 暂设为 AffineConstantExpr(0)
  //
  // count 追踪近似策略：
  //   step7 IR 中 data_copy_l2 的 count 操作数通常是 arith.minsi（tail 处理），
  //   如 %inner_size = arith.minsi(%remaining, %tb_n)。
  //   分析器取 minsi 的第二操作数（tile size，即 %tb_n → sym_TB_M 等）作为
  //   搬运量上界估算，等价于假设每次循环迭代均满载（保守上界，实际搬运量 ≤ 估算值）。
  struct PipeAccess {
    mlir::AffineExpr gm_to_ub;   // MTE2，如 sym_TB_M * d_N * 2（bytes）
    mlir::AffineExpr ub_to_gm;   // MTE3，如 sym_TB_M * 2（bytes）
    mlir::AffineExpr vec_flops;  // VEC，前期为 AffineConstantExpr(0)
  } pipe_access;

  // block_dim 表达式：ceildiv(M, TB_M)
  // 从 dim_arg0_0（M）/ TB_M 的语义关系推导
  // 其中 TB_M 是 AffineSymbolExpr，M 是 AffineDimExpr
  mlir::AffineExpr block_dim_expr;

  // op 类型列表（用于后期接 att api_perf 注册表）
  std::vector<std::string> op_types;
};

class MlirAnalyzer {
public:
  static llvm::Expected<KernelAnalysis> Analyze(mlir::ModuleOp module);

private:
  // 提取策略（step7 IR 特征）：
  //
  // 1. tiling_fields：找 func 参数中 emitasc.py_struct<"TilingData",...> 类型的参数，
  //    直接读取字段名列表和类型列表。
  //    区分 TilingParam vs ShapeDim：
  //      - 字段名以 "TB_" 开头 → TilingParam
  //      - 字段名以 "dim_arg" 开头 → ShapeDim（解析 arg_index 和 dim_index）
  //
  // 2. shape_dims：遍历 func 参数，对每个 memref<?...xT> 参数，
  //    将 dynamic dim（?）映射为 AffineDimExpr
  //
  // 3. pipe_access：遍历 func body 中的 ascendc.data_copy_l2：
  //    - src 为 GlobalTensor（来自 GM）→ count 操作数累加到 gm_to_ub
  //    - dst 为 GlobalTensor（写到 GM）→ count 操作数累加到 ub_to_gm
  //    count 操作数通常是 arith.muli 链，向上追踪直到 emitasc.member 调用
  //    （即 TilingData 字段），将其表示为 AffineExpr
  //
  // 4. block_dim_expr：从 TilingData 字段中找到多核轴参数（TB_M）和对应
  //    shape dim（dim_arg0_0=M），构造 ceildiv(d_M, sym_TB_M)
  //
  // 5. op_types：遍历 func body 收集 ascendc.* op 名称
};
```

**提取规则说明**（step7 IR 对应关系）：

| step7 IR 特征 | 提取内容 |
|---|---|
| `emitasc.py_struct<"TilingData", [...], ["TB_M",...]>` | tiling_fields 完整列表和字段顺序 |
| 字段名 `"TB_M"` / `"TB_N"` | TilingParam |
| 字段名 `"dim_arg0_0"` | ShapeDim，arg=0, dim=0 |
| func 参数 `%arg0: memref<?xf16>` | shape_dims：AffineDimExpr d0 → M |
| `ascendc.data_copy_l2 %local, %global, %count`（src=GM） | gm_to_ub += count 的 AffineExpr |
| `ascendc.data_copy_l2 %global, %local, %count`（dst=GM） | ub_to_gm += count 的 AffineExpr |
| `ascendc.add_l2`、`ascendc.reduce_sum_2d_l2` 等 | op_types 追加 |

### 5.3 HardwareProfile

SoC 硬件参数，从 JSON 加载。**`aiv_num` 等参数在 Solver 中作为编译期已知常量使用**（代入具体值后生成带常量的 AffineExpr，如 `floorDiv(M, 20)`），不作为 AffineExpr 的符号变量。

```cpp
// include/AutoTuner/HardwareProfile.h
struct HardwareProfile {
  std::string soc_name;
  int64_t     ub_size;          // bytes，如 200 * 1024
  int32_t     aiv_num;          // AIV core 数，如 20（编译期常量）
  double      mte2_bandwidth;   // GB/s，GM→UB
  double      mte3_bandwidth;   // GB/s，UB→GM
  // 扩展预留：per-op 吞吐（后期从 att api_perf_register 提取，手动维护）
  // std::map<std::string, double> op_throughput;

  // soc_name → hardware/{soc_name}.json
  // JSON 路径相对于编译器可执行文件所在目录（可通过 --autotuner-hw-dir 覆盖）
  static llvm::Expected<HardwareProfile> Load(const std::string& soc_name,
                                              const std::string& hw_dir = "");
};
```

`hardware/Ascend910B1.json` 示例：
```json
{
  "soc_name":       "Ascend910B1",
  "ub_size":        204800,
  "aiv_num":        20,
  "mte2_bandwidth": 800.0,
  "mte3_bandwidth": 800.0
}
```

### 5.4 TilingSpace

每个 example 提供 `tiling_space.json`，描述该 kernel 模板的搜索空间和约束。

约束分两类处理：

- **线性约束**（如 `TB_M <= M`、`TB_N <= N`）：使用 MLIR `mlir::presburger::IntegerRelation` 表示，支持快速可行性检查和区间推导。
- **非线性约束**（如 `TB_M * TB_N * 2 <= ub_size / 2`，含参数乘积）：`IntegerRelation` 不支持非线性项，此类约束在枚举时逐点代入数值后作为简单整数表达式直接求值（不构建符号系统）。

JSON 约束字符串由自定义解析器分类：若表达式含两变量之积则归入"逐点求值"路径；否则归入 `IntegerRelation` 路径。支持的语法：线性/非线性算术表达式 + `<=` / `>=` / `==`，变量绑定为 TilingParam 名或硬件常量（`ub_size`、`aiv_num`）或 shape 维度（`M`、`N`）。

```cpp
// include/AutoTuner/TilingSpace.h
struct TilingParam {
  std::string name;
  int64_t     min, max, step;
  int64_t     alignment;   // 候选值必须是 alignment 的倍数
};

struct TilingSpace {
  std::vector<TilingParam>  params;

  // 解析后的约束（两路径）：
  // linear_constraints：纯线性约束（IntegerRelation，快速剪枝）
  // nonlinear_exprs：含参数乘积的约束字符串，枚举时逐点代入数值求值
  // 变量顺序（linear）：[tiling params..., shape dims..., hw constants...]
  mlir::presburger::IntegerRelation linear_constraints;
  std::vector<std::string>          nonlinear_exprs;   // 原始字符串，枚举时逐点求值

  // 查找规则（优先级从高到低）：
  //   1. {func_name}_tiling_space.json（per-func 精确匹配，多模板场景）
  //   2. tiling_space.json（目录级默认，单 func 场景零配置）
  //   3. --autotuner-space 命令行显式指定（覆盖自动查找）
  // search_dir：step7 mlir 文件所在目录
  // hw：用于将 ub_size/aiv_num 等硬件常量代入约束（作为编译期常量，非符号）
  static llvm::Expected<TilingSpace> LoadFromJson(
      const std::string&     func_name,
      const std::string&     search_dir,
      const HardwareProfile& hw,
      mlir::MLIRContext*     ctx,
      const std::string&     explicit_path = "");
};
```

`examples/broadcast-add-reduce/tiling_space.json` 示例：
```json
{
  "params": [
    {"name": "TB_M", "min": 16, "max": 128, "step": 16, "alignment": 16},
    {"name": "TB_N", "min": 32, "max": 128, "step": 32, "alignment": 32}
  ],
  "constraints": [
    "TB_M * TB_N * 2 <= ub_size / 2",
    "TB_N <= N"
  ]
}
```

### 5.5 Solver

可插拔接口，前期实现 `EnumerateSolver`，后期替换为 att C++ solver。

```cpp
// include/AutoTuner/Solver.h

// 单个 impl_graph 的求解结果
struct TilingExprResult {
  // TilingParam → 其表达式（关于 shape_dims 的 AffineExpr）
  // 前期（EnumerateSolver）：常量 AffineExpr（如 AffineConstantExpr(64)）
  // 后期（AttSolver）：线性 AffineExpr（如 floorDiv(d0, 20) * 16）
  // 注意：硬件常量（aiv_num=20）在生成时代入为编译期常量，不作为符号
  std::map<std::string, mlir::AffineExpr> param_exprs;

  // 性能估算表达式（用于运行期模板选择打分）
  // 前期：perf = gm_to_ub_numerator(shape 符号) / mte2_bandwidth
  // numerator 为 AffineExpr（整数算术），bandwidth 为浮点常量（GB/s）；
  // 分离存储避免 AffineExpr 不支持浮点除法的问题，
  // TilingFuncEmitter 生成：static_cast<double>(numerator_expr) / bandwidth
  struct PerfExpr {
    mlir::AffineExpr numerator;   // 整数搬运量（bytes），含 shape 符号 AffineDimExpr
    double           bandwidth_gbs;  // 对应硬件带宽，如 800.0（GB/s）
  } perf_expr;

  // block_dim 表达式（代入 param_exprs 后的结果，用于 get_tiling.cpp 生成）
  mlir::AffineExpr block_dim_expr;
};

class SolverBase {
public:
  virtual ~SolverBase() = default;
  virtual llvm::Expected<TilingExprResult>
  Solve(const KernelAnalysis&  analysis,
        const TilingSpace&     space,
        const HardwareProfile& hw) = 0;
};

// 前期实现：枚举 + 剪枝
class EnumerateSolver : public SolverBase {
  // 流程：
  // 1. 枚举 TilingSpace 中所有 (TB_M, TB_N, ...) 参数组合
  // 2. 约束剪枝（双路径）：
  //    a. TilingSpace::linear_constraints 快速线性可行性检查
  //    b. TilingSpace::nonlinear_exprs 逐点代入候选值求值检查
  // 3. 对每个通过约束的候选：
  //    a. 将 TB_M 等常量代入 analysis.pipe_access.gm_to_ub
  //    b. 计算 perf_estimate = gm_to_ub_bytes / hw.mte2_bandwidth
  // 4. 选 perf_estimate 最小的候选
  // 5. 将选出的常量值包装为 AffineConstantExpr，填入 TilingExprResult
  //
  // 注意：前期 param_exprs 为常量，perf_expr 仍保留 shape 符号
  //       （如 perf = 64 * N * 2 / bandwidth），使运行期 ScoreTemplate 仍有意义
public:
  llvm::Expected<TilingExprResult>
  Solve(const KernelAnalysis&  analysis,
        const TilingSpace&     space,
        const HardwareProfile& hw) override;
};

// 预留：后期接 att solver
// class AttSolver : public SolverBase { ... };
```

### 5.6 TilingFuncEmitter

将 `TilingExprResult` 翻译为 Host C++ 代码。

```cpp
// include/AutoTuner/TilingFuncEmitter.h
class TilingFuncEmitter {
public:
  // 为单个 impl_graph 生成 tiling func
  // 输出：tiling_func_{graph_id}.cpp
  // 使用 KernelAnalysis::tiling_fields 生成正确的 struct 字段赋值
  llvm::Error EmitSingleGraphTilingFunc(
      const TilingExprResult& result,
      const KernelAnalysis&   analysis,
      int                     graph_id,
      const std::string&      output_dir);

  // 为多个 impl_graph 生成模板选择入口
  // 输出：get_tiling.cpp
  // 单 impl_graph 退化情况：直接转发，不生成 score 比较逻辑
  // ScoreTemplate 在单模板时标记为 [[maybe_unused]] 避免编译警告
  llvm::Error EmitGetTilingEntry(
      const std::vector<TilingExprResult>& results,
      const std::vector<KernelAnalysis>&   analyses,
      const std::string&                   output_dir);

  // 预留：运行期在线 tuning 入口（暂 emit 空 stub）
  llvm::Error EmitOnlineTuningStub(
      const KernelAnalysis& analysis,
      const std::string&    output_dir);
};
```

**生成的 `tiling_func_0.cpp` 结构示例**（单模板，`broadcast-add-reduce`）：

```cpp
// 由 AutoTuner 自动生成，勿手动修改
#include "tiling_func_0.h"

void GetTiling_broadcast_add_reducesum_0(
    int64_t M, int64_t N,
    int64_t* block_dim_out,
    TilingData* tiling) {
  // Solver 求解的 tiling 参数（前期为常量，后期为 shape 函数）
  const int64_t TB_M = 64;
  const int64_t TB_N = 32;
  // block_dim = ceil(M / TB_M)
  *block_dim_out = (M + TB_M - 1) / TB_M;
  // TilingData 字段赋值（字段顺序来自 KernelAnalysis::tiling_fields，与 emitasc.py_struct 一致）
  // broadcast-add-reduce step7 有 4 个字段：TB_M, TB_N, dim_arg0_0, dim_arg1_1
  tiling->TB_M        = TB_M;
  tiling->TB_N        = TB_N;
  tiling->dim_arg0_0  = M;
  tiling->dim_arg1_1  = N;
}

// 性能估算（用于多模板时的 get_tiling 选择，单模板时仍生成供调试）
// numerator 来自 PerfExpr::numerator（TB_M 已代入常量），bandwidth 来自 PerfExpr::bandwidth_gbs
double ScoreTemplate_0(int64_t M, int64_t N) {
  // MTE2 搬运量(bytes) / 带宽(GB/s) → 估算执行时间(s)
  // TB_M=64 已代入（EnumerateSolver 选出的常量），N 为运行期 shape 符号
  return static_cast<double>(64 * N * 2) / 800e9;
}
```

**生成的 `get_tiling.cpp` 结构示例**（多模板）：

```cpp
// 运行期模板选择入口
void GetTiling_broadcast_add_reducesum(
    int64_t M, int64_t N,
    int*     selected_kernel_id,
    int64_t* block_dim,
    TilingData* tiling) {
  double scores[] = {
    ScoreTemplate_0(M, N),
    ScoreTemplate_1(M, N),
  };
  // 选执行时间估算最小的模板
  *selected_kernel_id = (scores[0] <= scores[1]) ? 0 : 1;
  if (*selected_kernel_id == 0)
    GetTiling_broadcast_add_reducesum_0(M, N, block_dim, tiling);
  else
    GetTiling_broadcast_add_reducesum_1(M, N, block_dim, tiling);
}
```

**单 impl_graph 退化**（`get_tiling.cpp`）：

```cpp
// 单模板：无需 score 比较，直接转发
void GetTiling_broadcast_add_reducesum(
    int64_t M, int64_t N,
    int*     selected_kernel_id,
    int64_t* block_dim,
    TilingData* tiling) {
  *selected_kernel_id = 0;
  GetTiling_broadcast_add_reducesum_0(M, N, block_dim, tiling);
}
```

### 5.7 AutoTunerPass

集成入口，作为 `afir-opt` 的一个 Pass（`--autotuner`）。

```
Pass 选项：
  --autotuner-hw=Ascend910B1        指定 SoC 型号
  --autotuner-hw-dir=./hardware     hardware JSON 目录
                                    （默认：CMake 安装时配置的 share/autotuner/hardware/）
  --autotuner-output-dir=./out      tiling func 输出目录
  --autotuner-space=./space.json    显式指定 tiling 搜索空间（覆盖自动查找）
                                    省略时按 func 名自动查找（见 TilingSpace 查找规则）
```

```
流程：
1. HardwareProfile::Load(soc_name, hw_dir)
2. 遍历 module 中所有 func.func，跳过不含 ascendc.aicore 属性的 func（如 transform.named_sequence、helper func）：
   a. MlirAnalyzer::Analyze(func) → KernelAnalysis
   b. TilingSpace::LoadFromJson(func_name, mlir_file_dir, hw, ctx, explicit_path)
      → 按优先级查找：func_name_tiling_space.json > tiling_space.json > explicit_path
   c. EnumerateSolver::Solve(analysis, space, hw) → TilingExprResult
   d. TilingFuncEmitter::EmitSingleGraphTilingFunc(result, analysis, id, output_dir)
3. TilingFuncEmitter::EmitGetTilingEntry(all_results, all_analyses, output_dir)
```

---

## 6. 数据流总结

```
[编译期]
step7_kernel_{0,1,...}.mlir（含 emitasc.py_struct TilingData schema，func 参数含动态 memref）
        │
        ▼ AutoTunerPass
        │  MlirAnalyzer   → KernelAnalysis（tiling_fields from py_struct + shape_dims AffineExpr）
        │  HardwareProfile → SoC 参数（aiv_num 等作为编译期常量）
        │  TilingSpace    → 按 func 名自动查找 space.json + IntegerRelation 约束
        │  EnumerateSolver → TilingExprResult（param_exprs + perf_expr + block_dim_expr）
        │  TilingFuncEmitter → tiling_func_{i}.cpp + get_tiling.cpp
        ▼
tiling_func_{i}.cpp（per impl_graph）
get_tiling.cpp（模板选择 + block_dim 计算入口）

[运行期]
GetTiling(M=1024, N=512, &kernel_id, &block_dim, &tiling)
  → 计算各模板 score，选最优
  → 填充 TilingData + block_dim
  → rtKernelLaunch(kernel_{id}.bin, block_dim, tiling)

[调试工具]
sim-validator \
  --kernel step8_kernel.cpp \
  --tiling-params "TB_M=64,TB_N=32,..." \
  --inputs input_a.npy,input_b.npy \
  --expected expected.npy \
  --block-dim 16
  → Compiler → Executor(Simulation) → 对比验证
```

---

## 7. 实现顺序

**阶段一：SimValidator**（`lib/Runtime/`）
- Compiler.cpp：bisheng 编译封装，arch 参数透传
- Executor.cpp：libruntime_camodel 调用，RunArgs 含 workspace_size，`BackendMode::RealDevice` 接口预留
- SimValidator.cpp：compile + execute + compare
- `tools/sim-validator/`：命令行工具，支持 `--tiling-params` 和 `--tiling` 两种输入方式
- **验收**：`sim-validator` 能跑通 `examples/broadcast-add-reduce/`，与 `test_e2e.py` 结果一致

**阶段二：HardwareProfile + TilingSpace**（数据层，无依赖，可与阶段一并行）
- `hardware/Ascend910B1.json`
- HardwareProfile.cpp：JSON 加载，路径解析
- TilingSpace.cpp：JSON 加载 + 约束解析为 `IntegerRelation`

**阶段三：AutoTunerPass**（依赖阶段一、二）
- MlirAnalyzer.cpp：step7 IR 分析（emitasc.py_struct 解析、shape_dim 提取、data_copy_l2 搬运量追踪）
- Solver.cpp：EnumerateSolver（枚举 + IntegerRelation 剪枝 + MTE2 估算）
- TilingFuncEmitter.cpp：AffineExpr → C++ 代码生成，含 block_dim，处理单/多模板退化
- AutoTunerPass.cpp：Pass 集成
- **验收**：`afir-opt --autotuner` 对 `broadcast-add-reduce` 生成正确的 `tiling_func_0.cpp` + `get_tiling.cpp`，`sim-validator` 使用生成的 tiling 验证通过

---

## 8. 关键约束与说明

- `python/runtime/` 保持不变，C++ 实现平行存在，接口对应关系：`compiler.py` ↔ `Compiler`，`executor.py` ↔ `Executor`
- AutoTuner 输入为 **step7_kernel.mlir**，`emitasc.py_struct` 字段名列表是 TilingData schema 的唯一权威来源
- AutoTuner 不做轴切分，模板由上游 Transform 生成
- `aiv_num` 等硬件常量在 Solver 中代入为编译期常量，**不**作为 AffineExpr 符号变量（避免 AffineExpr 不支持除以符号的限制）
- 约束系统双路径：线性约束用 `mlir::presburger::IntegerRelation` 快速剪枝，含参数乘积的非线性约束在枚举时逐点代入数值求值
- 前期 `EnumerateSolver` 输出常量 `param_exprs`，`perf_expr` 保留 shape 符号（用于运行期打分）
- `TilingExprResult::block_dim_expr` 由 Solver 填充，`TilingFuncEmitter` 负责生成 `*block_dim_out = ...` 赋值
- 单 impl_graph 时 `EmitGetTilingEntry` 生成无 score 比较的直接转发版本
- `EmitOnlineTuningStub` 预留接口，emit 空函数体 stub
- `BackendMode::RealDevice` 预留接口，CPU 仿真跑通后再实现 NPU 实跑
- `Solver` 设计为可插拔（`SolverBase`），后期可接 att C++ solver 替换 `EnumerateSolver`
- `HardwareProfile` 前期手写 JSON，后期从 att `api_perf_register` 提取，手动维护更新
