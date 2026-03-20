# AscendC 开发工具链设计文档

**日期**：2026-03-20
**状态**：已批准
**范围**：compiler / validator / autotuner 三工具 + HostRunnerGen 库

---

## 1. 背景与目标

### 现状

现有 `sim-validator` 工具将编译、执行、验证三者合并，且只支持 CPU 仿真模式。`autotuner` 内部也直接包含编译逻辑。工具职责不清晰，手动调试单个 kernel 步骤繁琐，且缺乏 NPU 实跑验证路径和 msprof 性能采集集成。

### 目标

建立四工具职责分离的开发工具链：

1. **compiler**：kernel.cpp → .o + .bin + runner 可执行文件
2. **validator**：.bin + .npy → 功能/精度验证（--sim 或 --npu）
3. **msprof**（复用 CANN 自带）：runner → 性能数据采集
4. **autotuner**：tiling 参数空间搜索 → 最优 tiling_func.cpp

### 不在范围内

- tiling 模板生成（轴切分）：由上游 Transform Pass 负责
- ML-based 搜索策略：正式 autotuner 后续迭代，当前使用枚举
- 运行期在线 AutoTuning
- `--npu` 后端实现：本次预留接口，标注为 not implemented（Executor NPU 后端后续迭代）

---

## 2. 整体架构

```
Runtime 库 (include/Runtime/ + lib/Runtime/)
├── Compiler       bisheng 编译 + ld.lld 链接
├── Executor       dlopen camodel(sim) / CANN RT(npu，预留未实现)
├── Validator      精度对比，复用 Executor
├── NpyIO          .npy 读写
└── HostRunnerGen  生成 runner.cpp 并编译成可执行文件（新增）

CLI 工具 (tools/)
├── compiler/      Compiler + HostRunnerGen 的薄包装（新增）
├── validator/     Validator 的薄包装（新增，替代 sim-validator）
├── sim-validator/ 保留，兼容旧用法
└── autotuner/     扩展现有：新增 --perf-report
```

### 工具间依赖关系

```
compiler ──────────────────────────────────────┐
  (生成 kernel.o / kernel.bin / runner)         │
                                                ▼
validator ← [kernel.bin + tiling + .npy]    autotuner
  (--sim: Executor via camodel)                │
  (--npu: 预留，not implemented)              │ 内部调用
                                               │ Compiler (一次)
msprof ← [runner + tiling + .npy]             │ Validator (循环)
  (op simulator: CPU 仿真性能)                  │ msprof (最优解，--perf-report)
  (op: NPU 实跑性能)                            │
                                               ▼
                                          tiling_func.cpp
```

---

## 3. 组件设计

### 3.1 HostRunnerGen（新增 Runtime 库组件）

**职责**：根据 kernel 参数签名生成 runner.cpp，编译成独立可执行文件，供 `msprof op [simulator]` 包裹调用。

**接口**：
```cpp
// include/Runtime/HostRunnerGen.h
class HostRunnerGen {
public:
  struct Config {
    std::string kernel_name;
    std::string kernel_type;       // "vec" | "cube" | "mix"
    // kernel_type → magic 映射：
    //   "vec"  → MAGIC_ELF_AIVEC  (0x41415246)
    //   "cube" → MAGIC_ELF_AICUBE (0x41494343)
    //   "mix"  → MAGIC_ELF_AIVEC（保守默认，待确认）
    std::string soc_version;
    int         num_inputs;
    int         num_outputs;
    std::vector<std::string> tiling_layout;  // e.g. {"int64","int64","int32"}
    bool        verbose = false;
    // 注：arch 仅用于 bisheng 编译 kernel，runner 本身是 Host 可执行文件（x86/aarch64），
    // 由 g++ 编译，不需要 arch 字段。
  };

  // 生成 runner.cpp 并编译为 Host 可执行文件
  // 输出：output_dir/runner
  // runner 运行时通过 --bin 接受 kernel.bin 路径，无需硬编码
  llvm::Expected<std::string> Generate(const Config& cfg,
                                        const std::string& output_dir);
};
```

runner 运行时接口（固化在生成代码中）：
```
runner --bin    ./build/kernel.bin \
       --tiling-params "TB_M=16,TB_N=4,M=32,N=32" \
       --tiling-layout "int64,int64,int64,int64" \
       --inputs a.npy,b.npy \
       --output out.npy \
       --block-dim 2
```

`--bin` 为必选参数，运行时传入，不在生成代码中硬编码，支持 runner 独立移动和 .bin 重建。

---

### 3.2 compiler CLI（新增）

**职责**：一次性生成 kernel.o、kernel.bin、runner。

**产物命名**：`--name` 可选，默认从 kernel 文件名去掉扩展名推导（`step8_kernel.cpp` → `step8_kernel`）。产物为 `<name>.o`、`<name>.bin`、`runner`。

**接口**：
```
compiler --kernel        step8_kernel.cpp \
         --output        ./build/ \
         --name          my_kernel \        # 可选，默认从文件名推导
         --soc           Ascend910B1 \
         --arch          dav-c220-vec \     # 传给 bisheng，不影响 runner 编译
         --num-inputs    2 \                # runner 生成需要
         --num-outputs   1 \
         --tiling-layout "int64,int64,int64,int64" \
         --kernel-type   vec
```

**产物**：
```
build/
├── <name>.o      # bisheng 编译产物（msprof codeline 需要）
├── <name>.bin    # ld.lld 链接产物
└── runner        # msprof 可执行入口（Host 可执行，g++ 编译）
```

---

### 3.3 validator CLI（新增）

**职责**：加载 .bin，执行，精度对比。不做编译，不收集性能数据。

**`--npu` 模式**：接口预留，本次不实现，传入时打印 "not implemented" 并以退出码 4 退出。

**接口**：
```
validator --bin      ./build/my_kernel.bin \
          --name     my_kernel \
          --inputs   a.npy,b.npy \
          --expected out.npy \
          --tiling-params "TB_M=16,TB_N=4,M=32,N=32" \
          --tiling-layout "int64,int64,int64,int64" \
          --block-dim 2 \
          [--sim | --npu] \               # 默认 --sim；--npu 预留未实现
          --atol 0.1 \
          --rtol 1e-2
```

**stdout**：
```
max_abs_diff:  0.031
mean_abs_diff: 0.008
PASS           # 或 FAIL，退出码对应
```

---

### 3.4 autotuner CLI（扩展）

**接口**：
```
autotuner --space        tiling_space.json \
          --kernel       step8_kernel.cpp \
          --inputs       a.npy,b.npy \
          --expected     out.npy \
          --shape        "M=32,N=32" \
          --output       tiling_func.cpp \
          [--sim | --npu] \              # evaluator 后端，默认 --sim；--npu 预留未实现
          --perf-report \                # 对最优解跑 msprof（新增，替代 --sim-report）
          --perf-report-out ./perf_out
```

**内部流程**：
```
1. 模块A：解析 tiling_space.json + --shape
         → 枚举候选点（应用 constraints 剪枝）
2. 编译（一次）：
         Compiler → <name>.o + <name>.bin
         若 --perf-report：HostRunnerGen → runner
           Config 构造方式：
             kernel_name    = tiling_space.json "kernel" 字段
             kernel_type    = tiling_space.json "kernel_type" 字段
             soc_version    = --soc（或 JSON "soc" 字段）
             num_inputs     = len(--inputs 列表)
             num_outputs    = 1（当前约束，仅支持单输出）
             tiling_layout  = tiling_params[].type 字段按顺序收集
         RegisterBinary 一次
3. 内循环（每个 tiling 点）：
   a. Validator::ValidateBinary() → 精度 pass/fail
      精度 fail → 静默跳过，记录日志（--verbose 可见）
   b. 精度通过 → 记录 simulator cycle count（--sim）
                 或墙钟时间（--npu，预留）
4. 选最优点（cycle count / 执行时间最短）
5. 若 --perf-report（对最优解完整重跑，独立于步骤3）：
      --sim：msprof op simulator --soc-version=<soc> \
                 ./runner --bin <name>.bin --tiling-params <最优> ...
      --npu：msprof op \
                 ./runner --bin <name>.bin --tiling-params <最优> ...
      注：此步骤是完整重新执行，产出比 msopgen sim 更丰富的性能数据
          （PipeUtilization、ArithmeticUtilization、Roofline 等），
          代价是额外一次完整模拟执行（~10-20s），可接受。
6. 生成 tiling_func.cpp
```

**性能排序依据**：
| 模式 | 排序依据 | 说明 |
|------|---------|------|
| `--sim` | simulator cycle count（`rtDeviceSynchronize` 前后，camodel 内部计数） | 相对排序，非实时 |
| `--npu` | 墙钟时间（预留） | 真实硬件 |

---

### 3.5 msprof 调用方式

```bash
# CPU 仿真性能采集（--sim 模式）
msprof op simulator --soc-version=Ascend910B1 \
    ./build/runner --bin ./build/my_kernel.bin \
                   --tiling-params "TB_M=16,..." \
                   --tiling-layout "int64,..." \
                   --inputs a.npy,b.npy \
                   --output /dev/null \
                   --block-dim 2

# NPU 实跑性能采集（--npu 模式）
msprof op \
    ./build/runner --bin ./build/my_kernel.bin \
                   --tiling-params "TB_M=16,..." \
                   --tiling-layout "int64,..." \
                   --inputs a.npy,b.npy \
                   --output /dev/null \
                   --block-dim 2
```

msprof 路径优先级：`--msprof` 参数 > 环境变量 `$ASCEND_HOME_PATH/tools/profiler/bin/msprof` > `/usr/local/Ascend/ascend-toolkit/latest/tools/profiler/bin/msprof`

---

## 4. 数据格式

### tiling_space.json

```json
{
  "kernel": "broadcast_add_reducesum",
  "kernel_file": "step8_kernel.cpp",
  "kernel_type": "vec",
  "soc": "Ascend910B1",
  "block_dim_expr": "ceil(M/TB_M)",
  "tiling_params": [
    {"name": "TB_M",       "type": "int64", "min": 16, "max": 64, "step": 16},
    {"name": "TB_N",       "type": "int64", "min": 4,  "max": 64, "step": 4},
    {"name": "dim_arg0_0", "type": "int64", "fixed": true, "shape_key": "M"},
    {"name": "dim_arg1_1", "type": "int64", "fixed": true, "shape_key": "N"}
  ],
  "constraints": [
    "TB_M * TB_N * 2 <= 262144"
  ]
}
```

字段说明：

| 字段 | 必填 | 说明 |
|------|------|------|
| `kernel` | 是 | kernel 函数名 |
| `kernel_file` | 是 | .cpp 路径（相对于 JSON 文件），可被 `--kernel` 覆盖 |
| `kernel_type` | 是 | `vec\|cube\|mix`，传给 HostRunnerGen 和 msprof |
| `soc` | 是 | SoC 版本，可被 `--soc` 覆盖 |
| `block_dim_expr` | 是 | 支持 `ceil(X/Y)` 和 `X/Y`，变量来自 `--shape` 和搜索变量 |
| `tiling_params[].type` | 是 | `int64\|int32`，决定 tiling_layout 和内存布局 |
| `tiling_params[].min/max/step` | 搜索参数必填 | 枚举范围 |
| `tiling_params[].values` | 可选 | 枚举候选列表，优先于 min/max/step |
| `tiling_params[].fixed` | 可选 | true 时从 `--shape` 取值，不参与搜索 |
| `tiling_params[].shape_key` | fixed=true 时必填 | 对应 `--shape` 中的变量名 |
| `constraints` | 可选 | 枚举前剪枝表达式列表 |

**constraints 语法**：支持整数四则运算（`+ - * / %`）、比较（`<= >= < > == !=`）和括号，变量为 `tiling_params[].name` 中的名称。运算符优先级遵循标准 C 规则。模块A在枚举时对每个候选点求值，不满足任一 constraint 的点直接跳过。除零（`X / 0`）视为 constraint 不满足，跳过该点。

---

## 5. 错误处理

### 退出码约定（compiler / validator / autotuner 统一）

| 退出码 | 含义 |
|--------|------|
| 0 | 成功 |
| 1 | 精度验证失败（validator FAIL；autotuner 所有点均 fail） |
| 2 | 编译错误（bisheng/lld 失败） |
| 3 | 执行错误（rtKernelLaunch 失败） |
| 4 | 输入错误（文件不存在、参数非法、功能未实现） |
| 5 | autotuner 搜索空间耗尽（所有点均被 constraints 剪枝，无候选点） |

退出码 1 和 5 的区分：退出码 1 表示有候选点但全部精度不通过；退出码 5 表示 constraints 剪枝后搜索空间为空。

### autotuner 特殊处理

- 精度 fail 的点：静默跳过，日志记录（`--verbose` 可见）
- 全部点精度 fail：退出码 1，输出所有点的 diff 统计
- constraints 剪枝后空间为空：退出码 5，提示调整参数范围
- msprof 调用失败：退出码 0（不影响 tiling_func.cpp 生成），stderr 打印警告

---

## 6. 测试策略

| 层级 | 位置 | 方式 |
|------|------|------|
| compiler CLI | `test/tools/compiler/` | shell + FileCheck 验证产物存在、退出码 |
| validator CLI | `test/tools/validator/` | shell + FileCheck 验证 PASS/FAIL 输出 |
| autotuner 端到端 | `examples/broadcast-add-reduce/` | 现有集成测试脚本 |
| HostRunnerGen | 包含在 compiler 集成测试中 | 生成 runner 后执行，验证退出码 |

---

## 7. 目录结构变更

```
include/Runtime/
    HostRunnerGen.h          # 新增

lib/Runtime/
    HostRunnerGen.cpp        # 新增

tools/
    compiler/
        compiler_main.cpp    # 新增
        CMakeLists.txt       # 新增
    validator/
        validator_main.cpp   # 新增
        CMakeLists.txt       # 新增
    sim-validator/           # 保留，不改动
    autotuner/
        autotuner_main.cpp   # 扩展：--perf-report（替代 --sim-report）

test/tools/
    compiler/                # 新增
    validator/               # 新增
```

---

## 8. 与现有代码的关系

| 现有组件 | 处理方式 |
|---------|---------|
| `sim-validator` CLI | 保留，不改动，兼容旧脚本 |
| `Compiler` 库 | 不变 |
| `Executor` 库 | 不变（NPU 后端预留，未实现） |
| `SimValidator` 库 | `validator` CLI 直接复用 |
| `NpyIO` 库 | 不变 |
| autotuner `--sim-report` | 替换为 `--perf-report`，输出目录默认值从 `sim_report` 改为 `perf_out`，调用路径从 `msopgen sim`（后处理已有 dump）改为 `msprof op [simulator]`（完整重新执行）。实现时需同步删除 `--sim-report`/`--sim-report-out` 标志，新增 `--perf-report`/`--perf-report-out`。 |

---

## 9. 未来扩展点

- `--search-strategy grid|random|beam`：当前默认 grid，预留字段，不实现
- ML-based cost model（TVM Ansor 风格）：替换模块B的评估逻辑，Validator/Executor 接口不变
- `--npu` 路径完整实现：Executor NPU 后端 + warmup/repeat 参数（对应 mskpp.autotune）
- constraints 语法扩展：当前支持整数表达式，后续可支持 `values` 依赖其他变量的条件枚举
