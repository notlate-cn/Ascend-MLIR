# impl-06: AutoTuner 框架深度实现方案

**Date:** 2026-04-23  
**Status:** Implementation Spec  
**设计依据**: [00-architecture.md](./00-architecture.md), [05-codegen-design.md](./05-codegen-design.md) §7  
**前置**: impl-04（tiling.infos 已生成），impl-05（Solver 接口已定义）

---

## 1. C++ 核心调度与 JSON 接口边界

考虑到系统的执行效率与模块内聚度，本方案采用 **纯 C++ 构建 Runner 搜寻框架**。但为了将“算子图分析出来的先天搜索约束”与“动态调整超参”的诉求完美解耦，在与前端 MLIR 构建的交界处，我们**保留 JSON 格式作为唯一的跨域交互和参数传递标准**。这使得未来无论接驳基于 Python 的贝叶斯模型库，还是人工 Hack 修改搜索空间去定向调优，都具有极大的便利性。

### 1.1 数据流向与执行核心

```cpp
[ MLIR tiling.infos Attribute ] 
        ↓ (tiling-infos-to-json 工具/Pass 自动推导提取)
[ tiling_space.json (包含空间参数和初始值) ]  ← (调优边界，外部可干预)
        ↓
[ AutoTuner JSON 解析器 ] 
        ↓
[ struct TilingSpace (C++ 内存模型) ]
        ↓
  Solver Engine (内含 Pruner 剪枝)
        ↓
[ std::vector<TilingConfig> ]
        ↓
Runner Environment (借助 LLVM/MLIR ThreadPool 及 C++ std::future 并发)
        ↓ 编译与 CAModel 评估 (针对真实内核源码/产物)
[ 最优 TilingConfig 或 Fallback 默认值 ]
        ↓
  回推生成的 tiling_func.cpp
```

---

## 2. 核心模块一：Space Extractor (跨界 JSON 空间)

本层将 MLIR 内部的 `tiling.infos` 进行语义解读兵导出为工业标准的 `tiling_space.json` 形式。C++ Tiling Tuner 通过读取该 JSON 文件（或字符串流）构建内置的 `TilingSpace`。

### 2.1 JSON Schema 契约设计

导出的 `tiling_space.json` 样例结构：

```json
{
  "kernel_id": "group0_plan0",
  "fields": [
    {
      "id": "tile.xblock",
      "abi_name": "XBLOCK",
      "kind": "tunable",
      "search": {
        "candidates": [64, 128, 256],
        "alignment": 32,
        "upper_bound_expr": "ceil(M/BM)"
      },
      "default": 128
    },
    {
      "id": "tile.rblock_0",
      "abi_name": "RBLOCK_0",
      "kind": "fixed",
      "default": 16
    }
  ],
  "block_dim_exprs": [
    "ceildiv(dim_0_0, XBLOCK)"
  ]
}
```

### 2.2 收口为基于内存的 TilingSpace 数据结构

AutoTuner 加载该 JSON 并映射入持久层的 C++ 结构中开展探寻：

```cpp
namespace autotune {

enum class TileKind { Tunable, Fixed, ShapeDim };

struct TileCandidateSpace {
  std::string abi_name;
  TileKind kind;
  std::vector<int64_t> candidates;
  std::optional<int64_t> alignment;
  std::string upper_bound_expr; 
};

struct TilingSpace {
  std::string kernel_id;
  std::vector<TileCandidateSpace> fields;
  std::vector<std::string> block_dim_exprs;
};

using TilingConfig = std::unordered_map<std::string, int64_t>;

} // namespace autotune
```

---

## 3. 核心模块二：Solver Engine 与 剪枝接口 (C++ 实现)

为了保证无效的超参组合（如炸 UB）被拦截在编译之前，设计强制的剪枝约束，并内置基础探针。

### 3.1 Pruner 接口及内存约束检查示例

```cpp
namespace autotune {

class Pruner {
public:
  virtual ~Pruner() = default;
  // 返回 true 表明配置无效，应该剪掉
  virtual bool can_prune(const TilingConfig& cfg, const HardwareSpec& hw) = 0;
};

// 【具体实现示例】基础容量检查
class MemoryConstraintPruner : public Pruner {
public:
  bool can_prune(const TilingConfig& cfg, const HardwareSpec& hw) override {
    int64_t total_ub_alloc = 0;
    // ... 基于 cfg['XBLOCK_SUB'], cfg['RBLOCK_0'] 及 Linalg 算子输入规模
    // ... 公式等效推演单 Block 切片的 Bytes
    
    // UB 大小如 256KB 阈值
    if (total_ub_alloc > hw.UnifiedBufferCapacityLimit) {
      return true; // 超容，立刻剪掉
    }
    return false;
  }
};

} // namespace autotune
```

### 3.2 派发探针：Default Solver 代码骨架

`DefaultSolver` 直接取 Min/Mid/Max 的排列来获取极快的搜索收敛路径：

```cpp
namespace autotune {

class DefaultSolver : public Solver {
  std::vector<std::unique_ptr<Pruner>> pruners;
public:
  std::vector<TilingConfig> generate(const TilingSpace& space, 
                                     const HardwareSpec& hw) override {
    std::vector<TilingConfig> results;
    
    // (仅展示骨架) 针对每个 tunable param, 取 candidates 的 前/中/后 三点
    std::vector<std::vector<int64_t>> target_pool;
    for (auto& field : space.fields) {
      if (field.kind == TileKind::Tunable && !field.candidates.empty()) {
        auto& c = field.candidates;
        target_pool.push_back({c.front(), c[c.size()/2], c.back()});
      }
    }
    
    // 笛卡尔组合
    std::vector<TilingConfig> raw_configs = cartesianProduct(space, target_pool);
    
    // 执行前置剪枝过滤
    for (auto& cfg : raw_configs) {
      bool prune = false;
      for (auto& p : pruners) {
        if (p->can_prune(cfg, hw)) { prune = true; break; }
      }
      if (!prune) results.push_back(cfg);
    }
    
    return results;
  }
};

} // namespace autotune
```

---

## 4. 核心模块三：Runner Environment (C++ 执行池)

借助 LLVM/MLIR 内置的 ThreadPool 或纯 C++ `std::async` 实现大并发验证。

### 4.1 Compilation Cache (带指纹缓存)

```cpp
std::string buildCacheKey(const TilingSpace& space, const TilingConfig& cfg) {
  // MD5(Kernel特征签名, 算子运行时实际 Shape, 当前这套参数)
  llvm::MD5 Hasher;
  Hasher.update(space.kernel_id);
  for (auto& [k, v] : cfg) {
    Hasher.update(k); Hasher.update(std::to_string(v));
  }
  llvm::MD5::MD5Result Result; Hasher.final(Result);
  return Result.digest().str().str();
}

// 缓存查找，避免同一个等价配置在多次编译中复跑
std::optional<double> checkCompileCache(const std::string& key);
```

### 4.2 Executor Pool

基于 `std::async` 的异步评测机制栈：

```cpp
struct EvaluateResult {
    TilingConfig cfg;
    bool success;
    double measured_cycle;
};

// 执行单个组合的编译翻译与 Simulator(仿真器) 分数获取
EvaluateResult runCompilationAndSimulate(const TilingConfig& cfg, 
                                         mlir::ModuleOp base_module) {
    // 1. 克隆隔离 Module => Clone module
    // 2. 将 cfg 注射进 Pass 4 替换并行宏
    // 3. afir-translate 调用底层 Ascend C 编译，生成 .o 或汇编
    // 4. 发送到 CAModel (或者 NPU 真实版) 并返回分数
}
```

---

## 5. 核心模块四：Fallback & Watchdog 机制

利用 `std::future::wait_for` 严格控制探索时延保障稳定性。

```cpp
TilingConfig executeAutoTune(const TilingSpace& space, 
                             mlir::ModuleOp module,
                             const TilingConfig& default_cfg) {
  DefaultSolver solver;
  auto configs = solver.generate(space, getHardwareSpec());
  
  std::vector<std::future<EvaluateResult>> futures;
  for (auto& cfg : configs) {
    futures.push_back(std::async(std::launch::async, 
                      runCompilationAndSimulate, cfg, module));
  }
  
  EvaluateResult best_result;
  best_result.measured_cycle = std::numeric_limits<double>::max();
  best_result.success = false;

  // Watchdog Timeout 检测 (比如 120 秒)
  auto timeout = std::chrono::seconds(120);
  auto start_t = std::chrono::steady_clock::now();
  
  for (auto& fut : futures) {
    auto time_left = timeout - (std::chrono::steady_clock::now() - start_t);
    if (time_left <= std::chrono::seconds(0)) break; // 超时强行中断

    if (fut.wait_for(time_left) == std::future_status::ready) {
      auto res = fut.get();
      if (res.success && res.measured_cycle < best_result.measured_cycle) {
        best_result = res;
      }
    }
  }

  // Fallback 策略：如果全军覆没或全超时，立即启用 Default
  if (!best_result.success) {
    llvm::errs() << "Warning: AutoTune all failed or timeout, using Fallback defaults.\n";
    return default_cfg; // 由 JSON 中的 default 字段解析合成的安全配置
  }
  
  return best_result.cfg;
}
```

---

## 6. 与 tiling.infos 的迁移关系

本模块对应 [04-tile-info.md](./04-tile-info.md) 定义的 **Phase C 迁移**：

| Phase | AutoTuner 消费来源 | 状态 |
|-------|-------------------|------|
| Phase A–B | `tiling_space.json`（从 `tiling.infos` 转换而来） | 当前 |
| Phase C | 直接读取 `tiling.infos` attribute（不经 JSON） | 计划 |
| Phase D | `tiling.tiles` / `tiling.shapes` 降级为兼容视图 | 计划 |

Phase C 之后，Solver 的 `generate()` 方法将直接接收 `TileInfo` 结构（而非 `TilingSpace`），
省去 JSON 序列化/反序列化开销。Solver 接口定义见 [05-codegen-design.md](./05-codegen-design.md) §7。
