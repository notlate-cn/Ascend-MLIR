# 设计文档：Inductor → MLIR → AscendC 编译路径

**日期**: 2026-03-30
**状态**: Draft

## 动机

当前 torch → NPU 路径：
```
torch.export → torch-mlir → linalg → fuse-elementwise → tiling → AscendC → C++ → NPU
```

算子融合仅靠 `--linalg-fuse-elementwise-ops` 一条 pass，局限性大：
- 不能融合 reduction 作为 producer 的情况（如 sum → add）
- 不能融合 reduction + elementwise 双向
- 不能做 matmul epilogue 融合
- 无 cost model，仅判断 `hasOneUse()`
- 无水平融合（sibling fusion）

PyTorch Inductor 的融合能力远强于此（详见下方对比），且其融合决策有 4 层架构，其中前 3 层后端无关，第 4 层（SIMDScheduling.can_fuse）也是通用 SIMD 逻辑。

**目标**：复用 inductor 的融合能力，将融合结果输出为 linalg MLIR，接入现有 Ascend-MLIR pipeline。

## 融合能力对比

| 能力 | linalg-fuse-elementwise-ops | Inductor |
|------|---------------------------|----------|
| elementwise + elementwise | ✅ | ✅ |
| elementwise → reduction | ⚠️ 有限制 | ✅ |
| reduction → elementwise | ❌ | ✅ |
| reduction + reduction | ❌ | ✅ |
| matmul epilogue | ❌ | ✅ |
| 水平融合 | ❌ | ✅ |
| cost model | hasOneUse() | 共享内存评分 + 距离 + benchmark |
| tiling 感知 | ❌ | ✅ |

## 架构

```
┌─────────────────────────────────────────────────┐
│  torch.compile(model, backend="inductor")       │
│  + device="npu" (注册 MLIRScheduling)            │
└─────────────────┬───────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────┐
│  Inductor Pipeline (复用)                        │
│  1. Dynamo trace → FX Graph                     │
│  2. Decomposition + Lowering → LoopIR           │
│  3. Scheduler + Fusion → FusedSchedulerNode     │
└─────────────────┬───────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────┐
│  _post_fusion_custom_pass (新代码)               │
│  提取 FusedSchedulerNode 列表                    │
│  遍历 inner_fn → 自定义 OpsHandler trace         │
│  翻译为 linalg.generic MLIR                      │
│  输出 .mlir 文件                                 │
└─────────────────┬───────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────┐
│  现有 Ascend-MLIR Pipeline (复用 stage 2+)       │
│  transform tiling → bufferize → AscendC → C++   │
│  → bisheng → autotuner → NPU binary             │
└─────────────────────────────────────────────────┘
```

对比现有路径：
```
现有:  torch.export → torch-mlir → linalg → fuse-elementwise → tiling → ... → NPU
新增:  torch.compile → inductor融合 → MLIRScheduling → linalg → tiling → ... → NPU
```

## 组件设计

### 1. MLIRScheduling — 最小 Backend

继承 `SIMDScheduling`，只为让 inductor 的融合逻辑跑起来。

```python
from torch._inductor.codegen.simd import SIMDScheduling
from torch._inductor.codegen.common import register_backend_for_device

class MLIRScheduling(SIMDScheduling):
    # can_fuse_vertical/horizontal 完全继承 SIMDScheduling，零 override
    # codegen 相关方法空实现——不走 inductor codegen 流程

    def codegen_node(self, node):
        pass

    def codegen_template(self, *args, **kwargs):
        pass

    def codegen_node_schedule(self, *args, **kwargs):
        pass

    def flush(self):
        pass
```

**设计决策**：
- 融合逻辑零 override，完全复用 SIMDScheduling 的 4 层融合判断
- codegen 方法全部空实现，因为我们在 `_post_fusion_custom_pass` 中自己处理
- 不走 Triton 编译流程

### 2. Post-Fusion Pass — FusedSchedulerNode → linalg MLIR

通过 `torch._inductor.config._post_fusion_custom_pass` 注册，在融合完成后拦截节点列表。

#### Step 1: 提取 LoopIR 信息

```python
def post_fusion_pass(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    for node in nodes:
        if isinstance(node, FusedSchedulerNode):
            for sub in node.get_nodes():
                buf = sub.node          # ComputedBuffer
                data = buf.data         # Pointwise 或 Reduction
                ranges = data.ranges    # 迭代空间
                reduction_ranges = getattr(data, 'reduction_ranges', [])
                inner_fn = data.inner_fn  # Python closure
        else:
            buf = node.node
            # 同样处理
    return nodes  # 必须返回
```

#### Step 2: inner_fn trace → linalg.generic

自定义 `OpsHandler`，调用 inner_fn 时捕获所有操作，构建 MLIR：

```python
class MLIROpsHandler(OpsHandler):
    def load(self, name, index):          # → linalg.generic input tensor
    def store(self, name, index, value):  # → linalg.generic output
    def constant(self, value, dtype):     # → arith.constant
    def add(self, a, b):                  # → arith.addf / arith.addi
    def mul(self, a, b):                  # → arith.mulf
    def reduction(self, dtype, src_dtype, reduction_type, value):
        # → linalg.generic reduction iterator + combiner
    def to_dtype(self, value, dtype):     # → arith.extf / arith.truncf
    def where(self, cond, a, b):          # → arith.select
```

#### 映射关系

| Inductor LoopIR | linalg.generic |
|-----------------|----------------|
| `ranges` | parallel iterator dimensions |
| `reduction_ranges` | reduction iterator dimensions |
| `load(buf, index)` | input tensor + indexing_map |
| `store(buf, index, val)` | output tensor + indexing_map |
| 算术操作 (add, mul, ...) | generic body 内的 arith ops |
| index 表达式 (sympy) | affine_map |
| 融合节点的中间缓冲区 | generic body 内的 SSA 临时值 |

#### 主要挑战

1. **index 表达式 → affine_map**：inductor 用 sympy 表达式描述索引（如 `floor_div(x, 16)`），需转成 MLIR affine_map。POC 阶段先支持简单的线性索引和 broadcast 模式。

2. **多个 fused 子节点拼接**：一个 FusedSchedulerNode 里多个 ComputedBuffer，中间缓冲区需变成 linalg.generic 内部的 SSA 临时值而非独立 tensor。

3. **reduction 语义**：inductor 的 `Reduction` 节点有 `reduction_type` (sum/max/argmax 等) 和 `reduction_ranges`，需映射到 linalg.generic 的 reduction iterator + yield combiner body。

### 3. 接入现有 Pipeline

Post-fusion pass 输出的 `.mlir` 文件接入现有 pipeline 的 stage 2 之后：

```
输出的 linalg MLIR (已融合，未 tiled)
    ↓ stage 2: _generate_transform_script() + transform-interpreter
    ↓ stage 3: one-shot-bufferize
    ↓ stage 4-7: AscendC passes
    ↓ stage 8: afir-translate → C++ + tiling_space.json
    ↓ stage 9-10: autotuner
```

需要修改 `pipeline.py`，增加一个入口：
- 接收 inductor 输出的 `.mlir` 文件（替代 torch-mlir 生成的 step0/step1）
- 跳过 stage 0（torch-mlir 转换）和 stage 1（elementwise 融合）
- 从 stage 2（tiling）开始执行

## POC 范围

### 验证用例

使用现有的 `broadcast-add-reduce`：
```python
class Model(torch.nn.Module):
    def forward(self, a, b):
        return (a.unsqueeze(1) + b).sum(dim=1)
```

这个用例包含 elementwise (add) + reduction (sum) 的融合，正好是当前 linalg pass 做不到但 inductor 能做到的。

### POC 需要验证的问题

1. inductor 是否能将 add + sum 融合成一个 FusedSchedulerNode
2. inner_fn trace 能否正确捕获 broadcast + add + reduction 的操作序列
3. 生成的 linalg.generic MLIR 能否被现有 pipeline (stage 2+) 正确处理
4. 端到端数值正确性

### POC 不处理

- 动态 shape
- matmul / attention 等复杂算子
- 多 kernel 图（多个 FusedSchedulerNode 之间的依赖）
- 性能优化

## 文件组织

```
python/
  inductor_backend/
    __init__.py              # setup_inductor_backend() 入口
    scheduling.py            # MLIRScheduling 类
    post_fusion_pass.py      # _post_fusion_custom_pass 实现
    ops_handler.py           # MLIROpsHandler (inner_fn → MLIR)
    mlir_emitter.py          # MLIR 文本生成工具
test/
  inductor_e2e/
    test_broadcast_add_reduce.py  # POC 验证用例
```

## 依赖

- PyTorch >= 2.x (带 inductor)
- 现有 Ascend-MLIR pipeline (afir-opt, afir-translate, autotuner)
- 不依赖 torch_npu（我们自己注册 backend）
- 不依赖 torch-mlir（inductor 替代了它的角色）

## 风险

1. **SIMDScheduling 的 can_fuse 假设 GPU 硬件**：其 numel/rnumel 模型来自 GPU thread block，NPU 的 core 模型不同，可能导致过度融合或不足。POC 阶段先观察，后续可能需要 override。

2. **inner_fn 的 OpsHandler 协议可能变化**：inductor 内部 API 不稳定，跨版本可能 break。

3. **sympy index → affine_map 的完整性**：复杂索引表达式可能超出 affine_map 的表达能力，需要 fallback。