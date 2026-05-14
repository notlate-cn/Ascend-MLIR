## 2. 第一层：Normalize

第一层的任务是把上层 lowering 后的 IR 规范化为第二层可稳定分析的统一入口形态。

```mermaid
flowchart LR
    A[统一入口 Module]
    B[入口规范化]
    C[规范化 Module]

    A --> B --> C
```

### 2.1 输入、输出与附加结果

| 项         | 内容                                                         |
| ---------- | ------------------------------------------------------------ |
| 输入       | 上层 lowering 后的结构化 module；方言分两级管理，见 2.3.1 节 |
| 输出       | 满足统一入口约定的规范化 module                              |
| 主边界对象 | `Normalized Linalg/Tensor IR`                                |
| 附加结果   | `gather_dim` / `embedding_dim` 结构标记、符号等价约束标注、入口 diagnostics |

### 2.2 语义规范化表

| 语义                  | 第一层输出形态                                               | 典型来源                                 |
| --------------------- | ------------------------------------------------------------ | ---------------------------------------- |
| matmul                | 具名 `linalg` op，优先保留 `linalg.matmul` 等标准结构化形式  | `torch.aten.mm`、`onnx.MatMul`           |
| elementwise           | 可识别 indexing map 的 `linalg.generic`                      | `torch.aten.add`、`onnx.Relu`            |
| reduce                | 带 reduction iterator 的 `linalg.generic` 或具名 `linalg` op | `torch.aten.sum`、`onnx.ReduceMax`       |
| reshape               | `tensor.expand_shape` / `tensor.collapse_shape` / `tensor.reshape` | `torch.aten.view`、`onnx.Reshape`        |
| cast                  | 具名 cast 或 body 可识别的 `linalg.generic`                  | `torch.aten.to`、`onnx.Cast`             |
| compare / select      | `arith.cmp*` + `select`，或可识别 body 的 `linalg.generic`   | `torch.aten.where`、`onnx.Where`         |
| gather / index_select | 带 `tensor.extract` 的 `linalg.generic`，附加 `gather_dim` 或 `embedding_dim` 结构标记 | `torch.aten.index_select`、`onnx.Gather` |
| broadcast             | 显式 indexing map 表达的 broadcast 语义                      | `torch.aten.expand`、`onnx.Expand`       |
| transpose             | permutation 明确的 indexing map                              | `torch.aten.permute`、`onnx.Transpose`   |
| split / slice         | `tensor.extract_slice` / `tensor.insert_slice`               | `torch.aten.split`、`onnx.Slice`         |
| concat                | `tensor.concat` 或等价 slice/insert 组合                     | `torch.aten.cat`、`onnx.Concat`          |
| shape 查询            | `tensor.dim` + `arith`                                       | `torch.aten.size`、`onnx.Shape`          |

### 2.3 入口处理规则

#### 2.3.1 方言白名单

入口方言分两级管理：

**核心方言**（必须支持；第一层对其结构做完整规范化与合法性验证）：

| 方言     | 说明               |
| -------- | ------------------ |
| `linalg` | 主计算载体         |
| `tensor` | 值语义张量操作     |
| `arith`  | 标量算术与类型转换 |
| `math`   | 数学函数           |
| `func`   | 函数与调用边界     |

**允许透传方言**（不分析、不重写；第一层只验证其是否影响 kernel 候选闭包，影响则报错，不影响则透传至第二层）：

| 方言      | 说明                                             |
| --------- | ------------------------------------------------ |
| `index`   | 索引类型运算，MLIR 推荐用于替代 `i64` 的维度计算 |
| `shape`   | 动态形状计算，部分前端会保留少量 shape 计算 op   |
| `complex` | 复数运算，复数模型的合法入口                     |

此外，`cf.assert` 作为 **op 级例外** 允许透传，用于承载前端生成的动态 shape guard。该例外不表示 `cf` 方言整体进入白名单；`cf.br`、`cf.cond_br` 等控制流 op 仍然报错拒绝。

出现上述两级之外的方言或 op 级例外之外的操作，报错拒绝，不允许静默透传。

**透传方言的额外限制**：

| 方言      | 允许形态                                                     | 禁止形态                                                     |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `index`   | 仅作为 shape / 维度计算的中间值；不参与 kernel 内主计算路径   | 不允许出现在 `linalg.generic` 的 body 内                     |
| `shape`   | 仅作为 dynamic shape 表达；不参与 kernel 内主计算路径         | 不允许出现在 kernel 候选闭包内（详见 2.4 节）                |
| `complex` | 仅允许 `complex.constant` 等纯常量在 module 顶层透传；不允许 `complex.add` / `complex.mul` 等计算 op 出现在任何 kernel 候选闭包内 | 当前版本下游层（第二、三、四、五层）**均不接受** `complex` 计算 op；遇到时第一层 verifier 报 `DialectRejected`。复数计算的完整支持在 V2-1.4.3 节中作为预留扩展点 |

**op 级例外的额外限制**：

| op          | 允许形态                                                     | 禁止形态                                                     |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `cf.assert` | 仅作为动态 shape guard 透传；不参与第二层 kernel 候选构造；后续层可将其消费为 guard 诊断或保留为 host/runtime guard 输入 | 不允许作为一般控制流载体；不允许引入 branch / region；不允许参与 `linalg` body 内主计算 |

#### 2.3.2 结构规范化规则

| 处理项          | 规则                                                         |
| --------------- | ------------------------------------------------------------ |
| 具名 op 保留    | 对 `linalg.matmul` 等已具备稳定结构语义的具名 `linalg` op，保留具名形式，不退化为通用 `linalg.generic` |
| 属性裁剪        | 按 2.3.3 节的可执行规则处理                                  |
| shape 规范化    | 只允许 ranked symbolic shape；维度可以是编译期常量或符号变量；rank 必须已知且在入口中不变化 |
| 符号等价标注    | 对结构上等价的符号维度附加统一符号变量名，将等价关系记录为 `AscendSymbolConstraintAttr`；详见 2.3.4 节 |
| indexing 规范化 | broadcast 必须规范化为 indexing map 表达；transpose 必须规范化为 permutation indexing map；split/slice 必须规范化为 `tensor.extract_slice` / `tensor.insert_slice`；concat 必须规范化为 `tensor.concat` 或等价 slice/insert 组合；gather 必须规范化为 `linalg.generic + tensor.extract + gather_dim/embedding_dim` |
| gather 规范化   | 上层框架（torch-mlir / onnx-mlir）lowering 后的 gather 已为 `linalg.generic + tensor.extract` 形态；第一层不做进一步 lowering，只由 `mark-structured-ops` 附加 `gather_dim` 或 `embedding_dim` 结构标记，供第二层 `OpRoleClassifier` 识别 |
| canonicalize    | 只运行 2.5.1 节许可 pass 集内的 pass；不得跨 op 语义边界重写，不得引入新控制流，不得改变 kernel 候选闭包 |

#### 2.3.3 属性裁剪规则

属性裁剪按以下优先级顺序执行，规则互斥，匹配第一条即停止：

1. attr name 在属性保留表中，保留。
2. attr name 以 `ascend.` 为命名空间前缀，保留（本编译器自身标记）。
3. attr name 以已知前端命名空间前缀开头，删除。
4. 其余情况，保留并输出 warning，附加 `ascend.unknown_origin` 标记。

规则 1 和规则 3 均依赖静态表，可直接编程实现；规则 2 和规则 4 为前缀匹配，无需人工介入。

**属性保留表**（规则 1）：

| 属性                           | 所属 dialect                                  | 消费方                                                       |
| ------------------------------ | --------------------------------------------- | ------------------------------------------------------------ |
| `linalg.iterator_types`        | `linalg`                                      | 第二层 `OpRoleClassifier`、第三层 `ScheduleProblemBuilder`   |
| `linalg.indexing_maps`         | `linalg`                                      | 第二层 `FusionCandidateAnalyzer`、第三层 `ScheduleProblemBuilder` |
| `gather_dim` / `embedding_dim` | 本编译器（第一层 `mark-structured-ops` 附加） | 第二层 `OpRoleClassifier`                                    |
| `AscendSymbolConstraintAttr`   | 本编译器（第一层符号等价分析附加）            | 第三层 `ScheduleProblemBuilder`                              |

属性保留表由人工维护；新增条目的判断标准为：后续层有代码显式读取该 attr。

**已知前端命名空间前缀表**（规则 3）：

| 前缀     | 来源前端        | 说明                                                  |
| -------- | --------------- | ----------------------------------------------------- |
| `torch.` | torch-mlir      | 前端专属属性，无后续消费方；典型如 `torch.type_bound` |
| `onnx.`  | onnx-mlir       | 调试 / 溯源用途，不影响编译语义；典型如 `onnx.name`   |
| `tf.`    | TensorFlow 前端 | 设备与图语义由本编译器重新建立，原 attr 失效          |

前缀表随接入前端扩展，新增前端时同步补充，不允许静默透传。

**未知来源 attr 的 warning 格式**：

```
warning: unknown attr '<attr_name>' on op '<op_name>' at <loc>;
         not in preserve list and not from known frontend namespace;
         preserved but marked as 'ascend.unknown_origin'
```

附加 `ascend.unknown_origin` 标记的 attr 在第二层入口 verifier 中再次提示，并在 verifier 完成后**立即删除**，不进入第二层后续分析流程。删除时机提前至入口的原因：`ascend.unknown_origin` attr 不携带任何编译语义，若允许其存活至结构识别或候选分析阶段，将污染 fingerprint 计算和结构匹配结果。

#### 2.3.4 符号等价标注

动态 shape 下同一符号维度可能在多个 op 的 operand 中独立出现。第一层末尾执行一次轻量的符号等价分析，将等价关系显式记录，避免后续各层重复推导。

##### 2.3.4.1 基础类型定义

**`DimRef`**：标识某个 SSA 值的某一维度，是等价关系的原子单位。

```cpp
struct DimRef {
  Value  value;   // 必须是 ranked tensor 类型的 SSA 值
  int64_t dim;    // 维度下标，范围 [0, rank(value))；负数不合法
};
```

`DimRef` 的等价关系定义为：两个 `DimRef` 等价，当且仅当在所有可能的运行时输入下，它们所指维度的大小始终相等。

**`DimExpr`**：`OpSemanticSummary.resultShape` 中每个维度的表示类型，是静态常数或符号变量的并集：

```cpp
using DimExpr = std::variant<
  int64_t,      // 编译期已知的静态常数，如 128、1
  StringAttr    // 符号变量名，与 AscendSymbolConstraintAttr 中的 symName 对应
>;
```

符号变量名在同一 `func` 范围内唯一，由 2.3.4.2 节的分析算法统一分配；不同 `func` 之间的符号变量名独立，不互相干扰。

**`AscendSymbolConstraintAttr`**：附加在 `func` attribute 上的等价关系表，结构如下：

```
AscendSymbolConstraintAttr ::= {
  equivalenceClasses: List<EquivalenceClass>
}

EquivalenceClass ::= {
  symName: StringAttr          // 该等价类的符号变量名，如 "M", "K", "seq_len"
  members: List<DimRef>        // 属于该等价类的全部 DimRef
}
```

约束：
- 每个 `DimRef` 至多属于一个 `EquivalenceClass`；不在任何等价类中的维度视为独立符号，后续层按悲观假设处理（不与任何其他维度等价）
- `symName` 在同一 `AscendSymbolConstraintAttr` 内唯一
- `members` 非空；空等价类不合法，不允许写入

##### 2.3.4.2 分析算法

分析采用**带权 Union-Find** 结构，以 `DimRef` 为节点，合并等价的节点。分析在 `func` 范围内一次性完成，输入为规范化后的 IR（indexing 规范化、gather 规范化均已完成）。

**触发合并的规则（按 IR 拓扑序遍历每个 op）：**

| 规则编号 | 触发场景 | 合并的 DimRef 对 |
| -------- | -------- | --------------- |
| R1 | `linalg.generic` op：对每对 `(ins[i], outs[j])` 或 `(ins[i], ins[j])`，若它们的 indexing map 在某个 iterator 维度 `d` 上指向同一个 `(value, dimIdx)` 位置，则合并这两个 `DimRef` | `(ins[i], f_i(d))` ↔ `(ins[j], f_j(d))`，其中 `f_i` / `f_j` 为对应的 indexing map |
| R2 | 具名 contraction-like op（`linalg.matmul` 等）：按 op 的语义显式合并收缩维度；`matmul(A: MxK, B: KxN)` → 合并 `(A, 1)` ↔ `(B, 0)` | 由 op 的 `ContractionOpInterface` 或静态规则表给出，不依赖 indexing map 推导 |
| R3 | producer-consumer SSA 边：若 `op_b` 的 operand 直接来自 `op_a` 的 result（即 `op_b.operand[i] == op_a.result[j]`），则对所有维度 `d` 合并 `(op_a.result[j], d)` ↔ `(op_b.operand[i], d)` | 两者是同一 SSA 值的不同引用上下文，维度严格对应 |
| R4 | `tensor.extract_slice(src, offsets, sizes, strides)`：对每个非退化维度 `d`（stride = 1 且 size 来自 `tensor.dim(src, d)` 或静态等于 `src.dim(d)`），合并 `(src, d)` ↔ `(result, d)` | 退化维度（size = 1）和 stride ≠ 1 的维度不合并 |
| R5 | `tensor.dim(v, d)` 的结果被多处引用：以该 `tensor.dim` 的 SSA value 为根，将所有以该值为 `sizes` / `offsets` 参数的 `tensor.extract_slice` / `tensor.empty` 等 op 的对应维度合并 | 间接等价，通过 SSA def-use 链追踪 |
| R6 | `linalg.broadcast`：output 的非广播维度与 input 的对应维度合并；广播维度（input 中不存在的维度）不合并 | 按 `linalg.broadcast` 的 `dimensions` attr 确定哪些是广播维度 |

**符号变量名分配**：Union-Find 合并完成后，对每个连通分量分配唯一 `symName`：
1. 若分量内存在来自 `func` 参数的 `DimRef`（即 `value` 是 `BlockArgument`），优先用参数名 + 维度下标，如 `arg0_dim1`
2. 否则用编译器生成的稳定 ID，格式为 `sym_<function内唯一整数>`
3. 静态常数维度（已知为编译期常量）不进入等价类，直接在 `DimExpr` 中以 `int64_t` 表示

**分析边界**：
- 只在 `func` 内分析，不跨 `func` 边界
- 只处理 ranked tensor 类型的 SSA 值；scalar / index 类型不参与
- 分析不修改 IR，只构建 `AscendSymbolConstraintAttr` 并附加到 `func`
- 若某维度无法确定等价类（如来自不透明的外部调用），保留为独立符号，不强行合并

**`matmul(A:MxK, B:KxN) → add(result, bias:N) → reduce(sum, dim=N)` 分析示例：**

遍历顺序（拓扑序）：matmul → add → reduce

| 步骤 | 触发规则 | 合并操作 |
| ---- | -------- | -------- |
| matmul R2 | 收缩维度 | `(A,1)` ↔ `(B,0)` → 等价类 `K = {(A,1),(B,0)}` |
| matmul R1 | outs 维度 | `(A,0)` ↔ `(result_mm,0)` → 类 `M`；`(B,1)` ↔ `(result_mm,1)` → 类 `N` |
| add R3 | producer-consumer | `(result_mm,0)` ↔ `(add.operand[0],0)` → 并入 `M`；`(result_mm,1)` ↔ `(add.operand[0],1)` → 并入 `N` |
| add R1 | ins/outs 共享 iterator | `(bias,0)` ↔ `(add.result,1)` → 并入 `N`（broadcast 维度 dim=0 不合并） |
| add R3 | producer-consumer | `(add.result,0)` ↔ `(reduce.operand,0)` → 并入 `M`；`(add.result,1)` ↔ `(reduce.operand,1)` → 并入 `N` |

最终 `AscendSymbolConstraintAttr`：

```
equivalenceClasses:
  - symName: "M", members: [(A,0),(result_mm,0),(add.op[0],0),(add.result,0),(reduce.op,0)]
  - symName: "N", members: [(B,1),(result_mm,1),(add.op[0],1),(bias,0),(add.result,1),(reduce.op,1)]
  - symName: "K", members: [(A,1),(B,0)]
```

`OpSemanticSummary.resultShape`（由上述等价类填充）：

| op | resultShape |
| --- | --- |
| matmul | `[DimExpr("M"), DimExpr("N")]` |
| add | `[DimExpr("M"), DimExpr("N")]` |
| reduce | `[DimExpr("M")]`（N 轴被 reduction 消去） |

### 2.4 入口非法条件

| 情况                                                  | 处理                                 |
| ----------------------------------------------------- | ------------------------------------ |
| 出现核心方言、透传方言白名单和 op 级例外之外的方言 / 操作 | 报错                                 |
| 透传方言中的 op 影响 kernel 候选闭包                  | 报错                                 |
| unranked tensor                                       | 报错                                 |
| 无法解释的 shape 语义，或同一语义存在多种未规范化表达 | 报错；不区分子类型，不允许第二层补救 |
| 具有内存写入、I/O、状态更新或未知副作用的 op          | 报错                                 |

### 2.5 进入第二层前的 Pass 约束

在 `Dependency Analysis` 之前，只允许运行许可 pass 集内的社区 pass；这些 pass 只能清理 IR，不得改变第二层将要消费的结构语义。

#### 2.5.1 许可 pass 列表

| Pass                      | 允许范围                                                     | 禁止事项                                                     |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `cse`                     | 全局公共子表达式消除                                         | 不得跨 region 消除带副作用的 op                              |
| `canonicalize`（受限）    | 仅开启 `arith` fold、`tensor` fold 相关 pattern；通过 `PatternApplicator` filter 机制显式禁用所有 `linalg` pattern | 禁止触发任何会改写 `indexingMaps`、`iteratorTypes`、`resultShape` 的 linalg pattern（如 `foldUnitExtentDims`） |
| `tensor` 局部 fold        | `tensor.cast` fold、`tensor.dim` 常量折叠                    | 不得改变 ranked symbolic shape 语义                          |
| `arith` / `math` 常量折叠 | 纯标量常量折叠与表达式化简                                   | 不得引入新控制流，不得重写主计算拓扑                         |

#### 2.5.2 禁止的 pass 类型

| 类型                                                         | 原因                                    |
| ------------------------------------------------------------ | --------------------------------------- |
| bufferization / memref lowering                              | 会改变值语义和后续 kernel 划分边界      |
| loop / scf lowering                                          | 会破坏结构化计算图和 region 分析基础    |
| 会重写 `resultShape` / `indexingMaps` / `iteratorTypes` 的 pass | 会导致第二层分析对象不稳定              |
| 社区 fusion / partition pass，或其他会跨 region 重组计算边界的 pass | 会绕过第二层的 `KernelPattern` 划分逻辑 |

#### 2.5.3 执行规则

- 许可 pass 集内的 pass 只允许在第一层结束到第二层开始之间执行
- 这些 pass 执行后，IR 仍必须满足第一层的统一入口约束
- 一旦执行了会修改 `resultShape`、`indexingMaps`、`iteratorTypes`、结构属性或 region 边界的 pass，必须重新进入第二层分析窗口，不得复用已有分析结果

### 2.6 第一层 Verifier

第一层结束后由 `EntryNormalizationVerifier` 在进入第二层前统一验证；任一项失败即中止编译，不允许向后传递不合规 IR。检查项按顺序执行，前一项失败仍继续后续以收集完整诊断：

| 顺序 | 检查项                  | 检查内容                                                     | 失败时 `reasonKind` |
| ---- | ----------------------- | ------------------------------------------------------------ | ------------------- |
| 1    | 方言白名单              | 所有 op 所属 dialect 必须出现在 2.3.1 节的核心方言、允许透传方言列表中，或命中 `cf.assert` op 级 shape-guard 例外 | `DialectRejected`   |
| 2    | 透传方言闭包安全        | 透传方言中的 op 不得位于任何 kernel 候选闭包内（详见 2.4 节） | `DialectRejected`   |
| 3    | shape 规范化            | 所有 tensor / memref 类型必须为 ranked symbolic；不允许 unranked，rank 必须已知且不变 | `DialectRejected`   |
| 4    | 具名 op 保留            | `linalg.matmul` 等具名 op 未被退化为 `linalg.generic`        | `DialectRejected`   |
| 5    | 属性裁剪正确性          | 不存在前端命名空间前缀属性（`torch.` / `onnx.` / `tf.`）；所有 `ascend.unknown_origin` 标记的 attr 已删除 | `DialectRejected`   |
| 6    | indexing 规范化         | broadcast / transpose / split / slice / concat / gather 已落到 2.3.2 节规定的标准载体上 | `DialectRejected`   |
| 7    | 符号等价标注完整性      | `AscendSymbolConstraintAttr` 已附加在 `func` attribute 上；其内容覆盖 2.3.4 节列出的所有等价场景 | `DialectRejected`   |
| 8    | 副作用 op 排除          | 所有 op 必须为纯值语义；不存在内存写入、I/O、状态更新或未知副作用 op | `DialectRejected`   |
| 9    | MLIR 内置 verifier      | 运行 `mlir::verify(module)`                                  | MLIR 自身诊断       |

**Verifier 数据流约束**：

- `EntryNormalizationVerifier` 只读消费 IR，不修改 IR 或 attribute
- 失败诊断必须遵循 V2-7.4 节 diagnostics 规范（含 `stage = Normalize`、`objectId`、`reasonKind`、`message`、`isRecoverable`、`fallbackTaken` 字段）
- 第一层 Verifier 的 `isRecoverable` 默认均为 `false`：入口非法的 IR 不允许第二层补救（与 2.4 节"不区分子类型，不允许第二层补救"一致）

**核心接口**：

```cpp
class EntryNormalizationVerifier {
public:
  LogicalResult verify(ModuleOp module,
                       const TargetProfile &targetProfile,
                       DiagnosticEmitter &diag) const;
};
```
