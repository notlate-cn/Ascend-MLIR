# Ascend NPU MLIR 编译器问题建模 V2

> 本文档是 `Ascend-MLIR-Detailed-Implementation-V2.zh.md` 的**理论前言（Problem Formulation）**。
>
> - 实现文档回答 **How**：每个 pass 怎么写、每个 IR 怎么传。
> - 本文档回答 **What / Why**：我们到底在求解什么问题、为什么必须分阶段、每个阶段在经典计算机科学中对应哪个已知问题家族。
>
> 本文档的目标不是给出可证明最优的算法，而是把"为 Ascend NPU 构建 AI 编译器"这一工程目标，归约为一组**经典计算机科学问题的实例**，从而：
>
> 1. 让每一层的算法选择可以站在半个世纪的经典文献之上；
> 2. 让"为什么是这五层"成为有论证的结论，而不是工程直觉；
> 3. 让每一层有明确的 **Input / Output / DecisionVariables / Constraints / Objective**，可以独立替换求解器（精确、近似、启发式、学习）。

---

## 0. 记号

| 记号 | 含义 |
|---|---|
| $G = (V, E)$ | 输入计算图（DAG），$V$ 为算子节点，$E$ 为数据依赖边 |
| $H$ | 目标硬件描述（Ascend NPU：cube/vector 单元、UB/L1/L0/GM 多级存储、DMA 通道、SIMD 宽度等） |
| $\mathcal{T}$ | 张量集合，$t \in \mathcal{T}$ 具有 shape、dtype、layout |
| $\Omega$ | 全局决策空间 |
| $\omega \in \Omega$ | 一次完整的编译决策（一个具体的 binary 是其像） |
| $C(\omega; G, H)$ | 代价函数（典型为 end-to-end latency 的估计或实测） |
| $\mathcal{V}(\omega; G, H)$ | 合法性谓词：决策 $\omega$ 是否满足语义正确性与硬件约束 |
| $P$ | 全局编译问题 |
| $P_1, \dots, P_5$ | 五个子问题，分别对应 Normalize / Kernelize / Schedule / Realize / Translate |

---

## 1. 全局问题 P

### 1.1 非形式陈述

> 给定一个计算图 $G$ 和一台 Ascend NPU $H$，找到一组编译决策 $\omega \in \Omega$，将 $G$ 翻译为可在 $H$ 上执行的二进制工件 $\mathcal{B}(\omega)$，使得：
>
> (i) $\mathcal{B}(\omega)$ 在 $H$ 上的执行结果与 $G$ 在参考语义下的结果按 numerical contract 等价；
> (ii) $\mathcal{B}(\omega)$ 满足 $H$ 的所有硬件约束（存储容量、对齐、寄存器数、指令编码等）；
> (iii) 端到端代价 $C(\omega; G, H)$（典型为 latency）尽可能小。

### 1.2 形式定义

$$
P:\quad \min_{\omega \in \Omega} C(\omega; G, H) \quad \text{s.t.} \quad \mathcal{V}(\omega; G, H) = \text{true}
$$

其中决策空间 $\Omega$ 是 5 类决策的笛卡尔积：

$$
\Omega \;=\; \Omega_{\text{norm}} \times \Omega_{\text{part}} \times \Omega_{\text{sched}} \times \Omega_{\text{mem}} \times \Omega_{\text{cg}}
$$

- $\Omega_{\text{norm}}$：语义规范化决策（同一 op 的多种合法书写形式中选一种规范形）
- $\Omega_{\text{part}}$：图划分与融合决策（哪些算子合并进同一个 kernel、哪些 kernel 横向融合）
- $\Omega_{\text{sched}}$：调度决策（每个 kernel 的循环嵌套、tiling 参数、axis binding、流水线节拍）
- $\Omega_{\text{mem}}$：内存实现决策（buffer 在哪一级存储、生命周期、复用、DMA 搬运计划）
- $\Omega_{\text{cg}}$：代码生成决策（IR 节点到 AscendC / cube intrinsic / vector intrinsic 的 cover）

合法性谓词 $\mathcal{V}$ 是若干子谓词的合取：

$$
\mathcal{V} \;=\; \mathcal{V}_{\text{sem}} \wedge \mathcal{V}_{\text{cap}} \wedge \mathcal{V}_{\text{align}} \wedge \mathcal{V}_{\text{life}} \wedge \mathcal{V}_{\text{enc}}
$$

分别表示：语义保持、硬件容量约束、对齐与 layout 约束、buffer 生命周期不冲突、指令可编码。

### 1.3 复杂度论证

**命题 1.** $P$ 是 NP-hard。

**证明（草图）**：考虑 $\Omega_{\text{part}}$ 的一个特例 —— 在算子级权重图上做最小割划分，子集大小受 UB 容量上界约束。这等价于带容量约束的 **balanced graph partitioning**，已知是 NP-hard（Garey & Johnson, 1979）。由于 $\Omega_{\text{part}} \subseteq \Omega$，$P$ 也至少 NP-hard。$\square$

类似地，可分别从以下子问题向 $P$ 做归约：
- $\Omega_{\text{sched}}$ 中的 tiling 决策包含 **多维 bin packing**（NP-hard）；
- $\Omega_{\text{mem}}$ 中的 buffer 复用是 **interval graph coloring 的推广**，在多级存储与对齐约束下不再具有 interval graph 的可着色保证；
- $\Omega_{\text{cg}}$ 中的 DAG pattern covering 在 DAG 上已是 NP-hard（Aho-Johnson-Ullman 在 tree 上是 P，DAG 上是 NP-hard）。

### 1.4 不可解性：状态空间爆炸

即便不考虑 NP-hardness，**直接在 $\Omega$ 上搜索**也是不现实的。粗略估计：

| 子空间 | 典型规模（单个 transformer block） |
|---|---|
| $\|\Omega_{\text{norm}}\|$ | $\sim 10^{1}$（每个 op 少量规范化分支） |
| $\|\Omega_{\text{part}}\|$ | $\sim 2^{\|V\|}$（每条边切/不切） |
| $\|\Omega_{\text{sched}}\|$ | $\sim 10^{6}\text{--}10^{8}$（tiling × order × binding） |
| $\|\Omega_{\text{mem}}\|$ | $\sim 10^{4}$（放置 × 复用方案） |
| $\|\Omega_{\text{cg}}\|$ | $\sim 10^{2}$（每个 op 的 cover 选择） |

笛卡尔积可轻易达到 $10^{20}$ 以上，且 $C(\omega)$ 的单次评估需要完整下层编译 + 上板/模拟，耗时秒级到分钟级。**联合搜索 $\Omega$ 在工程上不可行**。

### 1.5 代价函数的不可微与非凸性

$C(\omega; G, H)$ 具有以下结构性困难：

1. **离散**：$\omega$ 几乎全部是离散变量，不可对 $C$ 直接做梯度优化。
2. **非凸**：tiling、融合、放置的局部最优通常不是全局最优（典型例子：扩大 tile 提升 cube 利用率，但溢出 UB 触发 spill，导致整体变慢）。
3. **不可加**：层间代价存在强耦合，例如融合决策改变后，下层 tiling 的最优解会突变 —— 因此**不能假设 $C = \sum_i C_i$**。
4. **昂贵**：真实测量需要端到端编译 + 执行，开销远高于评估一次目标函数的标准优化设定。

### 1.6 由 1.3–1.5 推出的工程结论

(C1) **必须分解** $\Omega$。无法在联合空间上搜索。
(C2) **必须分层近似**。每层在固定上层决策的条件下，求该层子问题的（近似）最优解。
(C3) **必须承认 sub-optimality**。任何分层都会放弃跨层的联合优化机会；分解的合理性来自"工程可行性"，而非"全局最优"。
(C4) **必须在每层保留可替换性**。每个子问题应归约到一个经典问题家族，使得求解器（精确 ILP、近似算法、启发式、ML 学习）可以独立演进。

第 2 节给出 $P$ 到 $\{P_1,\dots,P_5\}$ 的具体分解；第 3 节将每个 $P_i$ 归约到其对应的经典问题家族。

---

## 2. 分解：从 $P$ 到 $P_1 \circ P_2 \circ P_3 \circ P_4 \circ P_5$

### 2.1 分解的形式

由 §1.5(3) 的不可加性，**$P$ 不能被分解为五个独立子问题的和**。我们采用的分解是 **fixed-prefix 串行近似**：

$$
\omega^\star \;\approx\; \omega_5^\star \circ \omega_4^\star \circ \omega_3^\star \circ \omega_2^\star \circ \omega_1^\star
$$

其中每个 $\omega_i^\star$ 是在**已固定上游决策的条件下**求得的子问题最优解：

$$
\omega_i^\star \;=\; \arg\min_{\omega_i \in \Omega_i(\omega_{<i})}\, C_i\!\left(\omega_i;\, \omega_{<i},\, G, H\right)
\quad \text{s.t.}\quad \mathcal{V}_i\!\left(\omega_i;\, \omega_{<i}\right) = \text{true}
$$

记 $\omega_{<i} := (\omega_1^\star, \dots, \omega_{i-1}^\star)$，则有：

- $\Omega_i(\omega_{<i})$：在上游决策固定下，本层尚可选择的合法子空间（被上游收窄）；
- $C_i$：本层的**代理代价函数**（surrogate cost），并非真实端到端 latency，而是本层对端到端代价的局部近似；
- $\mathcal{V}_i$：本层负责验证的合法性子谓词，满足 $\mathcal{V} = \bigwedge_i \mathcal{V}_i$。

### 2.2 五个子问题的定位

| $i$ | 子问题 $P_i$ | 决策子空间 | V2 实现层 | 经典问题家族（§3） |
|---|---|---|---|---|
| 1 | **Normalize** | $\Omega_{\text{norm}}$：选择规范形 | L1 Normalize | Term Rewriting System |
| 2 | **Kernelize** | $\Omega_{\text{part}}$：图划分与融合 | L2 Kernelize | Graph / Hypergraph Partitioning |
| 3 | **Schedule** | $\Omega_{\text{sched}}$：tiling、order、binding | L3 Schedule | Job-Shop Scheduling + Multi-dim Bin Packing |
| 4 | **Realize** | $\Omega_{\text{mem}}$：placement、复用、DMA | L4 Realize | Register Allocation / Interval Coloring |
| 5 | **Translate** | $\Omega_{\text{cg}}$：DAG pattern covering | L5 Translate | Instruction Selection (DAG Covering) |

子问题排序由**依赖方向**决定：下游子问题的合法子空间 $\Omega_i(\omega_{<i})$ 严格依赖上游决策的产出。例如：

- Schedule 必须知道一个 kernel 包含哪些算子（Kernelize 的产出），才能定义其循环嵌套；
- Realize 必须知道 tile 形状（Schedule 的产出），才能计算 buffer 大小是否满足 UB 容量；
- Translate 必须知道 buffer 在哪一级存储（Realize 的产出），才能选择正确的 cube/vector intrinsic。

这种**单向依赖**是分阶段近似在工程上可行的前提：若依赖是双向的，则必须用 fixpoint 迭代或联合求解。

### 2.3 引入的 sub-optimality

Fixed-prefix 串行的代价是放弃跨层联合优化。本节明确列出**哪些跨层耦合被牺牲**，以便后续在每层设计中能识别"看起来本层最优、全局却次优"的情形。

**(S1) Kernelize ↔ Schedule 耦合。**
融合决策固定后，Schedule 只能在该融合分组内部 tile。但某些算子若**不融合**反而能用更激进的 tile 形状，从而获得更高 cube 利用率。例如：matmul + softmax 强行融合后，softmax 的 reduce 轴限制了 matmul 的 K 轴 tile。

**(S2) Schedule ↔ Realize 耦合。**
Tile 形状固定后，Realize 只能在该形状下安排 buffer 复用。但更小的 tile 可能让两个本来 lifetime 冲突的 buffer 变得可复用，从而避免一次 spill。这种"宁可减小 tile 换取更优内存"的全局权衡在分阶段中被遮蔽。

**(S3) Realize ↔ Translate 耦合。**
Placement 决策固定后，Translate 只能在既定的存储层级上选择指令。但某些 cube intrinsic 对 layout 有特殊要求（如 NZ 格式），若 Realize 阶段不知道 Translate 会选哪条 intrinsic，可能选错 layout，导致额外 transpose。

**(S4) 长程耦合（Kernelize ↔ Translate）。**
某些算子的最优融合分组依赖于"该融合能否被一条 cube intrinsic 一次性 cover"。但 Translate 在最后一层才决定 cover，因此 Kernelize 阶段只能用启发式（如 op-role 分类）预判，而非真实 cover 结果。

### 2.4 缓解 sub-optimality 的工程手段

分阶段分解的次优性无法被消除，但可以被**有界缓解**。本工程采用三类手段（在 V2 实现文档中均有对应 pass）：

1. **代理代价 $C_i$ 中嵌入下游成本模型**。
   例如 Kernelize 的代价函数不只考虑融合后的局部 op 数，还显式引入"该融合预期能否被 cube intrinsic cover"的启发分项；Schedule 的代价函数显式估算 Realize 阶段的 buffer 压力。
   → 这是 §1.5(3) 不可加性的局部反制：用 surrogate 把下游耦合**部分**回灌到上游。

2. **跨层 hints（非约束的弱信号）**。
   上游决策附带 hint 给下游，下游可参考也可忽略。例如 Kernelize 输出的 KernelPattern 携带"建议的轴优先级"，Schedule 可据此优先尝试某些 tiling 候选。
   → 形式化为：$\omega_i^\star$ 的输出除 $\omega_i$ 本身外，还包含 hint 集合 $\mathcal{H}_i$，作为 $C_{i+1}$ 的辅助输入。

3. **层内候选集 + 早失败（early reject）**。
   每层内部维护**候选集合**（如 Kernelize 的 `FusionCandidate[] → MergedCandidate[] → KernelPatternCandidate[]`），通过本层的合法性检查 + 代理代价排序，**在本层结束前**收敛到单一决策再向下传递。下游接收的是已决定的 IR，不接收候选集，也不向上游回退。
   → 形式化为：$P_i$ 内部存在中间候选集 $\mathcal{K}_i \subseteq \Omega_i(\omega_{<i})$，最终输出 $\omega_i^\star \in \mathcal{K}_i$；$\mathcal{K}_i$ 不出本层。
   → 不可解情形通过**显式兜底契约**处理而非回退：如 Kernelize 的 `FallbackSingleOpPattern` 保证全覆盖性（任何 op 至少有一种合法划分），Normalize 的入口拒绝（`DialectRejected`）保证不合法输入早失败。

### 2.5 分解的正确性保证

虽然分解牺牲了最优性，但**正确性必须无损**。形式化为：

$$
\bigwedge_{i=1}^{5} \mathcal{V}_i(\omega_i^\star;\, \omega_{<i}) = \text{true}
\;\Longrightarrow\;
\mathcal{V}(\omega^\star;\, G, H) = \text{true}
$$

这要求每层 $\mathcal{V}_i$ 的设计满足**合法性的可分解性**：

- $\mathcal{V}_{\text{sem}}$（语义保持）：每层都必须保证语义不变。L1 通过 confluent rewriting、L2–L5 通过 IR verifier 强制。
- $\mathcal{V}_{\text{cap}}$（容量约束）：主要由 L3 + L4 联合保证（tile 形状决定 buffer 大小，placement 决定其归属存储级）。
- $\mathcal{V}_{\text{align}}$（对齐/layout）：L4 + L5 联合保证。
- $\mathcal{V}_{\text{life}}$（生命周期）：L4 单独保证。
- $\mathcal{V}_{\text{enc}}$（指令可编码）：L5 单独保证。

这种**合法性子谓词到层的映射**是 §4 中"层间 contract"的形式基础。

### 2.6 与 V2 五层架构的对齐

V2 的 Normalize → Kernelize → Schedule → Realize → Translate 不是任意选择的层数。它是 §2.2 表格中**单向依赖链**的直接体现：

- 选择 5 而非 3 或 7，是因为这 5 个子空间是**自然解耦点** —— 每对相邻子空间之间的耦合可以用代理代价 + hint + 有限回退缓解，而**子空间内部**的耦合无法解耦（例如 tile 与 order 在 Schedule 内部高度耦合，不能再拆）。
- 五层的边界等价于"在哪些位置插入 IR snapshot 与 verifier"。每层结束时 IR 必须是**自洽且可验证**的，下层不需要回看上层中间状态。

第 3 节将逐一展开 $P_1, \dots, P_5$ 到经典问题的归约。

---

## 3. 五个子问题：到经典问题的归约

### 3.0 统一模板

每个子问题 $P_i$ 按以下七元结构展开：

1. **归约目标**：$P_i$ 是哪个经典问题家族的实例（或推广）。
2. **形式定义**：$\langle \text{Input}, \text{Output}, \text{DecisionVariables}, \text{Constraints}, \text{Objective} \rangle$。
3. **复杂度**：所属复杂度类，关键的已知结果。
4. **与经典问题的差异**：Ascend NPU 特有的扩展或弱化。
5. **求解策略家族**：精确（ILP/SMT）、近似（有 bound 的多项式算法）、启发式（greedy/local search）、学习（RL/cost model）。
6. **本工程的选择**：当前 V2 实现采用的策略。
7. **V2 实现对应**：对应到 `Ascend-MLIR-Detailed-Implementation-V2.zh.md` 的哪些 pass 和 IR 结构。

### 3.1 $P_1$ Normalize：归约到 Term Rewriting System

> *简要总结*
>
> **问题归类：项重写系统 (Term Rewriting System)**
>
> 前端 IR 存在多种语义等价的表达形式（不同方言、不同算子分解粒度）。下游处理需要每个算子有唯一的规范形式，否则要为每种变体重复实现处理逻辑。
>
> **原理：** 设计一组语义保持的重写规则，使其满足两个性质：
>
> - **终止性**：规则有限步内停止
> - **合流性**：无论应用顺序如何，最终都收敛到同一规范形
>
> 合流性意味着不需要搜索"最优应用顺序"——任何顺序都得到相同结果，这一层因此不是优化问题。
>
> **解决方案：**
>
> - 手工设计规则集（V2-2.3 规范化表），并通过 verifier 静态校验合流性
> - 入口拒绝不合法方言（`DialectRejected`），不尝试修复
> - 单遍 fixpoint 重写至规范形

#### 3.1.1 归约目标

$P_1$ 是 **Term Rewriting System (TRS)** 中的"求规范形"问题：

> 给定一组语义保持的重写规则 $R$ 和一个输入项 $t_0$，求 $t_0$ 在 $R$ 下的规范形 $\text{nf}_R(t_0)$。

经典文献：Knuth-Bendix (1970), Baader & Nipkow《Term Rewriting and All That》(1998)。

#### 3.1.2 形式定义

$$
\def\llbracket{[\![}
\def\rrbracket{]\!]}
P_1:\quad
\begin{array}{ll}
\text{Input} & G_0:\text{ 前端 IR（多方言混合）} \\
\text{Output} & G_1 = \text{nf}_R(G_0):\text{ 规范化后 IR} \\
\text{Decision} & \omega_1 \in \Omega_{\text{norm}}:\text{ 重写步骤的应用顺序与选择} \\
\text{Constraints} & \forall t.\; \llbracket t \rrbracket = \llbracket R(t) \rrbracket \quad \text{(语义保持)} \\
& R\text{ 终止 (terminating)} \\
& R\text{ 合流 (confluent, 至少局部合流)} \\
\text{Objective} & \text{无显式代价；输出是结构唯一的规范形}
\end{array}
$$

关键性质：

- **终止性**：$R$ 中任何重写序列有限步停止。
- **合流性 (Church-Rosser)**：$\omega_1$ 的选择不影响最终结果，即 $\text{nf}_R$ 是良定义的函数。

合流 + 终止 ⟹ $\Omega_{\text{norm}}$ 实际上是平凡的（任何遍历顺序得到同一规范形），这是 §1.4 中 $|\Omega_{\text{norm}}| \sim 10^{1}$ 估计的根源。

#### 3.1.3 复杂度

- **判定 TRS 是否合流且终止**：一般不可判定（停机问题归约）。
- **给定合流终止 TRS 求规范形**：多项式时间（每步 $R$ 的应用是局部的）。

实践中：通过**人工构造** $R$ + Knuth-Bendix completion 的人工校验，保证 $R$ 是合流终止的，然后运行时只做多项式时间的规范化。

#### 3.1.4 与经典 TRS 的差异

| 维度 | 经典 TRS | $P_1$ |
|---|---|---|
| 项 | 一阶项 | MLIR IR（有 region、attribute、type） |
| 等价语义 | 句法等价 | 数值契约下的等价（允许 dtype 升降级、layout 变换） |
| 透传方言 | 不存在 | 允许部分 op 不被重写直接透传（需 verifier 确认不影响 kernel 闭包） |
| 入口拒绝 | 不存在 | 有 `DialectRejected` 等显式 reject 路径（不合法输入早失败） |

#### 3.1.5 求解策略

- **精确**：MLIR canonicalize pattern + 自定义 RewritePattern，单遍 fixpoint。
- **启发式**：在多种合法规范形存在时（理论上不应存在，但工程上可能），选择"最有利于下游"的形态（如优先转 `linalg.generic` 而非 named op）。
- **学习**：不适用 —— TRS 的合流性要求不允许学习引入非确定性。

#### 3.1.6 本工程的选择

精确：手工设计 $R$（V2-2.3 节"语义规范化表"），通过 verifier（V2-2.6）+ 入口拒绝（V2-2.4）保证合流终止。无搜索。

#### 3.1.7 V2 实现对应

- 重写规则集 $R$：V2-2.3 语义规范化表 + V2-2.5 许可 pass 集
- 入口拒绝：V2-2.4 入口非法条件
- 出口验证：V2-2.6 第一层 Verifier

---

### 3.2 $P_2$ Kernelize：归约到带容量约束的图划分

> *简要总结*
>
> **问题归类：带容量约束的非循环图划分 (Constrained Acyclic Graph Partitioning)**
>
> 计算图节点数大、若每个算子独立成 kernel，中间结果都要在 GM 和片上存储之间往返搬运，DMA 流量主导执行时间。需要把节点划分为若干子集，每个子集成为一个 kernel，子集内部的中间结果片上传递。
>
> **核心约束：**
>
> - 划分后的商图必须是 DAG（否则 kernel 之间死锁）
> - 每个 kernel 的内存占用不超过片上存储容量
> - 每个 kernel 内部必须能被下游调度（含合法的 OpRole 组合、可匹配模板族）
>
> **复杂度：** NP-hard（归约自带容量约束的 balanced graph partitioning）。METIS / hMETIS 等成熟工具不能直接用，因为它们针对对称图，不处理 acyclic 约束。
>
> **解决方案：**
>
> - **Role-driven 启发式**：以重算力算子（cube 类）为种子，按 OpRole 规则向数据依赖邻居扩展
> - **代理代价**：跨 kernel 数据量 − λ·片上复用收益，等价于带权 min-cut
> - **HandwrittenPattern 短路**：FlashAttention 等高价值子图通过结构匹配直接整体识别为单一 kernel，绕过通用启发式
> - **FallbackSingleOpPattern 兜底**：任何无法融入大 kernel 的算子，单独成 kernel，保证全覆盖性

#### 3.2.1 归约目标

$P_2$ 是 **Constrained Graph / Hypergraph Partitioning** 问题：

> 给定带权 DAG $G_1 = (V, E, w_V, w_E)$，将 $V$ 划分为不相交子集 $\{K_1, \dots, K_m\}$（每个 $K_j$ 即一个 kernel），在满足容量与无环约束下最小化跨划分代价。

经典文献：Karp (1972, NP-hard 化归), Kernighan-Lin (1970), METIS (Karypis-Kumar 1998), hMETIS (hypergraph)。

#### 3.2.2 形式定义

$$
P_2:\quad
\begin{array}{ll}
\text{Input} & G_1 = (V, E),\quad H:\text{ NPU 描述},\quad \omega_1^\star \\
\text{Output} & \pi: V \to \{1, \dots, m\}\quad (\text{kernel id assignment}) \\
\text{Decision} & \omega_2 \in \Omega_{\text{part}} \\
\text{Constraints} & (\text{acyclic}) \quad G/\pi\text{ 在 kernel 粒度上仍是 DAG} \\
& (\text{capacity}) \quad \forall j.\; \sum_{v \in K_j} \text{mem}(v) \leq \text{Cap}_{\text{UB}} \\
& (\text{role-closure}) \quad \text{每个 kernel 必须包含合法 OpRole 组合} \\
& (\text{primitive-coverage}) \quad K_j \text{ 必须能匹配至少一个 primitive 或 HandwrittenPattern} \\
& (\text{shape-guard-budget}) \quad |\text{dynamicGuardSet}(K_j)| \leq B \\
\text{Objective} & C_2(\pi) = \sum_{(u,v)\in E,\,\pi(u)\neq\pi(v)} \text{traffic}(u,v) \;-\; \lambda \cdot \text{reuse\_gain}(\pi)
\end{array}
$$

其中：

- $\text{traffic}(u,v)$：跨 kernel 数据量（被切的边对应一次 GM ↔ on-chip 搬运）。
- $\text{reuse\_gain}(\pi)$：融合后片上复用的收益（被融合的中间值不下盘）。
- $\lambda$：两者权衡系数。

直觉：**最小化跨 kernel 流量 = 最大化片上复用**，与经典 min-cut 同构。

#### 3.2.3 复杂度

- 无约束图划分（min-cut 平衡划分）：NP-hard (Garey & Johnson, 1979)。
- 带容量约束 + DAG acyclic：仍 NP-hard，且 acyclic 约束使得 METIS 这类对称划分算法**不能直接用**。
- 在树上（特殊 DAG）：可多项式求解（动态规划）。

#### 3.2.4 与经典图划分的差异

| 维度 | 经典图划分 | $P_2$ |
|---|---|---|
| 边权 | 标量 | 张量大小（受 dtype × shape × layout 影响） |
| 节点权 | 标量 | 多维（compute cost + mem footprint + register pressure） |
| 划分数 $m$ | 通常给定 | 不给定，由算法决定 |
| Acyclic 约束 | 无 | 必须保证 quotient graph 是 DAG（否则产生死锁） |
| 划分内部结构 | 不关心 | 必须满足 role-closure / primitive coverage / handwritten pattern 完整性 |

最后一条最关键：**$P_2$ 不仅划分节点，还要求每个划分内部具有可被下游 $P_3$ 调度的结构**。这是 $P_2$ 比纯图划分更难的核心原因。

#### 3.2.5 求解策略

- **精确 ILP**：变量数 $O(|V|)$，约束数 $O(|E|)$，但 acyclic 约束需引入 $O(|V|^2)$ 拓扑变量，工业规模图不可行。
- **谱方法**：适合无 acyclic 约束的对称图，**不适用于 DAG**。
- **启发式（greedy + role-driven）**：从 anchor op（cube 类）出发，按 OpRole 规则扩展融合候选，逐步合并。这是 V2 的选择。
- **学习**：可用 GNN + RL（如 GraphPiece、REGAL），但当前算子集相对窄，启发式已能覆盖。

#### 3.2.6 本工程的选择

启发式 + 显式兜底：
- 主路径：role-driven candidate expansion（V2-3.6 `FusionCandidateAnalyzer`）+ 候选合并（V2-3.7 `CandidateMergeAnalyzer`）+ 水平融合（V2-3.8）。
- 兜底：`FallbackSingleOpPattern`（V2-3.12）保证全覆盖性 —— 任何 op 至少有一个单 op 划分作为合法解。
- 高价值结构特例：`HandwrittenPattern`（V2-3.4.2、V2-3.9）通过结构匹配预先识别 FlashAttention/GroupedMatMul 等子图，绕过通用启发式。

#### 3.2.7 V2 实现对应

- 依赖图与 acyclic 验证：V2-3.3 `DependencyAnalyzer`
- 候选构造：V2-3.6 `FusionCandidateAnalyzer`
- 候选合并：V2-3.7 `CandidateMergeAnalyzer`
- 水平融合：V2-3.8 `HorizontalFusionAnalyzer`
- 决策 IR 产物：V2-3.9 `KernelPattern` / V2-3.10 `KernelPartitionDecision`
- 出口验证：V2-3.11 第二层 Verifier

---

### 3.3 $P_3$ Schedule：归约到 Scheduling + Multi-dim Bin Packing + State-Space Search

> *简要总结*
>
> **问题归类：作业车间调度 + 多维装箱 + 状态空间搜索的耦合问题**
>
> 每个 kernel 内部的多重嵌套循环需要决定四件事：
>
> - **Tile 形状**（每个轴一次处理多少元素）
> - **循环顺序**
> - **轴绑定**（哪些轴并行化到 block/thread）
> - **流水线深度**（double buffer / triple buffer）
>
> **核心难点：** Tile 形状**同时是调度变量和资源消耗变量**。
>
> - 经典调度问题假设任务大小固定、决策只是顺序
> - NPU 上 tile 大小直接决定 buffer footprint、cube 单元利用率、reuse 模式、对齐效率
> - 这导致 scheduling 与 bin packing 强耦合，不能先 pack 再 schedule
>
> **复杂度：** NP-hard。三个子问题各自就 NP-hard：作业车间调度（Garey-Johnson-Sethi 1976）、多维装箱（Chlebík 2006 证明无 APTAS）、状态空间 $10^4 \sim 10^6$ 量级。
>
> **代理代价：** $\alpha \cdot t_{cube} + \beta \cdot t_{vector} + \gamma \cdot t_{dma} - \delta \cdot \text{overlap} + \epsilon \cdot \text{spillRisk}$。最后一项 `spillRisk` 是把下游 Realize 的内存压力**前向**回灌到本层代价，缓解 §2.3 S2 耦合。
>
> **解决方案：**
>
> - **模板化**：每个 kernel 根据其 templateFamilies 标签选择调度骨架，骨架定义参数空间
> - **受约束参数搜索**：在模板允许范围内枚举 tile/order 候选，按代理代价排序取最优
> - **AxisCoalescing 预处理**：合并可合并的轴，降低搜索维度
> - **HandwrittenPattern 短路**：命中高价值结构直接用预定义最优调度
> - **Compilation Cache**：相同 fingerprint 直接复用历史结果
>
> 不采用 polyhedral 框架（NPU 异构性超出仿射模型表达力）、不采用在线 RL（编译期成本不可接受）。

#### 3.3.1 归约目标

$P_3$ 是三类经典问题的**组合**，三者在 NPU 调度中不可独立求解：

1. **Job-Shop / Pipeline Scheduling**：决定 kernel 内多算子（cube job / vector job / DMA job）在异构执行单元上的执行顺序与流水线节拍。Garey-Johnson-Sethi (1976) 已证 $J_3 \| C_{\max}$（三机 job-shop）NP-hard。
2. **Multi-dimensional Bin Packing**：tile 形状选择，相当于把"算子的逻辑迭代空间"装进"片上存储容量盒子"，每个轴对应一维。多维 bin packing 已知 NP-hard 且不存在 APTAS（Chlebík-Chlebíková, 2006）。
3. **State-Space Search**：tile × order × binding × pipeline depth 的笛卡尔积，是典型的离散组合空间。

经典文献：Pinedo《Scheduling》(2016)、Coffman-Garey-Johnson (bin packing 综述)、Pluto (Bondhugula 2008, polyhedral scheduling)、TVM Ansor (Zheng 2020, learned cost model)。

#### 3.3.2 形式定义

固定 Kernelize 的产出 $\omega_2^\star$，对每个 kernel $K_j$ 分别求解 $P_3^{(j)}$：

$$
P_3^{(j)}:\quad
\begin{array}{ll}
\text{Input} & K_j,\; \text{scheduleContract}(K_j),\; H \\
\text{Output} & \sigma_j = \langle \mathcal{T}_j, \mathcal{O}_j, \mathcal{B}_j, \mathcal{P}_j \rangle \\
\text{Decision} & \omega_3^{(j)} \in \Omega_{\text{sched}}^{(j)}
\end{array}
$$

输出四元组 $\sigma_j$ 含义：

- $\mathcal{T}_j \in \mathbb{Z}_{>0}^{d_j}$：tile 形状，$d_j$ 为 kernel 的逻辑轴维度
- $\mathcal{O}_j \in \text{Perm}([d_j])$：循环顺序
- $\mathcal{B}_j: [d_j] \to \{\text{block}, \text{thread}, \text{serial}\}$：轴 binding（哪些轴并行化到 block/thread）
- $\mathcal{P}_j \in \mathbb{Z}_{>0}$：流水线深度（double / triple buffering）

**约束**：

$$
\begin{aligned}
& \text{(tileable)} \quad \mathcal{T}_j[k] \mid \text{axisExtent}(K_j, k),\quad \forall k \in \text{tileableAxes}(K_j) \\
& \text{(reduction-keep)} \quad \mathcal{T}_j[k] = \text{axisExtent}(K_j, k),\quad \forall k \in \text{requiredReductionAxes}(K_j) \\
& \text{(capacity)} \quad \text{bufferFootprint}(K_j, \mathcal{T}_j, \mathcal{P}_j) \leq \text{Cap}_{\text{UB}}(H) \\
& \text{(alignment)} \quad \mathcal{T}_j[k] \equiv 0 \pmod{\text{align}(H, k)} \\
& \text{(template-applicable)} \quad \sigma_j \in \text{validRange}(\text{templateFamilies}(K_j)) \\
& \text{(dynamic-guard)} \quad \sigma_j \models \text{dynamicGuardSet}(K_j)
\end{aligned}
$$

**目标函数（代理代价）**：

$$
C_3^{(j)}(\sigma_j) \;=\; \alpha\cdot t_{\text{cube}} \;+\; \beta\cdot t_{\text{vector}} \;+\; \gamma\cdot t_{\text{dma}} \;-\; \delta\cdot \text{overlap}(\sigma_j) \;+\; \epsilon\cdot \text{spillRisk}(\sigma_j)
$$

含义：

- $t_{\text{cube}}, t_{\text{vector}}, t_{\text{dma}}$：三类执行单元的估计耗时
- $\text{overlap}(\sigma_j)$：流水线下三类单元的并行重叠，是负代价（越大越好）
- $\text{spillRisk}(\sigma_j)$：下游 Realize 可能引发的 spill 风险预估（§2.4(1) 代理代价嵌入下游模型的体现）

整个 kernel 的调度问题为：

$$
\omega_3^\star = \bigsqcup_j \arg\min_{\sigma_j} C_3^{(j)}(\sigma_j) \quad \text{s.t. all constraints above}
$$

注意 kernel 之间在 $P_3$ 层面是独立的（融合决策已在 $P_2$ 固定）。

#### 3.3.3 复杂度

- 单 kernel 内的 tile 选择（仅 $\mathcal{T}_j$，固定其他）：多维 bin packing 子情形，NP-hard。
- 单 kernel 内的 order 选择（仅 $\mathcal{O}_j$）：在有循环依赖时是 NP-hard（polyhedral scheduling 已证）；无循环依赖时多项式。
- 联合 $(\mathcal{T}_j, \mathcal{O}_j, \mathcal{B}_j, \mathcal{P}_j)$：NP-hard，且**真实代价**（含 cost model）通常非凸非单调。
- 状态空间：单 kernel 典型 $|\Omega_{\text{sched}}^{(j)}| \sim 10^4\text{–}10^6$（参考 TVM Ansor 实测）。

#### 3.3.4 与经典问题的差异

| 维度 | 经典 Scheduling/Bin Packing | $P_3$ |
|---|---|---|
| 任务粒度 | 原子任务 | 多重嵌套循环，任务大小是 tile 形状的函数 |
| 资源 | 同构机器 | 异构（cube/vector/scalar + 多级存储 + DMA 通道） |
| 目标 | $C_{\max}$ 或 makespan | 流水重叠 + 容量利用 + 下游友好度的加权 |
| Tile 的副作用 | 不存在 | tile 选择同时影响 buffer footprint、reuse 模式、对齐效率 |
| Template 约束 | 不存在 | 必须落在 `templateFamilies` 声明的有效范围内 |
| 代价函数 | 显式可算 | 需要 cost model（解析模型 / 学习模型 / 实测） |

最关键差异：**tile 形状同时是调度变量和资源消耗变量**。这破坏了经典调度问题"任务大小给定、决策只是顺序"的假设，使得 $P_3$ 是 scheduling 与 packing 的**耦合**问题，不能先 pack 再 schedule。

#### 3.3.5 求解策略家族

| 家族 | 代表方法 | 适用情形 |
|---|---|---|
| **精确（ILP / Polyhedral）** | Pluto, isl scheduler | 仿射依赖、规则循环；对 NPU 异构性表达力不足 |
| **手写模板 + 参数搜索** | TVM Compute/Schedule, Halide schedule | 模板覆盖典型 op 族，对每个模板做参数枚举 |
| **自动调度（基于成本模型）** | TVM Ansor, MetaSchedule | 学习 cost model + 演化搜索 |
| **强化学习** | AutoTVM RL, REGAL | 状态空间过大时；当前 NPU 算子集规模下性价比不高 |
| **启发式 + Cache** | XLA SPMD partitioner, IREE schedules | 工程性强，可解释性高 |

#### 3.3.6 本工程的选择

**Template-based + 受约束参数化搜索 + 编译期 cache**：

1. **模板族（Template Families）**：由 $P_2$ 输出的 `templateFamilies` 标签声明本 kernel 适用的模板（如 `AnchorEpilogue`、`SoftmaxTemplate`）；模板是参数化的调度骨架。
2. **参数搜索**：在模板允许的参数空间（tile 形状候选、order 选项、pipeline depth）内，用代理代价 $C_3$ 排序，取最优单点。
3. **HandwrittenPattern 短路**：若 kernel 被 `HandwrittenPattern` 命中（如 FlashAttention），直接采用预定义最优调度，跳过搜索。
4. **Compilation Cache**：相同 fingerprint 的 kernel 直接复用历史调度结果（V2-4.8 `HandwrittenPattern 的缓存与参数化`、V2-4.9 `Compilation Cache and Runtime Selection`）。
5. **AxisCoalescing 预处理**：在搜索前合并可合并的轴（V2-4.3），缩小 $d_j$、降低搜索空间维度。

不采用：
- 完整 polyhedral 框架：NPU 异构性（cube intrinsic 的 NZ layout、固定 MAC 尺寸）超出仿射模型表达力。
- 在线 RL：编译期开销不可接受，且当前算子集启发式已足够。
- 全空间穷举：状态空间 $10^6$ 量级，单点代价评估若需 cost model 调用则总时间过长。

#### 3.3.7 V2 实现对应

- 轴空间预处理：V2-4.3 `AxisCoalescing`
- 调度问题构造：V2-4.4 `ScheduleProblem`（即 §3.3.2 的形式定义在工程层的实例化）
- Tiling 策略：V2-4.5 `TilingStrategy`
- 决策产物：V2-4.6 `ScheduleDecision`（即 $\sigma_j$）
- Structured Lowering：V2-4.7（将 $\sigma_j$ 物化到 IR）
- HandwrittenPattern 调度短路：V2-4.8
- Cache 与运行时选择：V2-4.9
- 出口验证：V2-4.10 第三层 Verifier

---

### 3.4 $P_4$ Realize：归约到 Register Allocation（区间图着色推广）

> *简要总结*
>
> **问题归类：寄存器分配（图着色）在多级存储 + 显式 DMA 下的推广**
>
> 调度结果给出每个 buffer 的生命周期区间和大小，需要决定：
>
> - **Placement**：每个 buffer 放到哪一级存储（L0A/B/C、L1、UB、GM）
> - **Offset**：在该存储级内部的物理偏移（生命周期不重叠的 buffer 可共享区域）
> - **DMA 计划**：跨存储级的搬运指令及其在流水线中的时序
>
> **核心难点 —— 与经典 register allocation 的本质差异：**
> 经典 CPU 寄存器分配中，spill 由 ISA 隐式触发，成本仅是带宽。NPU 上的搬运是**显式 DMA 指令**，必须排进流水线节拍，会和 cube/vector 单元争抢 issue slot。因此 placement 决策直接影响 §3.3 的流水线重叠。
>
> 此外还有 layout 约束：cube 单元的 L0A/B/C 只接受特定 layout（NZ 等），这破坏了 interval graph 着色的多项式可解性前提。
>
> **复杂度：** 在单一存储级 + 等大小情形下退化为区间图着色，多项式可解（Olariu 1991）。加入容量约束变 strip packing（NP-hard，无 PTAS）。加入多级存储 + layout + 显式 DMA 后整体 NP-hard。
>
> **解决方案（分阶段 greedy）：**
>
> 1. **Bufferization**：tensor SSA → 显式 buffer 集合，计算生命周期区间
> 2. **Placement**：按 locality（使用者距离）+ layout 兼容性 + `mustKeepOnChipValues` hint，从最近的 L0 开始自顶向下放置；放不下退到下一级
> 3. **Static Memory Planning**：每级内部用 First-Fit Decreasing 算 offset，生命周期不重叠的 buffer 复用
> 4. **Data Movement**：根据 placement 差异生成 DMA 指令并排进流水线
> 5. **Materialization + Verifier**：产出 `MemoryRealizationPlan` 并验证所有约束

#### 3.4.1 归约目标

$P_4$ 是 **Register Allocation via Graph Coloring** 在多级存储 + 显式 DMA 模型下的推广：

> 给定 buffer 集合 $\mathcal{B}$，每个 buffer 有生命周期区间 $[s_b, e_b]$ 和大小 $\text{sz}(b)$，将其放置到多级存储 $\mathcal{M} = \{\text{UB}, \text{L1}, \text{L0A}, \text{L0B}, \text{L0C}, \text{GM}\}$，并安排跨层 DMA 搬运计划，使得每级存储容量约束满足、生命周期不冲突的 buffer 可复用同一物理区域、总数据搬运代价最小。

经典基础：

- Chaitin (1982)：register allocation 归约为图着色 NP-hard。
- Poletto-Sarkar (1999)：linear scan / interval graph 着色，多项式可解的特殊情形。
- Belady (1966)：离线最优替换策略（MIN 算法）。
- Sethi-Ullman (1970)：表达式树最少寄存器数。

#### 3.4.2 形式定义

固定上游 $\omega_{<4}^\star$（含 schedule 决策 $\sigma$），对每个 kernel $K_j$ 求解：

$$
P_4^{(j)}:\quad
\begin{array}{ll}
\text{Input} & K_j,\; \sigma_j,\; H,\; \mathcal{B}_j = \{b_1, \dots, b_n\} \\
\text{Output} & \mu_j = \langle \pi_{\text{mem}}, \rho, \tau_{\text{dma}} \rangle \\
\text{Decision} & \omega_4^{(j)} \in \Omega_{\text{mem}}^{(j)}
\end{array}
$$

输出三元组 $\mu_j$ 含义：

- $\pi_{\text{mem}}: \mathcal{B}_j \to \mathcal{M}$：placement，每个 buffer 的归属存储级
- $\rho: \mathcal{B}_j \to \mathbb{Z}_{\geq 0}$：物理偏移（offset within memory level），允许多个生命周期不重叠的 buffer 共享同一偏移区域
- $\tau_{\text{dma}}$：DMA 搬运计划，序列 $\langle (b, m_{\text{src}}, m_{\text{dst}}, t) \rangle$，每条记录在时刻 $t$ 将 buffer $b$ 从 $m_{\text{src}}$ 搬到 $m_{\text{dst}}$

**约束**：

$$
\begin{aligned}
& \text{(capacity)} \quad \forall m \in \mathcal{M},\, \forall t.\;\; \sum_{b:\, \pi_{\text{mem}}(b)=m,\, t \in [s_b, e_b]} \text{sz}(b) \;\leq\; \text{Cap}_m(H) \\
& \text{(no-overlap)} \quad \forall b_1, b_2:\, \pi_{\text{mem}}(b_1) = \pi_{\text{mem}}(b_2) \wedge [s_{b_1}, e_{b_1}] \cap [s_{b_2}, e_{b_2}] \neq \emptyset \\
& \qquad\qquad \Longrightarrow [\rho(b_1), \rho(b_1)+\text{sz}(b_1)) \cap [\rho(b_2), \rho(b_2)+\text{sz}(b_2)) = \emptyset \\
& \text{(alignment)} \quad \rho(b) \equiv 0 \pmod{\text{align}(H, \pi_{\text{mem}}(b))} \\
& \text{(producer-locality)} \quad \text{producer}(b)\text{ 的执行单元可访问 } \pi_{\text{mem}}(b) \\
& \text{(consumer-locality)} \quad \forall \text{consumer } c \text{ of } b.\; \pi_{\text{mem}}(b) \in \text{readable}(c, H) \\
& \text{(layout)} \quad \pi_{\text{mem}}(b) = \text{L0A/L0B/L0C} \Longrightarrow \text{layout}(b) \in \text{supportedLayouts}(\pi_{\text{mem}}(b)) \\
& \text{(must-keep-on-chip)} \quad \forall b \in \text{mustKeepOnChipValues}.\; \pi_{\text{mem}}(b) \neq \text{GM}
\end{aligned}
$$

**目标函数（代理代价）**：

$$
C_4^{(j)}(\mu_j) \;=\; \sum_{(b, m_{\text{src}}, m_{\text{dst}}, t) \in \tau_{\text{dma}}} \text{dmaCost}(b, m_{\text{src}}, m_{\text{dst}}) \;+\; \kappa\cdot \text{fragLoss}(\rho)
$$

含义：

- $\text{dmaCost}$：根据 DMA 通道带宽与 burst 大小估算的搬运耗时。
- $\text{fragLoss}(\rho)$：内存碎片率（已分配但无法复用的"洞"），$\kappa$ 控制碎片的惩罚强度。

直觉：**最小化 DMA 总量 ≈ 最大化片上 buffer 复用 + 最优 placement**，与经典 register allocation 的目标（最少 spill）同构。

#### 3.4.3 复杂度

- **单一存储级 + 大小 1 的 buffer + 生命周期区间**：退化为 **interval graph coloring**，多项式可解（Olariu 1991, $O(n \log n)$）。
- **加入容量约束（buffer 大小 ≠ 1）**：变为 **strip packing** 子问题，NP-hard。
- **多级存储 + 显式 DMA 决策**：放置选择本身是离散变量，进一步耦合 placement 与 packing。
- **加 layout / locality 约束**：约束图不再是 interval graph，经典 Olariu 多项式算法失效。

整体 $P_4$ 是 NP-hard，且不存在多项式时间近似方案（PTAS 在 strip packing 上即已知不存在）。

#### 3.4.4 与经典 Register Allocation 的差异

| 维度 | 经典寄存器分配 | $P_4$ |
|---|---|---|
| 存储级 | 单一寄存器堆 | UB / L1 / L0A / L0B / L0C / GM 多级 |
| Buffer 大小 | 统一一字 | 张量大小不一，可达 MB 级 |
| 搬运 | spill/reload，由 ISA 隐式 | 显式 DMA，必须排进流水线 |
| Layout | 不存在 | L0A/B/C 各有 NZ/ZN 等专用 layout |
| 着色冲突图 | 区间图（理论上多项式） | 含 layout / locality 约束的推广，非 interval graph |
| 生命周期来源 | 寄存器变量活跃分析 | tile 调度产出的 buffer 区间（依赖 $\sigma_j$） |
| 替换策略 | LRU / Belady-MIN | 离线最优 + 多级层次决策（哪一级 spill 到哪一级） |

最关键差异：**显式 DMA 搬运**。经典 register allocation 中 spill 是隐式的（由 ISA 自动加载），代价在某种意义上是 free（只算带宽）；NPU 上 DMA 是**必须在调度中显式排进流水线的 op**，会和 cube/vector 单元抢占 issue slot，因此 placement 决策直接影响 §3.3 的流水线 overlap，形成 $P_3 \leftrightarrow P_4$ 的耦合（§2.3 S2）。

V2 采用的妥协：$P_3$ 在代理代价中预估 `spillRisk`（§3.3.2 中 $C_3$ 的 $\epsilon$ 项），让 $P_3$ 倾向于产生"易于 $P_4$ 实现"的 tile 形状，从而**前向**缓解这层耦合。

#### 3.4.5 求解策略家族

| 家族 | 代表方法 | 适用情形 |
|---|---|---|
| **精确 ILP** | 二次分配（QAP）公式化 | $n \leq$ 数十的小 kernel；工业规模 kernel 不可行 |
| **图着色启发式** | Chaitin-Briggs | 单一存储级；多级需大幅改造 |
| **Linear Scan** | Poletto-Sarkar | 生命周期形成 interval 时 $O(n \log n)$；layout 约束破坏前提 |
| **静态规划 + Greedy Packing** | First-Fit Decreasing / Best-Fit | 工程常用，简单且效果接近最优 |
| **离线最优替换** | Belady MIN | 单级 + 等大小 block；NPU 多级需推广 |
| **学习方法** | RL placement | 状态空间大但评估代价低时可行；NPU 上评估代价高，性价比一般 |

#### 3.4.6 本工程的选择

**Staged greedy + 多级 Belady 推广**：

1. **Bufferization**（V2-5.3）：把 tensor SSA 形式物化为 buffer 集合 $\mathcal{B}_j$，给出每个 buffer 的生命周期区间。
2. **Placement**（V2-5.4）：按 locality（哪条 op 访问）+ layout 兼容性 + `mustKeepOnChipValues` hint，将 buffer 自顶向下逐级放置（先尝试最近的 L0，失败则退到 UB，再失败到 GM）。
3. **Static Memory Planning**（V2-5.5）：在每级存储内部用 First-Fit Decreasing 算偏移 $\rho$，对生命周期不重叠的 buffer 复用偏移。
4. **Data Movement**（V2-5.6）：根据 placement 差异生成 $\tau_{\text{dma}}$，按调度时序排进流水线节拍。
5. **Materialization + Verifier**（V2-5.7）：产出最终 `MemoryRealizationPlan`，验证所有约束。

不采用：
- 完整 ILP：单 kernel 内 buffer 数可达数十到上百，ILP 求解时间不可接受。
- 在线 RL：编译期一次性决策，无需在线学习。
- 单存储级简化：必须显式建模 NPU 多级存储，否则模型与硬件脱节。

#### 3.4.7 V2 实现对应

- 流水线总览：V2-5.1 整体流水线与核心类
- Bufferization：V2-5.3
- Placement：V2-5.4
- 静态内存规划：V2-5.5
- 数据搬运：V2-5.6
- 物化与 Verifier：V2-5.7
- 决策产物：`BufferizedKernelIR` → `PlacementPlan` → `StaticMemoryPlan` → `MovementPlan` → `MemoryRealizationPlan`

---

### 3.5 $P_5$ Translate：归约到 DAG Instruction Selection（Pattern Covering）

> *简要总结*
>
> **问题归类：DAG 上的模式覆盖 (DAG Instruction Selection / Pattern Covering)**
>
> IR 上的每个节点需要被一条或多条目标指令覆盖。Pattern 库中每条规则包含：匹配的子图形状、适用条件 guard、发射的目标指令序列、估算代价。要求覆盖完整、guard 满足、可编码，总代价最小。
>
> **复杂度：**
>
> - 树上：Aho-Johnson 动态规划 $O(n)$ 最优
> - DAG 上：NP-hard（Ertl 1999，归约自 set cover）
>
> 实际中 $|V(D_j)|$ 通常 ≤ 数百，pattern 之间冲突有限（cube / vector / DMA 几乎不在节点上竞争），启发式可接近最优。
>
> **核心难点 —— 与经典指令选择的本质差异：**
> $P_5$ 不仅产出"IR → 指令"翻译，而是同时产出三类必须一致的工件：
>
> - **Kernel 侧**：AscendC 源码
> - **Host 侧**：tiling 计算函数（运行时根据实际 shape 算 tile 参数）
> - **Runtime Manifest**：kernel 元信息（参数布局、workspace 大小、launch 配置）
>
> 三者参数布局、workspace 大小、ABI 必须严格一致，这是经典指令选择不存在的**跨工件一致性约束**。
>
> **解决方案：**
>
> - **Pattern-driven rewriting**：用 MLIR DRR / C++ pattern 写规则，按"最具体优先"排序
> - **Greedy bottom-up 匹配**：从叶子向根传递最优覆盖（DAG 上是近似算法）
> - **HandwrittenPattern 短路**：命中高价值结构直接用预定义 emit 模板
> - **三工件联合产出**：Compute Lowering → AscendC 源码 + Kernel ABI + Host Tiling + Runtime Manifest 同步生成，由 V2-6.7 验证一致性

#### 3.5.1 归约目标

$P_5$ 是经典 **Instruction Selection via Tree/DAG Pattern Matching and Covering**：

> 给定一个 IR DAG 和一组目标指令的模式（pattern）库，将 IR 的每个节点用一条或多条目标指令"覆盖"，每个 pattern 有匹配条件和代价，使得总代价最小且覆盖完整。

经典文献：

- Aho-Johnson (1976)：树上指令选择，动态规划 $O(n)$ 最优。
- Aho-Ganapathi-Tjiang (1989)：bottom-up rewrite system (BURS) / `twig` / `iburg`。
- Ertl (1999)：DAG 上指令选择 NP-hard 的归约证明。
- Koes-Goldstein (2008)：近 - 最优 DAG 覆盖的实用启发式。

#### 3.5.2 形式定义

固定 $\omega_{<5}^\star$（kernel 划分、调度、内存方案都已确定），对每个 kernel $K_j$ 求解：

$$
P_5^{(j)}:\quad
\begin{array}{ll}
\text{Input} & \text{lowered IR DAG}\; D_j,\;\; \text{pattern library}\; \mathcal{L}(H) \\
\text{Output} & \chi_j:\text{ 覆盖方案，将 } D_j \text{ 的每个节点分配到某条 pattern} \\
\text{Decision} & \omega_5^{(j)} \in \Omega_{\text{cg}}^{(j)}
\end{array}
$$

**Pattern 库** $\mathcal{L}(H)$ 中每条 pattern $p$ 是四元组：

$$
p = \langle \text{shape}_p,\; \text{guard}_p,\; \text{emit}_p,\; \text{cost}_p \rangle
$$

- $\text{shape}_p$：IR 子图模板（如 `linalg.matmul` 配某种 layout/dtype）
- $\text{guard}_p$：适用条件（如"dtype = fp16 且 layout = NZ 且 K-轴对齐 16"）
- $\text{emit}_p$：发射的目标指令序列（AscendC / cube intrinsic / vector intrinsic / DMA primitive）
- $\text{cost}_p$：估算执行代价

**覆盖** $\chi_j$ 满足：

$$
\begin{aligned}
& \text{(complete)} \quad \bigcup_{p \in \chi_j} \text{nodes}(p) = V(D_j) \\
& \text{(disjoint or overlap-allowed)} \quad \text{pattern 之间节点不重叠（基本形式），或允许 overlap 共享子节点（DAG 推广形式）} \\
& \text{(guard-satisfied)} \quad \forall p \in \chi_j.\; \text{guard}_p(\text{context}) = \text{true} \\
& \text{(encodable)} \quad \text{emit}_p \text{ 中每条目标指令在 } H \text{ 上可编码（寄存器数、立即数范围、操作数模式均合法）}
\end{aligned}
$$

**目标函数**：

$$
C_5^{(j)}(\chi_j) \;=\; \sum_{p \in \chi_j} \text{cost}_p
$$

#### 3.5.3 复杂度

- **树上** instruction selection（Aho-Johnson）：动态规划 $O(n)$ 最优。
- **DAG 上**（共享子表达式）：NP-hard（Ertl 1999；归约自 set cover）。
- **加 guard / 加状态**（如寄存器类约束）：更复杂，但实用启发式（greedy bottom-up + tie-breaking）在多数情况下接近最优。

NPU 上 $D_j$ 是 DAG（buffer 复用、共享中间结果），原则上是 NP-hard 情形。但实践中：
- $|V(D_j)|$ 通常 $\leq 数百$；
- pattern 之间冲突有限（cube/vector/DMA 几乎不在节点上竞争）；
- 因此启发式可达接近最优。

#### 3.5.4 与经典 Instruction Selection 的差异

| 维度 | 经典指令选择 | $P_5$ |
|---|---|---|
| IR 形态 | 表达式树 / SSA DAG | MLIR 多方言（含 region、blockArg、attr） |
| 目标指令集 | 同构 ISA（RISC/CISC） | 异构：AscendC 高层 op + cube intrinsic + vector intrinsic + DMA primitive |
| Pattern 来源 | 编译器手写或 BURS 表 | 手写 + `HandwrittenPattern` registry（高价值结构特例） |
| 寄存器类 | 同构 GPR | 多级 buffer（已由 $P_4$ 决定，本层不再选择） |
| ABI | 隐式调用约定 | 显式 kernel ABI（host 参数布局、tiling args、workspace） |
| Host 配合 | 不存在 | 必须同时产出 host-side tiling 计算和 runtime manifest |
| 验证 | ISA 编码合法 | IR verifier + AscendC 源代码可编译 + runtime manifest 一致性 |

最关键差异：**$P_5$ 不是单一"IR → 指令"的翻译，而是同时产出三类 artifact**：

1. **Kernel 侧**：AscendC 源码（被 CCE 编译器进一步编为二进制）。
2. **Host 侧**：tiling 计算函数（运行时根据实际 shape 算出 tile 参数）。
3. **Runtime Manifest**：kernel 元信息（参数布局、workspace 大小、launch 配置）。

这三者必须**一致**：runtime 传给 kernel 的参数布局必须与 AscendC 源码的声明一致，与 manifest 中记录的一致。这是 $\mathcal{V}_{\text{enc}}$（指令可编码）之外的**跨工件一致性约束**，经典 instruction selection 不存在。

#### 3.5.5 求解策略家族

| 家族 | 代表方法 | 适用情形 |
|---|---|---|
| **树 DP（Aho-Johnson）** | `iburg` / `BURS` | IR 是树时最优；DAG 上不直接适用 |
| **DAG Greedy Bottom-Up** | LLVM `SelectionDAG` 风格 | 工程主流，接近最优 |
| **Pattern DSL + 优先级** | TableGen / MLIR `Pattern` | 声明式，易扩展 |
| **精确 ILP** | DAG 上小规模可行 | 工业规模 kernel 不可行 |
| **学习方法** | 不常见 | 工程上无优势，pattern 数量有限 |

#### 3.5.6 本工程的选择

**Pattern-driven rewriting + 显式优先级**：

1. **Compute Lowering**（V2-6.3）：linalg / arith / vector op → AscendC compute primitive。Pattern 用 MLIR DRR / C++ pattern 写，按"最具体优先"排序。
2. **Kernel ABI Translation**（V2-6.4）：根据 $P_4$ 的 `MemoryRealizationPlan` 生成 kernel 参数列表、workspace 布局。
3. **AscendC Source Translation**（V2-6.5）：IR → AscendC 文本源码。
4. **Host Tiling / Runtime Manifest**（V2-6.6）：生成 host 侧 tiling 函数 + manifest。
5. **HandwrittenPattern 短路**：若 kernel 是 `HandwrittenPattern` 命中（从 $P_2$ 一路携带的标记），直接使用预定义的 emit 模板，跳过通用 pattern 匹配。

不采用：
- ILP：pattern 数 × DAG 大小的乘积空间过大。
- 学习方法：pattern 集合可枚举，无学习必要。
- 单一通用 lowering：不同 op 族对应的目标指令差异大（cube vs vector），必须 pattern 化。

#### 3.5.7 V2 实现对应

- Compute Lowering：V2-6.3
- Kernel ABI：V2-6.4
- AscendC 源码翻译：V2-6.5
- Host tiling / Runtime manifest：V2-6.6
- 层级验证：V2-6.7
- pass 顺序与依赖：V2-6.8
- 社区能力复用边界：V2-6.9

---

## 4. 子问题间的接口（Contract）

### 4.1 接口的形式

子问题 $P_i$ 的输出严格等于 $P_{i+1}$ 的输入，且每对相邻层之间存在**双重接口**：

1. **IR snapshot**：当前阶段的 MLIR 模块状态，是主要的数据载体。
2. **附加结构（side-channel artifact）**：本阶段的决策摘要，作为 hint 或 contract 传给下游。

形式化：

$$
\text{Output}(P_i) \;=\; \langle \text{IR}_i,\; \mathcal{A}_i,\; \mathcal{H}_i \rangle
$$

- $\text{IR}_i$：阶段 $i$ 结束时的 IR 状态
- $\mathcal{A}_i$：本阶段产出的**契约性**附加结构（下游必须读取并遵守）
- $\mathcal{H}_i$：本阶段产出的**建议性** hint（下游可参考可忽略，对应 §2.4(2)）

### 4.2 五对接口

| 接口 | $\mathcal{A}_i$（契约） | $\mathcal{H}_i$（hint） |
|---|---|---|
| $P_1 \to P_2$ | 规范化后的 IR + 入口许可方言集 + `AscendSymbolConstraintAttr` | （无 —— L1 是平凡映射） |
| $P_2 \to P_3$ | `KernelPattern` / `KernelPartitionDecision` / `scheduleContract`（含 `tileableAxes`、`requiredReductionAxes`、`layoutConstraints`、`mustKeepOnChipValues`、`templateFamilies`、`dynamicGuardSet`） | 轴优先级建议、handwritten pattern 命中标记 |
| $P_3 \to P_4$ | `ScheduleDecision`（$\sigma_j$：tile / order / binding / pipeline depth） | spillRisk 评估、可选替代 tile 形状 |
| $P_4 \to P_5$ | `MemoryRealizationPlan`（placement + offset + DMA 计划 + layout） | DMA 通道占用模式、对齐结构 |
| $P_5 \to$ 运行时 | AscendC 源码 + Kernel ABI + Runtime Manifest + Host Tiling 函数 | （无 —— $P_5$ 终结） |

### 4.3 不变量（Invariants）

每层结束时，IR 必须满足**层间不变量**。这些不变量是 §2.5 中 $\bigwedge_i \mathcal{V}_i \Rightarrow \mathcal{V}$ 推导的具体形式。

**$I_1$（Normalize 出口）**：
- IR 中只剩许可方言集内的 op
- 所有 op 满足 V2-2.3 表中的规范形
- 无 `ascend.unknown_origin` 残留
- `AscendSymbolConstraintAttr` 已就位

**$I_2$（Kernelize 出口）**：
- 每个 op 归属唯一 kernel（`FallbackSingleOpPattern` 兜底）
- 跨 kernel 的 quotient graph 是 DAG
- 每个 kernel 携带合法的 `scheduleContract`
- 每个 kernel 在 `TemplateRegistry` 中存在对应 template family

**$I_3$（Schedule 出口）**：
- 每个 kernel 携带 `ScheduleDecision`，满足该 kernel 的 `scheduleContract`
- 所有 tile 形状满足对齐与容量预算
- Structured Lowering 已物化到 `scf` / `affine` 形式
- `dynamicGuardSet` 已转换为运行时 guard 或编译期消除

**$I_4$（Realize 出口）**：
- 所有 tensor SSA 已 buffer 化
- 每个 buffer 有唯一 placement + offset
- 容量约束全部满足
- DMA 搬运排进了流水线节拍
- Layout 与 cube intrinsic 要求一致（为 $P_5$ 的 emit 做铺垫）

**$I_5$（Translate 出口）**：
- AscendC 源码可被 CCE 编译器编译通过
- Kernel ABI 与 Runtime Manifest 与 Host Tiling 函数三者一致
- 无残留高层方言 op

### 4.4 验证职责的划分

§1.2 中合法性谓词 $\mathcal{V} = \mathcal{V}_{\text{sem}} \wedge \mathcal{V}_{\text{cap}} \wedge \mathcal{V}_{\text{align}} \wedge \mathcal{V}_{\text{life}} \wedge \mathcal{V}_{\text{enc}}$ 的具体分配：

| 子谓词 | 主要负责层 | 协助层 |
|---|---|---|
| $\mathcal{V}_{\text{sem}}$ | $P_1$（规范化保语义） | 全部层的 verifier |
| $\mathcal{V}_{\text{cap}}$ | $P_3$ + $P_4$ | $P_2$ 提供 `scheduleContract` 约束 |
| $\mathcal{V}_{\text{align}}$ | $P_4$ + $P_5$ | $P_3$ 保证 tile 形状对齐 |
| $\mathcal{V}_{\text{life}}$ | $P_4$ | — |
| $\mathcal{V}_{\text{enc}}$ | $P_5$ | — |

### 4.5 不变量的传递与失败处理

每层结束时**必须**通过其 verifier 才能进入下一层；不通过则按 V2-7.4 Diagnostics 规范报告失败：

```
stage = <P_i 对应层>
objectId = <失败对象的稳定 ID>
reasonKind = <失败原因枚举>
message = <人类可读描述>
isRecoverable = <bool>
fallbackTaken = <bool>
```

由 §2.4(3) 的"层内候选集 + 早失败"原则，跨层不回退：

- 若 $I_i$ 失败且 $P_i$ 仍有未尝试的候选 $\omega_i^{(k)}$，则在 **本层内** 切换候选；
- 若 $P_i$ 候选集穷尽仍不满足 $I_i$，则编译终止（除非有显式兜底契约如 `FallbackSingleOpPattern`）；
- $P_j (j > i)$ 不向 $P_i$ 请求重选。

---

## 5. 总结：本建模的工程意义

回到 §1.6 给出的四条工程结论：

| 结论 | 在本文的体现 |
|---|---|
| **(C1) 必须分解** | §1.3 NP-hardness + §1.4 状态空间爆炸 + §2.1 fixed-prefix 串行分解 |
| **(C2) 必须分层近似** | §2.1 每层 $\arg\min_{\omega_i}$ on $\Omega_i(\omega_{<i})$ + §3.x 每层的代理代价 $C_i$ |
| **(C3) 必须承认 sub-optimality** | §2.3 四类被牺牲耦合 (S1–S4) + §2.4 三类前向缓解手段 |
| **(C4) 必须保留每层可替换性** | §3.x 每层归约到独立的经典问题家族 + §3.x.5 各列出多种求解策略家族 |

由此本建模回答了原始问题"如何为 Ascend NPU 开发一款基于 MLIR 的 AI 编译器"中的"What/Why"部分：

- **What**：求解 $P = \min_\omega C(\omega; G, H)$ s.t. $\mathcal{V}$，其决策空间是 5 类决策的笛卡尔积，目标是端到端 latency。
- **Why 分五层**：每个子空间对应一个经典 CS 问题家族，且子问题之间是单向依赖（§2.2），五层是天然解耦点；分层不是最优解，而是工程可行的近似（§2.3）。
- **Why 每层这样设计**：每层归约到经典问题（§3）后，其代理代价、约束、求解策略都有半个世纪的文献基础；本工程的选择（启发式 / 模板 / cache / 兜底契约）在该经典问题家族的策略家族中做了明确取舍。

具体的 pass 实现、IR 结构、源码细节在 `Ascend-MLIR-Detailed-Implementation-V2.zh.md` 中给出，对应关系见各节末尾的 "V2 实现对应" 小节。
