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
