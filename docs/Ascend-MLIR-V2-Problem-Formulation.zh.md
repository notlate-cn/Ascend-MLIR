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
