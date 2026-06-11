# GPT-2 真机生成 (Phase 1) 设计文档

- 日期: 2026-06-02
- 状态: 设计待评审
- 范围: **仅 Phase 1** —— 在真机 910C 上用真实权重的 GPT-2 small 做交互式逐 token 生成,产出连贯英文。agent 壳、Qwen 支持均不在本文范围。

## 1. 目标与背景

### 1.1 总目标
做一个**工具链 demo / 炫技**:证明 "我们这套 MLIR→910C 的编译栈能端到端跑一个真实 LLM,并在真机上交互式逐 token 生成连贯文本"。模型本身不需要好用,核心价值在于 **pipeline 打通 + 真机部署 + host 生成编排** 这件事本身。

### 1.2 为什么是 GPT-2(而非 Qwen)
分阶段、风险隔离的决策(见对话):
- GPT-2 small 架构与现有 `MiniGPT` (`examples/gpt2-e2e/export_gpt2.py`) 完全匹配(pre-LN + GELU + learned pos-emb),现有 recognizer(LayerNorm / Attention / Embedding)可复用,**第一阶段建模工作几乎为零**。
- GPT-2 small 是 base 补全模型(非 instruct),做不了真 agent —— 但 Phase 1 不要 agent,只要"真机能跑真 LLM 并连贯生成"。
- "生成循环 + 真机部署 + host 编排" 这套基础设施 **两种模型通用、可复用**。Qwen 需要的 RMSNorm / RoPE / SwiGLU / GQA 四块编译工作作为隔离的后续阶段,视 Phase 1 结果再启动。

### 1.3 现状基线
- 现有 `examples/gpt2-e2e/` 跑的是 **随机权重、tiny(12层/2头/n_embd=64/vocab=1024/seq=8)、静态 shape** 的 MiniGPT,**仅 sim 验证**(camodel,max_diff=8.3e-7)。输出是乱码。
- 真机基建已就绪:`--backend npu`(phase 1-4 sim、phase-5 device)已在 **BERT tiny 全网真机 PASS(max_diff=1.407e-5)**、two-elewise(max_diff=0)上验证。
- aclnn device 路径(`aclnnMatmul/BatchMatMul/LayerNorm/Permute/FlashAttentionScore`)已 HW 验证 7/7;混合内存 bug 已修(host-in/host-out staging,`085ad7a2`)。

## 2. 已查证的关键风险(基于当前 HEAD 代码,非 memory 旧态)

### 2.1 ⚠️ Masked-FA device 路径缺失 ——「sim 正确 / 真机静默算错」(头号风险)
`lib/Runtime/AclnnOps.cpp` `run_FlashAttentionScore` (L328-402):
- 函数签名已收到 `TensorInfo mask`(L330),mask 已 plumb 到 runtime。
- **host 路径(sim)** L336 `sdpa_cpu(q,k,v,mask,out)` —— **正确应用 mask**。
- **device 路径(真机)** L379 `attenMaskOptional=nullptr` 写死 + `preTokens/nextTokens=65536`(全双向)。→ **真机上 FA 忽略因果 mask,算成双向注意力**。

后果:**GPT-2 在 sim 上会过(sdpa_cpu 应用了 mask),一上真机就因 attention 变双向而输出错乱**,且 camodel 看不见。这是典型的 encoder 式 "sim-pass/real-fail" 陷阱,必须在 Phase 1 修。

修法(二选一,均需真机验证):
- (a) **通用**:将 `mask` 暂存上 device,作为 `attenMaskOptional` 传入。
- (b) **纯因果优化**:用 `sparseMode` + `nextTokens=0` 让 aclnn 自身做因果,免 mask tensor。
推荐先做 (a)(与 sim 的 sdpa_cpu 语义一致、最易对齐验证),(b) 作为后续优化。

### 2.2 ⚠️ Embedding device 路径缺失
`run_Embedding` (L555-558):`if (!g_host_mode)` 直接 `fprintf(stderr, "device path not implemented")` 退出。→ 真机上 embedding 会 fail。
修法(二选一):
- (a) **简单**:让 embedding 强制留在 host(embedding lookup 放 host 很常见,不丢 demo 分)。
- (b) 实现 device 版(`aclnnEmbedding` 或等价 gather)。
推荐 (a) 起步。

### 2.3 「BERT 过、encoder 没过」的先例
更复杂的 encoder 在真机仍卡住:前端 buffer 接线读到未初始化数据(garbage)+ tiling 相关真机 kernel fault(wild scalar GM 地址 507035 / trap 507034),均 camodel 不可见。GPT-2 复杂度介于 BERT 与 encoder 之间 —— **真机能否干净跑通是 Phase 1 真正要赌的点**。

### 2.4 「编一次、跑多次」基础设施缺失
现有 `--backend npu` 是**一次性**(每次从 phase-1 重头跑)。逐 token 生成需要 **编译一次 device binary、host 端循环复用、每步喂新 input_ids**。这是当前没有的新基建,也正是本阶段要交付的可复用核心。

### 2.5 其它已知点
- **autotuner 在此网会崩** → 用 default tilings(tiny 上验过,124M 需复验)。
- **gelu_new vs erf-gelu**:HF GPT-2 用 tanh 近似的 gelu_new,我们用 erf-gelu,有微小数值差,demo 可接受(或后续对齐为 tanh 近似)。
- **vocab 50257**:lm_head matmul `[N,768]×[768,50257]` 与 embedding 表 `50257×768` 的 shape/显存需在真机复核。

## 3. 架构与组件

### 3.1 生成策略:固定窗口 + 全量重算(无 KV-cache)
固定 `seq=N`(默认 128)。每生成一个 token:host 把整个 `[1,N]` 前向**重跑一遍** → 取当前位置 logits → 采样(先 greedy)→ 写回下一格 → 重跑。因果 mask 天然处理未填充位。

理由:**完美贴合静态 shape 流水线** —— 只编译一个 binary、每步复用、不重编译;把全部风险压到"真机能否跑这一个前向"。KV-cache 需动态 shape(撞符号化 shape 那条线),Phase 1 不做。

### 3.2 组件拆分

1. **权重移植(host / PyTorch)**
   - `MiniGPT` 实例化为 GPT-2 small config:n_layer=12, n_head=12, n_embd=768, vocab=50257, seq=N。
   - `load_state_dict` key 重映射:HF `Conv1D` 转置布局 → 我们的 `nn.Linear`;weight-tied lm_head。
   - 输入/接口:复用 `export_gpt2.py` 路径(buffer→函数 arg,named_buffers 顺序)。
   - 边界:单元 = "给定 prompt token ids,PyTorch 产出 logits";可独立验证(关 0)。

2. **静态前向编译(编译栈)**
   - `export_gpt2.py` → linalg → 现有编译栈,固定 `[1,N]`。复用现有 recognizer。
   - 用 default tilings(`NETWORK_RUNNER_SKIP_AUTOTUNE=1` 或等价),规避 autotuner 崩溃。
   - 边界:单元 = "linalg → 可执行 device binary";依赖现有 pipeline。

3. **host 生成循环 +「编一次跑多次」(扩展 network_runner)**
   - 一次性 build:phases 1-4 + phase-5 link,产出可复用的 device 可执行体/session。
   - 循环:tokenize(GPT-2 BPE)→ `while`{ 复用 session 跑一次 `[1,N]` 前向 → 取最后真实 token logits → greedy 采样 → 追加 → 判 EOS } → detokenize。
   - 采样、tokenizer 全在 host。
   - 边界:单元 = "持有一个已编译 session,输入 ids → 输出 logits";接口清晰、可独立测。

4. **device 路径补齐(runtime)**
   - **masked-causal-FA**(§2.1 修法 a):`run_FlashAttentionScore` device 分支接 `attenMaskOptional`。
   - **embedding**(§2.2 修法 a):Phase 1 让 embedding 留在 host。
   - 边界:改 `AclnnOps.cpp`,host 路径(sim 参考)语义不变。

5. **真机部署 + 逐关验证**
   - 走 `scripts/sync-and-submit.sh` 或 `.claude/npu-relay/submit.sh`,device 7,`source examples/env_gser.sh`。
   - 遵守 §5 真机安全铁律。

### 3.3 数据流
```
prompt(str)
  └─host─ tokenize ─► input_ids[1,N] (pad/置位)
              │  ┌────────────── 生成循环(host)──────────────┐
              └─►│ 复用 device session: forward([1,N]) ─► logits[1,N,vocab]
                 │   take logits[pos] ─► greedy ─► next_id ─► 写回 ids[pos+1]
                 │   pos++; EOS? ──no──┘                              │
                 └──────────────────────yes──────────────────────────┘
  ◄─host─ detokenize(ids) ─► 连贯英文
```
device 每步只做一次静态前向;生成编排全在 host。

## 4. 成功判据(分三关,逐关验证)

- **关 0 / PyTorch golden**:MiniGPT 装真实 GPT-2 small 权重,PyTorch 里对给定 prompt 产出连贯英文续写(肉眼 + 与 HF 官方 GPT-2 logits 对齐)。
- **关 1 / sim**:124M 静态前向 `[1,N]` 过编译栈,**sim logits 对齐 PyTorch**(max_diff 小,量级参考 BERT 1e-5);host 生成循环在 sim 上吐连贯英文。
- **关 2 / 真机**:同一前向 + 生成循环上 device 7,**masked-FA 修复后输出连贯英文**(关键:验证 attention 不再退化为双向)。

每关绿了再进下一关;先 sim 全绿再上真机(风险隔离)。

## 5. 真机操作安全铁律(防误杀)

1. 任何 destructive 命令(`kill`/`pkill`/`docker rm`/`docker kill`/`docker stop`/`rm -rf`):**先说清杀什么、可能误伤什么、风险,等确认再跑**。唯一豁免 = 杀本 session 自己刚启、按精确 PID 的 job。
2. 绝不 `docker exec` 进别人容器、grep 宿主可见进程判归属(共享 PID namespace 会匹配所有人)。
3. 只读认领"我的"容器:`docker inspect` Mounts/Env 命中 `/data/gser` 才算我的;`/dev/davinci7` 与 `-v /data:/data` 不是归属证据。杀只 `docker stop <我的确切ID>`(优先 stop 让 EXIT trap 收 plog),绝不批杀、绝不按 builder image 批杀。
4. 每次跑加 timeout + 监控;本地 timeout ≠ 远端容器被杀,超时后精确清理自己的孤儿容器。
5. 失败时把 `logs/`(含 `plog/plog-errorStr.txt`)、manifest、出错 kernel 的 mlir/cpp 拉到**仓库外** `/tmp/npu-real-logs/<date>/<case>/`,绝不放 worktree(会被 sync)。

## 6. 工作分解与风险

| # | 任务 | 风险 |
|---|---|---|
| 1 | 权重移植:MiniGPT→GPT-2 small config + HF 权重重映射;gelu 差异 | 低 |
| 2 | 静态前向跑通 sim(124M、vocab 50k);default tilings | 中 |
| 3 | host 生成循环 +「编一次跑多次」device 复用(扩展 network_runner) | 中-高(新基建) |
| 4 | device 路径补齐:masked-causal-FA(必做)+ embedding(放 host) | 高(全新,camodel 看不见) |
| 5 | 真机部署 + 逐关验证(sync-and-submit / npu-relay,device 7) | 高(encoder 先例) |

## 7. 非目标(本阶段明确不做)
- agent 壳(ReAct / tool-calling)。
- Qwen 支持(RMSNorm / RoPE / SwiGLU / GQA)。
- KV-cache / 动态 shape / 变长序列。
- temperature / top-k 采样(先 greedy;后续易加)。
- autotuner 在此网的崩溃修复(用 default tilings 绕过)。
- embedding device 实现(Phase 1 放 host)。
- gelu_new 精确对齐(接受 erf-gelu 微小数值差)。

## 8. 开放问题
- 窗口 N 默认 128 是否够 demo(prompt + 生成长度)?可调。
- masked-FA 修法选 (a) 通用 attenMask 还是直接上 (b) sparseMode 因果?设计选 (a) 起步。
- 关 2 真机若复现 encoder 式前端 garbage / tiling fault,如何分流(codegen 另案 vs 本阶段范围)?

## 9. 相关引用
- 现有模型:`examples/gpt2-e2e/export_gpt2.py`
- runtime:`lib/Runtime/AclnnOps.cpp`(run_FlashAttentionScore L328 / run_Embedding L555)
- 真机基建:`scripts/sync-and-submit.sh`、`.claude/npu-relay/`、`examples/env_gser.sh`
- 先例 memory:BERT 真机 PASS、network-runner npu backend、real-npu aclnn direct、real-npu test checklist
