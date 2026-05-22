# BERT Bring-up: 用开源 BERT 压整条 pipeline

**日期**: 2026-05-22
**分支/worktree**: `bert-bringup`（基于 `dev-network` @ `5f2c754e`，worktree 在 `.claude/worktrees/bert-bringup`）

## 目标

拿一个**现成开源 transformer（BERT-base 真结构）**，从前端一路压过整个 project 的 pipeline，
最终在 **camodel 仿真**上跑出数值并跟 PyTorch 参考对 `max_diff`。

心态是**增量 bring-up（"先试能不能跑"）**：用极小配置先把整条路跑通，再逐步放大。不追求一上来就整网 12 层。

## 为什么基于 dev-network

BERT layer 的 attention / layernorm 要走 aclnn。`RecognizeAttentionPass.cpp` 与
`RecognizeLayerNormPass.cpp` **只在 dev-network 上，develop 没有**。matmul→aclnn、CoordEmitter
host-gen、network_runner phase 改进也都在 dev-network。因此基底只能是 dev-network。

dev-network 本身被 `encoder-robustness` worktree 占用（并行工作，正在修 transpose codegen），
所以我们另开 `bert-bringup` 分支 + worktree 与其隔离。BERT bring-up 主要是**新增代码**
（前端 helper + 新 example + 驱动 network_runner），预期不动到那条线在改的文件。

## 方案：HF BertLayer（极小配置）+ network_runner

选定 **方案 A**：用 HuggingFace 现成的 `BertLayer` / `BertEncoder`，
- 极小配置：`hidden=64, num_heads=1, intermediate=256(=4×hidden), seq_len=8, num_layers=1`（真 BERT 结构，仅缩规模）
- **直接喂 `hidden_states`**，跳过 embedding 查表（gather/整数索引前端大概率不支持，会卡第一步）和 attention mask（无 padding 时 mask 为全 0 加性项，可省）
- PyTorch 参考 = 同一个 `BertLayer.eval()` 的输出

否决的备选：
- **B（完整 BertModel 含 embedding）**：几乎必然立刻撞上不支持算子，bring-up 卡死。
- **C（手写 mini-BERT）**：用户要现成开源网络，否决。

### 数据流

```
HF BertLayer(tiny config, eval)
   │
   ├─ torch.export ──> step0_linalg.mlir        （复用 python/torch/torch2linalg 的 torch_to_linalg）
   ├─ 随机 hidden_states ──> input_0.npy
   └─ model(hidden_states) ──> expected_0.npy
                    │
        network_runner.py --input-linalg step0_linalg.mlir
                          --inputs input_0.npy --expected expected_0.npy
                          --backend sim --workdir <wd>
                    │
   phase1  auto-fuse-group-analysis + group-outline → 切 kernel（matmul→aclnn / attention→FA / LN→aclnn / vector→AscendC）
   phase2-3 codegen + compile（aclnn matmul + AscendC vector kernel）
   phase4   autotune
   phase5   camodel sim run → 跟 expected_0.npy 对 max_diff（atol/rtol）
```

## 组件

1. **`examples/bert-e2e/export_bert.py`**（新增）
   - 构造 `transformers` 的 `BertConfig`（tiny）+ 取 `BertLayer`（或 `BertEncoder`）
   - 生成 `step0_linalg.mlir`、`input_0.npy`、`expected_0.npy`
   - 配置项（hidden/heads/seq/layers/dtype）走命令行参数，方便后续放大
   - dtype 默认 fp16；精度过不去时回退 fp32
2. **`examples/bert-e2e/run.sh`**（新增）
   - `source examples/env.sh`（或 env_gser.sh）设置仿真 LD_LIBRARY_PATH
   - 调 `export_bert.py` 再调 `network_runner.py`，带 `--max-phase` 参数
3. **`examples/bert-e2e/README.md`**（新增）：怎么跑、各 bring-up 阶段、已知限制

不改动 dev-network 现有 pass / runner 代码——除非 bring-up 暴露出必须修的 bug（届时单独评估，尽量小改）。

## Bring-up 阶段（验证点）

| 阶段 | 命令 | 验证 |
|------|------|------|
| S1 前端→linalg | `export_bert.py` 单跑 | 生成 step0_linalg.mlir，无 export 报错；人工看出现哪些算子、有无不支持 |
| S2 group 切分 | `network_runner --max-phase 1` | group-analysis/outline 不崩，看切出哪些 kernel |
| S3 codegen 干净 | `network_runner --max-phase 3` | aclnn + AscendC codegen 全部编译通过 |
| S4 仿真数值 | `network_runner --max-phase 5 --backend sim` | camodel 出输出，`max_diff` 在容差内 |
| S5 放大 | 调大 config 重跑 S1–S4 | 2 层 / seq 32 / 多头逐步通过 |

每个阶段是独立可验证的 gate；卡在哪一步就地诊断（systematic-debugging）。

## 构建 / 二进制（待用户拍板）

新 worktree 没有 `build/`。两个选择：
- **(a) 复用 sibling 构建**：env 指向 `encoder-robustness/build/bin/*`（同 commit，含 recognize passes）。
  优点：立即可探。缺点：与并行工作耦合，对方重建会变；只读执行风险较低。
- **(b) 本 worktree 自建**：完全隔离，但 afir-opt 从头编多小时。

**推荐**：先用 (a) 快速验 S1–S3，同时后台起 (b) 自建；S4 仿真验证用自建二进制以保证隔离与可复现。
（请在评审时确认。）

## 风险 / 开放点

- **softmax**：BERT attention = `softmax(QKᵀ/√d)·V`。`recognize-attention` 折 `bmm→exp→bmm→FlashAttentionScore`；
  需确认 HF BERT 分解出的 softmax（max-sub + exp + sum + div）能被该 pass 匹配。**最大不确定点。**
- **GELU**：HF BERT 默认 `gelu`（erf 版）。需确认 vector 路径支持 erf-GELU，还是要切 tanh 近似 / 是否被 codegen 接受。
- **dtype 精度**：fp16 在 transformer 上累积误差可能偏大；atol/rtol 需放宽，必要时 fp32。
- **配置过小的副作用**：num_heads=1 / hidden=64 可能触发 unit-dim、退化 reduce 等边界（项目里已知雷区）。若卡住，调成 hidden=128 / 2 头再试。
- **camodel 速度**：即便 tiny，多 kernel 仿真仍可能慢；S4 先期望"能出数"，不追速度。

## 成功标准

- **最小成功**：tiny BertLayer 跑通 S1–S4，camodel 输出 vs PyTorch `max_diff` 在容差内（fp16 放宽容差）。
- **进一步**：S5 放大到 2 层 / seq 32 / 多头仍数值正确。
- 全程不破坏 dev-network / 并行 encoder 工作。
