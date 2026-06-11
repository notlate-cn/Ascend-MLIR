# GPT-2 真机生成 (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在真机 910C (device 7) 上,用真实权重的 GPT-2 small 做 host 端逐 token 生成,产出连贯英文 —— 证明 MLIR→910C 编译栈能端到端跑真实 LLM 并交互。

**Architecture:** 固定窗口 `[1,N]` 静态前向(无 KV-cache),每 token 重跑整网。embedding lookup 提到 host(网络从 `hidden_states[1,N,768]` 开始),sidestep 缺失的 embedding device 路径。因果 attention 需修 `run_FlashAttentionScore` 的 device 分支(当前写死 `attenMask=nullptr` → 真机退化为双向)。生成循环 = 编译一次 device binary,host driver 在远端 dev host 上循环调用同一 binary。

**Tech Stack:** PyTorch + HuggingFace `transformers`(取官方 GPT-2 权重做 golden)、torch-mlir(`torch_to_linalg`)、`python/network_runner.py`(5 阶段 pipeline)、`lib/Runtime/AclnnOps.cpp`(aclnn device 路径)、CANN 9.1.0 aclnn、真机经 `scripts/sync-and-submit.sh` / `.claude/npu-relay/`。

**设计文档:** `docs/superpowers/specs/2026-06-02-gpt2-real-npu-generation-design.md`

**验证三关:** 关0=PyTorch golden(vs HF 官方);关1=sim(logits 对齐 + 连贯生成);关2=真机(masked-FA 修复后连贯生成)。每关绿了再进下一关。

---

## File Structure

| 文件 | 责任 | 动作 |
|---|---|---|
| `examples/gpt2-e2e/port_gpt2_weights.py` | 把 HF 官方 GPT-2 权重灌进我们的 MiniGPT,做 golden + 导出 npy | Create |
| `examples/gpt2-e2e/export_gpt2.py` | 模型定义;新增 `--from-hidden`(embedding 提到 host)与权重加载 | Modify |
| `examples/gpt2-e2e/generate.py` | host 生成 driver:tokenize → 循环调 binary → argmax → detokenize | Create |
| `examples/gpt2-e2e/run_gpt2_real.sh` | 真机:build once + 在远端 host 跑 generate.py | Create |
| `lib/Runtime/AclnnOps.cpp` | `run_FlashAttentionScore` device 分支接 causal attenMask | Modify (L328-411) |
| `test/tools/runtime/test_masked_fa_device.cpp` | device 单测:causal FA vs sdpa_cpu | Create |

> 约定:`$REPO=/home/gser/code/Ascend-MLIR`。真机 dev host = `ssh -p 141 root@113.46.10.114`,只用 device 7,只碰 `/data/gser`。下文真机命令默认在已 `source examples/env_gser.sh` 后,经 `scripts/sync-and-submit.sh` 或 `.claude/npu-relay/submit.sh` 提交。

---

## Task 1: PyTorch golden —— 真实 GPT-2 权重灌进 MiniGPT,对齐 HF 官方

**Files:**
- Create: `examples/gpt2-e2e/port_gpt2_weights.py`
- Modify: `examples/gpt2-e2e/export_gpt2.py`(MLP gelu 改 tanh 近似,匹配 HF gelu_new)

**前置:** `pip show transformers` 确认可用;无网络时需预先有本地 `gpt2` 权重缓存。

- [ ] **Step 1: 写 golden 对齐脚本(失败测试)**

Create `examples/gpt2-e2e/port_gpt2_weights.py`:
```python
#!/usr/bin/env python3
"""Load official HF GPT-2 weights into our MiniGPT and assert logits match.

This is the Phase-1 "关 0" golden gate: if our from-scratch MiniGPT (with ported
weights) does not match HuggingFace GPT2LMHeadModel logits, nothing downstream
can be trusted.
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "examples" / "gpt2-e2e"))
from export_gpt2 import MiniGPT  # noqa: E402

# GPT-2 small config (must match HF "gpt2")
CFG = dict(vocab=50257, n_embd=768, n_head=12, n_layer=12, seq_len=64)


def port_weights(mini: MiniGPT, hf):
    """Copy HF GPT2LMHeadModel state_dict into MiniGPT.

    HF uses Conv1D (weight shape [in, out]); our nn.Linear wants [out, in],
    so c_attn/c_proj/c_fc/c_proj weights are transposed. lm_head is tied to wte.
    """
    h = hf.transformer
    sd = {}
    sd["tok_emb.weight"] = h.wte.weight
    sd["pos_emb.weight"] = h.wpe.weight[: CFG["seq_len"]]
    for i, blk in enumerate(h.h):
        p = f"blocks.{i}."
        sd[p + "ln1.weight"] = blk.ln_1.weight
        sd[p + "ln1.bias"] = blk.ln_1.bias
        sd[p + "attn.qkv.weight"] = blk.attn.c_attn.weight.t().contiguous()
        sd[p + "attn.qkv.bias"] = blk.attn.c_attn.bias
        sd[p + "attn.proj.weight"] = blk.attn.c_proj.weight.t().contiguous()
        sd[p + "attn.proj.bias"] = blk.attn.c_proj.bias
        sd[p + "ln2.weight"] = blk.ln_2.weight
        sd[p + "ln2.bias"] = blk.ln_2.bias
        sd[p + "mlp.fc.weight"] = blk.mlp.c_fc.weight.t().contiguous()
        sd[p + "mlp.fc.bias"] = blk.mlp.c_fc.bias
        sd[p + "mlp.proj.weight"] = blk.mlp.c_proj.weight.t().contiguous()
        sd[p + "mlp.proj.bias"] = blk.mlp.c_proj.bias
    sd["ln_f.weight"] = h.ln_f.weight
    sd["ln_f.bias"] = h.ln_f.bias
    sd["lm_head.weight"] = h.wte.weight  # weight-tied
    missing, unexpected = mini.load_state_dict(sd, strict=False)
    # pos_ids buffer is allowed missing/unexpected; nothing else should be.
    bad = [k for k in missing + unexpected if "pos_ids" not in k]
    assert not bad, f"state_dict mismatch: {bad}"


def main():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    mini = MiniGPT(vocab=CFG["vocab"], n_embd=CFG["n_embd"],
                   n_head=CFG["n_head"], n_layer=CFG["n_layer"],
                   seq_len=CFG["seq_len"]).eval()
    port_weights(mini, hf)

    text = "The capital of France is"
    ids = tok(text, return_tensors="pt").input_ids  # [1, T]
    T = ids.shape[1]
    pad = torch.zeros(1, CFG["seq_len"], dtype=torch.long)
    pad[0, :T] = ids[0]

    with torch.no_grad():
        ours = mini(pad)[0, T - 1]            # our logits at last real token
        ref = hf(ids).logits[0, T - 1]        # HF logits at same position
    max_diff = (ours - ref).abs().max().item()
    print(f"max_diff(ours,HF) = {max_diff:.3e}")
    print("our argmax token:", tok.decode([ours.argmax().item()]))
    print("HF  argmax token:", tok.decode([ref.argmax().item()]))
    assert max_diff < 1e-3, f"golden mismatch {max_diff}"
    print("GOLDEN PASS")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑一次确认它失败**

Run: `cd $REPO && python examples/gpt2-e2e/port_gpt2_weights.py`
Expected: FAIL —— `max_diff` 偏大(我们 MLP 用 erf-gelu,HF 用 gelu_new tanh 近似),断言 `max_diff < 1e-3` 不过(或 argmax token 不同)。

- [ ] **Step 3: 把 MLP 的 gelu 改成 tanh 近似以匹配 HF**

Modify `examples/gpt2-e2e/export_gpt2.py` 的 `MLP.forward`(当前 L65-66):
```python
    def forward(self, x):
        return self.proj(nn.functional.gelu(self.fc(x), approximate="tanh"))
```

- [ ] **Step 4: 跑 golden,确认通过**

Run: `cd $REPO && python examples/gpt2-e2e/port_gpt2_weights.py`
Expected: PASS —— 打印 `max_diff(ours,HF) = <~1e-5>`、两边 argmax token 一致(应为 " Paris"),末行 `GOLDEN PASS`。

- [ ] **Step 5: Commit**

```bash
cd $REPO
git add examples/gpt2-e2e/port_gpt2_weights.py examples/gpt2-e2e/export_gpt2.py
git commit -m "feat(gpt2): port HF GPT-2 small weights into MiniGPT, golden vs HF

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: embedding 提到 host —— 网络从 hidden_states 开始

**理由:** `g_host_mode` 是全局开关(见 `AclnnOps.cpp:105`),无法只让 embedding 走 host 而 matmul/FA 走 device;`run_Embedding` device 路径未实现(`AclnnOps.cpp:557`)。最干净的做法:把 token+pos embedding 移出导出图,网络入参直接是 `hidden_states[1,N,768]`,host 每步算 embedding。

**Files:**
- Modify: `examples/gpt2-e2e/export_gpt2.py`(MiniGPT 增加 `from_hidden` 路径)
- Modify: `examples/gpt2-e2e/port_gpt2_weights.py`(golden 也走 from_hidden 验证等价)

- [ ] **Step 1: 给 MiniGPT 增加 from-hidden 前向(失败测试)**

Modify `examples/gpt2-e2e/export_gpt2.py` 的 `MiniGPT`:把 embedding 拆成可单独调用,新增一个只吃 hidden_states 的前向。
```python
    def embed(self, input_ids):
        """Host-side embedding: returns hidden_states [B, S, C]."""
        return self.tok_emb(input_ids) + self.pos_emb(self.pos_ids)

    def forward_from_hidden(self, hidden):
        x = hidden
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def forward(self, input_ids):
        return self.forward_from_hidden(self.embed(input_ids))
```

在 `port_gpt2_weights.py` 的 `main()` 末尾(`GOLDEN PASS` 之前)加等价性断言:
```python
    with torch.no_grad():
        hidden = mini.embed(pad)
        ours2 = mini.forward_from_hidden(hidden)[0, T - 1]
    assert torch.allclose(ours, ours2, atol=1e-6), "from_hidden path diverged"
    print("FROM_HIDDEN EQUIV PASS")
```

- [ ] **Step 2: 跑确认等价**

Run: `cd $REPO && python examples/gpt2-e2e/port_gpt2_weights.py`
Expected: PASS,新增打印 `FROM_HIDDEN EQUIV PASS`(原 `GOLDEN PASS` 仍在)。

- [ ] **Step 3: 改导出 main(),按 from-hidden 导出 linalg + npy**

Modify `examples/gpt2-e2e/export_gpt2.py` 的 `main()`:加 `--from-hidden` 开关,为真时:用 `MiniGPT.embed` 在 host 算 hidden 作为 sample input,导出 `forward_from_hidden` 的 linalg,并把 hidden + 所有 buffer/输入按顺序存 npy。复用 Task 1 的权重加载(从 `port_gpt2_weights.port_weights` import,避免重复)。关键片段:
```python
    if args.from_hidden:
        from port_gpt2_weights import port_weights
        from transformers import GPT2LMHeadModel
        port_weights(model, GPT2LMHeadModel.from_pretrained("gpt2").eval())
        hidden = model.embed(input_ids).detach()          # host embedding
        np.save(outdir / "input_0.npy", hidden.numpy())   # network input = hidden
        with torch.no_grad():
            expected = model.forward_from_hidden(hidden)
        np.save(outdir / "expected_0.npy", expected.numpy())
        mlir_text = torch_to_linalg(model.forward_from_hidden, [hidden], None)
        (outdir / "step0_linalg.mlir").write_text(mlir_text)
        # weight buffers are captured by torch.export as constants; no extra inputs
        return
```
> 注意:`torch_to_linalg` 第一个参数原是 `model`;这里传 `model.forward_from_hidden`(一个 bound method)。若 `export_and_import` 不接受 method,改为包一个 `nn.Module` wrapper 暴露该 forward。Step 4 验证。

- [ ] **Step 4: 导出并确认产物**

Run:
```bash
cd $REPO && python examples/gpt2-e2e/export_gpt2.py \
  --n-layer 12 --n-head 12 --n-embd 768 --vocab 50257 --seq 64 \
  --from-hidden --outdir examples/gpt2-e2e/artifact_gpt2small
```
Expected: 生成 `step0_linalg.mlir`(开头是 layernorm,**无 embedding/gather op**)、`input_0.npy` shape `[1,64,768]`、`expected_0.npy` shape `[1,64,50257]`。
验证无 embedding:`grep -c "extract\|gather\|__aclnn_embedding" examples/gpt2-e2e/artifact_gpt2small/step0_linalg.mlir` 应为 0。

- [ ] **Step 5: Commit**

```bash
cd $REPO
git add examples/gpt2-e2e/export_gpt2.py examples/gpt2-e2e/port_gpt2_weights.py
git commit -m "feat(gpt2): lift embedding to host, export forward-from-hidden

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: 124M 静态前向跑通 sim,logits 对齐 PyTorch(关 1 第一半)

**Files:** 无新文件;用 `python/network_runner.py` + Task 2 产物。

- [ ] **Step 1: 跑 network_runner(sim,default tilings,跳过 autotuner)**

Run:
```bash
cd $REPO && NETWORK_RUNNER_SKIP_AUTOTUNE=1 python python/network_runner.py \
  --input-linalg examples/gpt2-e2e/artifact_gpt2small/step0_linalg.mlir \
  --inputs examples/gpt2-e2e/artifact_gpt2small/input_0.npy \
  --expected examples/gpt2-e2e/artifact_gpt2small/expected_0.npy \
  --backend sim --workdir /tmp/gpt2_sim_work --atol 1e-2 --rtol 1e-2 --log
```
> 注:确认 `network_runner.py` 实际 flag 名(`--input-linalg`/`--inputs`/`--expected`/`--workdir`/`--backend`/`--atol`)与 `examples/two-elewise-e2e/run.sh` 一致;以该 run.sh 为准对齐参数。`NETWORK_RUNNER_SKIP_AUTOTUNE=1` 规避此网 autotuner 崩溃(见 spec §2.5)。

Expected(可能失败):若通过 → 末尾 `network.output[0]: max_diff=<small> PASS`。若失败,转 Step 2 诊断。

- [ ] **Step 2: 若失败,定位是哪个 group/阶段**

按 spec §2.3 / [[project_real_npu_test_checklist]] #7:看 `session.error_stage`;sim 阶段失败 → 看 `/tmp/gpt2_sim_work/` 里 phase-3 的 `intermediates_default/` 逐 kernel dump,找第一个 max_diff 爆掉的 kernel。常见:vocab=50257 的 lm_head matmul / n_embd=768 tiling。把该 kernel 的 `groups/<k>.mlir` 拉出来分析。
> 这是诊断步,不是盲改;若属 codegen/tiling 缺陷,记录并按 [[feedback_session_role_realnpu_runner]] 分流(本计划范围是 bring-up,非新 codegen 修复)。

- [ ] **Step 3: sim PASS 后,确认 argmax token 正确**

Run:
```bash
cd $REPO && python -c "
import numpy as np
o=np.load('/tmp/gpt2_sim_work/outputs/out0.npy'); e=np.load('examples/gpt2-e2e/artifact_gpt2small/expected_0.npy')
T=5  # 'The capital of France is' = 5 tokens
print('sim argmax', o[0,T-1].argmax(), 'ref argmax', e[0,T-1].argmax())
assert o[0,T-1].argmax()==e[0,T-1].argmax()
print('SIM ARGMAX MATCH')
"
```
> `outputs/out0.npy` 路径以实际 workdir 输出为准(phase5 `work/outputs/out*.npy`)。
Expected: `SIM ARGMAX MATCH`,且 ref argmax 解码为 " Paris"。

- [ ] **Step 4: Commit(记录 sim 通过的运行配方)**

```bash
cd $REPO
git add docs/superpowers/plans/2026-06-02-gpt2-real-npu-generation.md
git commit -m "chore(gpt2): record sim forward bring-up config (124M, default tilings)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
> (本步无代码改动;若 Step 2 需要任何 example run.sh / 配置文件,在此一并提交。)

---

## Task 4: host 生成 driver(编一次跑多次),sim 上连贯生成(关 1 完成)

**Files:**
- Create: `examples/gpt2-e2e/generate.py`

**思路:** 复用 Task 3 已编译的 binary(phase-5 `work/network_test`,吃 `--input`/`--output` npy)。driver 在 host:tokenize → 循环{ host 算 embedding → 写 input_0.npy → 调 binary → 读 out0.npy → 取 pos 处 argmax → 追加 } → detokenize。

- [ ] **Step 1: 写 generate.py(失败:binary 路径/调用未定)**

Create `examples/gpt2-e2e/generate.py`:
```python
#!/usr/bin/env python3
"""Host-side token-by-token generation driver for GPT-2 on our pipeline.

Compile-once/run-many: assumes the network binary was already built (Task 3).
Each step: host computes embedding -> writes hidden npy -> invokes the binary
-> reads logits npy -> greedy-picks next token -> appends. No KV-cache; the full
[1,N] window is recomputed each step.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "examples" / "gpt2-e2e"))
from export_gpt2 import MiniGPT  # noqa: E402
from port_gpt2_weights import CFG, port_weights  # noqa: E402


def build_model():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    mini = MiniGPT(vocab=CFG["vocab"], n_embd=CFG["n_embd"], n_head=CFG["n_head"],
                   n_layer=CFG["n_layer"], seq_len=CFG["seq_len"]).eval()
    port_weights(mini, hf)
    return mini, tok


def run_forward(binary, hidden_np, io_dir):
    """Invoke the compiled network binary once; return logits [1, N, vocab]."""
    inp = io_dir / "input_0.npy"
    out = io_dir / "out0.npy"
    np.save(inp, hidden_np)
    if out.exists():
        out.unlink()
    subprocess.run([str(binary), "--input", str(inp), "--output", str(out)],
                   check=True, cwd=io_dir)
    return np.load(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True, help="compiled network_test binary")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=20)
    ap.add_argument("--io-dir", default="/tmp/gpt2_gen_io")
    args = ap.parse_args()

    io_dir = Path(args.io_dir)
    io_dir.mkdir(parents=True, exist_ok=True)
    mini, tok = build_model()
    N = CFG["seq_len"]

    ids = tok(args.prompt, return_tensors="pt").input_ids[0].tolist()
    for _ in range(args.max_new):
        pos = len(ids) - 1
        if pos >= N - 1:
            print("[generate] window full, stopping"); break
        pad = torch.zeros(1, N, dtype=torch.long)
        pad[0, : len(ids)] = torch.tensor(ids)
        with torch.no_grad():
            hidden = mini.embed(pad).numpy()           # host embedding
        logits = run_forward(Path(args.binary), hidden, io_dir)
        nxt = int(logits[0, pos].argmax())             # greedy
        ids.append(nxt)
        if nxt == tok.eos_token_id:
            break
    print(tok.decode(ids))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑 sim binary 的生成循环**

Run(binary 路径以 Task 3 的 workdir 为准):
```bash
cd $REPO && python examples/gpt2-e2e/generate.py \
  --binary /tmp/gpt2_sim_work/network_test \
  --prompt "The capital of France is" --max-new 10
```
> 若 `network_test` 不直接接 `--input/--output`,对照 `phase5_final_run_verify`(`network_runner.py:800-887`)里实际的 run 调用方式调整 `run_forward` 的命令行;以该函数为准。
Expected: 打印一句连贯英文(应含 " Paris"),例如 `The capital of France is Paris, and the city is ...`。

- [ ] **Step 3: 与 PyTorch greedy 续写对比(等价性)**

Run:
```bash
cd $REPO && python -c "
import torch,sys; sys.path.insert(0,'examples/gpt2-e2e')
from generate import build_model
m,t=build_model(); ids=t('The capital of France is',return_tensors='pt').input_ids
for _ in range(10):
  pos=ids.shape[1]-1; pad=torch.zeros(1,64,dtype=torch.long); pad[0,:ids.shape[1]]=ids[0]
  with torch.no_grad(): nxt=m.forward_from_hidden(m.embed(pad))[0,pos].argmax()
  ids=torch.cat([ids,nxt.view(1,1)],1)
print('TORCH:',t.decode(ids[0]))
"
```
Expected: sim 输出与 TORCH 输出**前若干 token 一致**(greedy 确定性;数值小漂移可能在后段分叉)。

- [ ] **Step 4: Commit**

```bash
cd $REPO
git add examples/gpt2-e2e/generate.py
git commit -m "feat(gpt2): host token-by-token generation driver (sim verified)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: masked-causal-FA device 单测(关 2 前置,真机)

**理由:** sim 用 `sdpa_cpu`(应用了 mask,正确);device 路径 `attenMask=nullptr`(双向,错)。先在 device 7 上用单测确定**正确的 causal 配置**(attenMask tensor 的极性/dtype,还是 sparseMode),再改 runtime —— 与既有 `test_aclnn_ops_device.cpp` 的 7/7 验证模式一致(见 [[project_real_npu_aclnn_direct]])。

**Files:**
- Create: `test/tools/runtime/test_masked_fa_device.cpp`

- [ ] **Step 1: 写 device 单测(对比 causal-FA 与 sdpa_cpu)**

Create `test/tools/runtime/test_masked_fa_device.cpp`,参照既有 `test/tools/runtime/test_aclnn_ops_device.cpp` 的骨架(aclInit + aclrtSetDevice(7) + stream;构造小 q/k/v [1,2,4,8] + causal additive mask [4,4]):
- 跑 A:`setHostMode(true)` → `run_FlashAttentionScore(q,k,v,mask,init,&hostOut,nullptr)` 得 sdpa_cpu 参考(已应用 mask)。
- 跑 B:`setHostMode(false)` → 同输入跑 device 路径,得 `devOut`。
- 断言 `max_abs_diff(hostOut, devOut) < 1e-3`。
关键:mask = 下三角 0 / 上三角 -inf 的 additive mask(rank-2 [S,S]),与 `export_gpt2.py:38-40` 一致。
> 实现细节(extern 声明、TensorInfo 构造、h2f/f2h)照搬既有 device 单测;此处只新增 mask 与 causal 断言。

- [ ] **Step 2: 在远端 dev host 直接编译运行(确认它失败)**

按 [[project_real_npu_aclnn_direct]] 的 host-direct 配方(dev host 自带 g++/CANN/devices,免容器):
```bash
# on 113.46.10.114, against synced /data/gser/Ascend-MLIR-current
TK=/data/nyh/Ascend/cann-9.1.0; source $TK/set_env.sh
g++ -std=c++17 -I include -I $TK/include \
  test/tools/runtime/test_masked_fa_device.cpp lib/Runtime/AclnnOps.cpp \
  -L $TK/lib64 -Wl,-rpath,$TK/lib64 \
  -lascendcl -lnnopbase -lopapi -lopapi_nn -lopapi_math -lopapi_transformer \
  -Wl,--allow-shlib-undefined -o /data/gser/aclnn-dev/t_mfa
ASCEND_DEVICE_ID=7 /data/gser/aclnn-dev/t_mfa
```
Expected: FAIL —— device 路径忽略 mask(双向),`max_abs_diff` 远大于 1e-3(尤其前几行被 mask 的位置)。

- [ ] **Step 3: Commit 单测(红)**

```bash
cd $REPO
git add test/tools/runtime/test_masked_fa_device.cpp
git commit -m "test(aclnn): device unit test for causal masked FlashAttention (red)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: 修 run_FlashAttentionScore device 分支,接 causal attenMask

**Files:**
- Modify: `lib/Runtime/AclnnOps.cpp`(`run_FlashAttentionScore` L335-405)

- [ ] **Step 1: device 分支按 mask 是否非平凡选择 causal 配置**

Modify `lib/Runtime/AclnnOps.cpp`。在 device 分支(L344 之后):判定 `mask` 是否为真实因果 mask(非 Q 占位:`mask.rank==4 && mask.shape[3]==q.shape[2]`,即 [1,1,S,S]),若是 → 走 causal。优先用 **sparseMode 因果**(免 attenMask tensor 的极性/dtype 坑):把 L383-388 改为按 causal 设置:
```cpp
  bool causal = (mask.data != q.data) && mask.rank == 4 &&
                mask.shape[3] == q.shape[2];   // real [1,1,S,S] mask, not Q placeholder
  // ... in the GetWorkspaceSize call:
  /*attenMaskOptional=*/nullptr,
  /*prefixOptional=*/nullptr,
  scale,
  /*keepProb=*/1.0,
  /*preTokens=*/causal ? 65536 : 65536,
  /*nextTokens=*/causal ? 0 : 65536,          // nextTokens=0 => causal
  numHeads,
  const_cast<char *>("BNSD"),
  /*innerPrecise=*/0,
  /*sparseMode=*/causal ? 3 : 0,              // 3 = causal band; verify vs CANN header
```
> `sparseMode` 的确切枚举值(causal=3?)以 dev host 上 CANN 9.1.0 头文件/文档为准 —— Task 5 单测就是用来确定它的:在 Step 2 里试 sparseMode∈{causal 候选值} 直到单测绿。若 sparseMode 路线对小 S 不生效,回退到"把 additive mask 转 bool/byte attenMask tensor、stageToDevice 后传 attenMaskOptional"。

- [ ] **Step 2: 重编译单测,迭代到绿**

Run(同 Task 5 Step 2 的 host-direct 命令):
```bash
ASCEND_DEVICE_ID=7 /data/gser/aclnn-dev/t_mfa
```
Expected: PASS —— `max_abs_diff(host,dev) < 1e-3`。若不过,调整 sparseMode/preTokens/nextTokens(或切 attenMask tensor 方案)重编重跑,直到绿。

- [ ] **Step 3: 回归既有 device 单测(别打破双向 FA)**

Run:
```bash
# rebuild & run existing device unit test (BERT-style bidirectional FA must still pass)
ASCEND_DEVICE_ID=7 /data/gser/aclnn-dev/t_dev   # from project_real_npu_aclnn_direct recipe
```
Expected: 既有 7/7 仍 PASS(无 mask 时 `nextTokens=65536`/`sparseMode=0` 路径不变)。

- [ ] **Step 4: Commit**

```bash
cd $REPO
git add lib/Runtime/AclnnOps.cpp
git commit -m "fix(aclnn): wire causal mask into FlashAttentionScore device path

Device branch hardcoded attenMask=nullptr (bidirectional) -> GPT-2 causal
attention silently wrong on real NPU (sim used sdpa_cpu, masked correctly).
Use sparseMode/nextTokens=0 for real [1,1,S,S] causal masks; Q-placeholder
(no-mask) path unchanged. Verified vs sdpa_cpu on device 7.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: 真机单次前向,logits 对齐(关 2 第一半)

**Files:** 无新文件;`scripts/sync-and-submit.sh` / `.claude/npu-relay/submit.sh`。

> **真机安全铁律(每次)**:加 timeout + 监控;失败把 `logs/`(含 `plog/plog-errorStr.txt`)拉到仓库外 `/tmp/npu-real-logs/<date>/gpt2/`;任何 kill/docker 操作先报风险等确认;只碰 `/data/gser`/device 7(spec §5)。

- [ ] **Step 1: 同步带 Task 6 修复的代码,跑 device 7 单次前向**

Run(从 x86 dev box,已 `source examples/env_gser.sh`):
```bash
cd $REPO && timeout 1800 scripts/real-npu-ci/sync-and-submit.sh \
  --cmd 'NETWORK_RUNNER_SKIP_AUTOTUNE=1 BACKEND=npu python python/network_runner.py \
    --input-linalg examples/gpt2-e2e/artifact_gpt2small/step0_linalg.mlir \
    --inputs examples/gpt2-e2e/artifact_gpt2small/input_0.npy \
    --expected examples/gpt2-e2e/artifact_gpt2small/expected_0.npy \
    --backend npu --workdir /data/gser/gpt2_npu_work --atol 1e-2 --rtol 1e-2 --log'
```
> 命令行以 `examples/two-elewise-e2e/run.sh` 的真机调用为准对齐(`--cmd 'BACKEND=npu ...'` 模式见 [[project_network_runner_npu_backend]])。

Expected(主要风险点):`network.output[0]: max_diff=<small> PASS`。若失败 → Step 2。

- [ ] **Step 2: 失败诊断(分类 plog 错误)**

按 [[project_real_npu_test_checklist]] #3:拉 `plog/plog-errorStr.txt` 分类——
- "GM address ... exceeds 48 bits"(507035)= 寻址/stride codegen 或 host-wiring 喂错指针;
- "timeout or trap error"(507034)= 另一类;
- 输出全 0 / 读到 version.info = 前端 buffer 接线 garbage(encoder 同款,spec §2.3)。
分类后:若是 masked-FA 数值错 → 回 Task 6;若是 encoder 式 codegen/wiring → 记录并按 [[feedback_session_role_realnpu_runner]] 分流(超出本 bring-up 范围)。证据存 `/tmp/npu-real-logs/<date>/gpt2/`。

- [ ] **Step 3: 真机 PASS 后,确认 argmax 正确**

Run: 把 `/data/gser/gpt2_npu_work/outputs/out0.npy` scp 回本地,复用 Task 3 Step 3 的 argmax 脚本。
Expected: 真机 argmax == ref argmax(" Paris"),`NPU ARGMAX MATCH`。

- [ ] **Step 4: Commit(记录真机通过配方)**

```bash
cd $REPO
git add docs/superpowers/plans/2026-06-02-gpt2-real-npu-generation.md
git commit -m "chore(gpt2): record real-NPU single-forward pass config (device 7)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: 真机生成循环 —— 连贯英文(关 2 完成,Phase 1 收尾)

**Files:**
- Create: `examples/gpt2-e2e/run_gpt2_real.sh`

**思路:** binary 已在 Task 7 编好(`/data/gser/gpt2_npu_work/network_test`),在**远端 dev host 上直接**跑 `generate.py`(每步本地调 binary,不走 per-token sync-and-submit;dev host 自带 CANN/devices)。

- [ ] **Step 1: 写远端运行脚本**

Create `examples/gpt2-e2e/run_gpt2_real.sh`:
```bash
#!/usr/bin/env bash
# Run the GPT-2 generation loop on the remote dev host, against the already-built
# NPU binary. Invoke FROM the dev host (113.46.10.114), inside /data/gser source.
set -euo pipefail
TK=${ASCEND_HOME_PATH:-/data/nyh/Ascend/cann-9.1.0}
source "$TK/set_env.sh"
export ASCEND_DEVICE_ID=7
# strip simulator/devlib so the real driver HAL is used (see network-runner-npu-backend)
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' \
  | grep -v '/simulator/' | grep -v '/devlib/' | paste -sd: -)
WORK=${1:-/data/gser/gpt2_npu_work}
python examples/gpt2-e2e/generate.py \
  --binary "$WORK/network_test" \
  --prompt "${PROMPT:-The capital of France is}" \
  --max-new "${MAX_NEW:-20}" \
  --io-dir /data/gser/gpt2_gen_io
```

- [ ] **Step 2: 在 dev host 上跑生成循环**

Run(经 sync-and-submit 的 `--cmd`,或在 server session 上直接执行;`generate.py` 每步本地调 NPU binary):
```bash
cd $REPO && timeout 1800 scripts/real-npu-ci/sync-and-submit.sh \
  --cmd 'PROMPT="The capital of France is" MAX_NEW=20 bash examples/gpt2-e2e/run_gpt2_real.sh'
```
> 注意 LD_LIBRARY_PATH 在 `bash -lc`/login shell 下可能被重置丢掉 driver(见 [[project_real_npu_aclnn_direct]] 的 aclInit 500000 gotcha);脚本里已重 source set_env.sh,若仍 aclInit 失败,在脚本顶部显式补回 `/usr/local/Ascend/driver/lib64`。
Expected: 真机输出一句**连贯英文**(含 " Paris"),且与 sim(Task 4)/PyTorch(Task 4 Step 3)前若干 token 一致。

- [ ] **Step 3: 与 sim 输出对比验收**

人工对比 Task 8 Step 2(真机)与 Task 4 Step 2(sim)的输出文本:greedy 下应高度一致(数值小漂移可能后段分叉)。记录两段输出到 plan/handoff。
Expected: 真机文本连贯且与 sim 一致到 mask 修复生效的程度(关键:attention 不再双向退化 → 不是乱码)。

- [ ] **Step 4: Commit + 写 handoff**

```bash
cd $REPO
git add examples/gpt2-e2e/run_gpt2_real.sh
git commit -m "feat(gpt2): real-NPU token-by-token generation loop (Phase 1 done)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
然后写 `docs/superpowers/notes/2026-06-XX-gpt2-real-npu-phase1-handoff.md`:三关结果、masked-FA 修法实测、真机/sim/PyTorch 三方输出对照、剩余风险(autotuner、KV-cache、embedding device、Qwen 四块)。更新 memory(新建 `project_gpt2_real_npu_phase1`)。

---

## Self-Review

**Spec 覆盖核查:**
- spec §2.1 masked-FA → Task 5/6 ✅
- spec §2.2 embedding 缺失 → Task 2(提到 host,比 spec 的 host-mode seam 更干净;原因已在 Task 2 说明)✅
- spec §2.3 encoder 先例风险 → Task 7 Step 2 诊断+分流 ✅
- spec §2.4 编一次跑多次 → Task 4 + Task 8(发现 binary 已可复用,driver 即可)✅
- spec §2.5 autotuner 崩溃 → 全程 `NETWORK_RUNNER_SKIP_AUTOTUNE=1`;gelu → Task 1 Step 3;vocab → Task 3 Step 2 ✅
- spec §3.1 固定窗口无 KV-cache → Task 4/8 ✅
- spec §4 三关 → 关0=Task 1/2,关1=Task 3/4,关2=Task 5-8 ✅
- spec §5 真机安全 → Task 7/8 抬头铁律 ✅

**偏离说明:** embedding 由"global host-mode seam"改为"host-precompute / network-from-hidden"——因为 `g_host_mode` 全局、无法 per-op(Explore 查证),host-precompute 是同 intent 下最干净的实现,且天然契合生成循环。已记录于 Task 2 抬头。

**待执行者注意的不确定点(已在对应 step 标注,非占位符):** `network_runner.py` 的确切 flag 名 / `network_test` 的确切 `--input/--output` 调用 / `sparseMode` 因果枚举值 —— 均给了"以哪个现存文件为准"的查证锚点 + 验证 gate,不是 TODO。
