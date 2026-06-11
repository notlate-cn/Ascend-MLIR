# 多输出 Vector kernel 真机死锁 — 机制 + 正式方案委托 (2026-05-25)

委托背景:多输出**必须**能融到同一 kernel(拆 kernel 不可接受)。本文讲清死锁机制 + 两个候选为何不够,给正式方案 session 起点。运行/验证已全回基线、零提交。

## 现象
单层 kernel `add(a,b)→out0 + fill0→out1` (2x8x64),sim PASS max_diff=0,真机死锁(block_dim=1 也挂,plog 无 aicore 错=纯 que 死等)。

## 隔离结论
- 单输出 add=6 que(2VECIN+1VECOUT 主+尾)→ **真机过**。
- 双输出同 kernel=8 que(2VECIN+2VECOUT 主+尾)→ **真机挂**。差别只在第二条 VECOUT。
- two-elewise 双输出过,只因被拆成两个单输出 kernel。**融合 ≠ 拆**,所以真问题=同 kernel 多 VECOUT 共存。
- 排除:空 store(补 Duplicate 不解决)、多核、host 接线(t1[3]/t2[2]/outputs[0,1] 全对)、que 配对(8 que 全 1:1)。

## 两候选为何不够
1. **补 Duplicate(0) 给空 VECOUT**:`ComputeConversion.cpp:2305` `ms==10` 把 fill 当冗余 erase,留无 producer 空 store;补回 dup 后 sim 仍 0、真机仍挂。空 tile 不是死因。
2. **合 VECOUT que(8→6,round-robin)**:lit 85/85,但 sim out1 max_diff=5.39。g18 两输出 `add`/`fill` 别名同一 `tensor.empty %e`,合 que 后 fill 的 0-写被丢 → out1 garbage。共享概念对,栽在输出别名。

## 死锁机制(本质)
camodel 容忍多 VECOUT 共存,真机两条 VECOUT 队列在同 kernel body 同存时 event/MTE 死等。codegen 起点:`InsertTileBuffers.cpp:242` 每 result 一条 VECOUT alloc → `LinalgToAscendCPass.cpp:350` 1 alloc=1 que → result 数翻倍 que。多输出路径无真机覆盖,藏到现在。

## 正式方案要满足
- 多输出留同一 kernel(融合必需)。
- 输出别名同 init 时不丢写(g18 病态)。
- 不超真机 VECOUT/event 上界:候选=共一 VECOUT 但保留各输出的 producer 写(fill→Duplicate 不能丢)、或正确序列化两输出 enque/deque/free。
- 验:sim 双 max_diff=0 + lit + **真机 device7 不挂**(camodel 测不出,必须真机)。

## 根因坐实(2026-05-25,排除法)
单输出 4 输入 add4 = **10 que 1 VECOUT 真机不挂**;c2/g18 双输出 8–12 que 挂。→ 死锁**专属并发 ≥2 条 VECOUT**,非总队列数。reuse-tail 减 VECIN 无效;1 que=1 buf,共享 que 即共享 buf,故正式方案=单 VECOUT 串行 + fill 补 Duplicate(ms==10 当前直接 erase 致 out1 garbage)。

## 修复进展(2026-05-25)
单VECOUT共享(InsertTileBuffers 按elemtype+dom复用1 alloc)+ fill不再erase: c2 真机12→9que **死锁消失**(根因确认)。残:共buf两输出无间隔,真机mul覆写在add copy0(MTE3)前→out0丢 1.0(add4同类), sim顺序测不出。下一步=输出store间补PipeBarrier串行。lit95, sim双0。

## 收口(2026-05-25): encoder 真机 PASS
g18 来源=matmul/bmm 累加器 fill 被分给无关 add 组(matmul→aclnn 不剥零)→死输出双VECOUT。短期修 GroupAnalysis Step0 剥 fill→空(10b9344f);+ 并发VECOUT串行(0f7451bc)兜真多输出。encoder 整网 910C max_diff=7.15e-7 无回归, lit95, group18 单输出。长期=开 CV 融合让 fill 进 cube。

复现 /tmp/g18-repro/。相关 [[project_real_npu_aclnn_direct]] + handoff 2026-05-25-group18-*。
