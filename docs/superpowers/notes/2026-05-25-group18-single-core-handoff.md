# kernel_group18 single-core test — handoff (2026-05-25)

Runner forced kernel_group18 to single-core to prove whether the hang is multicore/tiling vs kernel codegen. Branch `dev-network`@746ba742 (worktree encoder-robustness), device 7.

## Result: block_dim=1 STILL HANGS → kernel-codegen deadlock, NOT tiling
- group18 = dual-output Vector: `add(arg0,arg1)→out0` AND `fill(0.0)→out1`, shape 2×8×64=1024, two results. extent=1024, `block_dim_expr=ceil(1024/XBLOCK)`, params XBLOCK+XBLOCK_SUB ∈{16..256}.
- Default picks XBLOCK=256 → block_dim=4 → hangs ~9 min.
- Forced XBLOCK=1024 → block_dim=1 (single core) → **also hangs ~9 min**, `plog-errorStr.txt` empty (no aicore trap). hang, not fault.
- Conclusion: not multicore (block_dim=2/4) and not autotuner — a kernel-codegen deadlock in the dual-output kernel. Hand to codegen session, not tiling. group2/20/26 PASS on this rev (old leading-bcast faults gone).

## Two-step forced-tiling recipe (reusable, no resume flag exists)
1. `--max-phase 4` (SKIP_AUTOTUNE) writes `<workdir>/tilings_default.json` — no device.
2. ssh-patch one kernel: `tilings_default.json["kernel_group18__v0"]={XBLOCK,XBLOCK_SUB,_block_dim}`.
3. phase5-only via python snippet (avoids phase3 overwrite): `python3 -c "import network_runner as nr; nr.phase5_final_run_verify(work,work/'groups',work/'artifacts',work/'tilings_default.json',net,argparse.Namespace(inputs=[..],expected=[..],soc='Ascend910B1',atol=1e-2,rtol=1e-2,backend='npu'))"`.

Evidence: `/tmp/npu-real-logs/20260525-g18-single/` (logs/, kernel_group18.mlir). See [[project_real_npu_aclnn_direct]], [[project_real_npu_test_checklist]].
