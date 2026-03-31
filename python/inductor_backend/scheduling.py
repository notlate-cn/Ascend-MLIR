"""
MLIRScheduling + setup_inductor_backend

拦截方式：patch Scheduler._codegen，在融合完成后截获节点列表。

原因：inductor 的 tensor 在 CPU 上时走 CppScheduling，注册 npu backend 无效。
直接 patch Scheduler._codegen 可以跨设备类型拦截，不依赖特定 device 注册。
"""

import torch._inductor.scheduler as _sched_mod

_post_fusion_hook = None  # 全局 hook，由 setup_inductor_backend 设置
_orig_codegen = None       # 保存原始 _codegen 以便恢复


def _patched_codegen(scheduler_self, nodes):
    """
    替换 Scheduler._codegen，在融合完成后（节点已是 FusedSchedulerNode 或
    SchedulerNode）调用我们的 post_fusion_hook，然后继续原始 codegen。
    """
    global _post_fusion_hook, _orig_codegen
    if _post_fusion_hook is not None:
        try:
            nodes = _post_fusion_hook(nodes)
        except Exception as e:
            import traceback
            print(f"[MLIRBackend] post_fusion_hook failed: {e}")
            traceback.print_exc()
    return _orig_codegen(scheduler_self, nodes)


def setup_inductor_backend(post_fusion_pass=None):
    """
    安装 post-fusion hook，在 inductor 完成算子融合后截获节点列表。

    不依赖特定 device 注册（npu/cpu 都有效）。
    patch Scheduler._codegen，在那里节点已完成融合（FusedSchedulerNode 已生成）。

    Args:
        post_fusion_pass: 接收 list[BaseSchedulerNode] 的函数，必须返回该列表。
    """
    global _post_fusion_hook, _orig_codegen

    if _orig_codegen is None:
        _orig_codegen = _sched_mod.Scheduler._codegen
        _sched_mod.Scheduler._codegen = _patched_codegen

    _post_fusion_hook = post_fusion_pass

    # 必须禁用 fx graph cache，否则 _codegen 不会被调用
    import torch._inductor.config as cfg
    cfg.fx_graph_cache = False
    cfg.fx_graph_remote_cache = False