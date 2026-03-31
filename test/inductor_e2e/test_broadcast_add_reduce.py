import os
import sys
import torch
import torch._dynamo

# 每个测试独立运行，reset dynamo 缓存避免互相影响
def setup_function():
    torch._dynamo.reset()


def test_inductor_backend_registration():
    """测试 MLIRScheduling 能正确注册并运行 inductor 融合，并验证 LoopIR 提取"""
    from inductor_backend import setup_inductor_backend, create_post_fusion_pass

    output_dir = "/tmp/inductor_e2e_output"
    os.makedirs(output_dir, exist_ok=True)

    post_fusion_pass = create_post_fusion_pass(output_dir, verbose=True)
    setup_inductor_backend(post_fusion_pass=post_fusion_pass)

    class SimpleModel(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    model = SimpleModel()
    example_inputs = (torch.randn(4, 4), torch.randn(4, 4))

    compiled = torch.compile(model, backend="inductor")
    result = compiled(*example_inputs)

    expected = model(*example_inputs)
    assert torch.allclose(result, expected, atol=1e-5), "Inductor backend output mismatch"

    print("✓ Backend registration and inductor fusion work")


def test_broadcast_add_reduce_fusion():
    """
    验证 inductor 能将 elementwise + reduction 融合成一个 kernel。

    这是当前 linalg-fuse-elementwise-ops 做不到的，
    但 inductor 可以实现。
    """
    from inductor_backend import setup_inductor_backend, create_post_fusion_pass

    output_dir = "/tmp/inductor_e2e_output"
    os.makedirs(output_dir, exist_ok=True)

    post_fusion_pass = create_post_fusion_pass(output_dir, verbose=True)
    setup_inductor_backend(post_fusion_pass=post_fusion_pass)

    class BroadcastAddReduceModel(torch.nn.Module):
        def forward(self, a, b):
            # a: [M], b: [M, N] → broadcast → add → reduce
            return (a.unsqueeze(1) + b).sum(dim=1)

    model = BroadcastAddReduceModel()
    example_inputs = (torch.randn(128), torch.randn(128, 16))

    compiled = torch.compile(model, backend="inductor")
    result = compiled(*example_inputs)

    expected = model(*example_inputs)
    assert torch.allclose(result, expected, atol=1e-5), "Output mismatch"

    print("✓ broadcast-add-reduce fusion works")


def test_broadcast_add_reduce_mlir_pipeline():
    """
    端到端测试：inductor 融合 → linalg MLIR → stage 2-8 pipeline。
    验证生成的 MLIR 能通过 afir-opt 各阶段处理，产出 step8_kernel.cpp。
    """
    import shutil
    from pathlib import Path
    from inductor_backend import setup_inductor_backend, create_post_fusion_pass
    from framework.pipeline import _run_mlir_pipeline_from_inductor

    output_dir = "/tmp/inductor_e2e_pipeline"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    post_fusion_pass = create_post_fusion_pass(output_dir, verbose=True)
    setup_inductor_backend(post_fusion_pass=post_fusion_pass)

    class BroadcastAddReduceModel(torch.nn.Module):
        def forward(self, a, b):
            return (a.unsqueeze(1) + b).sum(dim=1)

    model = BroadcastAddReduceModel()
    compiled = torch.compile(model, backend="inductor")
    compiled(torch.randn(128), torch.randn(128, 16))

    mlir_path = Path(output_dir) / "inductor_linalg.mlir"
    assert mlir_path.exists(), "inductor_linalg.mlir not generated"
    print(f"\nGenerated MLIR:\n{mlir_path.read_text()}")

    ok = _run_mlir_pipeline_from_inductor(Path(output_dir))
    assert ok, "MLIR pipeline (inductor path) failed"

    cpp_path = Path(output_dir) / "step8_kernel.cpp"
    assert cpp_path.exists(), "step8_kernel.cpp not generated"
    print(f"\n✓ step8_kernel.cpp generated: {cpp_path}")
    print("✓ broadcast-add-reduce inductor → MLIR pipeline works")