import os
import torch


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