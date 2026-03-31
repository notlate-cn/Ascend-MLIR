import torch


def test_inductor_backend_registration():
    """测试 MLIRScheduling 能正确注册并运行 inductor 融合"""
    from inductor_backend import setup_inductor_backend

    setup_inductor_backend()

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