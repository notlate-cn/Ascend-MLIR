"""End-to-end fusion case exercising the linalg-level symbolic-shape path.

`(a * b + a) * b` -- three elementwise ops on dynamic-shape inputs (mul, add,
mul; `a` is reused).  Through the auto-fuse-codegen pipeline these fuse into a
single linalg.generic, whose two parallel axes collapse into one (extent
`s0*s1`); the AscendC kernel ends up with a de-duped TilingData
(`[XBLOCK, XBLOCK_SUB, dim_arg0_0, dim_arg0_1]` -- the second input's and the
output's dims fold onto the first input's) and a populated
`block_dim_expr = ceil((arg0_dim0 * arg0_dim1)/XBLOCK)` in tiling_space.json,
which the autotuner evaluates to `block_dim = ceil(M*N / XBLOCK)`.
"""

import torch
from framework import torch_e2e_test, TensorSpec


@torch_e2e_test(verify_shapes={"M": 128, "N": 64})
def test_fused_elementwise():
    class Model(torch.nn.Module):
        def forward(self, a, b):
            return (a * b + a) * b

    return Model(), [TensorSpec(("M", "N"), torch.float16),
                     TensorSpec(("M", "N"), torch.float16)]


@torch_e2e_test(verify_shapes={"M": 128, "N": 64})
def test_fused_relu():
    # relu(a*b + a): torch.export lowers relu to arith.cmpf-ugt + arith.select;
    # LinalgToAscendC's SelectToMinMaxPattern rewrites it to arith.maximumf so it
    # actually reaches the kernel (it was being silently dropped before).
    class Model(torch.nn.Module):
        def forward(self, a, b):
            return torch.relu(a * b + a)

    return Model(), [TensorSpec(("M", "N"), torch.float16),
                     TensorSpec(("M", "N"), torch.float16)]
