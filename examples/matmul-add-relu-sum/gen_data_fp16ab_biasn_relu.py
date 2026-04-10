import numpy as np
from pathlib import Path

out_dir = Path(__import__('sys').argv[1])
out_dir.mkdir(parents=True, exist_ok=True)
M, K, N = 128, 64, 128
rng = np.random.default_rng(42)
A = rng.integers(-3, 4, (M, K)).astype(np.float16)
B = rng.integers(-3, 4, (K, N)).astype(np.float16)
bias = rng.uniform(-0.1, 0.1, (N,)).astype(np.float32)
out = np.maximum((A.astype(np.float32) @ B.astype(np.float32)) + bias, 0).astype(np.float32)
np.save(out_dir / 'input_a.npy', A)
np.save(out_dir / 'input_b.npy', B)
np.save(out_dir / 'input_bias.npy', bias)
np.save(out_dir / 'output.npy', out)
print('done')
