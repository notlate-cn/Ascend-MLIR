import numpy as np
from pathlib import Path
base = Path('/tmp/final_mix_npy')
base.mkdir(parents=True, exist_ok=True)
A = np.fromfile('/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/final_sample_target_graph/input/x1_gm.bin', dtype=np.float16).reshape(128, 64)
B = np.fromfile('/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/final_sample_target_graph/input/x2_gm.bin', dtype=np.float16).reshape(64, 128)
Bias = np.fromfile('/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/final_sample_target_graph/input/bias.bin', dtype=np.float32).reshape(128, 128)
Out = np.fromfile('/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/final_sample_target_graph/output/golden.bin', dtype=np.float32).reshape(128, 128)
np.save(base / 'input_a.npy', A)
np.save(base / 'input_b.npy', B)
np.save(base / 'input_bias.npy', Bias)
np.save(base / 'output.npy', Out)
print(base)
