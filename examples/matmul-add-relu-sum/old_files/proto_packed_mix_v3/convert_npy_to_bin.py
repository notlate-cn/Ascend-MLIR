import numpy as np
from pathlib import Path
base = Path('/tmp/packed_raw_inputs')
base.mkdir(parents=True, exist_ok=True)
for name in ['input_a', 'input_b', 'input_bias', 'output']:
    np.load('/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/runner_probe2/data/' + name + '.npy').tofile(base / f'fc_relu_{name}.bin')
for name in ['input_a', 'input_b', 'input_bias', 'output']:
    np.load('/tmp/fc_leakyrelu_split_data/' + name + '.npy').tofile(base / f'fc_leakyrelu_{name}.bin')
print('done')
