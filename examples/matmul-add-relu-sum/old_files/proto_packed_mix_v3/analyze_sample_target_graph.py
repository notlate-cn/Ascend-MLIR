import numpy as np
out = np.fromfile('/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/sample_target_graph/output/output.bin', dtype=np.float32).reshape(128, 128)
gold = np.fromfile('/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/sample_target_graph/output/golden.bin', dtype=np.float32).reshape(128, 128)
mask = np.abs(out - gold) > 1e-5
rows = mask.any(axis=1)
print('bad rows count', int(rows.sum()))
print('first bad rows', np.where(rows)[0][:20].tolist())
print('last bad rows', np.where(rows)[0][-20:].tolist())
print('bad elements first half', int(mask[:64].sum()), 'second half', int(mask[64:].sum()))
print('row sums out 60-68', [float(out[r].sum()) for r in range(60, 69)])
print('row sums gold 60-68', [float(gold[r].sum()) for r in range(60, 69)])
