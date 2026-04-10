from pathlib import Path
base = Path('/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/sample_relu_same_name')

p = base / 'baremix_custom.cpp'
s = p.read_text()
s = s.replace('AscendC::LeakyRelu(reluOutLocal, reluInLocal, (float)0.001, tiling.singleCoreM * tiling.singleCoreN /2);', 'AscendC::Relu(reluOutLocal, reluInLocal, tiling.singleCoreM * tiling.singleCoreN / 2);')
p.write_text(s)

r = base / 'scripts' / 'gen_data.py'
g = r.read_text()
g = g.replace('    alpha = 0.001\n    golden = (np.matmul(input_a.astype(np.float32), input_b.astype(np.float32)) + input_bias).astype(np.float32)\n    golden = np.where(golden >= 0, golden, golden * alpha)', '    golden = (np.matmul(input_a.astype(np.float32), input_b.astype(np.float32)) + input_bias).astype(np.float32)\n    golden = np.maximum(golden, 0)')
r.write_text(g)
print('patched')
