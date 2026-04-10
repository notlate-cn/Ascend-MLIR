from pathlib import Path
base = Path('/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/sample_relu_singleline')

p = base / 'baremix_custom.cpp'
s = p.read_text()
old = '    AscendC::LeakyRelu(reluOutLocal, reluInLocal, (float)0.001, tiling.singleCoreM * tiling.singleCoreN /2);'
new = '    AscendC::Relu(reluOutLocal, reluInLocal, tiling.singleCoreM * tiling.singleCoreN / 2);'
if old not in s:
    raise SystemExit('missing leakyrelu call')
s = s.replace(old, new, 1)
p.write_text(s)

r = base / 'scripts' / 'gen_data.py'
g = r.read_text()
oldg = '    alpha = 0.001\n    golden = (np.matmul(input_a.astype(np.float32), input_b.astype(np.float32)) + input_bias).astype(np.float32)\n    golden = np.where(golden >= 0, golden, golden * alpha)'
newg = '    golden = (np.matmul(input_a.astype(np.float32), input_b.astype(np.float32)) + input_bias).astype(np.float32)\n    golden = np.maximum(golden, 0)'
if oldg not in g:
    raise SystemExit('missing golden leakyrelu block')
g = g.replace(oldg, newg, 1)
r.write_text(g)
print('patched')
