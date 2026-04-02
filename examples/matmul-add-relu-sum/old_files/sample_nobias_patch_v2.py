from pathlib import Path
base = Path('/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/sample_nobias')

# data: zero bias and golden without bias term
r = base / 'scripts' / 'gen_data.py'
g = r.read_text()
g = g.replace('    input_bias = np.random.randint(1, 10, [N]).astype(np.float32)', '    input_bias = np.zeros([N], dtype=np.float32)', 1)
g = g.replace('    golden = (np.matmul(input_a.astype(np.float32), input_b.astype(np.float32)) + input_bias).astype(np.float32)', '    golden = np.matmul(input_a.astype(np.float32), input_b.astype(np.float32)).astype(np.float32)', 1)
r.write_text(g)

# kernel: keep signatures intact; only stop using bias tensor in AIC matmul path
p = base / 'baremix_custom.cpp'
s = p.read_text()
s = s.replace('    AscendC::GlobalTensor<biasType> biasGlobal;\n', '')
s = s.replace('    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling.N);\n', '')
s = s.replace('    biasGlobal = biasGlobal[offsetBias];\n', '')
s = s.replace('    matmulObj.SetBias(biasGlobal);\n', '')
# keep CalcOffset signature, just make offsetBias unused and set harmlessly
s = s.replace('    offsetBias = nCoreIndx * tiling.singleCoreN;\n', '    offsetBias = 0;\n')
p.write_text(s)
print('patched')
