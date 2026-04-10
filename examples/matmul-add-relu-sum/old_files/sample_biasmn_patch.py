from pathlib import Path
base = Path('/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/sample_biasmn')

# kernel: keep everything else, only change bias handling from [N] to [M,N]
p = base / 'baremix_custom.cpp'
s = p.read_text()
# AIC side: make bias tensor full M*N and keep per-core offseting logic already present via offsetBias
s = s.replace('    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling.N);',
              '    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling.M * tiling.N);')
p.write_text(s)

# golden data: bias becomes [M,N]
r = base / 'scripts' / 'gen_data.py'
g = r.read_text()
g = g.replace('    input_bias = np.random.randint(1, 10, [N]).astype(np.float32)',
              '    input_bias = np.random.randint(1, 10, [M, N]).astype(np.float32)')
r.write_text(g)
print('patched')
