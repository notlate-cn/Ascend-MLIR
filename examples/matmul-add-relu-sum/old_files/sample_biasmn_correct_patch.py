from pathlib import Path
base = Path('/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/sample_biasmn_correct')

# 1) bias data becomes [M,N]
r = base / 'scripts' / 'gen_data.py'
g = r.read_text()
g = g.replace('    input_bias = np.random.randint(1, 10, [N]).astype(np.float32)',
              '    input_bias = np.random.randint(1, 10, [M, N]).astype(np.float32)')
r.write_text(g)

# 2) kernel bias tensor should cover full M*N and offset per core over rows
p = base / 'baremix_custom.cpp'
s = p.read_text()
s = s.replace('    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling.N);',
              '    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling.M * tiling.N);')
s = s.replace('    offsetBias = nCoreIndx * tiling.singleCoreN;',
              '    offsetBias = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;')
p.write_text(s)

print('patched')
