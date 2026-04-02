from pathlib import Path
base = Path('/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/sample_biasmn_aivadd')
r = base / 'scripts' / 'gen_data.py'
g = r.read_text()
old = '    input_bias = np.random.randint(1, 10, [N]).astype(np.float32)'
new = '    input_bias = np.random.randint(1, 10, [M, N]).astype(np.float32)'
if old not in g:
    raise SystemExit('missing bias[N] generator line')
g = g.replace(old, new, 1)
r.write_text(g)
print('patched')
