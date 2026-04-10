from pathlib import Path
base = Path('/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/sample_k64')

# data generator: K 256 -> 64
r = base / 'scripts' / 'gen_data.py'
g = r.read_text().replace('    K = 256', '    K = 64', 1)
r.write_text(g)

# tiling generator: K 256 -> 64
p = base / 'baremix_custom_tiling.cpp'
s = p.read_text()
s = s.replace('    int K = 256;', '    int K = 64;', 1)
s = s.replace('    TPosition biasPosition = TPosition::GM;', '    TPosition biasPosition = TPosition::GM;', 1)
p.write_text(s)
print('patched')
