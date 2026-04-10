from pathlib import Path
base = Path('/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/sample_target_graph')

# K -> 64 in data and tiling
r = base / 'scripts' / 'gen_data.py'
g = r.read_text()
g = g.replace('    K = 256', '    K = 64', 1)
r.write_text(g)

p = base / 'baremix_custom_tiling.cpp'
s = p.read_text()
s = s.replace('    int K = 256;', '    int K = 64;', 1)
p.write_text(s)
print('patched')
