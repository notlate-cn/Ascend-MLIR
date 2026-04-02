from pathlib import Path
base = Path('/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/sample_biasmn_correct')
p = base / 'main.cpp'
s = p.read_text()
s = s.replace('    size_t biasFileSize = 640 * sizeof(float);', '    size_t biasFileSize = 16384 * sizeof(float);', 1)
p.write_text(s)
print('patched')
