from pathlib import Path
for name in ['sample_biasmn_aivadd', 'sample_biasmn_aivadd_2d']:
    p = Path('/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum') / name / 'main.cpp'
    s = p.read_text()
    s = s.replace('    size_t biasFileSize = 640 * sizeof(float);', '    size_t biasFileSize = 16384 * sizeof(float);', 1)
    p.write_text(s)
print('patched')
