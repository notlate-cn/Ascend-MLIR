from pathlib import Path
base = Path('/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/sample_f32ab')

# kernel: half,half -> float  => float,float -> float
p = base / 'baremix_custom.cpp'
s = p.read_text()
s = s.replace('MatmulLeakyKernel<half, half, float, float>', 'MatmulReluKernel<float, float, float, float>')
s = s.replace('MatmulReluKernel<half, half, float, float>', 'MatmulReluKernel<float, float, float, float>')
p.write_text(s)

# tiling: K=256 unchanged, but dtype declarations become float32 for A/B
q = base / 'baremix_custom_tiling.cpp'
t = q.read_text()
t = t.replace('DataType leftDtype = DataType::DT_FLOAT16;', 'DataType leftDtype = DataType::DT_FLOAT;')
t = t.replace('DataType rightDtype = DataType::DT_FLOAT16;', 'DataType rightDtype = DataType::DT_FLOAT;')
q.write_text(t)

# data: fp32 A/B, bias[N], ReLU
r = base / 'scripts' / 'gen_data.py'
g = r.read_text()
g = g.replace('input_a = np.random.randint(-10, 10, [M, K]).astype(np.float16)', 'input_a = np.random.randint(-10, 10, [M, K]).astype(np.float32)')
g = g.replace('input_b = np.random.randint(-10, 10, [K, N]).astype(np.float16)', 'input_b = np.random.randint(-10, 10, [K, N]).astype(np.float32)')
r.write_text(g)
print('patched')
