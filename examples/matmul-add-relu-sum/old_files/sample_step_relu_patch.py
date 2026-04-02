from pathlib import Path
base = Path('/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/sample_step_relu')

p = base / 'baremix_custom.cpp'
s = p.read_text()
s = s.replace('LeakyReluKernel', 'ReluKernel')
s = s.replace('LeakyReluCopyIn', 'ReluCopyIn')
s = s.replace('LeakyReluCompute', 'ReluCompute')
s = s.replace('LeakyReluCopyOut', 'ReluCopyOut')
s = s.replace('AscendC::LeakyRelu(reluOutLocal, reluInLocal, (float)0.001, tiling.singleCoreM * tiling.singleCoreN /2);', 'AscendC::Relu(reluOutLocal, reluInLocal, tiling.singleCoreM * tiling.singleCoreN / 2);')
s = s.replace('MatmulLeakyKernel', 'MatmulReluKernel')
s = s.replace('matmulLeakyKernel', 'matmulReluKernel')
s = s.replace('baremix_custom', 'baremix_relu_custom')
p.write_text(s)

q = base / 'baremix_custom_tiling.cpp'
q.write_text(q.read_text().replace('baremix_custom', 'baremix_relu_custom'))

r = base / 'scripts' / 'gen_data.py'
g = r.read_text()
g = g.replace('    alpha = 0.001\n    golden = (np.matmul(input_a.astype(np.float32), input_b.astype(np.float32)) + input_bias).astype(np.float32)\n    golden = np.where(golden >= 0, golden, golden * alpha)', '    golden = (np.matmul(input_a.astype(np.float32), input_b.astype(np.float32)) + input_bias).astype(np.float32)\n    golden = np.maximum(golden, 0)')
r.write_text(g)

m = base / 'main.cpp'
ms = m.read_text()
ms = ms.replace('aclrtlaunch_baremix_custom.h', 'aclrtlaunch_baremix_relu_custom.h')
ms = ms.replace('extern "C" void baremix_custom', 'extern "C" void baremix_relu_custom')
ms = ms.replace('ICPU_RUN_KF(baremix_custom', 'ICPU_RUN_KF(baremix_relu_custom')
ms = ms.replace('ACLRT_LAUNCH_KERNEL(baremix_custom)', 'ACLRT_LAUNCH_KERNEL(baremix_relu_custom)')
m.write_text(ms)
print('patched')
