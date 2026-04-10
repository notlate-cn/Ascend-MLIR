from pathlib import Path
base = Path('/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/sample_nobias')

# data: no bias contribution, still emit a dummy bias file for host ABI compatibility
r = base / 'scripts' / 'gen_data.py'
g = r.read_text()
g = g.replace('    input_bias = np.random.randint(1, 10, [N]).astype(np.float32)', '    input_bias = np.zeros([N], dtype=np.float32)', 1)
g = g.replace('    golden = (np.matmul(input_a.astype(np.float32), input_b.astype(np.float32)) + input_bias).astype(np.float32)', '    golden = np.matmul(input_a.astype(np.float32), input_b.astype(np.float32)).astype(np.float32)', 1)
r.write_text(g)

# kernel: remove bias tensor setup and SetBias call, keep structure unchanged
p = base / 'baremix_custom.cpp'
s = p.read_text()
s = s.replace('    AscendC::GlobalTensor<biasType> biasGlobal;\n', '')
s = s.replace('    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling.N);\n', '')
s = s.replace('    biasGlobal = biasGlobal[offsetBias];\n', '')
s = s.replace('    matmulObj.SetBias(biasGlobal);\n', '')
# offsetBias becomes unused, simplify signature and call
s = s.replace('    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB,\n                                      int32_t &offsetC, int32_t &offsetBias);',
              '    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB,\n                                      int32_t &offsetC);')
s = s.replace('    int32_t offsetA, offsetB, offsetC, offsetBias;\n    CalcOffset(AscendC::GetBlockIdx(), tiling, offsetA, offsetB, offsetC, offsetBias); // Calculate the gm offset based on the blockidx.\n',
              '    int32_t offsetA, offsetB, offsetC;\n    CalcOffset(AscendC::GetBlockIdx(), tiling, offsetA, offsetB, offsetC); // Calculate the gm offset based on the blockidx.\n')
s = s.replace('                                                             int32_t &offsetC,\n                                                             int32_t &offsetBias)',
              '                                                             int32_t &offsetC)')
s = s.replace('    offsetBias = nCoreIndx * tiling.singleCoreN;\n', '')
p.write_text(s)
print('patched')
