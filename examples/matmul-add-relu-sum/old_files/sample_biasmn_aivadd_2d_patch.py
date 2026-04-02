from pathlib import Path
base = Path('/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/sample_biasmn_aivadd_2d')

# bias data is [M,N]
r = base / 'scripts' / 'gen_data.py'
g = r.read_text()
old = '    input_bias = np.random.randint(1, 10, [N]).astype(np.float32)'
new = '    input_bias = np.random.randint(1, 10, [M, N]).astype(np.float32)'
if old not in g:
    raise SystemExit('missing bias[N] line')
g = g.replace(old, new, 1)
r.write_text(g)

# kernel: use full bias tile on AIV and 2D DataCopy back to GM
p = base / 'baremix_custom.cpp'
s = p.read_text()
# remove bias from AIC matmul path
s = s.replace('    AscendC::GlobalTensor<biasType> biasGlobal;\n', '')
s = s.replace('    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling.N);\n', '')
s = s.replace('    biasGlobal = biasGlobal[offsetBias];\n', '')
s = s.replace('    matmulObj.SetBias(biasGlobal);\n', '')
# AIV kernel gets bias GM and uses 2D copy-out
s = s.replace('    __aicore__ inline void Init(GM_ADDR c, const TCubeTiling &tiling, AscendC::TPipe *pipe);',
              '    __aicore__ inline void Init(GM_ADDR c, GM_ADDR bias, const TCubeTiling &tiling, AscendC::TPipe *pipe);')
s = s.replace('    __aicore__ inline void LeakyReluCopyIn(const TCubeTiling &tiling);',
              '    __aicore__ inline void LeakyReluCopyIn(const TCubeTiling &tiling);')
s = s.replace('    __aicore__ inline void LeakyReluCopyOut(const TCubeTiling &tiling);',
              '    __aicore__ inline void LeakyReluCopyOut(const TCubeTiling &tiling);')
s = s.replace('    AscendC::GlobalTensor<cType> cGlobal;\n', '    AscendC::GlobalTensor<cType> cGlobal;\n    AscendC::GlobalTensor<cType> biasGlobal;\n')
s = s.replace('    AscendC::TQue<AscendC::TPosition::VECIN, 1> reluInQueue_;\n    AscendC::TQue<AscendC::TPosition::VECOUT, 1> reluOutQueue_;',
              '    AscendC::TQue<AscendC::TPosition::VECIN, 1> reluInQueue_;\n    AscendC::TQue<AscendC::TPosition::VECIN, 1> biasQueue_;\n    AscendC::TQue<AscendC::TPosition::VECOUT, 1> reluOutQueue_;')
s = s.replace('__aicore__ inline void LeakyReluKernel<cType>::Init(GM_ADDR c, const TCubeTiling &tiling, AscendC::TPipe *pipe)',
              '__aicore__ inline void LeakyReluKernel<cType>::Init(GM_ADDR c, GM_ADDR bias, const TCubeTiling &tiling, AscendC::TPipe *pipe)')
s = s.replace('    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c) + AscendC::GetBlockIdx() * tiling.M * tiling.N / 2); //c:v = 1:2, split into 2 parts, for vector calculation\n\n    pipe->InitBuffer(reluInQueue_, 1, tiling.singleCoreM * tiling.singleCoreN * sizeof(cType) /2); // Init input buffer.\n    pipe->InitBuffer(reluOutQueue_, 1, tiling.singleCoreM * tiling.singleCoreN * sizeof(cType)/2); // Init output buffer.',
              '    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c) + AscendC::GetBlockIdx() * tiling.M * tiling.N / 2);\n    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(bias) + AscendC::GetBlockIdx() * tiling.M * tiling.N / 2);\n\n    pipe->InitBuffer(reluInQueue_, 1, tiling.singleCoreM * tiling.singleCoreN * sizeof(cType) / 2);\n    pipe->InitBuffer(biasQueue_, 1, tiling.singleCoreM * tiling.singleCoreN * sizeof(cType) / 2);\n    pipe->InitBuffer(reluOutQueue_, 1, tiling.singleCoreM * tiling.singleCoreN * sizeof(cType) / 2);')
s = s.replace('    AscendC::LocalTensor<float> reluInLocal = reluInQueue_.AllocTensor<float>();\n    AscendC::DataCopy(reluInLocal, cGlobal, tiling.singleCoreM * tiling.singleCoreN / 2);\n    reluInQueue_.EnQue<float>(reluInLocal);',
              '    AscendC::LocalTensor<float> reluInLocal = reluInQueue_.AllocTensor<float>();\n    AscendC::LocalTensor<float> biasLocal = biasQueue_.AllocTensor<float>();\n    AscendC::DataCopy(reluInLocal, cGlobal, tiling.singleCoreM * tiling.singleCoreN / 2);\n    AscendC::DataCopy(biasLocal, biasGlobal, tiling.singleCoreM * tiling.singleCoreN / 2);\n    reluInQueue_.EnQue<float>(reluInLocal);\n    biasQueue_.EnQue<float>(biasLocal);')
s = s.replace('    AscendC::LocalTensor<float> reluInLocal = reluInQueue_.DeQue<float>();\n    AscendC::LocalTensor<float> reluOutLocal = reluOutQueue_.AllocTensor<float>();\n    AscendC::Relu(reluOutLocal, reluInLocal, tiling.singleCoreM * tiling.singleCoreN / 2);\n    reluOutQueue_.EnQue<float>(reluOutLocal);\n    reluInQueue_.FreeTensor(reluInLocal);',
              '    AscendC::LocalTensor<float> reluInLocal = reluInQueue_.DeQue<float>();\n    AscendC::LocalTensor<float> biasLocal = biasQueue_.DeQue<float>();\n    AscendC::Add(reluInLocal, reluInLocal, biasLocal, tiling.singleCoreM * tiling.singleCoreN / 2);\n    biasQueue_.FreeTensor(biasLocal);\n    AscendC::LocalTensor<float> reluOutLocal = reluOutQueue_.AllocTensor<float>();\n    AscendC::Relu(reluOutLocal, reluInLocal, tiling.singleCoreM * tiling.singleCoreN / 2);\n    reluOutQueue_.EnQue<float>(reluOutLocal);\n    reluInQueue_.FreeTensor(reluInLocal);')
s = s.replace('    AscendC::LocalTensor<float> reluOutLocal = reluOutQueue_.DeQue<float>();\n    AscendC::DataCopy(cGlobal, reluOutLocal, tiling.singleCoreM * tiling.singleCoreN / 2);\n    reluOutQueue_.FreeTensor(reluOutLocal);',
              '    AscendC::LocalTensor<float> reluOutLocal = reluOutQueue_.DeQue<float>();\n    AscendC::DataCopyParams copyParam = {(uint16_t)tiling.singleCoreM, (uint16_t)(tiling.singleCoreN * sizeof(cType) / AscendC::DEFAULT_C0_SIZE), 0, 0};\n    AscendC::DataCopy(cGlobal, reluOutLocal, copyParam);\n    reluOutQueue_.FreeTensor(reluOutLocal);')
s = s.replace('        LeakyReluKernel<float> leakyReluKernel;\n        leakyReluKernel.Init(c, tiling, &pipe);',
              '        LeakyReluKernel<float> leakyReluKernel;\n        leakyReluKernel.Init(c, bias, tiling, &pipe);')
p.write_text(s)
print('patched')
