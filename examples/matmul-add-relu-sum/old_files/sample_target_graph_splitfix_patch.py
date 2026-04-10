from pathlib import Path
base = Path('/home/niu/code/Ascend-MLIR/examples/matmul-add-relu-sum/sample_target_graph')
p = base / 'baremix_custom.cpp'
s = p.read_text()
# Introduce splitRowSize in Init and Process and use it for 2D copy-out repeat
s = s.replace('    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c) + AscendC::GetBlockIdx() * tiling.M * tiling.N / 2);\n    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(bias) + AscendC::GetBlockIdx() * tiling.M * tiling.N / 2);\n\n    pipe->InitBuffer(reluInQueue_, 1, tiling.singleCoreM * tiling.singleCoreN * sizeof(cType) / 2);\n    pipe->InitBuffer(biasQueue_, 1, tiling.singleCoreM * tiling.singleCoreN * sizeof(cType) / 2);\n    pipe->InitBuffer(reluOutQueue_, 1, tiling.singleCoreM * tiling.singleCoreN * sizeof(cType) / 2);',
'''    const uint32_t splitRowSize = tiling.singleCoreM / 2;
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c) + AscendC::GetBlockIdx() * splitRowSize * tiling.singleCoreN);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(bias) + AscendC::GetBlockIdx() * splitRowSize * tiling.singleCoreN);

    pipe->InitBuffer(reluInQueue_, 1, splitRowSize * tiling.singleCoreN * sizeof(cType));
    pipe->InitBuffer(biasQueue_, 1, splitRowSize * tiling.singleCoreN * sizeof(cType));
    pipe->InitBuffer(reluOutQueue_, 1, splitRowSize * tiling.singleCoreN * sizeof(cType));''')

s = s.replace('    AscendC::DataCopy(reluInLocal, cGlobal, tiling.singleCoreM * tiling.singleCoreN / 2);\n    AscendC::DataCopy(biasLocal, biasGlobal, tiling.singleCoreM * tiling.singleCoreN / 2);',
'''    const uint32_t splitCount = (tiling.singleCoreM / 2) * tiling.singleCoreN;
    AscendC::DataCopy(reluInLocal, cGlobal, splitCount);
    AscendC::DataCopy(biasLocal, biasGlobal, splitCount);''')

s = s.replace('    AscendC::Add(reluInLocal, reluInLocal, biasLocal, tiling.singleCoreM * tiling.singleCoreN / 2);\n    biasQueue_.FreeTensor(biasLocal);\n    AscendC::LocalTensor<float> reluOutLocal = reluOutQueue_.AllocTensor<float>();\n    AscendC::Relu(reluOutLocal, reluInLocal, tiling.singleCoreM * tiling.singleCoreN / 2);',
'''    const uint32_t splitCount = (tiling.singleCoreM / 2) * tiling.singleCoreN;
    AscendC::Add(reluInLocal, reluInLocal, biasLocal, splitCount);
    biasQueue_.FreeTensor(biasLocal);
    AscendC::LocalTensor<float> reluOutLocal = reluOutQueue_.AllocTensor<float>();
    AscendC::Relu(reluOutLocal, reluInLocal, splitCount);''')

s = s.replace('    AscendC::DataCopyParams copyParam = {(uint16_t)tiling.singleCoreM, (uint16_t)(tiling.singleCoreN * sizeof(cType) / AscendC::DEFAULT_C0_SIZE), 0, 0};',
              '    const uint32_t splitRowSize = tiling.singleCoreM / 2;\n    AscendC::DataCopyParams copyParam = {(uint16_t)splitRowSize, (uint16_t)(tiling.singleCoreN * sizeof(cType) / AscendC::DEFAULT_C0_SIZE), 0, 0};')
p.write_text(s)
print('patched')
