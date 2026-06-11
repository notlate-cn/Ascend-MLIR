//===- AfirConfusionTransposeKernelSource.h -------------------------------===//
//
// Device-side AscendC source for the ConfusionTranspose (ND2ND_ONLY) transpose,
// vendored from AutoFuse (ge-eco .../autofuse/ascendc/api/transpose.h, trimmed
// to the rank-2 [1,0] scene) and ported to compute its tiling device-side.
//
// Emitted INLINE into a generated kernel's preamble by CannTranslation when the
// kernel uses ConfusionTranspose (gated on the `afir.uses_confusion_transpose`
// func attr that LinalgToAscendC stamps). Inlining avoids any kernel-JIT include
// path plumbing — the code is self-contained in the .cpp.
//
// Why vendored: CANN 9.0.0 ships the ConfusionTranspose device adv_api but NOT
// its host tiling (`GetConfusionTransposeOnlyTilingInfo` — impl dir empty). AF
// builds directly on the low-level `AscendC::TransDataTo5HD` intrinsic (shipped)
// and computes the tiling itself; this is that self-contained path with the
// tiling derived device-side from the tile [H,W]. Validated f16+f32 on camodel
// (examples/confusion-transpose-spike).
//
//===----------------------------------------------------------------------===//
#ifndef TARGET_CANNKERNEL_AFIRCONFUSIONTRANSPOSEKERNELSOURCE_H
#define TARGET_CANNKERNEL_AFIRCONFUSIONTRANSPOSEKERNELSOURCE_H

namespace mlir::afir {

// Raw device source. Requires (from kernel_operator.h, already included by the
// kernel preamble): AscendC::{LocalTensor, TransDataTo5HD, TransDataTo5HDParams,
// half}, BLOCK_CUBE(=16), ONE_BLK_SIZE(=32), NCHW_CONV_ADDR_LIST_SIZE(=16).
inline constexpr const char *kAfirConfusionTransposeSource = R"AFIRCT(
namespace codegen {
using namespace AscendC;

struct ConfusionTransposeLastTiling {
  uint32_t height;
  uint32_t width;
  uint32_t highBlock;
  uint32_t stride;
  uint32_t blockSize;
  uint32_t repeat;
  uint32_t firstAxisAlign;
  uint32_t firstAxisRem;
  uint32_t secondAxisAlign;
  uint32_t secondAxisRem;
};

__aicore__ inline uint32_t AfirAlignUp(uint32_t origin, uint32_t align) {
  return (origin % align == 0) ? origin : (origin + align - origin % align);
}

template <typename T>
__aicore__ inline void AfirFillTranspose2DTiling(uint32_t height, uint32_t width,
                                                 ConfusionTransposeLastTiling &tiling) {
  uint32_t blockSize = ONE_BLK_SIZE / sizeof(T);
  tiling.height = height;
  tiling.width = width;
  tiling.highBlock = height / BLOCK_CUBE;
  tiling.blockSize = blockSize;
  tiling.repeat = width / blockSize;
  tiling.firstAxisAlign = AfirAlignUp(height, BLOCK_CUBE);
  tiling.firstAxisRem = height % BLOCK_CUBE;
  tiling.secondAxisAlign = AfirAlignUp(width, 16);
  tiling.secondAxisRem = width % blockSize;
  tiling.stride = tiling.firstAxisAlign;
}

template <typename T>
__aicore__ inline void Transpose10ConfigMatrixA(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const ConfusionTransposeLastTiling &tiling) {
  uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  AscendC::TransDataTo5HDParams transParams;
  uint32_t loopIdx, i;
  transParams.repeatTimes = tiling.repeat;
  transParams.srcRepStride = tiling.repeat > 1 ? 1 : 0;
  transParams.dstRepStride = tiling.repeat > 1 ? tiling.stride : 0;
  for (loopIdx = 0; loopIdx < tiling.highBlock; loopIdx++) {
    if constexpr (sizeof(T) == sizeof(half)) {
      for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * i].GetPhyAddr());
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[loopIdx * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
      }
      TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
    } else if constexpr (sizeof(T) == sizeof(float)) {
      for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i = i + 2) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2)].GetPhyAddr());
        dstLocalList[i + 1] = reinterpret_cast<uint64_t>(dstTensor[loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2) + tiling.blockSize].GetPhyAddr());
      }
      for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[loopIdx * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
      }
      TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
    }
  }
}

template <typename T>
__aicore__ inline void Transpose10ConfigMatrixB(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const ConfusionTransposeLastTiling &tiling) {
  uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  AscendC::TransDataTo5HDParams transParams;
  uint32_t i;
  transParams.repeatTimes = tiling.repeat;
  transParams.srcRepStride = tiling.repeat > 1 ? 1 : 0;
  transParams.dstRepStride = tiling.repeat > 1 ? tiling.stride : 0;
  if constexpr (sizeof(T) == sizeof(half)) {
    for (i = 0; i < tiling.firstAxisRem; i++) {
      dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * i].GetPhyAddr());
      srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
    }
    for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
      dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * i].GetPhyAddr());
      srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * (tiling.firstAxisRem - 1)].GetPhyAddr());
    }
    TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
  } else if constexpr (sizeof(T) == sizeof(float)) {
    for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i = i + 2) {
      dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2)].GetPhyAddr());
      dstLocalList[i + 1] = reinterpret_cast<uint64_t>(dstTensor[tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2) + tiling.blockSize].GetPhyAddr());
    }
    for (i = 0; i < tiling.firstAxisRem; i++) {
      srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
    }
    for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
      srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * (tiling.firstAxisRem - 1)].GetPhyAddr());
    }
    TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
  }
}

template <typename T>
__aicore__ inline void Transpose10ConfigMatrixC(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const ConfusionTransposeLastTiling &tiling, const LocalTensor<uint8_t> &tmpbuf) {
  uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  AscendC::TransDataTo5HDParams transParams;
  const LocalTensor<T> tmpTensor = tmpbuf.ReinterpretCast<T>();
  const uint64_t dstAddrOffset = tiling.repeat * tiling.blockSize * tiling.firstAxisAlign;
  const uint64_t srcAddrOffset = tiling.repeat * tiling.blockSize;
  uint32_t loopIdx, i;
  transParams.repeatTimes = 1;
  transParams.srcRepStride = 0;
  transParams.dstRepStride = 0;
  for (loopIdx = 0; loopIdx < tiling.highBlock; loopIdx++) {
    if constexpr (sizeof(T) == sizeof(half)) {
      for (i = 0; i < tiling.secondAxisRem; i++) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstAddrOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * i].GetPhyAddr());
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcAddrOffset + loopIdx * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
      }
      for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem) * tiling.blockSize].GetPhyAddr());
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcAddrOffset + loopIdx * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
      }
      TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
    } else if constexpr (sizeof(T) == sizeof(float)) {
      for (i = 0; i < tiling.secondAxisRem * 2; i = i + 2) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstAddrOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2)].GetPhyAddr());
        dstLocalList[i + 1] = reinterpret_cast<uint64_t>(dstTensor[dstAddrOffset + loopIdx * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2) + tiling.blockSize].GetPhyAddr());
      }
      for (; i < NCHW_CONV_ADDR_LIST_SIZE; i = i + 2) {
        dstLocalList[i] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem * 2) * tiling.blockSize].GetPhyAddr());
        dstLocalList[i + 1] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem * 2 + 1) * tiling.blockSize].GetPhyAddr());
      }
      for (i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
        srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcAddrOffset + loopIdx * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
      }
      TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
    }
  }
}

template <typename T>
__aicore__ inline void Transpose10ConfigMatrixD(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const ConfusionTransposeLastTiling &tiling, const LocalTensor<uint8_t> &tmpbuf) {
  uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
  AscendC::TransDataTo5HDParams transParams;
  const LocalTensor<T> tmpTensor = tmpbuf.ReinterpretCast<T>();
  const uint64_t dstAddrOffset = tiling.repeat * tiling.blockSize * tiling.firstAxisAlign;
  const uint64_t srcAddrOffset = tiling.repeat * tiling.blockSize;
  uint32_t i;
  transParams.repeatTimes = 1;
  transParams.srcRepStride = 0;
  transParams.dstRepStride = 0;
  if constexpr (sizeof(T) == sizeof(half)) {
    for (i = 0; i < tiling.secondAxisRem; i++) {
      dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstAddrOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * i].GetPhyAddr());
    }
    for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
      dstLocalList[i] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem) * tiling.blockSize].GetPhyAddr());
    }
    for (i = 0; i < tiling.firstAxisRem; i++) {
      srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcAddrOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
    }
    for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
      srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcAddrOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * (tiling.firstAxisRem - 1)].GetPhyAddr());
    }
    TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
  } else if constexpr (sizeof(T) == sizeof(float)) {
    for (i = 0; i < tiling.secondAxisRem * 2; i = i + 2) {
      dstLocalList[i] = reinterpret_cast<uint64_t>(dstTensor[dstAddrOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2)].GetPhyAddr());
      dstLocalList[i + 1] = reinterpret_cast<uint64_t>(dstTensor[dstAddrOffset + tiling.highBlock * BLOCK_CUBE + tiling.firstAxisAlign * (i / 2) + tiling.blockSize].GetPhyAddr());
    }
    for (; i < NCHW_CONV_ADDR_LIST_SIZE; i = i + 2) {
      dstLocalList[i] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem * 2) * tiling.blockSize].GetPhyAddr());
      dstLocalList[i + 1] = reinterpret_cast<uint64_t>(tmpTensor[(i - tiling.secondAxisRem * 2 + 1) * tiling.blockSize].GetPhyAddr());
    }
    for (i = 0; i < tiling.firstAxisRem; i++) {
      srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcAddrOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * i].GetPhyAddr());
    }
    for (; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
      srcLocalList[i] = reinterpret_cast<uint64_t>(srcTensor[srcAddrOffset + tiling.highBlock * BLOCK_CUBE * tiling.secondAxisAlign + tiling.secondAxisAlign * (tiling.firstAxisRem - 1)].GetPhyAddr());
    }
    TransDataTo5HD<T>(dstLocalList, srcLocalList, transParams);
  }
}

template <typename T>
__aicore__ inline void ConfusionTranspose10Compute(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const LocalTensor<uint8_t> &tmpbuf,
                                                   const ConfusionTransposeLastTiling &tiling) {
  if (tiling.highBlock > 0) {
    if (tiling.repeat > 0) {
      Transpose10ConfigMatrixA(dstTensor, srcTensor, tiling);
    }
    if (tiling.secondAxisRem > 0) {
      Transpose10ConfigMatrixC(dstTensor, srcTensor, tiling, tmpbuf);
    }
  }
  if (tiling.firstAxisRem > 0) {
    if (tiling.repeat > 0) {
      Transpose10ConfigMatrixB(dstTensor, srcTensor, tiling);
    }
    if (tiling.secondAxisRem > 0) {
      Transpose10ConfigMatrixD(dstTensor, srcTensor, tiling, tmpbuf);
    }
  }
}

template <typename T>
__aicore__ inline void AfirConfusionTranspose2D(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
                                                const LocalTensor<uint8_t> &tmpbuf, uint32_t height, uint32_t width) {
  ConfusionTransposeLastTiling tiling;
  AfirFillTranspose2DTiling<T>(height, width, tiling);
  ConfusionTranspose10Compute<T>(dstTensor, srcTensor, tmpbuf, tiling);
}

} // namespace codegen
)AFIRCT";

} // namespace mlir::afir

#endif // TARGET_CANNKERNEL_AFIRCONFUSIONTRANSPOSEKERNELSOURCE_H
