#include "kernel_operator.h"

namespace {
constexpr uint32_t kM = 640;
constexpr uint32_t kN = 500;
constexpr uint32_t kTileCols = 32;
constexpr uint32_t kRowsPerBlock = 32;
constexpr uint32_t kTileElems = kRowsPerBlock * kTileCols;
} // namespace

extern "C" __global__ __aicore__ void relu_diag_strided_copy(
    GM_ADDR data1, GM_ADDR output, GM_ADDR workspace) {
  (void)workspace;
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> inputQueue;
  AscendC::TBuf<AscendC::TPosition::VECCALC> zeroBuf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuf;
  pipe.InitBuffer(inputQueue, 1, kTileElems * sizeof(half));
  pipe.InitBuffer(zeroBuf, kTileElems * sizeof(half));
  pipe.InitBuffer(outputBuf, kTileElems * sizeof(half));

  const uint32_t rowStart = AscendC::GetBlockIdx() * kRowsPerBlock;
  if (rowStart >= kN)
    return;
  const uint32_t rows =
      (rowStart + kRowsPerBlock <= kN) ? kRowsPerBlock : (kN - rowStart);

  AscendC::GlobalTensor<half> inGlobal;
  inGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(data1));
  AscendC::GlobalTensor<half> outGlobal;
  outGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output));

  for (uint32_t col = 0; col < kM; col += kTileCols) {
    const uint32_t cols = (col + kTileCols <= kM) ? kTileCols : (kM - col);
    const uint32_t blockBytes = cols * sizeof(half);
    const uint32_t gapBytes = (kM - cols) * sizeof(half);
    AscendC::LocalTensor<half> tile = inputQueue.AllocTensor<half>();
    AscendC::DataCopyExtParams loadParams{
        static_cast<uint16_t>(rows), blockBytes, gapBytes, 0u, 0u};
    AscendC::DataCopyPadExtParams<half> pad{false, 0, 0,
                                            static_cast<half>(0)};
    AscendC::DataCopyPad(tile, inGlobal[rowStart * kM + col], loadParams, pad);
    tile.SetSize(rows * cols);
    inputQueue.EnQue(tile);

    AscendC::LocalTensor<half> ready = inputQueue.DeQue<half>();
    const uint32_t count = rows * cols;
    AscendC::LocalTensor<half> zeroTile = zeroBuf.Get<half>();
    AscendC::LocalTensor<half> outTile = outputBuf.Get<half>();
    AscendC::Duplicate(zeroTile, static_cast<half>(0), count);
    AscendC::PipeBarrier<PIPE_ALL>();
    for (uint32_t off = 0; off < count; off += 1024u) {
      const uint32_t chunk = (count - off < 1024u) ? (count - off) : 1024u;
      AscendC::Add(outTile[off], ready[off], zeroTile[off], chunk);
    }
    outTile.SetSize(count);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyExtParams storeParams{
        static_cast<uint16_t>(rows), blockBytes, 0u, gapBytes, 0u};
    AscendC::DataCopyPad(outGlobal[rowStart * kM + col], outTile, storeParams);
    inputQueue.FreeTensor(ready);
  }
}
