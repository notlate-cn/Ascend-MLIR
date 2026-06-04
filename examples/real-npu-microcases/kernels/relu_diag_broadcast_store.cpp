#include "adv_api/broadcast/broadcast.h"
#include "kernel_operator.h"

namespace {
constexpr uint32_t kM = 640;
constexpr uint32_t kN = 500;
constexpr uint32_t kTileCols = 32;
constexpr uint32_t kRowsPerBlock = 32;
constexpr uint32_t kTileElems = kRowsPerBlock * kTileCols;
} // namespace

extern "C" __global__ __aicore__ void relu_diag_broadcast_store(
    GM_ADDR data0, GM_ADDR output, GM_ADDR workspace) {
  (void)workspace;
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> inputQueue;
  AscendC::TBuf<AscendC::TPosition::VECCALC> broadcastBuf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> zeroBuf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuf;
  pipe.InitBuffer(inputQueue, 1, kTileCols * sizeof(half));
  pipe.InitBuffer(broadcastBuf, kTileElems * sizeof(half));
  pipe.InitBuffer(zeroBuf, kTileElems * sizeof(half));
  pipe.InitBuffer(outputBuf, kTileElems * sizeof(half));

  const uint32_t rowStart = AscendC::GetBlockIdx() * kRowsPerBlock;
  if (rowStart >= kN)
    return;
  const uint32_t rows =
      (rowStart + kRowsPerBlock <= kN) ? kRowsPerBlock : (kN - rowStart);

  AscendC::GlobalTensor<half> data0Global;
  data0Global.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(data0));
  AscendC::GlobalTensor<half> outGlobal;
  outGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output));

  for (uint32_t col = 0; col < kM; col += kTileCols) {
    const uint32_t cols = (col + kTileCols <= kM) ? kTileCols : (kM - col);
    AscendC::LocalTensor<half> data0Local = inputQueue.AllocTensor<half>();
    AscendC::DataCopy(data0Local, data0Global[col], cols);
    data0Local.SetSize(cols);
    inputQueue.EnQue(data0Local);

    AscendC::LocalTensor<half> data0Tile = inputQueue.DeQue<half>();
    AscendC::LocalTensor<half> broadcastTile = broadcastBuf.Get<half>();
    uint32_t dstShape[2] = {rows, cols};
    uint32_t srcShape[2] = {1, cols};
    AscendC::Broadcast<half, 2, 0>(broadcastTile, data0Tile, dstShape,
                                   srcShape);
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::LocalTensor<half> zeroTile = zeroBuf.Get<half>();
    AscendC::LocalTensor<half> outTile = outputBuf.Get<half>();
    const uint32_t count = rows * cols;
    AscendC::Duplicate(zeroTile, static_cast<half>(0), count);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Max(outTile, broadcastTile, zeroTile, count);
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::DataCopyExtParams params{
        static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(half)),
        0u, static_cast<uint32_t>((kM - cols) * sizeof(half)), 0u};
    AscendC::DataCopyPad(outGlobal[rowStart * kM + col], outTile, params);
    inputQueue.FreeTensor(data0Tile);
  }
}
