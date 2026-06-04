#include "adv_api/broadcast/broadcast.h"
#include "kernel_operator.h"

namespace {
constexpr uint32_t kM = 640;
constexpr uint32_t kN = 500;
constexpr uint32_t kTileCols = 32;
constexpr uint32_t kRowsPerBlock = 32;
constexpr uint32_t kTileElems = kRowsPerBlock * kTileCols;
} // namespace

extern "C" __global__ __aicore__ void relu_diag_broadcast_add(
    GM_ADDR data0, GM_ADDR data1, GM_ADDR output, GM_ADDR workspace) {
  (void)workspace;
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> data0Queue;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> data1Queue;
  AscendC::TBuf<AscendC::TPosition::VECCALC> broadcastBuf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> zeroBuf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuf;
  pipe.InitBuffer(data0Queue, 1, kTileCols * sizeof(half));
  pipe.InitBuffer(data1Queue, 1, kTileElems * sizeof(half));
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
  AscendC::GlobalTensor<half> data1Global;
  data1Global.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(data1));
  AscendC::GlobalTensor<half> outGlobal;
  outGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output));

  for (uint32_t col = 0; col < kM; col += kTileCols) {
    const uint32_t cols = (col + kTileCols <= kM) ? kTileCols : (kM - col);
    const uint32_t count = rows * cols;
    AscendC::LocalTensor<half> data0Local = data0Queue.AllocTensor<half>();
    AscendC::DataCopy(data0Local, data0Global[col], cols);
    data0Local.SetSize(cols);
    data0Queue.EnQue(data0Local);

    AscendC::LocalTensor<half> data0Tile = data0Queue.DeQue<half>();
    AscendC::LocalTensor<half> broadcastTile = broadcastBuf.Get<half>();
    uint32_t dstShape[2] = {rows, cols};
    uint32_t srcShape[2] = {1, cols};
    AscendC::Broadcast<half, 2, 0>(broadcastTile, data0Tile, dstShape,
                                   srcShape);
    AscendC::PipeBarrier<PIPE_ALL>();

    const uint32_t blockBytes = cols * sizeof(half);
    const uint32_t gapBytes = (kM - cols) * sizeof(half);
    AscendC::LocalTensor<half> data1Local = data1Queue.AllocTensor<half>();
    AscendC::DataCopyExtParams loadParams{
        static_cast<uint16_t>(rows), blockBytes, gapBytes, 0u, 0u};
    AscendC::DataCopyPadExtParams<half> pad{false, 0, 0,
                                            static_cast<half>(0)};
    AscendC::DataCopyPad(data1Local, data1Global[rowStart * kM + col],
                         loadParams, pad);
    data1Local.SetSize(count);
    data1Queue.EnQue(data1Local);

    AscendC::LocalTensor<half> data1Tile = data1Queue.DeQue<half>();
    AscendC::LocalTensor<half> zeroTile = zeroBuf.Get<half>();
    AscendC::LocalTensor<half> outTile = outputBuf.Get<half>();
    AscendC::Duplicate(zeroTile, static_cast<half>(0), count);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Max(outTile, broadcastTile, zeroTile, count);
    AscendC::PipeBarrier<PIPE_ALL>();
    for (uint32_t off = 0; off < count; off += 1024u) {
      const uint32_t chunk = (count - off < 1024u) ? (count - off) : 1024u;
      AscendC::Add(outTile[off], outTile[off], data1Tile[off], chunk);
    }
    outTile.SetSize(count);
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::DataCopyExtParams storeParams{
        static_cast<uint16_t>(rows), blockBytes, 0u, gapBytes, 0u};
    AscendC::DataCopyPad(outGlobal[rowStart * kM + col], outTile, storeParams);
    data0Queue.FreeTensor(data0Tile);
    data1Queue.FreeTensor(data1Tile);
  }
}
