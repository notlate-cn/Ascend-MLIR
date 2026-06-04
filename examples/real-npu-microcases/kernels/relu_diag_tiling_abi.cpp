#include "adv_api/broadcast/broadcast.h"
#include "kernel_operator.h"

#include <cstdint>

struct TilingData {
  int64_t TB_M;
  int64_t TB_N;
  int64_t dim_arg0_0;
  int64_t dim_arg1_0;
  int64_t dim_arg1_1;
  int64_t dim_arg0_1;
};

namespace {
__aicore__ inline uint32_t align32(uint32_t bytes) {
  if (bytes == 0)
    return 0;
  uint32_t aligned = ((bytes + 31u) / 32u) * 32u;
  return aligned < 32u ? 32u : aligned;
}
} // namespace

extern "C" __global__ __aicore__ void relu_diag_tiling_abi(
    GM_ADDR data0, GM_ADDR data1, GM_ADDR output, GM_ADDR workspace,
    TilingData tiling) {
  (void)workspace;
  const uint32_t rowsPerBlock = static_cast<uint32_t>(tiling.TB_M);
  const uint32_t tileCols = static_cast<uint32_t>(tiling.TB_N);
  const uint32_t m = static_cast<uint32_t>(tiling.dim_arg0_0);
  const uint32_t n = static_cast<uint32_t>(tiling.dim_arg1_0);
  const uint32_t data1Stride = static_cast<uint32_t>(tiling.dim_arg1_1);
  const uint32_t data0Stride = static_cast<uint32_t>(tiling.dim_arg0_1);
  if (rowsPerBlock == 0 || tileCols == 0)
    return;

  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> data1Queue;
  AscendC::TBuf<AscendC::TPosition::VECCALC> zeroBuf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> broadcastBuf;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> data0Queue;
  const uint32_t tileBytes = align32(rowsPerBlock * tileCols * sizeof(half));
  pipe.InitBuffer(zeroBuf, tileBytes);
  pipe.InitBuffer(broadcastBuf, tileBytes);
  pipe.InitBuffer(data1Queue, 1, tileBytes);
  pipe.InitBuffer(outputQueue, 1, tileBytes);
  pipe.InitBuffer(data0Queue, 1, align32(tileCols * sizeof(half)));

  const uint32_t rowStart = AscendC::GetBlockIdx() * rowsPerBlock;
  if (rowStart >= n)
    return;
  const uint32_t rows =
      (rowStart + rowsPerBlock <= n) ? rowsPerBlock : (n - rowStart);

  AscendC::GlobalTensor<half> data0Global;
  data0Global.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(data0));
  AscendC::GlobalTensor<half> data1Global;
  data1Global.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(data1));
  AscendC::GlobalTensor<half> outGlobal;
  outGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output));

  for (uint32_t col = 0; col < m; col += tileCols) {
    const uint32_t cols = (col + tileCols <= m) ? tileCols : (m - col);
    const uint32_t count = rows * cols;
    AscendC::LocalTensor<half> data0Local = data0Queue.AllocTensor<half>();
    AscendC::DataCopy(data0Local, data0Global[col * data0Stride], cols);
    data0Local.SetSize(cols);
    data0Queue.EnQue(data0Local);

    AscendC::LocalTensor<half> data0Tile = data0Queue.DeQue<half>();
    AscendC::LocalTensor<half> broadcastTile = broadcastBuf.Get<half>();
    uint32_t dstShape[2] = {rows, cols};
    uint32_t srcShape[2] = {1u, cols};
    AscendC::Broadcast<half, 2, 0>(broadcastTile, data0Tile, dstShape,
                                   srcShape);
    AscendC::PipeBarrier<PIPE_ALL>();

    const uint32_t blockBytes = cols * sizeof(half);
    const uint32_t loadGapBytes = (data1Stride - cols) * sizeof(half);
    AscendC::LocalTensor<half> data1Local = data1Queue.AllocTensor<half>();
    AscendC::DataCopyExtParams loadParams{
        static_cast<uint16_t>(rows), blockBytes, loadGapBytes, 0u, 0u};
    AscendC::DataCopyPadExtParams<half> pad{false, 0, 0,
                                            static_cast<half>(0)};
    AscendC::DataCopyPad(data1Local, data1Global[rowStart * data1Stride + col],
                         loadParams, pad);
    data1Local.SetSize(count);
    data1Queue.EnQue(data1Local);

    AscendC::LocalTensor<half> data1Tile = data1Queue.DeQue<half>();
    AscendC::LocalTensor<half> outTile = outputQueue.AllocTensor<half>();
    AscendC::LocalTensor<half> zeroTile = zeroBuf.Get<half>();
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
    data0Queue.FreeTensor(data0Tile);
    data1Queue.FreeTensor(data1Tile);
    outputQueue.EnQue(outTile);

    AscendC::LocalTensor<half> ready = outputQueue.DeQue<half>();
    const uint32_t storeGapBytes = (m - cols) * sizeof(half);
    AscendC::DataCopyExtParams storeParams{
        static_cast<uint16_t>(rows), blockBytes, 0u, storeGapBytes, 0u};
    AscendC::DataCopyPad(outGlobal[rowStart * m + col], ready, storeParams);
    outputQueue.FreeTensor(ready);
  }
}
