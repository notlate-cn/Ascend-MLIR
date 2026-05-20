#include "kernel_operator.h"
#include "adv_api/broadcast/broadcast.h"

extern "C" __global__ __aicore__ void broadcast_add(GM_ADDR input,
                                                     GM_ADDR bias,
                                                     GM_ADDR output,
                                                     GM_ADDR workspace) {
  (void)workspace;
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> inputQueue;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> biasQueue;
  AscendC::TBuf<AscendC::TPosition::VECCALC> broadcastBuf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf;
  pipe.InitBuffer(inputQueue, 1, 640 * sizeof(half));
  pipe.InitBuffer(biasQueue, 1, 1280 * sizeof(half));
  pipe.InitBuffer(broadcastBuf, 1280 * sizeof(half));
  pipe.InitBuffer(outBuf, 1280 * sizeof(half));

  AscendC::GlobalTensor<half> inputGlobal;
  inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(input));
  AscendC::LocalTensor<half> inputLocal = inputQueue.AllocTensor<half>();
  AscendC::DataCopy(inputLocal, inputGlobal, 640);
  inputQueue.EnQue(inputLocal);

  AscendC::GlobalTensor<half> biasGlobal;
  biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(bias));
  AscendC::LocalTensor<half> biasLocal = biasQueue.AllocTensor<half>();
  AscendC::DataCopy(biasLocal, biasGlobal, 1280);
  event_t loadEvent =
      static_cast<event_t>(GetTPipePtr()->FetchEventID(
          AscendC::HardEvent::MTE2_V));
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(loadEvent);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(loadEvent);
  biasQueue.EnQue(biasLocal);

  AscendC::LocalTensor<half> base = inputQueue.DeQue<half>();
  AscendC::LocalTensor<half> biasTile = biasQueue.DeQue<half>();
  AscendC::LocalTensor<half> broadcastTile = broadcastBuf.Get<half>();
  AscendC::LocalTensor<half> outTile = outBuf.Get<half>();
  uint32_t dstShape[2] = {2, 640};
  uint32_t srcShape[2] = {1, 640};
  AscendC::Broadcast<half, 2, 0>(broadcastTile, base, dstShape, srcShape);
  AscendC::Add(outTile, broadcastTile, biasTile, 1280);
  event_t storeEvent =
      static_cast<event_t>(GetTPipePtr()->FetchEventID(
          AscendC::HardEvent::V_MTE3));
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(storeEvent);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(storeEvent);

  AscendC::GlobalTensor<half> outGlobal;
  outGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output));
  AscendC::DataCopy(outGlobal, outTile, 1280);
  inputQueue.FreeTensor(base);
  biasQueue.FreeTensor(biasTile);
}
