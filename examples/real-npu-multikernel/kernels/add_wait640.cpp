#include "kernel_operator.h"

extern "C" __global__ __aicore__ void add_wait640(GM_ADDR lhs, GM_ADDR rhs,
                                                   GM_ADDR output,
                                                   GM_ADDR workspace) {
  (void)workspace;
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> lhsQueue;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> rhsQueue;
  AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf;
  pipe.InitBuffer(lhsQueue, 1, 640 * sizeof(half));
  pipe.InitBuffer(rhsQueue, 1, 640 * sizeof(half));
  pipe.InitBuffer(outBuf, 640 * sizeof(half));

  AscendC::GlobalTensor<half> lhsGlobal;
  lhsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(lhs));
  AscendC::LocalTensor<half> lhsLocal = lhsQueue.AllocTensor<half>();
  AscendC::DataCopy(lhsLocal, lhsGlobal, 640);

  AscendC::GlobalTensor<half> rhsGlobal;
  rhsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(rhs));
  AscendC::LocalTensor<half> rhsLocal = rhsQueue.AllocTensor<half>();
  AscendC::DataCopy(rhsLocal, rhsGlobal, 640);

  event_t loadEvent =
      static_cast<event_t>(GetTPipePtr()->FetchEventID(
          AscendC::HardEvent::MTE2_V));
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(loadEvent);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(loadEvent);

  lhsQueue.EnQue(lhsLocal);
  rhsQueue.EnQue(rhsLocal);
  AscendC::LocalTensor<half> lhsTile = lhsQueue.DeQue<half>();
  AscendC::LocalTensor<half> rhsTile = rhsQueue.DeQue<half>();
  AscendC::LocalTensor<half> outTile = outBuf.Get<half>();
  AscendC::Add(outTile, lhsTile, rhsTile, 640);

  event_t storeEvent =
      static_cast<event_t>(GetTPipePtr()->FetchEventID(
          AscendC::HardEvent::V_MTE3));
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(storeEvent);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(storeEvent);

  AscendC::GlobalTensor<half> outGlobal;
  outGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output));
  AscendC::DataCopy(outGlobal, outTile, 640);
  lhsQueue.FreeTensor(lhsTile);
  rhsQueue.FreeTensor(rhsTile);
}
