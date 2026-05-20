#include "kernel_operator.h"

extern "C" __global__ __aicore__ void relu_only(GM_ADDR input, GM_ADDR output,
                                                 GM_ADDR workspace) {
  (void)workspace;
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
  AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf;
  pipe.InitBuffer(inQueue, 1, 640 * sizeof(half));
  pipe.InitBuffer(outBuf, 640 * sizeof(half));

  AscendC::GlobalTensor<half> inGlobal;
  inGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(input));
  AscendC::LocalTensor<half> inLocal = inQueue.AllocTensor<half>();
  AscendC::DataCopy(inLocal, inGlobal, 640);
  event_t loadEvent =
      static_cast<event_t>(GetTPipePtr()->FetchEventID(
          AscendC::HardEvent::MTE2_V));
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(loadEvent);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(loadEvent);
  inQueue.EnQue(inLocal);

  AscendC::LocalTensor<half> x = inQueue.DeQue<half>();
  AscendC::LocalTensor<half> y = outBuf.Get<half>();
  half zeroValue = 0.0;
  AscendC::Maxs(y, x, zeroValue, 640);
  event_t storeEvent =
      static_cast<event_t>(GetTPipePtr()->FetchEventID(
          AscendC::HardEvent::V_MTE3));
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(storeEvent);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(storeEvent);

  AscendC::GlobalTensor<half> outGlobal;
  outGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output));
  AscendC::DataCopy(outGlobal, y, 640);
  inQueue.FreeTensor(x);
}
