#include "kernel_operator.h"

extern "C" __global__ __aicore__ void copy_wait640(GM_ADDR input,
                                                    GM_ADDR output,
                                                    GM_ADDR workspace) {
  (void)workspace;
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
  pipe.InitBuffer(inQueue, 1, 640 * sizeof(half));

  AscendC::GlobalTensor<half> inGlobal;
  inGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(input));
  AscendC::LocalTensor<half> inLocal = inQueue.AllocTensor<half>();
  AscendC::DataCopy(inLocal, inGlobal, 640);
  event_t eventId =
      static_cast<event_t>(GetTPipePtr()->FetchEventID(
          AscendC::HardEvent::MTE2_MTE3));
  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
  inQueue.EnQue(inLocal);

  AscendC::LocalTensor<half> copied = inQueue.DeQue<half>();
  AscendC::GlobalTensor<half> outGlobal;
  outGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output));
  AscendC::DataCopy(outGlobal, copied, 640);
  inQueue.FreeTensor(copied);
}
