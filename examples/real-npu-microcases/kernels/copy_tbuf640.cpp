#include "kernel_operator.h"

extern "C" __global__ __aicore__ void copy_tbuf640(GM_ADDR input,
                                                    GM_ADDR output,
                                                    GM_ADDR workspace) {
  (void)workspace;
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> buf;
  pipe.InitBuffer(buf, 640 * sizeof(half));

  AscendC::GlobalTensor<half> inGlobal;
  inGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(input));
  AscendC::LocalTensor<half> local = buf.Get<half>();
  AscendC::DataCopy(local, inGlobal, 640);
  event_t eventId =
      static_cast<event_t>(GetTPipePtr()->FetchEventID(
          AscendC::HardEvent::MTE2_MTE3));
  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);

  AscendC::GlobalTensor<half> outGlobal;
  outGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output));
  AscendC::DataCopy(outGlobal, local, 640);
}
