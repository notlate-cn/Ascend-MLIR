#include "kernel_operator.h"

extern "C" __global__ __aicore__ void relu_only(GM_ADDR input, GM_ADDR output,
                                                 GM_ADDR workspace) {
  (void)workspace;
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
  AscendC::TBuf<AscendC::TPosition::VECCALC> zeroBuf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf;
  pipe.InitBuffer(inQueue, 1, 640 * sizeof(half));
  pipe.InitBuffer(zeroBuf, 640 * sizeof(half));
  pipe.InitBuffer(outBuf, 640 * sizeof(half));

  AscendC::GlobalTensor<half> inGlobal;
  inGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(input));
  AscendC::LocalTensor<half> inLocal = inQueue.AllocTensor<half>();
  AscendC::DataCopy(inLocal, inGlobal, 640);
  inQueue.EnQue(inLocal);

  AscendC::LocalTensor<half> x = inQueue.DeQue<half>();
  AscendC::LocalTensor<half> zero = zeroBuf.Get<half>();
  AscendC::LocalTensor<half> y = outBuf.Get<half>();
  half zeroValue = 0.0;
  AscendC::Duplicate(zero, zeroValue, 640);
  AscendC::Max(y, x, zero, 640);

  AscendC::GlobalTensor<half> outGlobal;
  outGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output));
  AscendC::DataCopy(outGlobal, y, 640);
  inQueue.FreeTensor(x);
}
