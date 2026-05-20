#include "kernel_operator.h"

extern "C" __global__ __aicore__ void const640(GM_ADDR output,
                                                GM_ADDR workspace) {
  (void)workspace;
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf;
  pipe.InitBuffer(outBuf, 640 * sizeof(half));
  AscendC::LocalTensor<half> outLocal = outBuf.Get<half>();
  half one = 1.0;
  AscendC::Duplicate(outLocal, one, 640);

  AscendC::GlobalTensor<half> outGlobal;
  outGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output));
  AscendC::DataCopy(outGlobal, outLocal, 640);
}
