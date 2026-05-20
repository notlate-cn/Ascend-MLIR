#include "kernel_operator.h"

extern "C" __global__ __aicore__ void
const_with_input640(GM_ADDR input, GM_ADDR output, GM_ADDR workspace) {
  (void)input;
  (void)workspace;
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf;
  pipe.InitBuffer(outBuf, 640 * sizeof(half));
  AscendC::LocalTensor<half> outLocal = outBuf.Get<half>();
  half two = 2.0;
  AscendC::Duplicate(outLocal, two, 640);

  AscendC::GlobalTensor<half> outGlobal;
  outGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output));
  AscendC::DataCopy(outGlobal, outLocal, 640);
}
