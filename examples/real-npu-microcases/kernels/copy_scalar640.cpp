#include "kernel_operator.h"

extern "C" __global__ __aicore__ void copy_scalar640(GM_ADDR input,
                                                      GM_ADDR output,
                                                      GM_ADDR workspace) {
  (void)workspace;
  AscendC::GlobalTensor<half> inGlobal;
  inGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(input));
  AscendC::GlobalTensor<half> outGlobal;
  outGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output));
  for (uint32_t i = 0; i < 640; ++i)
    outGlobal.SetValue(i, inGlobal.GetValue(i));
}
