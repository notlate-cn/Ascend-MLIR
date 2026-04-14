// Minimal mix fixture source used only to verify the retained legacy Compiler
// mix compile path.
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void auto_gen_fc_relu_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR bias,
    GM_ADDR out, GM_ADDR workspace, GM_ADDR tilingGm) {
  (void)a;
  (void)b;
  (void)bias;
  (void)out;
  (void)workspace;
  (void)tilingGm;
}
