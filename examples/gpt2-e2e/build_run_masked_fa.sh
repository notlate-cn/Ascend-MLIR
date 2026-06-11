#!/usr/bin/env bash
# Build + run the masked-FA device unit test on the real 910C (device 7).
# Invoked from repo root via sync-and-submit.sh --cmd. Re-sources set_env.sh
# inside the script because the CI --cmd login shell resets LD_LIBRARY_PATH and
# drops the driver libs (see project_real_npu_aclnn_direct: aclInit 500000).
set -euo pipefail

TK="${ASCEND_HOME_PATH:-/data/nyh/Ascend/latest}"
[ -f "$TK/set_env.sh" ] && source "$TK/set_env.sh"
# Make sure the real driver libs are reachable for aclInit.
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}"

ARCH="$(uname -m)"
INC="$TK/include";  [ -d "$TK/$ARCH-linux/include" ] && INC="$TK/$ARCH-linux/include"
LIB="$TK/lib64";    [ -d "$TK/$ARCH-linux/lib64" ]   && LIB="$TK/$ARCH-linux/lib64"
echo "TK=$TK ARCH=$ARCH INC=$INC LIB=$LIB"

OUT=/data/gser/aclnn-dev
mkdir -p "$OUT"
g++ -std=c++17 -I include -I "$INC" \
  test/tools/runtime/test_masked_fa_device.cpp lib/Runtime/AclnnOps.cpp \
  -L "$LIB" -Wl,-rpath,"$LIB" \
  -lascendcl -lnnopbase -lopapi -lopapi_nn -lopapi_math -lopapi_transformer \
  -Wl,--allow-shlib-undefined -o "$OUT/t_mfa"

echo "=== running masked-FA device test on device ${ASCEND_DEVICE_ID:-0} ==="
ASCEND_DEVICE_ID="${ASCEND_DEVICE_ID:-7}" "$OUT/t_mfa"
