export ASCEND_HOME_PATH=/home/gser/Ascend/cann/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "${PROJECT_ROOT}/scripts/resolve_ascend_env.sh"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
export PATH=$PATH:${PROJECT_ROOT}/build/bin

CANN_BASE="$(resolve_ascend_home || true)"
if [ -z "$CANN_BASE" ]; then
  echo "Set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME before sourcing examples/env.sh" >&2
  return 1 2>/dev/null || exit 1
fi

export ASCEND_HOME_PATH="$CANN_BASE"
SOC_VERSION="${SOC_VERSION:-Ascend910B1}"
CANN_ARCH="$(resolve_cann_arch_dir)"
SIM_LIB="$CANN_BASE/$CANN_ARCH/simulator/$SOC_VERSION/lib"
BASE_LIB="$CANN_BASE/$CANN_ARCH/lib64"
# devlib provides libascend_hal.so needed by runtime-session; listed after
# BASE_LIB so that lib64's libmetadef.so takes precedence over devlib's.
DEV_LIB="$CANN_BASE/$CANN_ARCH/devlib/linux/$(uname -m)"
export LD_LIBRARY_PATH="$SIM_LIB:$BASE_LIB:$DEV_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
