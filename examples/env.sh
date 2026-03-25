SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
export PATH=$PATH:${PROJECT_ROOT}/build/bin
# CANN toolkit
CANN_BASE="${HOME}/Ascend/latest"
if [ -d "$CANN_BASE" ]; then
  export ASCEND_HOME_PATH="$CANN_BASE"
  # Simulator runtime libraries (required for runner / validator dlopen)
  SOC_VERSION="${SOC_VERSION:-Ascend910B1}"
  # Detect host architecture for CANN paths
  CANN_ARCH="$(uname -m)"
  case "$CANN_ARCH" in
    x86_64)  CANN_ARCH="x86_64-linux" ;;
    aarch64) CANN_ARCH="aarch64-linux" ;;
    *)       CANN_ARCH="aarch64-linux" ;;  # default to ARM
  esac
  SIM_LIB="$CANN_BASE/$CANN_ARCH/simulator/$SOC_VERSION/lib"
  BASE_LIB="$CANN_BASE/$CANN_ARCH/lib64"
  export LD_LIBRARY_PATH="$SIM_LIB:$BASE_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi