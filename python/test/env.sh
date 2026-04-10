# simulator
export SOC_VERSION=Ascend910B1
export ASCEND_CPU_SIMULATION=1
export ASCEND_DEVICE_ID=0
export ASCEND_CLEAN_DUMP=${ASCEND_CLEAN_DUMP:-1}  # 默认自动清理仿真器产生的core*dump文件，设为0禁用

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/resolve_ascend_env.sh"

CANN_BASE="$(resolve_ascend_home || true)"
if [ -z "${CANN_BASE}" ]; then
  echo "Set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME before sourcing python/test/env.sh" >&2
  return 1 2>/dev/null || exit 1
fi
export ASCEND_HOME_PATH="${CANN_BASE}"
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
source "${ASCEND_HOME_PATH}/set_env.sh"
ARCH_DIR="$(resolve_cann_arch_dir)"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/runtime/lib64/stub/linux/${ARCH_DIR%%-linux}:${ASCEND_HOME_PATH}/${ARCH_DIR}/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH"

# autofuse
export ASCEND_SLOG_PRINT_TO_STDOUT=1 # 0: 不打屏，1: 打屏
export ASCEND_GLOBAL_LOG_LEVEL=3 # 0: DEBUG  1: INFO  2: WARN  3: ERROR
export DUMP_GRAPH_LEVEL=1 # 0:不dump 1:dump根图在所有阶段的图 2:dump白名单阶段的图 3:dump最后阶段的生成图
export DUMP_GRAPH_PATH=./dump_graph