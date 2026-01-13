export ASCEND_HOME_PATH=~/Ascend/latest
source ${ASCEND_HOME_PATH}/../set_env.sh
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/runtime/lib64/stub/linux/aarch64:/home/niu/Ascend/20251209_newest/ascend-toolkit/latest/aarch64-linux/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH
export ASCEND_SLOG_PRINT_TO_STDOUT=1 # 0: 不打屏，1: 打屏
export ASCEND_GLOLOG_LEVEL=3 # 0: DEBUG  1: INFO  2: WARN  3: ERROR
export SOC_VERSION=Ascend910B1
export ASCEND_CPU_SIMULATION=1
export ASCEND_DEVICE_ID=0