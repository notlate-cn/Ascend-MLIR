# simulator
export SOC_VERSION=Ascend910B1
export ASCEND_CPU_SIMULATION=1
export ASCEND_DEVICE_ID=0

# cann environment
export ASCEND_HOME_PATH=~/Ascend/latest
source ${ASCEND_HOME_PATH}/../set_env.sh
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/runtime/lib64/stub/linux/$(arch):${ASCEND_HOME_PATH}/$(arch)-linux/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH

# autofuse
export ASCEND_SLOG_PRINT_TO_STDOUT=1 # 0: 不打屏，1: 打屏
export ASCEND_GLOBAL_LOG_LEVEL=0 # 0: DEBUG  1: INFO  2: WARN  3: ERROR
export DUMP_GRAPH_LEVEL=1 # 0:不dump 1:dump根图在所有阶段的图 2:dump白名单阶段的图 3:dump最后阶段的生成图
export DUMP_GRAPH_PATH=./dump_graph