export ASCEND_INSTALL_PATH=~/Ascend/latest
source ${ASCEND_INSTALL_PATH}/../set_env.sh
export LD_LIBRARY_PATH=${ASCEND_INSTALL_PATH}/runtime/lib64/stub/linux/aarch64:$LD_LIBRARY_PATH
export ASCEND_SLOG_PRINT_TO_STDOUT=1 # 0: 不打屏，1: 打屏
export ASCEND_GLOLOG_LEVEL=3 # 0: DEBUG  1: INFO  2: WARN  3: ERROR