export PATH=$PATH:/home/niu/code/Ascend-MLIR/build/bin

# CANN toolkit（通过 Mac 挂载路径访问）
CANN_BASE="/mnt/mac/Volumes/GM9/cann/Ascend/20251209_newest/ascend-toolkit/latest"
if [ -d "$CANN_BASE" ]; then
  export ASCEND_HOME_PATH="$CANN_BASE"
fi