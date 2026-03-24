export PATH=$PATH:/home/niu/code/Ascend-MLIR/build/bin

# CANN toolkit（通过 Mac 挂载路径访问）
CANN_BASE="/mnt/mac/Volumes/GM9/cann/Ascend/20251209_newest/ascend-toolkit/latest"
if [ -d "$CANN_BASE" ]; then
  export ASCEND_HOME_PATH="$CANN_BASE"
  # Simulator runtime libraries (required for runner / validator dlopen)
  SOC_VERSION="${SOC_VERSION:-Ascend910B1}"
  SIM_LIB="$CANN_BASE/aarch64-linux/simulator/$SOC_VERSION/lib"
  BASE_LIB="$CANN_BASE/aarch64-linux/lib64"
  export LD_LIBRARY_PATH="$SIM_LIB:$BASE_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi