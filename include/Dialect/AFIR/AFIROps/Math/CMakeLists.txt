# Math operations TableGen definitions

# Ensure TableGen can find generated Types.td.inc file
set(MLIR_TABLEGEN_FLAGS "${MLIR_TABLEGEN_FLAGS} -I${PYASC_DIR}/include/ascir/API")

set(LLVM_TARGET_DEFINITIONS Elementwise.td)
mlir_tablegen(Elementwise.h.inc -gen-op-decls)
mlir_tablegen(Elementwise.cpp.inc -gen-op-defs)
add_public_tablegen_target(AFIRElementwiseOpsIncGen)
add_dependencies(AFIRElementwiseOpsIncGen AFIROpsIncGen)