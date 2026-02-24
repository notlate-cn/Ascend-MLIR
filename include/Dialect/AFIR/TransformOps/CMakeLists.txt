set(LLVM_TARGET_DEFINITIONS AFIRTransformOps.td)
mlir_tablegen(AFIRTransformOps.h.inc -gen-op-decls)
mlir_tablegen(AFIRTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(AFIRTransformOpsIncGen)
