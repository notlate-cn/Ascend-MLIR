set(LLVM_TARGET_DEFINITIONS ShapeHelperOpInterface.td)
mlir_tablegen(ShapeHelperOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeHelperOpInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(AFIRShapeHelperOpInterfaceIncGen)
