set(LLVM_TARGET_DEFINITIONS AFIRDialect.td)
mlir_tablegen(AFIRDialect.h.inc -gen-dialect-decls)
mlir_tablegen(AFIRDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(AFIRDialectIncGen)

set(LLVM_TARGET_DEFINITIONS AFIROps.td)
mlir_tablegen(AFIROps.h.inc -gen-op-decls)
mlir_tablegen(AFIROps.cpp.inc -gen-op-defs)
add_public_tablegen_target(AFIROpsIncGen)

add_mlir_doc(AFIRDialect AFIRDialect Dialects/ -gen-dialect-doc)
add_mlir_doc(AFIROps AFIROps Dialects/ -gen-op-doc)

add_subdirectory(Transforms)
