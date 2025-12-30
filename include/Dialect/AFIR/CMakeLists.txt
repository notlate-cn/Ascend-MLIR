include_directories(${PYASC_DIR}/include/ascir/Dialect/Asc/IR)
include_directories(${PYASC_DIR}/include/ascir/API)
include_directories(${CMAKE_BINARY_DIR}/externals/pyasc/include)

set(LLVM_TARGET_DEFINITIONS AFIRDialect.td)
mlir_tablegen(AFIRDialect.h.inc -gen-dialect-decls)
mlir_tablegen(AFIRDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(AFIRDialectIncGen)

set(LLVM_TARGET_DEFINITIONS AFIROps.td)
mlir_tablegen(AFIROps.h.inc -gen-op-decls)
mlir_tablegen(AFIROps.cpp.inc -gen-op-defs)
add_public_tablegen_target(AFIROpsIncGen)
add_dependencies(AFIROpsIncGen AscIRAPITypesIncGen)

# 确保TableGen能够找到生成的Types.td.inc文件
set(MLIR_TABLEGEN_FLAGS "${MLIR_TABLEGEN_FLAGS} -I${PYASC_DIR}/include/ascir/API")

# Add AFIROps subdirectory
add_subdirectory(AFIROps)

add_mlir_doc(AFIRDialect AFIRDialect Dialects/ -gen-dialect-doc)
add_mlir_doc(AFIROps AFIROps Dialects/ -gen-op-doc)

add_subdirectory(Transforms)
