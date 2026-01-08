include_directories(${PYASC_DIR}/include/ascir/Dialect/Asc/IR)
include_directories(${PYASC_DIR}/include/ascir/API)
include_directories(${CMAKE_BINARY_DIR}/externals/pyasc/include)

# Dialect definitions
set(LLVM_TARGET_DEFINITIONS Dialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(AFIRDialectIncGen)

# Enum definitions (independent from AttrDefs to avoid redefinition)
set(LLVM_TARGET_DEFINITIONS Enums.td)
mlir_tablegen(Enums.h.inc -gen-enum-decls)
mlir_tablegen(Enums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(AFIREnumsIncGen)
add_dependencies(AFIREnumsIncGen AFIRDialectIncGen)

# Attribute definitions (independent from enums in TableGen, but uses enum types in C++)
set(LLVM_TARGET_DEFINITIONS Attrs.td)
mlir_tablegen(Attrs.h.inc -gen-attrdef-decls)
mlir_tablegen(Attrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(AFIRAttrsIncGen)
add_dependencies(AFIRAttrsIncGen AFIRDialectIncGen)

# Operation definitions
set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(AFIROpsIncGen)
add_dependencies(AFIROpsIncGen AFIRDialectIncGen AFIRAttrsIncGen AFIREnumsIncGen MLIRAscIncGen AscIRAPITypesIncGen)

# 确保TableGen能够找到生成的Types.td.inc文件
set(MLIR_TABLEGEN_FLAGS "${MLIR_TABLEGEN_FLAGS} -I${PYASC_DIR}/include/ascir/API")

add_mlir_doc(AFIRDialect AFIRDialect Dialects/ -gen-dialect-doc)
add_mlir_doc(AFIRAttrs AFIRAttrs Dialects/ -gen-attrdef-doc)
add_mlir_doc(AFIROps AFIROps Dialects/ -gen-op-doc)

# Add Ops subdirectory
add_subdirectory(Ops)

add_subdirectory(Transforms)
