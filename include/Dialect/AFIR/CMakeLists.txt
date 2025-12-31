include_directories(${PYASC_DIR}/include/ascir/Dialect/Asc/IR)
include_directories(${PYASC_DIR}/include/ascir/API)
include_directories(${CMAKE_BINARY_DIR}/externals/pyasc/include)

# Dialect definitions
set(LLVM_TARGET_DEFINITIONS AFIRDialect.td)
mlir_tablegen(AFIRDialect.h.inc -gen-dialect-decls)
mlir_tablegen(AFIRDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(AFIRDialectIncGen)

# Type definitions
set(LLVM_TARGET_DEFINITIONS AFIRTypes.td)
mlir_tablegen(AFIRTypes.h.inc -gen-typedef-decls)
mlir_tablegen(AFIRTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(AFIRTypesIncGen)
add_dependencies(AFIRTypesIncGen AFIRDialectIncGen)

# Enum definitions (independent from AttrDefs to avoid redefinition)
set(LLVM_TARGET_DEFINITIONS AFIREnums.td)
mlir_tablegen(AFIREnums.h.inc -gen-enum-decls)
mlir_tablegen(AFIREnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(AFIREnumsIncGen)
add_dependencies(AFIREnumsIncGen AFIRDialectIncGen)

# Attribute definitions (independent from enums in TableGen, but uses enum types in C++)
set(LLVM_TARGET_DEFINITIONS AFIRAttrs.td)
mlir_tablegen(AFIRAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(AFIRAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(AFIRAttrsIncGen)
add_dependencies(AFIRAttrsIncGen AFIRDialectIncGen)

# Operation definitions
set(LLVM_TARGET_DEFINITIONS AFIROps.td)
mlir_tablegen(AFIROps.h.inc -gen-op-decls)
mlir_tablegen(AFIROps.cpp.inc -gen-op-defs)
add_public_tablegen_target(AFIROpsIncGen)
add_dependencies(AFIROpsIncGen AFIRDialectIncGen AFIRTypesIncGen AFIRAttrsIncGen AFIREnumsIncGen AscIRAPITypesIncGen)

# 确保TableGen能够找到生成的Types.td.inc文件
set(MLIR_TABLEGEN_FLAGS "${MLIR_TABLEGEN_FLAGS} -I${PYASC_DIR}/include/ascir/API")

# Add AFIROps subdirectory
add_subdirectory(AFIROps)

add_mlir_doc(AFIRDialect AFIRDialect Dialects/ -gen-dialect-doc)
add_mlir_doc(AFIRTypes AFIRTypes Dialects/ -gen-typedef-doc)
add_mlir_doc(AFIRAttrs AFIRAttrs Dialects/ -gen-attrdef-doc)
add_mlir_doc(AFIROps AFIROps Dialects/ -gen-op-doc)

add_subdirectory(Transforms)
