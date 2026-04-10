set(LLVM_LINK_COMPONENTS Support)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(afir-translate
  afir-translate.cpp
)

target_link_libraries(afir-translate
  PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    ${translation_libs}
    MLIRTranslateLib
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRIR
    MLIRAsc
    MLIREmitAsc
    MLIRTargetAsc
    MLIRLinalgTransformOps
    AFIRTransforms
    CannKernelTranslation
)

mlir_check_all_link_libraries(afir-translate)
