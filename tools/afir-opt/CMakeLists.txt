set(LLVM_LINK_COMPONENTS
  Core
  Support
)

add_llvm_executable(afir-opt
  afir-opt.cpp
)

target_link_libraries(afir-opt
  PRIVATE
    MLIROptLib
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRRegisterAllExtensions

    AFIRDialect
    AFIRTransforms
    AFIRInterface
    AFIRConversion
    AFIRUtils
)

mlir_check_all_link_libraries(afir-opt)