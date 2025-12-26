add_executable(afir-tool afir.cpp)

target_link_libraries(afir-tool
  PRIVATE
    LLVMSupport
    LLVMCore
    MLIRParser
    MLIRSupport
    MLIRTransforms
    MLIRPass
    MLIRIR

    AFIRDialect
    AFIRInterface
    AFIRConversion
    AFIRTransforms
    AFIRUtils

    MLIRCastInterfaces
)