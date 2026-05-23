# Conversion passes TableGen definitions
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AFIRConversion)
add_public_tablegen_target(AFIRConversionPassIncGen)

set(LLVM_TARGET_DEFINITIONS Ascend/Passes.td)
mlir_tablegen(Ascend/Passes.h.inc -gen-pass-decls -name AscendConversion)
add_public_tablegen_target(AscendConversionPassIncGen)

set(LLVM_TARGET_DEFINITIONS Ascend/Kernelize/KernelizeOpInterfaces.td)
mlir_tablegen(Ascend/Kernelize/KernelizeOpInterfaces.h.inc
  -gen-op-interface-decls)
mlir_tablegen(Ascend/Kernelize/KernelizeOpInterfaces.cpp.inc
  -gen-op-interface-defs)
add_public_tablegen_target(AscendKernelizeOpInterfacesIncGen)
