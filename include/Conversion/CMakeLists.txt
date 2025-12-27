# Conversion passes TableGen definitions
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AFIRConversion)
add_public_tablegen_target(AFIRConversionPassIncGen)
