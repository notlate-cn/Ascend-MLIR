set(LLVM_TARGET_DEFINITIONS TorchFrontendPasses.td)
mlir_tablegen(TorchFrontendPasses.h.inc -gen-pass-decls -name TorchFrontend)
add_public_tablegen_target(TorchFrontendPassIncGen)