set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AFIR)
add_public_tablegen_target(AFIRTransformsPassIncGen)

add_mlir_doc(Passes AFIRPasses Transforms/ -gen-pass-doc)
