add_mlir_dialect_library(AFIRTransforms
  Canonicalize.cpp
  AFIRInferShapes.cpp
  AFIRAddAxis.cpp
  LinalgInferShape.cpp
  LinalgMark.cpp
  LinalgAddBroadcast.cpp
  AFIRSymbolizeShapes.cpp
  AFIRVerifySymbolicShapes.cpp
  EmitNetworkJsonPass.cpp

  ADDITIONAL_HEADER_DIRS
        ${CMAKE_SOURCE_DIR}/include/Dialect/AFIR/Transforms

  DEPENDS
  AFIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  AFIRDialect
  AFIRSymbolicShape
  MLIRPass
  MLIRTransforms
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRBufferizationDialect
  MLIRArithDialect
  MLIRAffineDialect
  MLIRNetworkJsonEmitter
)
