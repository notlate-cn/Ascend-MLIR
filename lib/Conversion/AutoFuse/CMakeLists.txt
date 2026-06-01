add_mlir_library(MLIRNetworkJsonEmitter
  PARTIAL_SOURCES_INTENDED
  GroupOutline/NetworkJsonEmitter.cpp
  GroupOutline/OpRoleClassifier.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Conversion

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRTensorDialect
  LLVMSupport
)

add_mlir_library(MLIRAutoFuse
  PARTIAL_SOURCES_INTENDED
  GroupAnalysis/AxisLattice.cpp
  GroupAnalysis/CanFuse.cpp
  GroupAnalysis/GroupAnalysisPass.cpp
  GroupOutline/GroupOutlinePass.cpp
  GroupOutline/IsolateKernelOutputs.cpp
  GroupOutline/SplitRCoreGroup.cpp
  TileFuse/TileFusePass.cpp
  TileFuse/InsertTileBuffers.cpp
  TileFuse/FoldShadowAlloc.cpp
  TileFuse/BroadcastAbsorb.cpp
  TileFuse/Collapse.cpp
  TileFuse/TileFuseUtils.cpp
  TileFuse/TilePlanGen.cpp
  TileFuse/TilePlanSchema.cpp
  TileFuse/TilePlanCost.cpp
  TileFuse/LoopNestBuilder.cpp
  TileFuse/SliceComputer.cpp
  TileFuse/GroupEmitter.cpp
  TileFuse/CubeEmitter.cpp
  TileFuse/RestoreMatmul.cpp
  TileInfo/TilePlanToTileInfo.cpp
  Verify/VerifyTilingInfoSchemaPass.cpp
  Pipeline.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Conversion
  ${CMAKE_CURRENT_SOURCE_DIR}/GroupAnalysis

  DEPENDS
  AFIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRBufferizationDialect
  MLIRPass
  AFIRTransforms
  AFIRSymbolicShape
  MLIRNetworkJsonEmitter
  LLVMSupport
)
