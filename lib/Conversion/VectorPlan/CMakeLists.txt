add_mlir_library(MLIRVectorPlan
  GroupAnalysis/AxisLattice.cpp
  GroupAnalysis/CanFuse.cpp
  GroupAnalysis/GroupAnalysisPass.cpp
  GroupOutline/GroupOutlinePass.cpp
  TileFuse/TileFusePass.cpp
  TileFuse/BroadcastAbsorb.cpp
  TileFuse/Collapse.cpp
  TileInfo/TilePlanToTileInfo.cpp
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
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRPass
)
