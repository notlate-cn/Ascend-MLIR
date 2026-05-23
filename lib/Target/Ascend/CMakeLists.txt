add_mlir_library(AscendTargetProfile
  TargetProfile.cpp
  TargetMemoryModel.cpp
  TargetIntrinsicModel.cpp
  TargetCostModel.cpp
  TargetModelVerifier.cpp
  CannTargetProfileLoader.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Target
  ${CMAKE_SOURCE_DIR}/include/Conversion

  DEPENDS
  AscendConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
)
