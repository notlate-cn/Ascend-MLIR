add_mlir_library(AscendTargetProfile
  TargetProfile.cpp
  TargetMemoryModel.cpp
  TargetIntrinsicModel.cpp
  CannTargetProfileLoader.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Target
  ${CMAKE_SOURCE_DIR}/include/Conversion

  DEPENDS
  AFIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
)
