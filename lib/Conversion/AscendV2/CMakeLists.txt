add_mlir_library(AscendV2Conversion
  Debug/DebugOptions.cpp
  Normalize/NormalizePass.cpp
  Kernelize/KernelizePass.cpp
  Schedule/SchedulePass.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Conversion

  DEPENDS
  AFIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
)
