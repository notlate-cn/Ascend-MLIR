add_mlir_library(AscendV2Conversion
  Kernelize/CandidateClosure.cpp
  Debug/DebugOptions.cpp
  Kernelize/DependencyAnalysis.cpp
  Kernelize/FusionCandidateAnalysis.cpp
  Normalize/NormalizePass.cpp
  Kernelize/KernelizePass.cpp
  Kernelize/OpRoleClassification.cpp
  Kernelize/StructuralMarking.cpp
  Schedule/SchedulePass.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Conversion

  DEPENDS
  AFIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRSupport
)
