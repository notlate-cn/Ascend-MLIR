add_mlir_library(AscendV2Conversion
  Schedule/AxisCoalescer.cpp
  Schedule/ScheduleCache.cpp
  Schedule/ScheduleDecision.cpp
  Schedule/ScheduleProblemBuilder.cpp
  Schedule/ScheduleSearch.cpp
  Schedule/StructuredLoweringDriver.cpp
  Kernelize/CandidateClosure.cpp
  Kernelize/CandidateMergeAnalysis.cpp
  Debug/DebugOptions.cpp
  Kernelize/DependencyAnalysis.cpp
  Kernelize/FusionCandidateAnalysis.cpp
  Kernelize/HorizontalFusionAnalysis.cpp
  Kernelize/KernelPattern.cpp
  Schedule/KernelPatternView.cpp
  Schedule/TemplateRegistry.cpp
  Normalize/NormalizePass.cpp
  Kernelize/KernelizePass.cpp
  Kernelize/OpRoleClassification.cpp
  Kernelize/StructuralMarking.cpp
  Realize/RealizePass.cpp
  Realize/RealizeReport.cpp
  Schedule/SchedulePass.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Conversion

  DEPENDS
  AFIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSupport
)
