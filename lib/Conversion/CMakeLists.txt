add_mlir_library(AFIRConversion
  AFIRToASCIR.cpp

  ADDITIONAL_HEADER_DIRS
        ${CMAKE_SOURCE_DIR}/include/Conversion

  DEPENDS
  AFIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AFIRDialect
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRCastInterfaces
        MLIRIR
        MLIRSupport
        MLIRAsc

)

#target_link_libraries(AFIRConversion PUBLIC
#    MLIRAsc
#)
