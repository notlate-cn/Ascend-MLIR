add_library(AFIRRuntimeCAPI SHARED
  Runtime.cpp
)
target_link_libraries(AFIRRuntimeCAPI PUBLIC
  AscendCRuntime
  LLVMSupport
)
