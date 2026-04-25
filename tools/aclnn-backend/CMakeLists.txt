set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(aclnn-backend
  aclnn-backend.cpp
)

target_link_libraries(aclnn-backend
  PRIVATE
    AclnnBackend
    LLVMSupport
)

mlir_check_all_link_libraries(aclnn-backend)