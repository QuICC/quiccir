option(QUICCIR_BUILD_MINIAPP "Build quiccir-miniapp." ON)
if(NOT QUICCIR_BUILD_MINIAPP)
  return()
endif()
message(STATUS "Building quiccir-miniapp")

add_llvm_executable(quiccir-miniapp quiccir-miniapp.cpp)

llvm_update_compile_flags(quiccir-miniapp)
target_link_libraries(quiccir-miniapp PRIVATE QuiccirJITLib quiccir_external)

mlir_check_all_link_libraries(quiccir-miniapp)
