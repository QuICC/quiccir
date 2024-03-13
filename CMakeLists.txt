cmake_minimum_required(VERSION 3.20.0)
project(quiccir-dialect LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

include(GNUInstallDirs)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR 17 REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
# include_directories(${PROJECT_BINARY_DIR}/include)
set(QUICCIR_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${QUICCIR_INCLUDE_DIR})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
if(MLIR_ENABLE_BINDINGS_PYTHON)
  message(STATUS "Enabling Python API")
  add_subdirectory(python)
endif()
add_subdirectory(test)
add_subdirectory(quiccir-opt)
add_subdirectory(quiccir-translate)
add_subdirectory(quiccir-miniapp)
add_subdirectory(external)

# Install headers
install(DIRECTORY include/Quiccir include/Quiccir-c
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT mlir-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "LICENSE.TXT"
)

install(DIRECTORY ${QUICCIR_INCLUDE_DIR}/Quiccir ${QUICCIR_INCLUDE_DIR}/Quiccir-c
DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT mlir-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.gen"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "config.h" EXCLUDE
)

add_llvm_install_targets(install-mlir-headers
                             DEPENDS mlir-headers
                             COMPONENT mlir-headers)

# Install libraries
add_custom_target(mlir-libraries)
set_target_properties(mlir-libraries PROPERTIES FOLDER "Misc")

add_llvm_install_targets(install-mlir-libraries
                          DEPENDS mlir-libraries
                          COMPONENT mlir-libraries)

# \todo: clean this up
# Attach include folders
target_include_directories(MLIRQuiccirDialect
  INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_include_directories(MLIRQuiccirDialect
  INTERFACE
    ${LLVM_INCLUDE_DIRS}
)
target_include_directories(MLIRQuiccirDialect
  INTERFACE
    ${MLIR_INCLUDE_DIRS}
)
target_include_directories(QuiccirCAPI
  INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# Export targets
include(cmake/modules/AddQuiccir.cmake)
add_quiccir_library_install(MLIRQuiccirDialect)
add_quiccir_library_install(QuiccirCAPI)
add_quiccir_library_install(QuiccirCAPITestLib)
add_subdirectory(cmake/modules)
