# Install headers
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/Quiccir ${PROJECT_SOURCE_DIR}/include/Quiccir-c
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

# Export targets
install(EXPORT quiccirTargets
  FILE quiccirTargets.cmake
  # NAMESPACE quiccir
  DESTINATION ${CMAKE_INSTALL_PREFIX}/share/quicc/cmake
)

# Create config file
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/quiccirConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/quiccirConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}/share/quicc/cmake
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/quiccirConfigVersion.cmake
  VERSION 0.0.0
  COMPATIBILITY SameMajorVersion )

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/quiccirConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/quiccirConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/share/quicc/cmake
)
