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
