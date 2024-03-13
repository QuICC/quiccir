# Export targets
install(EXPORT quiccirTargets
  FILE quiccirTargets.cmake
  NAMESPACE quiccir
  DESTINATION ${CMAKE_INSTALL_PREFIX}/share/quicc/cmake
)