#
# Export target
#
function (add_quiccir_library_install TGT)
  # parse inputs
  set(oneValueArgs COMPONENT FILES_MATCHING_PATTERN)
  set(multiValueArgs DIRECTORIES)
  cmake_parse_arguments(QET "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  message(DEBUG "add_quiccir_library_install")

  # Export info
  install(TARGETS ${TGT}
    EXPORT quiccirTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    # PUBLIC_HEADER DESTINATION ${_dest}
  )
  list(POP_BACK CMAKE_MESSAGE_INDENT)
endfunction (add_quiccir_library_install)
