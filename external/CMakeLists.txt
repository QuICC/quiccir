add_library(quiccir_external SHARED
    jwOp.cpp
)

target_include_directories(quiccir_external PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)