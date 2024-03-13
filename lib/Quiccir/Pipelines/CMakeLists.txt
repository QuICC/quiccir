add_mlir_dialect_library(MLIRQuiccirPipelines
    Passes.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Quiccir

    LINK_LIBS PUBLIC
    MLIRPass
    MLIRReconcileUnrealizedCasts
    MLIRQuiccirPasses

)