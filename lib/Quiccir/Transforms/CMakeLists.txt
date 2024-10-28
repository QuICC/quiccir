add_mlir_dialect_library(MLIRQuiccirPasses
    QuiccirLowerToCall.cpp
    QuiccirFinalizeViewToLLVM.cpp
    # QuiccirLowerToAffine.cpp
    # QuiccirLowerToLinalg.cpp
    TypeConverter.cpp
    QuiccirLowerAlloc.cpp
    QuiccirViewDeallocation.cpp
    ConvertToLLVM.cpp
    SetViewLayout.cpp
    SetDimensions.cpp
    Utils.cpp
    ViewWrapper.cpp
    SetImplptr.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Quiccir

    DEPENDS
    MLIRQuiccirPassesIncGen

    LINK_LIBS PUBLIC
    MLIRQuiccirDialect
    MLIRIR
    MLIRInferTypeOpInterface
    )

