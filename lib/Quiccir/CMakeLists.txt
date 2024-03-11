add_mlir_dialect_library(MLIRQuiccir
        QuiccirDialect.cpp
        QuiccirOps.cpp
        QuiccirTypes.cpp
        QuiccirLowerToCall.cpp
        QuiccirFinalizeViewToLLVM.cpp
        # QuiccirLowerToAffine.cpp
        # QuiccirLowerToLinalg.cpp
        Transforms/TypeConverter.cpp
        Transforms/QuiccirLowerAlloc.cpp
        Transforms/QuiccirViewDeallocation.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Quiccir

        DEPENDS
        MLIRQuiccirOpsIncGen
        MLIRQuiccirPassesIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)


