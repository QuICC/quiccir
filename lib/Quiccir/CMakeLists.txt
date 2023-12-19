add_mlir_dialect_library(MLIRQuiccir
        QuiccirDialect.cpp
        QuiccirOps.cpp
        # QuiccirLowerToAffine.cpp
        # QuiccirLowerToLinalg.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Quiccir

        DEPENDS
        MLIRQuiccirOpsIncGen
        MLIRQuiccirPassesIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)


