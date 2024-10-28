[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Tests](https://github.com/QuICC/quiccir/actions/workflows/lit.yml/badge.svg?branch=dev)](https://github.com/QuICC/quiccir/actions/workflows/lit.yml)

# quiccir

`quiccir` is an out-of-tree [MLIR](https://mlir.llvm.org/) dialect.
This dialect provides operators and types that can be lowered to library calls to use the spectral operators in [QuICC](https://github.com/QuICC/QuICC).

It implements
- a `opt`-like tool to operate on quiccir
- a self contained library to JIT quiccir operators
- a miniapp that prototypes the usage in QuICC

## Building quiccir

`quiccir` depends on LLVM 17 and MLIR and assuming they are installed the to `$LLVM_ROOT` one can build `quiccir` as follows
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$LLVM_ROOT/lib/cmake/mlir -DQUICCIR_BUILD_TEST=OFF
```

To build and launch the tests, there is an additional dependency on `lit`, for instance
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$LLVM_ROOT/lib/cmake/mlir  \
-DLLVM_EXTERNAL_LIT=$LLVM_BUILD/bin/llvm-lit -DQUICCIR_BUILD_TEST=ON
cmake --build . --target check-quiccir
```

## Miniapp example

`quiccir-miniapp` is a mock application that shows how a computational graph can be lowered and used to call c++ operators
```sh
./bin/quiccir-miniapp ../examples/jw-lower-to-lib-call-2ops.mlir --emit=jit -opt
```

## Documentation

To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```

