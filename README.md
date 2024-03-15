[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Tests](https://github.com/QuICC/quiccir/actions/workflows/lit.yml/badge.svg)](https://github.com/QuICC/quiccir/actions/workflows/lit.yml)

# quiccir: an out-of-tree MLIR dialect

`quiccir` is an out-of-tree [MLIR](https://mlir.llvm.org/) dialect.
This dialect provides operators and types that can be lowered to library calls to use the spectral operators in [QuICC](https://github.com/QuICC/QuICC).

It implements
- a `opt`-like tool to operate on quiccir
- a self contained library to JIT quiccir operators
- a miniapp that prototypes the usage in QuICC

## Building

This setup assumes that you have built LLVM 17 and MLIR in `$LLVM_BUILD` and installed them to `$LLVM_ROOT`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$LLVM_ROOT/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$LLVM_BUILD/bin/llvm-lit
cmake --build . --target check-quiccir
```
To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

