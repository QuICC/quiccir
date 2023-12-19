//===- quiccir-miniapp.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"


#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
// #include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
// #include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "mlir-c/Support.h"


#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"


template<typename T, size_t N>
struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
};


namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum InputType { MLIR };
}
static cl::opt<enum InputType> inputType(
    "x", cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(MLIR, "mlir",
                          "load the input file as an MLIR file")));

namespace {
enum Action {
  None,
  DumpMLIR,
  DumpLLVMIR,
  RunJIT
};
}
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
    cl::values(
        clEnumValN(RunJIT, "jit",
                   "JIT the code and run it by invoking the main function")));

// Optimization options
static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));
static cl::opt<bool> enableUnroll("unroll", cl::desc("Enable optimizations - only unroll"));
static cl::opt<bool> enableVectorize("vectorize", cl::desc("Enable optimizations - only vectorize"));

// Shared libs
llvm::cl::OptionCategory clOptionsCategory{"linking options"};
static llvm::cl::list<std::string> sharedLibs{
      "shared-libs", llvm::cl::desc("Libraries to link dynamically"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::cat(clOptionsCategory)};


int loadMLIR(mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp>  &module) {
  // Otherwise, the input is '.mlir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code EC = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

int processMLIR(mlir::MLIRContext &context,
                       mlir::OwningOpRef<mlir::ModuleOp>  &module) {

  mlir::PassManager pm(&context);
  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);

  // // Function pass
  // mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();


  pm.addPass(mlir::createConvertFuncToLLVMPass());
  // pm.addPass(mlir::createConvertFuncToLLVMPass());

  // run the pass manager
  if (mlir::failed(pm.run(*module)))
    return 4;

  module->dump();

  return 0;
}

int dumpLLVMIR(mlir::ModuleOp module) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}

int runJit(mlir::ModuleOp module) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Shared library with full path
  llvm::SmallVector<llvm::StringRef, 4> executionEngineLibs;
  for (auto &lib : sharedLibs) {
    executionEngineLibs.push_back(lib);
  }

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  std::string symbol = "wrap_batched_bwd";
  symbol = "_mlir_ciface_"+symbol;
  auto funSym = engine->lookup(symbol);
  if (auto E = funSym.takeError()) {
    llvm::errs() << "JIT invocation failed " << toString(std::move(E)) << "\n";
    return -1;
  }


  // number of elements/batch size
  // uval_e = b0 * a_e * b1^T
  constexpr std::size_t E = 16;
  // problem size
  constexpr std::size_t nQuad = 4;
  constexpr std::size_t nMod = 2;

  std::array<double, nQuad*nMod> b0{1,0.75,0.5,0.0, 0.125,0.25,0.375,0.5};
  std::array<double, nQuad*nMod> b1{b0};
  std::array<double, nMod*nMod> umod{1,1, 1,1};
  std::array<double, nQuad*nQuad> uval{};
  std::array<double, nQuad*nQuad> ref{
    1.265625, 1.125, 0.984375, 0.5625,
    1.125, 1, 0.875, 0.5,
    0.984375, 0.875, 0.765625, 0.4375,
    0.5625, 0.5, 0.4375, 0.25};


  // Call kernel
  // c_api calling convention
  // this works also without static sizes

  using mem3_t = MemRefDescriptor<double, 3>;
  using mem2_t = MemRefDescriptor<double, 2>;

  mem2_t memRef_b0{b0.data(), b0.data(), 0, {nQuad, nMod}, {nMod, 1}};
  mem2_t memRef_b1{b1.data(), b1.data(), 0, {nQuad, nMod}, {nMod, 1}};
  mem3_t memRef_umod{umod.data(), umod.data(), 0, {E, nMod, nMod},
    {nMod, nMod, 1}};
  mem3_t memRef_uval{uval.data(), uval.data(), 0, {E, nQuad, nQuad},
    {nQuad, nQuad, 1}};

  auto fun = (void (*)(mem2_t*, mem2_t*, mem3_t*, mem3_t*))funSym.get();
  fun(&memRef_b1, &memRef_b1, &memRef_umod, &memRef_uval);


  llvm::outs() << "ref out diff" << '\n';
  // check
  auto checkSuccess{true};
  for(size_t j = 0; j < nQuad; ++j)
  {
    for(size_t i = 0; i < nQuad; ++i)
    {
      auto ij = i+j*nQuad;
      auto diff = ref[ij]-uval[ij];
      if (diff != 0.0)
      {
        llvm::outs() << ij<< '\t' << ref[ij] << '\t' << uval[ij] << '\t'
          << diff << '\n';
        checkSuccess = false;
      }
    }
  }

  if(checkSuccess)
  {
    llvm::outs() << "test passed!\n";
  }

  return 0;
}

int main(int argc, char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "mlir jitter\n");

  // If we aren't dumping the AST, then we are compiling with/to MLIR.

  mlir::MLIRContext context;
  // Load our Dialect in this MLIR Context.
  context.loadDialect<
    mlir::scf::SCFDialect,
    mlir::arith::ArithDialect,
    mlir::memref::MemRefDialect,
    mlir::bufferization::BufferizationDialect,
    mlir::AffineDialect,
    mlir::linalg::LinalgDialect,
    mlir::cf::ControlFlowDialect,
    mlir::vector::VectorDialect,
    mlir::tensor::TensorDialect,
    mlir::func::FuncDialect,
    mlir::LLVM::LLVMDialect
    >();


  // Load
  mlir::OwningOpRef<mlir::ModuleOp>  module;
  if (int error = loadMLIR(context, module))
    return error;

  // Check to see if we want to echo MLIR
  if (emitAction == Action::DumpMLIR) {
    module->dump();
    return 0;
  }

  // // Check to see if we are compiling to LLVM IR.
  // if (emitAction == Action::DumpLLVMIR)
  //   return dumpLLVMIR(*module);

  // Otherwise, we must be running the jit.
  if (emitAction == Action::RunJIT)
    if (int error = processMLIR(context, module))
      return error;
    return runJit(*module);

  llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  return -1;
}
