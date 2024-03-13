//===- quiccir-miniapp.cpp --------------------------------------*- C++ -*-===//
//
//   Copyright (c) 2024,
//   Earth and Planetary Magnetism group, ETH Zurich
//
//===---------------------------------------------------------------------===//

#include "Quiccir/IR/QuiccirDialect.h"
#include "Quiccir/Transforms/QuiccirPasses.h"
#include "Quiccir/Pipelines/Passes.h"
#include "Quiccir-c/Utils.h"
#include "jwOp.hpp"

#include "mlir/Dialect/Func/Extensions/AllExtensions.h"

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
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
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
  TestPipeLine,
  DumpLLVMIR,
  RunJIT
};
}
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(TestPipeLine, "test", "output the MLIR dump after calling the pipeline")),
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

  // Top level (module) pass manager
  mlir::PassManager pm(&context);
  // Apply any generic pass manager command line options and run the pipeline.
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
      return 4;

  // Lower to view rapresentation
  mlir::quiccir::quiccLibCallPipelineBuilder(pm);

  // run the pass manager
  if (mlir::failed(pm.run(*module)))
    return 4;

  return 0;
}

int dumpLLVMIR(mlir::ModuleOp module) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
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

  // Create target machine and configure the LLVM Module
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Could not create JITTargetMachineBuilder\n";
    return -1;
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Could not create TargetMachine\n";
    return -1;
  }
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                        tmOrError.get().get());

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

// rand float
template <class T>
inline T randf()
{
    return 2.0*static_cast<T>(std::rand()) / static_cast<T>(RAND_MAX) - 1.0;
}


int runJit(mlir::ModuleOp module) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
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
  engineOptions.sharedLibPaths = executionEngineLibs;
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  std::string symbol = "entry";
  // symbol = "_mlir_ciface_"+symbol;
  auto funSym = engine->lookup(symbol);
  if (auto E = funSym.takeError()) {
    llvm::errs() << "JIT invocation failed " << toString(std::move(E)) << "\n";
    return -1;
  }

  // number of elements/batch size
  // uval = op * umod
  // problem size
  constexpr std::size_t nLayer = 1;
  constexpr std::size_t nMod0 = 2;
  constexpr std::size_t nMod1 = 3;
  constexpr std::size_t nQuad0 = 3;

  std::array<double, nLayer*nQuad0*nMod0> proj;
  std::array<double, nLayer*nMod0*nQuad0> intg;
  std::array<double, nLayer*nMod0*nMod1> umod;
  std::array<double, nLayer*nMod0*nMod1> out;
  std::array<double, nLayer*nMod0*nMod1> ref;
  std::array<double, nLayer*nQuad0*nMod1> uval;
  // std::array<double, nLayer*nQuad0*nMod1> ref;

  // Call kernel
  view3_t viewRef_proj{{nQuad0, nMod0, nLayer}, nullptr, 0, nullptr, 0, proj.data(), proj.size()};
  view3_t viewRef_intg{{nMod0, nQuad0, nLayer}, nullptr, 0, nullptr, 0, intg.data(), intg.size()};
  view3_t viewRef_umod{{nMod0, nMod1, nLayer}, nullptr, 0, nullptr, 0, umod.data(), umod.size()};
  view3_t viewRef_out{{nMod0, nMod1, nLayer}, nullptr, 0, nullptr, 0, out.data(), out.size()};
  // view3_t viewRef_uval{{nQuad0, nMod1, nLayer}, nullptr, 0, nullptr, 0, uval.data(), uval.size()};

  // instantiate mock projector
  JWOp jwp;
  jwp.getOp() = viewRef_proj;
  for(size_t i = 0; i < nLayer*nQuad0*nMod0; ++i)
  {
    proj[i] = randf<double>();
  }

  JWOp jwi;
  jwi.getOp() = viewRef_intg;
  for(size_t i = 0; i < nLayer*nMod0*nQuad0; ++i)
  {
    intg[i] = randf<double>();
  }

  // init input data
  for(size_t i = 0; i < nLayer*nMod0*nMod1; ++i)
  {
    umod[i] = randf<double>();
  }

  auto fun = (void (*)(void*, view3_t*, view3_t*))funSym.get();

  std::array<void*, 2> thisArr;
  thisArr[0] = &jwp;
  thisArr[1] = &jwi;

  fun(thisArr.data(), &viewRef_out, &viewRef_umod);

  llvm::outs() << "ref out diff" << '\n';
  // check
  cpu_op(uval.data(), proj.data(), umod.data(), nLayer, nQuad0, nMod0, nMod1);
  cpu_op(ref.data(), intg.data(), uval.data(), nLayer, nMod0, nQuad0, nMod1);

  auto checkSuccess{true};
  for(size_t k = 0; k < nLayer; ++k)
  {
    for(size_t j = 0; j < nMod0; ++j)
    {
      for(size_t i = 0; i < nMod1; ++i)
      {
        auto ijk = i + j*nMod0 + k*nMod0*nMod1;
        auto diff = ref[ijk]-out[ijk];
        if (diff != 0.0)
        {
          llvm::outs() << ijk<< '\t' << ref[ijk] << '\t' << uval[ijk] << '\t'
            << diff << '\n';
          checkSuccess = false;
        }
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
  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  registerAllDialects(registry);

  mlir::MLIRContext context(registry);
  // Load our Dialect in this MLIR Context.
  context.loadDialect<mlir::quiccir::QuiccirDialect>();

  // Load
  mlir::OwningOpRef<mlir::ModuleOp>  module;
  if (int error = loadMLIR(context, module))
    return error;

  // Check to see if we want to echo MLIR
  if (emitAction == Action::DumpMLIR) {
    module->dump();
    return 0;
  }

  if (int error = processMLIR(context, module)) {
    return error;
  }
  if (emitAction == Action::TestPipeLine) {
    module->dump();
    return 0;
  }

  // Check to see if we are compiling to LLVM IR.
  if (emitAction == Action::DumpLLVMIR)
    return dumpLLVMIR(*module);

  // Otherwise, we must be running the jit.
  if (emitAction == Action::RunJIT) {
    return runJit(*module);
  }

  llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  return -1;
}
