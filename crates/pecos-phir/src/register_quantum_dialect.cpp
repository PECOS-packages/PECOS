//===- register_quantum_dialect.cpp - Register quantum dialect with MLIR ---===//
//
// This file shows how to register the quantum dialect and lowering pass
// with MLIR so it can be used by mlir-opt.
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

// Forward declarations for our dialect and pass
namespace mlir {
namespace quantum {
class QuantumDialect;
}
std::unique_ptr<OperationPass<ModuleOp>> createConvertQuantumToLLVMPass();
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register all standard dialects
  mlir::registerAllDialects(registry);

  // Register our quantum dialect
  registry.insert<mlir::quantum::QuantumDialect>();

  // Register all standard passes
  mlir::registerAllPasses();

  // Register our quantum to LLVM lowering pass
  mlir::PassRegistration<> quantumToLLVMPass(
      "convert-quantum-to-llvm",
      "Convert Quantum dialect to LLVM dialect",
      []() { return mlir::createConvertQuantumToLLVMPass(); });

  return mlir::asMainReturnOnFailure(
      mlir::MlirOptMain(argc, argv, "Quantum MLIR optimizer driver\n", registry));
}
