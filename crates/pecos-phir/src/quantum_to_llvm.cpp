//===- QuantumToLLVM.cpp - Quantum to LLVM dialect conversion -------------===//
//
// This file implements the lowering of Quantum dialect operations to LLVM
// function calls that match the QIR specification.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVM/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace quantum {

/// Returns the LLVM type for opaque Qubit pointer
static Type getQubitPtrType(MLIRContext *context) {
  return LLVM::LLVMPointerType::get(
      LLVM::LLVMStructType::getOpaque("Qubit", context));
}

/// Returns the LLVM type for opaque Result pointer
static Type getResultPtrType(MLIRContext *context) {
  return LLVM::LLVMPointerType::get(
      LLVM::LLVMStructType::getOpaque("Result", context));
}

//===----------------------------------------------------------------------===//
// Lowering patterns
//===----------------------------------------------------------------------===//

/// Lower quantum.alloc to @__quantum__rt__qubit_allocate()
struct AllocOpLowering : public OpConversionPattern<AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(AllocOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    // Get or insert the allocation function
    auto allocFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("__quantum__rt__qubit_allocate");
    if (!allocFunc) {
      auto qubitPtrTy = getQubitPtrType(rewriter.getContext());
      auto funcTy = LLVM::LLVMFunctionType::get(qubitPtrTy, {});
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      allocFunc = rewriter.create<LLVM::LLVMFuncOp>(loc, "__quantum__rt__qubit_allocate", funcTy);
    }

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(loc, allocFunc, ValueRange{});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

/// Lower quantum.dealloc to @__quantum__rt__qubit_release()
struct DeallocOpLowering : public OpConversionPattern<DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(DeallocOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    // Get or insert the deallocation function
    auto deallocFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("__quantum__rt__qubit_release");
    if (!deallocFunc) {
      auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
      auto qubitPtrTy = getQubitPtrType(rewriter.getContext());
      auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {qubitPtrTy});
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      deallocFunc = rewriter.create<LLVM::LLVMFuncOp>(loc, "__quantum__rt__qubit_release", funcTy);
    }

    // Create the call
    rewriter.create<LLVM::CallOp>(loc, deallocFunc, adaptor.getQubit());
    rewriter.eraseOp(op);
    return success();
  }
};

/// Template for lowering single-qubit gates
template <typename QuantumOp>
struct SingleQubitGateLowering : public OpConversionPattern<QuantumOp> {
  using OpConversionPattern<QuantumOp>::OpConversionPattern;

  StringRef getFunctionName() const;

  LogicalResult matchAndRewrite(QuantumOp op, typename QuantumOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->template getParentOfType<ModuleOp>();

    // Get or insert the gate function
    auto funcName = getFunctionName();
    auto gateFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!gateFunc) {
      auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
      auto qubitPtrTy = getQubitPtrType(rewriter.getContext());
      auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {qubitPtrTy});
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      gateFunc = rewriter.create<LLVM::LLVMFuncOp>(loc, funcName, funcTy);
    }

    // Create the call
    rewriter.create<LLVM::CallOp>(loc, gateFunc, adaptor.getQubit());
    rewriter.eraseOp(op);
    return success();
  }
};

// Specializations for each gate
template <> StringRef SingleQubitGateLowering<HOp>::getFunctionName() const {
  return "__quantum__qis__h__body";
}
template <> StringRef SingleQubitGateLowering<XOp>::getFunctionName() const {
  return "__quantum__qis__x__body";
}
template <> StringRef SingleQubitGateLowering<YOp>::getFunctionName() const {
  return "__quantum__qis__y__body";
}
template <> StringRef SingleQubitGateLowering<ZOp>::getFunctionName() const {
  return "__quantum__qis__z__body";
}

/// Lower quantum.cx to @__quantum__qis__cnot__body()
struct CXOpLowering : public OpConversionPattern<CXOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(CXOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    // Get or insert the CNOT function
    auto cnotFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("__quantum__qis__cnot__body");
    if (!cnotFunc) {
      auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
      auto qubitPtrTy = getQubitPtrType(rewriter.getContext());
      auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {qubitPtrTy, qubitPtrTy});
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      cnotFunc = rewriter.create<LLVM::LLVMFuncOp>(loc, "__quantum__qis__cnot__body", funcTy);
    }

    // Create the call
    rewriter.create<LLVM::CallOp>(loc, cnotFunc,
                                  ValueRange{adaptor.getControl(), adaptor.getTarget()});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lower quantum.measure to QIR measurement calls
struct MeasureOpLowering : public OpConversionPattern<MeasureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(MeasureOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    // Get or insert result allocation function
    auto getZeroFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("__quantum__rt__result_get_zero");
    if (!getZeroFunc) {
      auto resultPtrTy = getResultPtrType(rewriter.getContext());
      auto funcTy = LLVM::LLVMFunctionType::get(resultPtrTy, {});
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      getZeroFunc = rewriter.create<LLVM::LLVMFuncOp>(loc, "__quantum__rt__result_get_zero", funcTy);
    }

    // Get or insert measurement function
    auto measureFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("__quantum__qis__mz__body");
    if (!measureFunc) {
      auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
      auto qubitPtrTy = getQubitPtrType(rewriter.getContext());
      auto resultPtrTy = getResultPtrType(rewriter.getContext());
      auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {qubitPtrTy, resultPtrTy});
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      measureFunc = rewriter.create<LLVM::LLVMFuncOp>(loc, "__quantum__qis__mz__body", funcTy);
    }

    // Allocate result and perform measurement
    auto resultAlloc = rewriter.create<LLVM::CallOp>(loc, getZeroFunc, ValueRange{});
    rewriter.create<LLVM::CallOp>(loc, measureFunc,
                                  ValueRange{adaptor.getQubit(), resultAlloc.getResult()});
    rewriter.replaceOp(op, resultAlloc.getResult());
    return success();
  }
};

/// Lower quantum.read_result to @__quantum__qis__read_result__body()
struct ReadResultOpLowering : public OpConversionPattern<ReadResultOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ReadResultOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    // Get or insert read result function
    auto readFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("__quantum__qis__read_result__body");
    if (!readFunc) {
      auto i1Ty = IntegerType::get(rewriter.getContext(), 1);
      auto resultPtrTy = getResultPtrType(rewriter.getContext());
      auto funcTy = LLVM::LLVMFunctionType::get(i1Ty, {resultPtrTy});
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      readFunc = rewriter.create<LLVM::LLVMFuncOp>(loc, "__quantum__qis__read_result__body", funcTy);
    }

    // Create the call
    auto call = rewriter.create<LLVM::CallOp>(loc, readFunc, adaptor.getResult());
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct ConvertQuantumToLLVMPass
    : public PassWrapper<ConvertQuantumToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertQuantumToLLVMPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<QuantumDialect>();

    LLVMTypeConverter typeConverter(&getContext());
    // Add type conversions for quantum types
    typeConverter.addConversion([&](QubitType type) {
      return getQubitPtrType(&getContext());
    });
    typeConverter.addConversion([&](ResultType type) {
      return getResultPtrType(&getContext());
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<AllocOpLowering, DeallocOpLowering,
                 SingleQubitGateLowering<HOp>, SingleQubitGateLowering<XOp>,
                 SingleQubitGateLowering<YOp>, SingleQubitGateLowering<ZOp>,
                 CXOpLowering, MeasureOpLowering, ReadResultOpLowering>(
        typeConverter, &getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace quantum
} // namespace mlir

/// Create a pass to convert Quantum dialect to LLVM
std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertQuantumToLLVMPass() {
  return std::make_unique<quantum::ConvertQuantumToLLVMPass>();
}
