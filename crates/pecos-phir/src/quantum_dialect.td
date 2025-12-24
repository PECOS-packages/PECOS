//===- QuantumDialect.td - Quantum dialect definition -------*- tablegen -*-===//
//
// This file defines the Quantum dialect for MLIR, which provides operations
// for quantum computing that can be lowered to QIR (Quantum Intermediate
// Representation) LLVM calls.
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_DIALECT
#define QUANTUM_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Quantum dialect definition
//===----------------------------------------------------------------------===//

def Quantum_Dialect : Dialect {
  let name = "quantum";
  let summary = "Quantum computing dialect for PECOS";
  let description = [{
    This dialect provides quantum computing operations that can be lowered
    to QIR (Quantum Intermediate Representation) LLVM function calls.
  }];
  let cppNamespace = "::mlir::quantum";
}

//===----------------------------------------------------------------------===//
// Quantum types
//===----------------------------------------------------------------------===//

class Quantum_Type<string name, string typeMnemonic>
    : TypeDef<Quantum_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Quantum_QubitType : Quantum_Type<"Qubit", "qubit"> {
  let summary = "Quantum bit type";
  let description = [{
    A qubit represents a quantum bit that can be in superposition.
  }];
}

def Quantum_ResultType : Quantum_Type<"Result", "result"> {
  let summary = "Measurement result type";
  let description = [{
    A result represents the outcome of a quantum measurement.
  }];
}

//===----------------------------------------------------------------------===//
// Quantum operation base classes
//===----------------------------------------------------------------------===//

class Quantum_Op<string mnemonic, list<Trait> traits = []> :
    Op<Quantum_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Quantum allocation/deallocation operations
//===----------------------------------------------------------------------===//

def Quantum_AllocOp : Quantum_Op<"alloc", [Pure]> {
  let summary = "Allocate a qubit";
  let description = [{
    Allocates a new qubit initialized to |0⟩ state.

    Example:
      %qubit = quantum.alloc : !quantum.qubit
  }];

  let results = (outs Quantum_QubitType:$qubit);
  let assemblyFormat = "attr-dict `:` type($qubit)";
}

def Quantum_DeallocOp : Quantum_Op<"dealloc"> {
  let summary = "Deallocate a qubit";
  let description = [{
    Deallocates a qubit, releasing its resources.

    Example:
      quantum.dealloc %qubit : !quantum.qubit
  }];

  let arguments = (ins Quantum_QubitType:$qubit);
  let assemblyFormat = "$qubit attr-dict `:` type($qubit)";
}

//===----------------------------------------------------------------------===//
// Single-qubit gates
//===----------------------------------------------------------------------===//

def Quantum_HOp : Quantum_Op<"h"> {
  let summary = "Hadamard gate";
  let description = [{
    Applies a Hadamard gate to a qubit.
    H = 1/√2 * [[1, 1], [1, -1]]

    Example:
      quantum.h %qubit : !quantum.qubit
  }];

  let arguments = (ins Quantum_QubitType:$qubit);
  let assemblyFormat = "$qubit attr-dict `:` type($qubit)";
}

def Quantum_XOp : Quantum_Op<"x"> {
  let summary = "Pauli-X gate";
  let description = [{
    Applies a Pauli-X (NOT) gate to a qubit.
    X = [[0, 1], [1, 0]]

    Example:
      quantum.x %qubit : !quantum.qubit
  }];

  let arguments = (ins Quantum_QubitType:$qubit);
  let assemblyFormat = "$qubit attr-dict `:` type($qubit)";
}

def Quantum_YOp : Quantum_Op<"y"> {
  let summary = "Pauli-Y gate";
  let description = [{
    Applies a Pauli-Y gate to a qubit.
    Y = [[0, -i], [i, 0]]

    Example:
      quantum.y %qubit : !quantum.qubit
  }];

  let arguments = (ins Quantum_QubitType:$qubit);
  let assemblyFormat = "$qubit attr-dict `:` type($qubit)";
}

def Quantum_ZOp : Quantum_Op<"z"> {
  let summary = "Pauli-Z gate";
  let description = [{
    Applies a Pauli-Z gate to a qubit.
    Z = [[1, 0], [0, -1]]

    Example:
      quantum.z %qubit : !quantum.qubit
  }];

  let arguments = (ins Quantum_QubitType:$qubit);
  let assemblyFormat = "$qubit attr-dict `:` type($qubit)";
}

//===----------------------------------------------------------------------===//
// Two-qubit gates
//===----------------------------------------------------------------------===//

def Quantum_CXOp : Quantum_Op<"cx"> {
  let summary = "Controlled-X (CNOT) gate";
  let description = [{
    Applies a controlled-X gate with control and target qubits.

    Example:
      quantum.cx %control, %target : !quantum.qubit, !quantum.qubit
  }];

  let arguments = (ins Quantum_QubitType:$control, Quantum_QubitType:$target);
  let assemblyFormat = "$control `,` $target attr-dict `:` type($control) `,` type($target)";
}

def Quantum_CZOp : Quantum_Op<"cz"> {
  let summary = "Controlled-Z gate";
  let description = [{
    Applies a controlled-Z gate with control and target qubits.

    Example:
      quantum.cz %control, %target : !quantum.qubit, !quantum.qubit
  }];

  let arguments = (ins Quantum_QubitType:$control, Quantum_QubitType:$target);
  let assemblyFormat = "$control `,` $target attr-dict `:` type($control) `,` type($target)";
}

//===----------------------------------------------------------------------===//
// Measurement operations
//===----------------------------------------------------------------------===//

def Quantum_MeasureOp : Quantum_Op<"measure", [Pure]> {
  let summary = "Measure a qubit in Z basis";
  let description = [{
    Measures a qubit in the computational (Z) basis.

    Example:
      %result = quantum.measure %qubit : !quantum.qubit -> !quantum.result
  }];

  let arguments = (ins Quantum_QubitType:$qubit);
  let results = (outs Quantum_ResultType:$result);
  let assemblyFormat = "$qubit attr-dict `:` type($qubit) `->` type($result)";
}

def Quantum_ReadResultOp : Quantum_Op<"read_result", [Pure]> {
  let summary = "Read measurement result as classical bit";
  let description = [{
    Converts a measurement result to a classical bit (i1).

    Example:
      %bit = quantum.read_result %result : !quantum.result -> i1
  }];

  let arguments = (ins Quantum_ResultType:$result);
  let results = (outs I1:$bit);
  let assemblyFormat = "$result attr-dict `:` type($result) `->` type($bit)";
}

//===----------------------------------------------------------------------===//
// Reset operation
//===----------------------------------------------------------------------===//

def Quantum_ResetOp : Quantum_Op<"reset"> {
  let summary = "Reset qubit to |0⟩ state";
  let description = [{
    Resets a qubit to the |0⟩ state.

    Example:
      quantum.reset %qubit : !quantum.qubit
  }];

  let arguments = (ins Quantum_QubitType:$qubit);
  let assemblyFormat = "$qubit attr-dict `:` type($qubit)";
}

#endif // QUANTUM_DIALECT
