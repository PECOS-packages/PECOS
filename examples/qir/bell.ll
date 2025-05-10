%Result = type opaque
%Qubit = type opaque

declare void @__quantum__qis__h__body(%Qubit*)
declare void @__quantum__qis__cx__body(%Qubit*, %Qubit*)
declare void @__quantum__qis__m__body(%Qubit*, %Result*)
declare void @__quantum__rt__result_record_output(%Result*, i8*)

@.str.c = constant [2 x i8] c"c\00"

define void @main() #0 {
    ; Apply Hadamard to first qubit using H gate
    call void @__quantum__qis__h__body(%Qubit* null)

    ; Apply CX between qubits
    call void @__quantum__qis__cx__body(%Qubit* null, %Qubit* inttoptr (i64 1 to %Qubit*))

    ; Measure both qubits
    call void @__quantum__qis__m__body(%Qubit* null, %Result* inttoptr (i64 0 to %Result*))
    call void @__quantum__qis__m__body(%Qubit* inttoptr (i64 1 to %Qubit*), %Result* inttoptr (i64 1 to %Result*))

    ; Record results with a name that aligns with PHIR and QASM
    ; We record both measurements with same name "c" to match the PHIR/QASM approach
    ; The QIR engine will combine these into the "c" variable in output
    call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 0 to %Result*), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.c, i32 0, i32 0))
    call void @__quantum__rt__result_record_output(%Result* inttoptr (i64 1 to %Result*), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.c, i32 0, i32 0))

    ret void
}

attributes #0 = { "EntryPoint" }
