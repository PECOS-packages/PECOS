; QIR slot convention from selene_request_tests.rs: use QIR gates and reads.
declare void @__quantum__qis__reset__body(i64)
declare i32 @__quantum__qis__m__body(i64, i64)
declare i32 @__quantum__rt__result_get_one(i64)

define i64 @qmain(i64 %shot) {
entry:
    call void @__quantum__qis__reset__body(i64 0)
    %measurement = call i32 @__quantum__qis__m__body(i64 0, i64 7)
    %value = call i32 @__quantum__rt__result_get_one(i64 7)
    ret i64 0
}
