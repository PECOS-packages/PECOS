(module
  ;; Init function
  (func $init (export "init"))

  ;; Multiple functions with different signatures
  (func $add (export "add") (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add
  )

  (func $multiply (export "multiply") (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.mul
  )

  (func $negate (export "negate") (param i32) (result i32)
    i32.const 0
    local.get 0
    i32.sub
  )

  ;; Void function (no return value)
  (func $void_func (export "void_func") (param i32 i32))

  ;; Function with i64 parameters
  (func $add64 (export "add64") (param i64 i64) (result i64)
    local.get 0
    local.get 1
    i64.add
  )
)
