(module
  (func $init (export "init"))
  (func $identity (export "identity") (param i32) (result i32)
    local.get 0)
  (func $add (export "add") (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add)
  (memory (export "memory") 1))
