(module
      (func $init)
      (func $add (param i32 i32) (result i32) (local.get 0) (local.get 1) (i32.add))
      (memory (;0;) 1)
      (export "init" (func $init))
      (export "add" (func $add))
      (export "memory" (memory 0))
    )
