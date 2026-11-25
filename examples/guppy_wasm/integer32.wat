(module
  (memory (export "memory") 1)
  (global $value (mut i32) (i32.const 0))
  (func (export "add") (param i32 i32) (result i32)
    local.get 0 local.get 1 i32.add)
  (func (export "set") (param i32)
    local.get 0 global.set $value)
  (func (export "get") (result i32)
    global.get $value))
