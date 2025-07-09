(module
  (type (;0;) (func))
  (type (;1;) (func (param i32 i32) (result i32)))
  (func $init (type 0))
  (func $add (type 1) (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add)
  (export "init" (func $init))
  (export "add" (func $add)))
