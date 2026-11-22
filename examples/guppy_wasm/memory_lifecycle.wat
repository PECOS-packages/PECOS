(module
  (memory 1 2)
  (type $binary (func (param i64 i64) (result i64)))
  (table 1 funcref)
  (func (export "init")
    i32.const 0
    i32.const 0 i64.load i64.const 31 i64.add
    i64.store)
  (func (export "get") (result i64)
    i32.const 0 i64.load)
  (func (export "increment") (result i64)
    i32.const 0
    i32.const 0 i64.load i64.const 1 i64.add
    i64.store
    i32.const 0 i64.load)
  (func (export "grow") (result i64)
    i32.const 1 memory.grow i64.extend_i32_s)
  (func (export "pages") (result i64)
    memory.size i64.extend_i32_u)
  (func (export "read_new") (result i64)
    i32.const 65536 i64.load)
  (func (export "write_new") (param i64)
    i32.const 65536 local.get 0 i64.store)
  (func (export "dispatch") (param i64 i64) (result i64)
    local.get 0 local.get 1 i32.const 0 call_indirect (type $binary))
  (func $sum (type $binary) (param i64 i64) (result i64)
    local.get 0 local.get 1 i64.add)
  (elem (i32.const 0) $sum)
  (func $start
    i32.const 0 i64.const 11 i64.store)
  (start $start)
  (export "memory" (memory 0)))
