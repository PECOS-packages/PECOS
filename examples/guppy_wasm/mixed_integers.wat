(module
  (func (export "add32") (param i32 i32) (result i32) local.get 0 local.get 1 i32.add)
  (func (export "add64") (param i64 i64) (result i64) local.get 0 local.get 1 i64.add))
