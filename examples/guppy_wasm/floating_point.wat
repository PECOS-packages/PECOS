(module
  (global $value (mut f64) (f64.const 0))
  (func (export "affine") (param f64 i64 f64) (result f64)
    local.get 0
    local.get 1
    f64.convert_i64_s
    f64.mul
    local.get 2
    f64.add)
  (func (export "remember") (param f64)
    local.get 0
    global.set $value)
  (func (export "recall") (result f64)
    global.get $value)
  (func (export "echo") (param f64) (result f64)
    local.get 0)
  (func (export "truncate") (param f64) (result i64)
    local.get 0
    i64.trunc_f64_s))
