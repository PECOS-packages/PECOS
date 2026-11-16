(module
  (global $counter (mut i64) (i64.const 0))
  (func (export "increment") (result i64)
    global.get $counter
    i64.const 1
    i64.add
    global.set $counter
    global.get $counter)
  (func (export "clear")
    i64.const 0
    global.set $counter)
  (func (export "trap") unreachable)
)
