(module
 (global $n (mut i64) (i64.const 0))
 (func (export "increment") (result i64)
   global.get $n i64.const 1 i64.add global.set $n global.get $n))
