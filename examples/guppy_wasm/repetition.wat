;; Three-bit repetition-code syndrome lookup, independently specified by
;; s0 = q0 XOR q1 and s1 = q1 XOR q2. At most one data bit is flipped.
(module
  (global $calls (mut i64) (i64.const 0))
  (func (export "init") (global.set $calls (i64.const 0)))
  (func (export "decode") (param $syndrome i64) (result i64)
    global.get $calls i64.const 1 i64.add global.set $calls
    local.get $syndrome i64.const 1 i64.eq
    if (result i64) i64.const 1
    else
      local.get $syndrome i64.const 3 i64.eq
      if (result i64) i64.const 2
      else
        local.get $syndrome i64.const 2 i64.eq
        if (result i64) i64.const 4 else i64.const 0 end
      end
    end)
  (func (export "calls") (result i64) global.get $calls))
