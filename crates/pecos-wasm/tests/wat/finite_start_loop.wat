(module
  ;; Keep instantiation active long enough for the epoch timer thread to tick,
  ;; while remaining comfortably inside the one-second test timeout.
  (func $finite_start_loop
    (local $remaining i32)
    i32.const 50000000
    local.set $remaining
    (loop $spin
      local.get $remaining
      i32.const 1
      i32.sub
      local.tee $remaining
      br_if $spin
    )
  )
  (start $finite_start_loop)
  (func (export "init"))
)
