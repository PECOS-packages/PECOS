(module
  ;; Global mutable state
  (global $counter (mut i32) (i32.const 0))

  ;; Init function - resets counter
  (func $init (export "init")
    i32.const 0
    global.set $counter
  )

  ;; Increment and return counter
  (func $increment (export "increment") (result i32)
    global.get $counter
    i32.const 1
    i32.add
    global.set $counter
    global.get $counter
  )

  ;; Get current counter value
  (func $get_counter (export "get_counter") (result i32)
    global.get $counter
  )
)
