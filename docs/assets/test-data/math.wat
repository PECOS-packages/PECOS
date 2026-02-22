(module
  ;; Global state for accumulator example
  (global $accumulator (mut i32) (i32.const 0))

  ;; Required init function
  (func $init)

  ;; Optional shot reset - resets accumulator to 0
  (func $shot_reinit
    i32.const 0
    global.set $accumulator)

  ;; Add two numbers
  (func $add (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add)

  ;; Subtract two numbers
  (func $sub (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.sub)

  ;; Multiply two numbers
  (func $mul (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.mul)

  ;; Accumulate a value and return the total
  (func $accumulate (param i32) (result i32)
    local.get 0
    global.get $accumulator
    i32.add
    global.set $accumulator
    global.get $accumulator)

  ;; Compute threshold (example from docs)
  (func $compute_threshold (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add
    i32.const 2
    i32.div_s)

  ;; Memory (required for some WASM operations)
  (memory (;0;) 1)

  ;; Exports
  (export "init" (func $init))
  (export "shot_reinit" (func $shot_reinit))
  (export "add" (func $add))
  (export "sub" (func $sub))
  (export "mul" (func $mul))
  (export "accumulate" (func $accumulate))
  (export "compute_threshold" (func $compute_threshold))
  (export "memory" (memory 0))
)
