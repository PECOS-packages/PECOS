;; Integer arithmetic for the default Guppy Helios WASM platform (int = i64).
(module
  ;; Required by PECOS's WasmForeignObject; this module has no mutable state.
  (func (export "init"))

  (func (export "add") (param $a i64) (param $b i64) (result i64)
    local.get $a
    local.get $b
    i64.add)

  (func (export "multiply") (param $a i64) (param $b i64) (result i64)
    local.get $a
    local.get $b
    i64.mul)
)
