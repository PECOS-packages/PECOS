;; Compile-time scalar interface for frontends that cannot inspect GC types.
;; Every body traps: execution must supply managed_objects.wat explicitly.
(module
  (func (export "get") (result i64) unreachable)
  (func (export "increment") (result i64) unreachable)
  (func (export "sum") (param i64 i64) (result i64) unreachable))
