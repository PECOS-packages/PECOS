(module
  (type $cell (struct (field (mut i64))))
  (type $values (array i64))
  (global $state (mut (ref null $cell)) (ref.null $cell))
  (func $start
    i64.const 7 struct.new $cell global.set $state)
  (start $start)
  (func (export "get") (result i64)
    global.get $state struct.get $cell 0)
  (func (export "increment") (result i64)
    global.get $state
    global.get $state struct.get $cell 0 i64.const 1 i64.add
    struct.set $cell 0
    global.get $state struct.get $cell 0)
  (func (export "sum") (param i64 i64) (result i64)
    (local $items (ref null $values))
    local.get 0 local.get 1 array.new_fixed $values 2 local.set $items
    local.get $items i32.const 0 array.get $values
    local.get $items i32.const 1 array.get $values
    i64.add))
