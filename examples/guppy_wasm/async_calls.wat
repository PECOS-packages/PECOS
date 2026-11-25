(module
  (func (export "work") (param $n i64) (result i64)
    (local $state i64)
    (block $done
      local.get $n i64.eqz br_if $done
      (loop $again
        local.get $state i64.const 6364136223846793005 i64.mul i64.const 1 i64.add local.set $state
        local.get $n i64.const 1 i64.sub local.tee $n i64.eqz
        if br $done end
        br $again))
    local.get $state))
