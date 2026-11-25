//! Platform-correct IR for Selene's C ABI in tests.
//!
//! 16-byte results use two registers on `SysV` and `AArch64`, including Windows on
//! Arm, and a hidden pointer on Windows x64. 8-byte results such as
//! `SeleneBoolResult` are declared `i64` on every target; the 4-byte
//! `SeleneVoidResult` is declared `i32`.

const HIDDEN_POINTER_RESULT: bool = cfg!(all(windows, target_arch = "x86_64"));

/// Declare an export that returns a 16-byte `{i32, i64}` result.
pub(crate) fn declaration(symbol: &str, parameters: &str) -> String {
    declaration_for_platform(symbol, parameters, HIDDEN_POINTER_RESULT)
}

/// Call a 16-byte result export, binding the `{i32, i64}` value to `%result`
/// on every platform. `result` must be unique within the IR function, because
/// the hidden-pointer form names its return buffer after it. Call from the entry
/// block: that form allocates the buffer in place, and an `alloca` in a loop grows
/// the stack on every iteration.
pub(crate) fn call(result: &str, symbol: &str, arguments: &str) -> String {
    call_for_platform(result, symbol, arguments, HIDDEN_POINTER_RESULT)
}

fn declaration_for_platform(symbol: &str, parameters: &str, hidden_pointer: bool) -> String {
    if hidden_pointer {
        format!("declare void @{symbol}(ptr sret({{i32, i64}}), {parameters})")
    } else {
        format!("declare {{i32, i64}} @{symbol}({parameters})")
    }
}

fn call_for_platform(result: &str, symbol: &str, arguments: &str, hidden_pointer: bool) -> String {
    if hidden_pointer {
        format!(
            "%{result}.out = alloca {{i32, i64}}\n\
             call void @{symbol}(ptr sret({{i32, i64}}) %{result}.out, {arguments})\n\
             %{result} = load {{i32, i64}}, ptr %{result}.out"
        )
    } else {
        format!("%{result} = call {{i32, i64}} @{symbol}({arguments})")
    }
}

mod tests {
    use super::*;

    #[test]
    fn declarations_match_both_platforms() {
        assert_eq!(
            declaration_for_platform("example", "ptr, i64", false),
            "declare {i32, i64} @example(ptr, i64)"
        );
        assert_eq!(
            declaration_for_platform("example", "ptr, i64", true),
            "declare void @example(ptr sret({i32, i64}), ptr, i64)"
        );
    }

    #[test]
    fn calls_match_both_platforms_with_distinct_buffers() {
        assert_eq!(
            call_for_platform("first", "example", "ptr null, i64 7", false),
            "%first = call {i32, i64} @example(ptr null, i64 7)"
        );
        assert_eq!(
            call_for_platform("second", "example", "ptr null, i64 8", false),
            "%second = call {i32, i64} @example(ptr null, i64 8)"
        );
        assert_eq!(
            call_for_platform("first", "example", "ptr null, i64 7", true),
            "%first.out = alloca {i32, i64}\n\
             call void @example(ptr sret({i32, i64}) %first.out, ptr null, i64 7)\n\
             %first = load {i32, i64}, ptr %first.out"
        );
        assert_eq!(
            call_for_platform("second", "example", "ptr null, i64 8", true),
            "%second.out = alloca {i32, i64}\n\
             call void @example(ptr sret({i32, i64}) %second.out, ptr null, i64 8)\n\
             %second = load {i32, i64}, ptr %second.out"
        );
    }
}
