//! Comprehensive tests for Zluppy grammar and parser.

use crate::parser::parse;

// =============================================================================
// Helper macros
// =============================================================================

/// Assert that parsing succeeds
macro_rules! assert_parses {
    ($src:expr) => {
        match parse($src) {
            Ok(_) => {}
            Err(e) => panic!("Failed to parse:\n{}\nError: {}", $src, e),
        }
    };
}

/// Assert that parsing fails
macro_rules! assert_parse_fails {
    ($src:expr) => {
        assert!(
            parse($src).is_err(),
            "Expected parse to fail but it succeeded:\n{}",
            $src
        );
    };
}

// =============================================================================
// Empty and minimal programs
// =============================================================================

#[test]
fn test_empty_program() {
    assert_parses!("");
}

#[test]
fn test_whitespace_only() {
    assert_parses!("   \n\t\n   ");
}

#[test]
fn test_comments_only() {
    assert_parses!("// line comment\n");
    assert_parses!("/* block comment */");
    assert_parses!("// comment 1\n// comment 2\n");
    assert_parses!("/* multi\nline\ncomment */");
}

// =============================================================================
// Const declarations
// =============================================================================

#[test]
fn test_const_with_type() {
    assert_parses!("x: u32 = 42;");
    assert_parses!("pi: f64 = 3.14159;");
    assert_parses!("flag: bool = true;");
}

#[test]
fn test_const_inferred_type() {
    assert_parses!("x := 42;");
    assert_parses!("name := \"hello\";");
}

#[test]
fn test_const_pub() {
    assert_parses!("pub API_VERSION: u32 = 1;");
}

#[test]
fn test_const_expressions() {
    assert_parses!("sum := 1 + 2;");
    assert_parses!("product := 3 * 4;");
    assert_parses!("complex := (1 + 2) * 3;");
}

// =============================================================================
// Var declarations
// =============================================================================

#[test]
fn test_var_with_type() {
    assert_parses!("mut count: u32 = 0;");
    assert_parses!("mut name: []u8 = undefined;");
}

#[test]
fn test_var_undefined() {
    assert_parses!("mut buffer: [1024]u8 = undefined;");
}

#[test]
fn test_var_pub() {
    assert_parses!("pub mut global_state: i32 = 0;");
}

// =============================================================================
// Function declarations
// =============================================================================

#[test]
fn test_fn_no_params_void() {
    assert_parses!("fn main() -> unit {}");
}

#[test]
fn test_fn_with_params() {
    assert_parses!("fn add(a: i32, b: i32) -> i32 { return a + b; }");
}

#[test]
fn test_fn_pub() {
    assert_parses!("pub fn process() -> unit {}");
}

#[test]
fn test_fn_inline() {
    assert_parses!("inline fn fast_add(a: i32, b: i32) -> i32 { return a + b; }");
}

#[test]
fn test_extern_fn_basic() {
    // Basic extern function with C calling convention
    assert_parses!(r#"extern "C" fn decode(data: [*]u8, len: usize) -> i32;"#);
}

#[test]
fn test_extern_fn_no_params() {
    // Extern function with no parameters
    assert_parses!(r#"extern "C" fn get_version() -> u32;"#);
}

#[test]
fn test_extern_fn_no_return() {
    // Extern function with no return type (returns unit)
    assert_parses!(r#"extern "C" fn init();"#);
}

#[test]
fn test_extern_fn_pub() {
    // Public extern function
    assert_parses!(r#"pub extern "C" fn mwpm_decode(syndrome: [*]const u8, n: usize) -> i32;"#);
}

#[test]
fn test_extern_fn_rust_abi() {
    // Extern function with Rust calling convention
    assert_parses!(r#"extern "Rust" fn pecos_decode(data: *const u8) -> DecoderResult;"#);
}

#[test]
fn test_fn_comptime_param() {
    assert_parses!("fn make_array(comptime T: type, comptime N: usize) -> unit {}");
}

#[test]
fn test_fn_with_body() {
    assert_parses!(
        r#"
        fn factorial(n: u64) -> u64 {
            if (n <= 1) {
                return 1;
            }
            return n * factorial(n - 1);
        }
        "#
    );
}

// =============================================================================
// Struct declarations
// =============================================================================

#[test]
fn test_struct_empty() {
    assert_parses!("Empty := struct {};");
}

#[test]
fn test_struct_with_fields() {
    assert_parses!(
        r#"
        Point := struct {
            x: f64,
            y: f64,
        };
        "#
    );
}

#[test]
fn test_struct_with_defaults() {
    assert_parses!(
        r#"
        Config := struct {
            width: u32 = 800,
            height: u32 = 600,
            fullscreen: bool = false,
        };
        "#
    );
}

#[test]
fn test_struct_with_methods() {
    assert_parses!(
        r#"
        Counter := struct {
            value: u32,

            fn increment(&mut self) -> unit {
                self.value = self.value + 1;
            }

            fn get(&mut self) -> u32 {
                return self.value;
            }
        };
        "#
    );
}

#[test]
fn test_struct_pub() {
    assert_parses!(
        r#"
        pub PublicStruct := struct {
            data: u32,
        };
        "#
    );
}

#[test]
fn test_keywords_as_member_names() {
    // Keywords can be used as struct field names
    assert_parses!(
        r#"
        MyStruct := struct {
            set: bool,
            union: u32,
            type: usize,
        };
        "#
    );

    // Keywords can be used as method names
    assert_parses!(
        r#"
        Container := struct {
            data: u32,

            pub fn set(&mut self, value: u32) -> void {
                self.data = value;
                return unit;
            }

            pub fn union(&mut self, other: *Self) -> void {
                self.data = self.data + other.data;
                return unit;
            }
        };
        "#
    );

    // Keywords can be accessed as field names
    assert_parses!(
        r#"
        fn use_keywords(s: *MyStruct) -> void {
            x := s.set;
            y := s.union;
            z := s.type;
            return unit;
        }
        "#
    );

    // Keywords in range bounds (field access)
    assert_parses!(
        r#"
        fn iterate(s: *Container) -> void {
            for i in 0..s.set {
                process(i);
            }
        }
        "#
    );
}

#[test]
fn test_block_attributes() {
    // Block with single attribute
    assert_parses!(
        r#"
        fn main() -> unit {
            @attr(kind, "init")
            {
                h q[0];
            }
            return unit;
        }
        "#
    );

    // Block with multiple attributes
    assert_parses!(
        r#"
        fn main() -> unit {
            @attrs({round: 0, kind: "syndrome"})
            {
                cx (q[0], q[1]);
            }
            return unit;
        }
        "#
    );

    // Labeled block with attributes
    assert_parses!(
        r#"
        fn main() -> unit {
            @attr(priority, 1)
            setup: {
                pz q;
                h q[0];
            }
            return unit;
        }
        "#
    );

    // Block expression with attributes
    assert_parses!(
        r#"
        fn compute() -> u32 {
            result := @attr(optimized, true) compute_block: {
                x := 1 + 2;
                x * 3
            };
            return result;
        }
        "#
    );
}

// =============================================================================
// Enum declarations
// =============================================================================

#[test]
fn test_enum_simple() {
    assert_parses!(
        r#"
        Color := enum {
            Red,
            Green,
            Blue,
        };
        "#
    );
}

#[test]
fn test_enum_with_values() {
    assert_parses!(
        r#"
        Status := enum(u8) {
            Ok = 0,
            Error = 1,
            Pending = 2,
        };
        "#
    );
}

#[test]
fn test_union_tagged() {
    // Auto-tagged union with union(enum)
    assert_parses!(
        r#"
        Value := union(enum) {
            Int: i32,
            Float: f64,
            Bool: bool,
            None,
        };
        "#
    );
}

#[test]
fn test_union_untagged() {
    // Untagged union
    assert_parses!(
        r#"
        RawValue := union {
            Int: i32,
            Float: f64,
        };
        "#
    );
}

#[test]
fn test_union_external_tag() {
    // Externally-tagged union
    assert_parses!(
        r#"
        MyTag := enum { A, B, C };
        MyUnion := union(MyTag) {
            A: u32,
            B: f32,
            C,
        };
        "#
    );
}

// =============================================================================
// Control flow statements
// =============================================================================

#[test]
fn test_if_simple() {
    assert_parses!(
        r#"
        fn run() -> unit {
            if (x > 0) {
                do_something();
            }
        }
        "#
    );
}

#[test]
fn test_if_else() {
    assert_parses!(
        r#"
        fn run() -> unit {
            if (x > 0) {
                positive();
            } else {
                non_positive();
            }
        }
        "#
    );
}

#[test]
fn test_if_else_if() {
    assert_parses!(
        r#"
        fn run() -> unit {
            if (x > 0) {
                positive();
            } else if (x < 0) {
                negative();
            } else {
                zero();
            }
        }
        "#
    );
}

#[test]
fn test_for_loop_range() {
    assert_parses!(
        r#"
        fn run() -> unit {
            for i in 0..10 {
                process(i);
            }
        }
        "#
    );
}

#[test]
fn test_for_loop_inline() {
    assert_parses!(
        r#"
        fn run() -> unit {
            inline for i in 0..4 {
                unrolled(i);
            }
        }
        "#
    );
}

#[test]
fn test_for_loop_runtime_bound() {
    // Runtime-bounded for loop (bound is a runtime value)
    assert_parses!(
        r#"
        fn process(n: usize) -> unit {
            for i in 0..n {
                do_work(i);
            }
        }
        "#
    );

    // Runtime bound with method call on self
    assert_parses!(
        r#"
        fn iterate(&mut self) -> unit {
            for i in 0..self.len {
                process(self.items[i]);
            }
        }
        "#
    );

    // Mixed: comptime capacity bound with early break on runtime value
    assert_parses!(
        r#"
        Container := fn(comptime capacity: usize) -> type {
            struct {
                items: [capacity]u32 = undefined,
                len: usize = 0,

                fn search(&mut self, target: u32) -> ?usize {
                    for i in 0..capacity {
                        if (i >= self.len) {
                            break;
                        }
                        if (self.items[i] == target) {
                            return i;
                        }
                    }
                    return none;
                }
            }
        };
        "#
    );
}

#[test]
fn test_switch() {
    assert_parses!(
        r#"
        fn handle(x: u32) -> unit {
            switch (x) {
                0 => handle_zero(),
                1 => handle_one(),
                else => handle_other(),
            }
        }
        "#
    );
}

#[test]
fn test_return() {
    assert_parses!("fn run() -> u32 { return 42; }");
    assert_parses!("fn run() -> unit { return; }");
}

#[test]
fn test_break_continue() {
    assert_parses!(
        r#"
        fn run() -> unit {
            for i in 0..100 {
                if (done) {
                    break;
                }
                if (skip) {
                    continue;
                }
            }
        }
        "#
    );
}

#[test]
fn test_labeled_break() {
    assert_parses!(
        r#"
        fn run() -> unit {
            outer: for _ in 0..100 {
                for _ in 0..100 {
                    break :outer;
                }
            }
        }
        "#
    );
}

#[test]
fn test_defer() {
    assert_parses!(
        r#"
        fn run() -> unit {
            defer cleanup();
            do_work();
        }
        "#
    );
}

#[test]
fn test_errdefer() {
    // Basic errdefer without capture
    assert_parses!(
        r#"
        fn run() -> unit {
            errdefer cleanup();
            do_work();
        }
        "#
    );

    // errdefer with capture
    assert_parses!(
        r#"
        fn run() -> unit {
            errdefer |err| {
                log_error(err);
            }
            do_work();
        }
        "#
    );

    // errdefer with block body
    assert_parses!(
        r#"
        fn run() -> unit {
            errdefer {
                cleanup();
                notify();
            }
            do_work();
        }
        "#
    );
}

// =============================================================================
// Expressions
// =============================================================================

#[test]
fn test_literals() {
    assert_parses!("a := 42;");
    assert_parses!("b := 3.14;");
    assert_parses!("c := true;");
    assert_parses!("d := false;");
    assert_parses!("e := none;");
    assert_parses!(r#"f := "hello";"#);
    assert_parses!("g := 'x';");
}

#[test]
fn test_number_formats() {
    assert_parses!("dec := 42;");
    assert_parses!("hex := 0xFF;");
    assert_parses!("bin := 0b1010;");
    assert_parses!("oct := 0o77;");
    assert_parses!("with_underscore := 1_000_000;");
    assert_parses!("float := 3.14159;");
    assert_parses!("exp := 1e10;");
}

#[test]
fn test_binary_operators() {
    assert_parses!("a := 1 + 2;");
    assert_parses!("b := 3 - 1;");
    assert_parses!("c := 2 * 3;");
    assert_parses!("d := 10 / 2;");
    assert_parses!("e := 10 % 3;");
    assert_parses!("f := a == b;");
    assert_parses!("g := a != b;");
    assert_parses!("h := a < b;");
    assert_parses!("i := a <= b;");
    assert_parses!("j := a > b;");
    assert_parses!("k := a >= b;");
    assert_parses!("l := a and b;");
    assert_parses!("m := a or b;");
    assert_parses!("n := a & b;");
    assert_parses!("o := a | b;");
    assert_parses!("p := a ^ b;");
    assert_parses!("q := a << 2;");
    assert_parses!("r := a >> 2;");
}

#[test]
fn test_unary_operators() {
    assert_parses!("a := -x;");
    assert_parses!("b := !flag;");
    assert_parses!("c := ~bits;");
    assert_parses!("d := &value;");
    assert_parses!("e := *ptr;");
}

#[test]
fn test_operator_precedence() {
    assert_parses!("a := 1 + 2 * 3;");
    assert_parses!("b := (1 + 2) * 3;");
    assert_parses!("c := a and b or c;");
    assert_parses!("d := a == b and c != d;");
}

#[test]
fn test_function_calls() {
    assert_parses!("a := foo();");
    assert_parses!("b := bar(1);");
    assert_parses!("c := baz(1, 2, 3);");
    assert_parses!("d := nested(inner(x));");
}

#[test]
fn test_field_access() {
    assert_parses!("a := obj.field;");
    assert_parses!("b := obj.nested.field;");
    assert_parses!("c := get_obj().field;");
}

#[test]
fn test_index_access() {
    assert_parses!("a := arr[0];");
    assert_parses!("b := arr[i];");
    assert_parses!("c := matrix[i][j];");
}

#[test]
fn test_if_expression() {
    // If expressions require parentheses around condition to disambiguate from the block
    // Blocks support trailing expressions like Rust
    assert_parses!("max := if (a > b) { a } else { b };");
}

#[test]
fn test_break_with_expression() {
    // Simple break with expression
    assert_parses!("fn run() -> unit { break 42; }");
    // Break with label and expression
    assert_parses!("fn run() -> unit { break :lbl 42; }");
}

#[test]
fn test_block_expression() {
    assert_parses!(
        r#"
        fn run() -> u32 {
            result := blk: {
                temp := compute();
                break :blk temp * 2;
            };
            return result;
        }
        "#
    );
}

#[test]
fn test_builtin_calls() {
    assert_parses!(r#"std := @import("std");"#);
    // Self is now a keyword (like Rust), no need for @This() alias
    assert_parses!("x := Self;");  // Self can be used as a type value
    assert_parses!("size := @sizeOf(u32);");
}

#[test]
fn test_struct_init() {
    // Rust-style struct init: `Type { field: value }`
    assert_parses!("p := Point { x: 1.0, y: 2.0 };");
    assert_parses!("empty := .{};");
    assert_parses!("anon := .{ a: 1, b: 2 };");
    // Shorthand when variable name matches field name
    assert_parses!("x := 1; y := 2; p := Point { x, y };");
}

#[test]
fn test_array_init() {
    assert_parses!("arr := .{ 1, 2, 3 };");
    assert_parses!("typed := [3]u32{ 1, 2, 3 };");
}

// =============================================================================
// Types
// =============================================================================

#[test]
fn test_primitive_types() {
    assert_parses!("a: u8 = 0;");
    assert_parses!("b: u16 = 0;");
    assert_parses!("c: u32 = 0;");
    assert_parses!("d: u64 = 0;");
    assert_parses!("e: i8 = 0;");
    assert_parses!("f: i16 = 0;");
    assert_parses!("g: i32 = 0;");
    assert_parses!("h: i64 = 0;");
    assert_parses!("i: f32 = 0.0;");
    assert_parses!("j: f64 = 0.0;");
    assert_parses!("k: bool = true;");
    assert_parses!("l: usize = 0;");
}

#[test]
fn test_array_types() {
    assert_parses!("a: [10]u8 = undefined;");
    assert_parses!("b: []u8 = undefined;");
    assert_parses!("c: [_]u8 = undefined;");
}

#[test]
fn test_array_size_expressions() {
    // Simple identifier
    assert_parses!("a: [N]u8 = undefined;");
    // Simple literal
    assert_parses!("b: [64]u8 = undefined;");
    // Addition expression
    assert_parses!("c: [N + 1]u8 = undefined;");
    // Subtraction expression
    assert_parses!("d: [N - 1]u8 = undefined;");
    // Multiplication expression
    assert_parses!("e: [N * 2]u8 = undefined;");
    // Division expression
    assert_parses!("f: [N / 2]u8 = undefined;");
    // Complex expression
    assert_parses!("g: [N + M + 1]u8 = undefined;");
    // Parenthesized expression
    assert_parses!("h: [(N + 1) * 2]u8 = undefined;");
    // In function returning comptime type
    assert_parses!(r#"
        pub MyArray := fn(comptime N: usize) -> type {
            struct {
                data: [N + 1]u8 = undefined,
            }
        };
    "#);
}

#[test]
fn test_pointer_types() {
    assert_parses!("a: *u32 = undefined;");
    // Note: When type is explicit, use = not :=
    assert_parses!("b: *u32 = undefined;");
    assert_parses!("c: [*]u8 = undefined;");
}

#[test]
fn test_sentinel_terminated_pointers() {
    // Null-terminated string (sentinel = 0)
    assert_parses!("s: [*:0]u8 = undefined;");
    // Sentinel with different value
    assert_parses!("t: [*:255]u8 = undefined;");
    // Sentinel with const
    assert_parses!("u: [*:0]const u8 = undefined;");
    // Sentinel with expression
    assert_parses!("v: [*:0xFF]u8 = undefined;");
}

#[test]
fn test_sentinel_terminated_arrays() {
    // Array with sentinel value
    assert_parses!("a: [10:0]u8 = undefined;");
    // Array with different sentinel
    assert_parses!("b: [5:255]u8 = undefined;");
    // Slice with sentinel (no size, just sentinel)
    assert_parses!("c: [:0]u8 = undefined;");
}

#[test]
fn test_optional_types() {
    assert_parses!("a: ?u32 = none;");
    assert_parses!("b: ?*Node = none;");
}

#[test]
fn test_error_union_types() {
    assert_parses!("fn run() -> Error!u32 { return 42; }");
}

#[test]
fn test_error_value() {
    // error.Name syntax for error literals
    // Use := when type is inferred (walrus operator)
    assert_parses!("e := error.OutOfMemory;");
    assert_parses!("e := error.InvalidArgument;");
    assert_parses!("e := error.FileNotFound;");
}

#[test]
fn test_error_set_definition() {
    // Error set definition: MyError := error { ... }
    assert_parses!("MyError := error { OutOfMemory };");
    assert_parses!("FileError := error { NotFound, AccessDenied, Busy };");
    assert_parses!(
        r#"
        DivisionError := error {
            DivisionByZero,
            Overflow,
            Underflow,
        };
        "#
    );
    // Public error set
    assert_parses!("pub IoError := error { ReadError, WriteError };");
}

#[test]
fn test_fault_set_definition() {
    // Fault set definition: MyFault := fault { ... }
    assert_parses!("QuantumFault := fault { Leakage };");
    assert_parses!("GateFault := fault { Leakage, QubitLoss, GateFailure };");
    assert_parses!(
        r#"
        QuantumFault := fault {
            Leakage,
            QubitLoss,
            GateFailure,
        };
        "#
    );
    // Fault with associated data (using type references)
    assert_parses!(
        r#"
        QuantumFault := fault {
            Leakage: LeakageInfo,
            QubitLoss: QubitLossInfo,
        };
        "#
    );
    // Public fault set
    assert_parses!("pub QuantumFault := fault { Leakage, QubitLoss };");
}

#[test]
fn test_set_literal() {
    // Set literals with set keyword (consistent with struct, enum, union)
    assert_parses!("primes := set { 2, 3, 5, 7 };");
    assert_parses!("empty := set {};");
    assert_parses!("single := set { 42 };");
    // Set with explicit type
    assert_parses!("nums: Set(i64) = set { 1, 2, 3 };");
}

#[test]
fn test_batch_apply() {
    // Batch gate apply: h { q[0], q[1] } - set semantics (order doesn't matter)
    assert_parses!(
        r#"
        fn main() -> unit {
            mut q := qalloc(4);
            h { q[0], q[1], q[2] };
        }
        "#
    );
    // Parameterized gate batch
    assert_parses!(
        r#"
        fn main() -> unit {
            mut q := qalloc(4);
            rz(pi/4) { q[0], q[1] };
        }
        "#
    );
    // Two-qubit gate batch with pairs
    assert_parses!(
        r#"
        fn main() -> unit {
            mut q := qalloc(4);
            cx { (q[0], q[1]), (q[2], q[3]) };
        }
        "#
    );
}

#[test]
fn test_measure_syntax() {
    // New measurement syntax: mz(T) targets
    // Inline array (ordered results)
    assert_parses!(
        r#"
        fn main() -> unit {
            mut q := qalloc(4);
            results := mz(u1) [q[0], q[1], q[2]];
        }
        "#
    );
    // Single qubit measurement
    assert_parses!(
        r#"
        fn main() -> unit {
            mut q := qalloc(2);
            r := mz(u1) q[0];
        }
        "#
    );
    // Different result types
    assert_parses!(
        r#"
        fn main() -> unit {
            mut q := qalloc(2);
            r8 := mz(u8) [q[0]];
            r64 := mz(u64) [q[0], q[1]];
        }
        "#
    );
}

#[test]
fn test_try_expression() {
    // try expr - error propagation
    assert_parses!(
        r#"
        fn process() -> Error!u32 {
            result := try doSomething();
            return result;
        }
        "#
    );
}

#[test]
fn test_catch_expression() {
    // expr catch handler
    assert_parses!(
        r#"
        fn safe_divide(a: u32, b: u32) -> u32 {
            return divide(a, b) catch 0;
        }
        "#
    );
    // expr catch |err| handler
    assert_parses!(
        r#"
        fn safe_op() -> u32 {
            return risky() catch |err| handleError(err);
        }
        "#
    );
}

#[test]
fn test_error_unwrap() {
    // .! postfix operator for error unwrap
    assert_parses!(
        r#"
        fn unwrap_result(r: Error!u32) -> u32 {
            return r.!;
        }
        "#
    );
}

#[test]
fn test_try_block_collect() {
    // try { } - collect quantum faults (QEC pattern)
    assert_parses!(
        r#"
        fn qec_round(q: []qubit) try -> []QuantumFault!void {
            h q[0];
            cx (q[0], q[1]);
        }
        "#
    );
    // try { } as statement (no semicolon after block)
    assert_parses!(
        r#"
        fn circuit() -> unit {
            mut q := qalloc(4);
            try {
                h q[0];
                cx (q[0], q[1]);
            }
        }
        "#
    );
    // try { } with catch
    assert_parses!(
        r#"
        fn circuit() -> unit {
            mut q := qalloc(4);
            try {
                h q[0];
            } catch |err| {
                log(err);
            }
        }
        "#
    );
}

#[test]
fn test_try_block_propagate() {
    // try! { } - stop on first fault/error (traditional)
    assert_parses!(
        r#"
        fn strict_circuit(q: []qubit) try! -> QuantumFault!void {
            h q[0];
            cx (q[0], q[1]);
        }
        "#
    );
    // try! { } as statement with catch (no semicolon after block)
    assert_parses!(
        r#"
        fn circuit() -> unit {
            mut q := qalloc(4);
            try! {
                h q[0];
                cx (q[0], q[1]);
            } catch |err| {
                abort();
            }
        }
        "#
    );
}

#[test]
fn test_try_block_expression() {
    // try { } as expression (for assignment)
    assert_parses!(
        r#"
        fn collect_errors() -> unit {
            mut q := qalloc(4);
            errors := try {
                h q[0];
                cx (q[0], q[1]);
            };
        }
        "#
    );
    // try! { } as expression with catch providing default
    assert_parses!(
        r#"
        fn safe_measure() -> u1 {
            mut q := qalloc(1);
            result := try! {
                h q[0];
                mz(u1) q[0]
            } catch |err| { 0 };
            return result;
        }
        "#
    );
}

#[test]
fn test_function_types() {
    assert_parses!("callback: fn(u32) -> unit = undefined;");
    assert_parses!("binary_fn: fn(i32, i32) -> i32 = undefined;");
}

#[test]
fn test_named_types() {
    assert_parses!("a: MyStruct = undefined;");
    assert_parses!("b: std.mem.Allocator = undefined;");
}

// =============================================================================
// Quantum-specific syntax
// =============================================================================

#[test]
fn test_quantum_types() {
    assert_parses!("q: qubit = undefined;");
    assert_parses!("b: bit = undefined;");
    assert_parses!("mut alloc: Alloc = undefined;");
}

#[test]
fn test_quantum_allocator() {
    assert_parses!(
        r#"
        fn main() -> unit {
            mut base := qalloc(100);
            mut data := base.child(9);
            mut ancilla := base.child(8);
        }
        "#
    );
}

#[test]
fn test_quantum_prepare() {
    assert_parses!(
        r#"
        fn main() -> unit {
            mut q := qalloc(4);
            pz q;
            pz {q[0], q[1], q[2]};
        }
        "#
    );
}

#[test]
fn test_quantum_gates() {
    assert_parses!(
        r#"
        fn main() -> unit {
            mut q := qalloc(4);
            pz q;
            H(q[0]);
            X(q[1]);
            CX(q[0], q[1]);
            RZ(q[0], 0.5);
        }
        "#
    );
}

#[test]
fn test_swap_and_toffoli_gates() {
    // SWAP gate
    assert_parses!(
        r#"
        fn main() -> unit {
            q := qalloc(2);
            pz q;
            swap (q[0], q[1]);
        }
        "#
    );

    // iSWAP gate
    assert_parses!(
        r#"
        fn main() -> unit {
            q := qalloc(2);
            pz q;
            iswap (q[0], q[1]);
        }
        "#
    );

    // CCX (Toffoli) gate
    assert_parses!(
        r#"
        fn main() -> unit {
            q := qalloc(3);
            pz q;
            ccx (q[0], q[1], q[2]);
        }
        "#
    );
}

#[test]
fn test_quantum_measure() {
    assert_parses!(
        r#"
        fn main() -> unit {
            mut q := qalloc(2);
            pz q;
            H(q[0]);
            CX(q[0], q[1]);
            result := measure(q[0]);
            all_results := measure(q);
        }
        "#
    );
}

#[test]
fn test_quantum_conditional() {
    assert_parses!(
        r#"
        fn main() -> unit {
            mut q := qalloc(2);
            pz q;
            H(q[0]);
            m := measure(q[0]);
            if (m) {
                X(q[1]);
            }
        }
        "#
    );
}

#[test]
fn test_quantum_bell_state() {
    assert_parses!(
        r#"
        pub fn bell_state() -> unit {
            mut base := qalloc(2);
            mut q := base.child(2);

            pz q;

            H(q[0]);
            CX(q[0], q[1]);

            results := measure(q);
        }
        "#
    );
}

#[test]
fn test_quantum_teleportation() {
    assert_parses!(
        r#"
        pub fn teleport() -> unit {
            mut base := qalloc(10);
            mut state := base.child(1);
            mut epr := base.child(2);

            pz state;
            pz epr;

            H(epr[0]);
            CX(epr[0], epr[1]);

            CX(state[0], epr[0]);
            H(state[0]);

            m1 := measure(state[0]);
            m2 := measure(epr[0]);

            if (m2) {
                X(epr[1]);
            }
            if (m1) {
                Z(epr[1]);
            }
        }
        "#
    );
}

// =============================================================================
// Comptime features
// =============================================================================

#[test]
fn test_comptime_expression() {
    assert_parses!("size := comptime 4 * 8;");
}

#[test]
fn test_comptime_block() {
    assert_parses!(
        r#"
        value := comptime {
            mut result: u32 = 0;
            result = 42;
        };
        "#
    );
}

#[test]
fn test_comptime_type_function() {
    assert_parses!(
        r#"
        fn make_array(comptime T: type, comptime N: usize) -> type {
            return [N]T;
        }
        "#
    );
}

// =============================================================================
// Test declarations
// =============================================================================

#[test]
fn test_test_decl() {
    assert_parses!(
        r#"
        test "basic addition" {
            result := 1 + 1;
        }
        "#
    );
}

// =============================================================================
// Documentation comments
// =============================================================================

#[test]
fn test_doc_comments() {
    assert_parses!(
        r#"
        /// This is a documented function.
        /// It does important things.
        pub fn documented() -> unit {}
        "#
    );
}

// =============================================================================
// Complex programs
// =============================================================================

#[test]
fn test_surface_code_skeleton() {
    assert_parses!(
        r#"
        pub fn surface_code(comptime distance: u32) -> type {
            num_data := distance * distance;
            num_ancilla := (distance - 1) * (distance - 1) * 2;

            return struct {
                data: Alloc,
                ancilla: Alloc,

                // Self is implicitly available in struct methods (Rust-style)

                pub fn init(base: *Alloc) -> Self {
                    return Self {
                        data: base.child(num_data),
                        ancilla: base.child(num_ancilla),
                    };
                }

                pub fn syndrome_round(&mut self) -> unit {
                    pz self.ancilla;

                    inline for i in 0..num_ancilla {
                        h(self.ancilla[i]);
                    }

                    syndrome := measure(self.ancilla);
                }
            };
        }
        "#
    );
}

// =============================================================================
// Error cases (should fail to parse)
// =============================================================================

#[test]
fn test_missing_semicolon() {
    assert_parse_fails!("x := 42");
}

#[test]
fn test_invalid_identifier() {
    assert_parse_fails!("const 123abc = 1;");
}

#[test]
fn test_unbalanced_braces() {
    assert_parse_fails!("fn run() -> unit {");
    assert_parse_fails!("fn run() -> unit }");
}

#[test]
fn test_unbalanced_parens() {
    assert_parse_fails!("x := (1 + 2;");
}

// =============================================================================
// Zig-style syntax should fail (we use Rust-style)
// =============================================================================

#[test]
fn test_zig_style_struct_init_fails() {
    // Old Zig-style: `.field = value` - should fail, use `field: value`
    assert_parse_fails!("p := Point { .x = 1, .y = 2 };");
    assert_parse_fails!("anon := .{ .a = 1, .b = 2 };");
}

#[test]
fn test_zig_style_no_space_before_brace_ok() {
    // `Type{ ... }` without space is still valid
    assert_parses!("p := Point{ x: 1, y: 2 };");
}

#[test]
fn test_rust_style_struct_init_ok() {
    // Rust-style: `field: value` - should work
    assert_parses!("p := Point { x: 1, y: 2 };");
    assert_parses!("anon := .{ a: 1, b: 2 };");
    // Shorthand
    assert_parses!("x := 1; p := Point { x };");
}

#[test]
fn test_zig_style_this_parses_as_builtin() {
    // @This() still parses (as builtin call) but Self is preferred
    // Semantic analysis could reject it, but parsing accepts it
    assert_parses!("T := @This();");
}

#[test]
fn test_zig_style_null_parses_as_identifier() {
    // null parses as identifier, but will fail semantic analysis
    // (undefined symbol). Use none instead.
    assert_parses!("x := null;");
}

#[test]
fn test_explicit_self_type_still_valid() {
    // Explicit self: *Type is still valid (for advanced cases)
    // But &mut self is the idiomatic style
    assert_parses!("fn foo(self: *Foo) -> unit {}");
}

#[test]
fn test_rust_style_self_ok() {
    // Rust-style Self and &mut self should work
    assert_parses!("fn foo() -> Self { return Self; }");
    assert_parses!("Foo := struct { fn bar(&mut self) -> unit { return unit; } };");
    assert_parses!("Foo := struct { fn baz(&self) -> unit { return unit; } };");
}

#[test]
fn test_rust_style_none_ok() {
    // none instead of null
    assert_parses!("x := none;");
    assert_parses!("opt: ?u32 = none;");
}

#[test]
fn test_python_style_fstring_ok() {
    // Python-style f-strings
    assert_parses!(r#"msg := f"Hello";"#);
    assert_parses!(r#"name := "World"; msg := f"Hello {name}";"#);
    assert_parses!(r#"x := 42; msg := f"value = {x}";"#);
    assert_parses!(r#"a := 1; b := 2; msg := f"sum = {a + b}";"#);
    assert_parses!(r#"msg := f"escaped \{ brace \}";"#);
}

#[test]
fn test_python_style_fstring_format_spec_ok() {
    // F-strings with format specifiers (Python-style)
    assert_parses!(r#"x := 3.14159; msg := f"pi = {x:.2f}";"#);
    assert_parses!(r#"n := 42; msg := f"padded = {n:08d}";"#);
    assert_parses!(r#"name := "Alice"; msg := f"name = {name:>10}";"#);
    assert_parses!(r#"val := 255; msg := f"hex = {val:x}";"#);
    // Multiple format specifiers in one string
    assert_parses!(r#"x := 1.0; y := 2.0; msg := f"({x:.1f}, {y:.1f})";"#);
}

#[test]
fn test_python_style_raw_string_ok() {
    // Python-style raw strings (no escape processing)
    assert_parses!(r#"path := r"C:\Users\test";"#);
    assert_parses!(r#"regex := r"\d+\.\d+";"#);
    assert_parses!(r#"s := r"no escapes: \n \t \\";"#);
}

#[test]
fn test_python_style_in_operator_ok() {
    // Python-style in/not in operators
    assert_parses!("items := set { 1, 2, 3 }; x := 5 in items;");
    assert_parses!("items := set { 1, 2, 3 }; x := 5 not in items;");
}

#[test]
fn test_python_style_multiline_string_ok() {
    // Python-style triple-quoted multi-line strings
    assert_parses!(r#"text := """hello""";"#);
    assert_parses!(
        r#"text := """line 1
line 2
line 3""";"#
    );
    assert_parses!(
        r#"sql := """
SELECT *
FROM users
WHERE active = true
""";"#
    );
    // Escape sequences still work inside multi-line strings
    assert_parses!(r#"text := """tab:\there""";"#);
}

// =============================================================================
// Standard Library Tests
// =============================================================================

#[test]
fn test_std_bits_parses() {
    let src = include_str!("../std/bits.zlup");
    match parse(src) {
        Ok(_) => {}
        Err(e) => panic!("Failed to parse std/bits.zlup:\n{}", e),
    }
}

#[test]
fn test_std_containers_parses() {
    let src = include_str!("../std/containers.zlup");
    match parse(src) {
        Ok(_) => {}
        Err(e) => panic!("Failed to parse std/containers.zlup:\n{}", e),
    }
}

#[test]
fn test_std_qec_parses() {
    let src = include_str!("../std/qec.zlup");
    match parse(src) {
        Ok(_) => {}
        Err(e) => panic!("Failed to parse std/qec.zlup:\n{}", e),
    }
}

#[test]
fn test_std_math_parses() {
    let src = include_str!("../std/math.zlup");
    match parse(src) {
        Ok(_) => {}
        Err(e) => panic!("Failed to parse std/math.zlup:\n{}", e),
    }
}

#[test]
fn test_std_main_parses() {
    let src = include_str!("../std/std.zlup");
    match parse(src) {
        Ok(_) => {}
        Err(e) => panic!("Failed to parse std/std.zlup:\n{}", e),
    }
}

#[test]
fn test_std_f64_parses() {
    let src = include_str!("../std/f64.zlp");
    match parse(src) {
        Ok(_) => {}
        Err(e) => panic!("Failed to parse std/f64.zlp:\n{}", e),
    }
}

#[test]
fn test_std_a64_parses() {
    let src = include_str!("../std/a64.zlp");
    match parse(src) {
        Ok(_) => {}
        Err(e) => panic!("Failed to parse std/a64.zlp:\n{}", e),
    }
}

// =============================================================================
// Deprecated gate names rejected at grammar level
// =============================================================================

#[test]
fn test_deprecated_s_gate_rejected() {
    // "s" is no longer a valid gate keyword — use "sz" instead
    assert_parse_fails!("fn main() -> unit { s q[0]; }");
}

#[test]
fn test_deprecated_sdg_gate_rejected() {
    // "sdg" is no longer a valid gate keyword — use "szdg" instead
    assert_parse_fails!("fn main() -> unit { sdg q[0]; }");
}

// =============================================================================
// Gate name suggestion helpers
// =============================================================================

#[test]
fn test_suggest_gate_name_deprecated() {
    use crate::parser::suggest_gate_name;
    assert_eq!(suggest_gate_name("s"), Some("sz"));
    assert_eq!(suggest_gate_name("sdg"), Some("szdg"));
}

#[test]
fn test_suggest_gate_name_typo() {
    use crate::parser::suggest_gate_name;
    // Close misspellings should suggest the correct gate
    assert_eq!(suggest_gate_name("cx"), Some("cx")); // exact match (dist 0)
    assert_eq!(suggest_gate_name("hh"), Some("h"));  // edit distance 1
    assert_eq!(suggest_gate_name("swp"), Some("swap")); // edit distance 1
}

#[test]
fn test_suggest_gate_name_no_match() {
    use crate::parser::suggest_gate_name;
    assert_eq!(suggest_gate_name("foobar"), None);
}

#[test]
fn test_edit_distance() {
    use crate::parser::edit_distance;
    assert_eq!(edit_distance("", ""), 0);
    assert_eq!(edit_distance("abc", "abc"), 0);
    assert_eq!(edit_distance("abc", "abd"), 1);
    assert_eq!(edit_distance("abc", "ab"), 1);
    assert_eq!(edit_distance("abc", "abcd"), 1);
    assert_eq!(edit_distance("kitten", "sitting"), 3);
}

// =============================================================================
// Custom gate declarations
// =============================================================================

#[test]
fn test_parse_declare_gate() {
    assert_parses!(
        "declare gate my_gate(theta)(q);"
    );
}

#[test]
fn test_parse_declare_gate_no_params() {
    assert_parses!(
        "declare gate my_x()(q);"
    );
}

#[test]
fn test_parse_declare_gate_multiple_qubits() {
    assert_parses!(
        "declare gate cnot()(control, target);"
    );
}

#[test]
fn test_parse_declare_gate_multiple_params() {
    assert_parses!(
        "declare gate u3(theta, phi, lambda)(q);"
    );
}

#[test]
fn test_parse_declare_gate_pub() {
    assert_parses!(
        "pub declare gate rz(angle)(q);"
    );
}

#[test]
fn test_parse_composite_gate() {
    assert_parses!(
        r#"
        gate my_h()(q) {
            h q;
        }
        "#
    );
}

#[test]
fn test_parse_composite_gate_with_params() {
    assert_parses!(
        r#"
        gate rx(theta)(q) {
            h q;
        }
        "#
    );
}

#[test]
fn test_parse_composite_gate_multi_qubit() {
    assert_parses!(
        r#"
        gate bell()(q0, q1) {
            h q0;
            cx q0, q1;
        }
        "#
    );
}

#[test]
fn test_parse_pub_composite_gate() {
    assert_parses!(
        r#"
        pub gate swap()(a, b) {
            cx a, b;
            cx b, a;
            cx a, b;
        }
        "#
    );
}

#[test]
fn test_parse_declare_gate_and_fn_together() {
    assert_parses!(
        r#"
        declare gate custom_rx(theta)(q);

        pub fn apply_custom(q: qubit) -> unit {
            h q;
            return;
        }
        "#
    );
}

#[test]
fn test_parse_composite_gate_with_typed_param() {
    assert_parses!(
        "gate rz(theta: a64)(q) { h q; }"
    );
}
