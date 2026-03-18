//! Macro for concise circuit construction.
//!
//! Provides a DSL for building circuits with minimal boilerplate.
//!
//! # Example
//!
//! ```
//! use pecos_neo::prelude::*;
//!
//! let (q0, q1) = (QubitId(0), QubitId(1));
//!
//! let seq = pecos_neo::circuit! {
//!     pz(q0);
//!     pz(q1);
//!     h(q0);
//!     cx(q0, q1);
//!     let r0 = mz(q0);
//!     let r1 = mz(q1);
//! };
//! ```

/// Macro for building circuits with a concise DSL.
///
/// # Syntax
///
/// ```text
/// circuit! {
///     // Preparations (short form)
///     pz(qubit);    // prep Z
///     px(qubit);    // prep X
///     py(qubit);    // prep Y
///
///     // Single-qubit gates
///     h(qubit);
///     x(qubit);
///     y(qubit);
///     z(qubit);
///     s(qubit);
///     t(qubit);
///
///     // Rotations
///     rx(qubit, angle);
///     ry(qubit, angle);
///     rz(qubit, angle);
///
///     // Two-qubit gates
///     cx(control, target);
///     cz(q0, q1);
///     swap(q0, q1);
///
///     // Measurements (let binds result to identifier)
///     let r = mz(qubit);    // measure Z
///     let r = mx(qubit);    // measure X
///
///     // Conditionals
///     if result { ... }
///     if result { ... } else { ... }
///
///     // Convenience
///     bell(q0, q1);     // Bell state preparation
///     ghz(q0, q1, q2);  // GHZ state preparation
/// }
/// ```
#[macro_export]
macro_rules! circuit {
    // =========================================================================
    // INTERNAL @munch PATTERNS (must come FIRST to avoid catch-all matching)
    // =========================================================================

    // Base case - no more tokens
    (@munch $b:ident, $rc:ident ;) => {};

    // === Preparations (short names) ===
    (@munch $b:ident, $rc:ident ; pz($q:expr); $($rest:tt)*) => {
        $b = $b.pz($q);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };
    (@munch $b:ident, $rc:ident ; px($q:expr); $($rest:tt)*) => {
        $b = $b.px($q);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };
    (@munch $b:ident, $rc:ident ; py($q:expr); $($rest:tt)*) => {
        $b = $b.prep_y($q);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };

    // === Single-qubit gates ===
    (@munch $b:ident, $rc:ident ; h($q:expr); $($rest:tt)*) => {
        $b = $b.h($q);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };
    (@munch $b:ident, $rc:ident ; x($q:expr); $($rest:tt)*) => {
        $b = $b.x($q);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };
    (@munch $b:ident, $rc:ident ; y($q:expr); $($rest:tt)*) => {
        $b = $b.y($q);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };
    (@munch $b:ident, $rc:ident ; z($q:expr); $($rest:tt)*) => {
        $b = $b.z($q);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };
    (@munch $b:ident, $rc:ident ; s($q:expr); $($rest:tt)*) => {
        $b = $b.s($q);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };
    (@munch $b:ident, $rc:ident ; sdg($q:expr); $($rest:tt)*) => {
        $b = $b.sdg($q);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };
    (@munch $b:ident, $rc:ident ; t($q:expr); $($rest:tt)*) => {
        $b = $b.t($q);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };
    (@munch $b:ident, $rc:ident ; tdg($q:expr); $($rest:tt)*) => {
        $b = $b.tdg($q);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };
    (@munch $b:ident, $rc:ident ; sx($q:expr); $($rest:tt)*) => {
        $b = $b.sx($q);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };
    (@munch $b:ident, $rc:ident ; sxdg($q:expr); $($rest:tt)*) => {
        $b = $b.sxdg($q);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };

    // === Rotation gates ===
    (@munch $b:ident, $rc:ident ; rx($q:expr, $a:expr); $($rest:tt)*) => {
        $b = $b.rx($q, $a);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };
    (@munch $b:ident, $rc:ident ; ry($q:expr, $a:expr); $($rest:tt)*) => {
        $b = $b.ry($q, $a);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };
    (@munch $b:ident, $rc:ident ; rz($q:expr, $a:expr); $($rest:tt)*) => {
        $b = $b.rz($q, $a);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };

    // === Two-qubit gates ===
    (@munch $b:ident, $rc:ident ; cx($c:expr, $t:expr); $($rest:tt)*) => {
        $b = $b.cx($c, $t);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };
    (@munch $b:ident, $rc:ident ; cy($c:expr, $t:expr); $($rest:tt)*) => {
        $b = $b.cy($c, $t);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };
    (@munch $b:ident, $rc:ident ; cz($q0:expr, $q1:expr); $($rest:tt)*) => {
        $b = $b.cz($q0, $q1);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };
    (@munch $b:ident, $rc:ident ; swap($q0:expr, $q1:expr); $($rest:tt)*) => {
        $b = $b.swap($q0, $q1);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };

    // === Convenience operations ===
    (@munch $b:ident, $rc:ident ; bell($q0:expr, $q1:expr); $($rest:tt)*) => {
        $b = $b.prep_bell($q0, $q1);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };
    (@munch $b:ident, $rc:ident ; ghz($($q:expr),+); $($rest:tt)*) => {
        $b = $b.prep_ghz(&[$($q),+]);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };

    // === Measurements with result binding ===
    (@munch $b:ident, $rc:ident ; let $r:ident = mz($q:expr); $($rest:tt)*) => {
        let $r = $crate::ResultId($rc);
        $rc += 1;
        $b = $b.mz($q, $r);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };
    (@munch $b:ident, $rc:ident ; let $r:ident = mx($q:expr); $($rest:tt)*) => {
        let $r = $crate::ResultId($rc);
        $rc += 1;
        $b = $b.mx($q, $r);
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };

    // === Conditionals ===
    // if with else (must come before if-only to match first)
    (@munch $b:ident, $rc:ident ; if $cond:ident { $($if_body:tt)* } else { $($else_body:tt)* } $($rest:tt)*) => {
        $b = $b.if_then_else(
            $cond,
            |__inner| {
                let mut __b = __inner;
                let mut __rc = $rc;
                $crate::circuit!(@munch __b, __rc ; $($if_body)*);
                __b
            },
            |__inner| {
                let mut __b = __inner;
                let mut __rc = $rc;
                $crate::circuit!(@munch __b, __rc ; $($else_body)*);
                __b
            }
        );
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };
    // if without else
    (@munch $b:ident, $rc:ident ; if $cond:ident { $($body:tt)* } $($rest:tt)*) => {
        $b = $b.if_one($cond, |__inner| {
            let mut __b = __inner;
            let mut __rc = $rc;
            $crate::circuit!(@munch __b, __rc ; $($body)*);
            __b
        });
        $crate::circuit!(@munch $b, $rc ; $($rest)*);
    };

    // Catch-all for debugging - shows what failed to match
    (@munch $b:ident, $rc:ident ; $($other:tt)+) => {
        compile_error!(concat!("circuit! macro: unrecognized tokens: ", stringify!($($other)+)));
    };

    // =========================================================================
    // ENTRY POINT (must come LAST to avoid matching internal @munch calls)
    // =========================================================================
    ($($tokens:tt)*) => {{
        let mut __builder = $crate::OpBuilder::new();
        let mut __rc: u16 = 0;
        $crate::circuit!(@munch __builder, __rc ; $($tokens)*);
        __builder.build()
    }};
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_circuit_macro_basic() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);

        let seq = circuit! {
            pz(q0);
            pz(q1);
            h(q0);
            cx(q0, q1);
        };

        assert_eq!(seq.ops.len(), 4);
    }

    #[test]
    fn test_circuit_macro_measurement() {
        let q0 = QubitId(0);

        let seq = circuit! {
            pz(q0);
            h(q0);
            let r = mz(q0);
        };

        assert_eq!(seq.ops.len(), 3);
        assert_eq!(seq.result_count, 1);
    }

    #[test]
    fn test_circuit_macro_conditional() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);

        let seq = circuit! {
            pz(q0);
            pz(q1);
            h(q0);
            let r0 = mz(q0);
            if r0 {
                x(q1);
            }
        };

        assert_eq!(seq.ops.len(), 5);
    }

    #[test]
    fn test_circuit_macro_bell() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);

        let seq = circuit! {
            bell(q0, q1);
            let _r0 = mz(q0);
            let _r1 = mz(q1);
        };

        // bell = pz, pz, h, cx (4) + 2 meas
        assert_eq!(seq.ops.len(), 6);
    }

    #[test]
    fn test_circuit_macro_ghz() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);
        let q2 = QubitId(2);

        let seq = circuit! {
            ghz(q0, q1, q2);
        };

        // 3 preps + H + 2 CX = 6
        assert_eq!(seq.ops.len(), 6);
    }

    #[test]
    fn test_circuit_macro_rotation() {
        let q0 = QubitId(0);

        let seq = circuit! {
            pz(q0);
            rz(q0, Angle64::QUARTER_TURN);
            let _r = mz(q0);
        };

        assert_eq!(seq.ops.len(), 3);
    }

    #[test]
    fn test_circuit_macro_teleportation() {
        let (msg, alice, bob) = (QubitId(0), QubitId(1), QubitId(2));

        let seq = circuit! {
            // Prepare message
            pz(msg);

            // Bell pair
            bell(alice, bob);

            // Teleport
            cx(msg, alice);
            h(msg);
            let m0 = mz(msg);
            let m1 = mz(alice);

            // Corrections
            if m1 { x(bob); }
            if m0 { z(bob); }
        };

        assert!(!seq.ops.is_empty());
    }

    #[test]
    fn test_circuit_macro_if_else() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);

        let seq = circuit! {
            pz(q0);
            pz(q1);
            h(q0);
            let r = mz(q0);
            if r {
                x(q1);
            } else {
                z(q1);
            }
        };

        assert_eq!(seq.ops.len(), 5);
    }

    #[test]
    fn test_circuit_macro_empty() {
        let seq = circuit! {};
        assert_eq!(seq.ops.len(), 0);
        assert_eq!(seq.result_count, 0);
    }

    #[test]
    fn test_circuit_macro_all_preps() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);
        let q2 = QubitId(2);

        let seq = circuit! {
            pz(q0);
            px(q1);
            py(q2);
        };

        assert_eq!(seq.ops.len(), 3);
    }

    #[test]
    fn test_circuit_macro_all_single_qubit_gates() {
        let q = QubitId(0);

        let seq = circuit! {
            pz(q);
            h(q);
            x(q);
            y(q);
            z(q);
            s(q);
            sdg(q);
            t(q);
            tdg(q);
            sx(q);
            sxdg(q);
        };

        assert_eq!(seq.ops.len(), 11);
    }

    #[test]
    fn test_circuit_macro_all_rotations() {
        let q = QubitId(0);

        let seq = circuit! {
            pz(q);
            rx(q, Angle64::QUARTER_TURN);
            ry(q, Angle64::HALF_TURN);
            rz(q, Angle64::THREE_QUARTERS_TURN);
        };

        assert_eq!(seq.ops.len(), 4);
    }

    #[test]
    fn test_circuit_macro_all_two_qubit_gates() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);

        let seq = circuit! {
            pz(q0);
            pz(q1);
            cx(q0, q1);
            cy(q0, q1);
            cz(q0, q1);
            swap(q0, q1);
        };

        assert_eq!(seq.ops.len(), 6);
    }

    #[test]
    fn test_circuit_macro_x_basis_measurement() {
        let q = QubitId(0);

        let seq = circuit! {
            px(q);
            let _r = mx(q);
        };

        assert_eq!(seq.ops.len(), 2);
        assert_eq!(seq.result_count, 1);
    }

    #[test]
    fn test_circuit_macro_multiple_measurements() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);
        let q2 = QubitId(2);

        let seq = circuit! {
            pz(q0);
            pz(q1);
            pz(q2);
            let _r0 = mz(q0);
            let _r1 = mz(q1);
            let _r2 = mz(q2);
        };

        assert_eq!(seq.ops.len(), 6);
        assert_eq!(seq.result_count, 3);
    }

    #[test]
    fn test_circuit_macro_nested_conditionals() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);
        let q2 = QubitId(2);

        let seq = circuit! {
            pz(q0);
            pz(q1);
            pz(q2);
            h(q0);
            let r0 = mz(q0);
            if r0 {
                h(q1);
                let r1 = mz(q1);
                if r1 {
                    x(q2);
                }
            }
        };

        // 3 preps + h + meas + conditional(h + meas + conditional(x))
        assert!(!seq.ops.is_empty());
    }

    #[test]
    fn test_circuit_macro_chained_conditionals() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);

        let seq = circuit! {
            pz(q0);
            pz(q1);
            h(q0);
            let r0 = mz(q0);
            if r0 {
                x(q1);
            }
            h(q1);
            let r1 = mz(q1);
            if r1 {
                z(q0);
            } else {
                y(q0);
            }
        };

        // Complex control flow with multiple conditionals
        assert!(!seq.ops.is_empty());
    }

    #[test]
    fn test_circuit_macro_conditional_with_multiple_ops() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);

        let seq = circuit! {
            bell(q0, q1);
            let r = mz(q0);
            if r {
                x(q1);
                z(q1);
                h(q1);
            } else {
                y(q1);
                s(q1);
            }
        };

        assert!(!seq.ops.is_empty());
    }

    #[test]
    fn test_circuit_macro_expressions_in_qubits() {
        // Test that we can use expressions, not just simple identifiers
        let qubits = [QubitId(0), QubitId(1), QubitId(2)];

        let seq = circuit! {
            pz(qubits[0]);
            pz(qubits[1]);
            cx(qubits[0], qubits[1]);
        };

        assert_eq!(seq.ops.len(), 3);
    }

    #[test]
    fn test_circuit_macro_inline_qubit_ids() {
        // Test using QubitId directly in the macro
        let seq = circuit! {
            pz(QubitId(0));
            pz(QubitId(1));
            h(QubitId(0));
            cx(QubitId(0), QubitId(1));
        };

        assert_eq!(seq.ops.len(), 4);
    }

    #[test]
    fn test_circuit_macro_ghz_various_sizes() {
        // GHZ with 2 qubits (same as Bell)
        let seq2 = circuit! {
            ghz(QubitId(0), QubitId(1));
        };
        // 2 preps + H + 1 CX = 4
        assert_eq!(seq2.ops.len(), 4);

        // GHZ with 4 qubits
        let seq4 = circuit! {
            ghz(QubitId(0), QubitId(1), QubitId(2), QubitId(3));
        };
        // 4 preps + H + 3 CX = 8
        assert_eq!(seq4.ops.len(), 8);

        // GHZ with 5 qubits
        let seq5 = circuit! {
            ghz(QubitId(0), QubitId(1), QubitId(2), QubitId(3), QubitId(4));
        };
        // 5 preps + H + 4 CX = 10
        assert_eq!(seq5.ops.len(), 10);
    }

    #[test]
    fn test_circuit_macro_computed_angles() {
        let q = QubitId(0);
        let angle = Angle64::QUARTER_TURN;
        let double_angle = Angle64::HALF_TURN;

        let seq = circuit! {
            pz(q);
            rz(q, angle);
            rz(q, double_angle);
        };

        assert_eq!(seq.ops.len(), 3);
    }

    #[test]
    fn test_circuit_macro_deeply_nested_conditionals() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);
        let q2 = QubitId(2);
        let q3 = QubitId(3);

        let seq = circuit! {
            pz(q0);
            pz(q1);
            pz(q2);
            pz(q3);
            h(q0);
            let r0 = mz(q0);
            if r0 {
                h(q1);
                let r1 = mz(q1);
                if r1 {
                    h(q2);
                    let r2 = mz(q2);
                    if r2 {
                        x(q3);
                    } else {
                        z(q3);
                    }
                }
            }
        };

        assert!(!seq.ops.is_empty());
    }

    #[test]
    fn test_circuit_macro_result_used_multiple_times() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);
        let q2 = QubitId(2);

        let seq = circuit! {
            pz(q0);
            pz(q1);
            pz(q2);
            h(q0);
            let r = mz(q0);
            // Same result conditions multiple operations
            if r {
                x(q1);
            }
            if r {
                z(q2);
            }
        };

        assert!(!seq.ops.is_empty());
    }

    #[test]
    fn test_circuit_macro_mixed_bases() {
        // Prepare in Z, measure in X (and vice versa)
        let q0 = QubitId(0);
        let q1 = QubitId(1);

        let seq = circuit! {
            pz(q0);  // Prepare |0>
            px(q1);  // Prepare |+>
            let _r0 = mx(q0);  // Measure in X basis
            let _r1 = mz(q1);  // Measure in Z basis
        };

        assert_eq!(seq.ops.len(), 4);
        assert_eq!(seq.result_count, 2);
    }

    #[test]
    fn test_circuit_macro_comprehensive() {
        // Use every single operation type in one circuit
        let q0 = QubitId(0);
        let q1 = QubitId(1);
        let q2 = QubitId(2);

        let seq = circuit! {
            // All prep types
            pz(q0);
            px(q1);
            py(q2);

            // Single qubit gates
            h(q0);
            x(q0);
            y(q0);
            z(q0);
            s(q0);
            sdg(q0);
            t(q0);
            tdg(q0);
            sx(q0);
            sxdg(q0);

            // Rotation gates
            rx(q0, Angle64::QUARTER_TURN);
            ry(q0, Angle64::HALF_TURN);
            rz(q0, Angle64::THREE_QUARTERS_TURN);

            // Two qubit gates
            cx(q0, q1);
            cy(q0, q1);
            cz(q0, q1);
            swap(q0, q1);

            // Measurements
            let r0 = mz(q0);
            let _r1 = mx(q1);

            // Conditional
            if r0 {
                x(q2);
            }
        };

        // 3 preps + 10 single + 3 rot + 4 two-qubit + 2 meas + 1 cond = 23
        assert_eq!(seq.ops.len(), 23);
    }

    #[test]
    fn test_circuit_macro_swap_in_context() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);

        let seq = circuit! {
            pz(q0);
            px(q1);
            // q0 = |0>, q1 = |+>
            swap(q0, q1);
            // Now q0 = |+>, q1 = |0>
            let _r0 = mx(q0);  // Should give 0
            let _r1 = mz(q1);  // Should give 0
        };

        assert_eq!(seq.ops.len(), 5);
    }

    #[test]
    fn test_circuit_macro_bell_measurement() {
        // Create Bell state and measure both qubits
        let q0 = QubitId(0);
        let q1 = QubitId(1);

        let seq = circuit! {
            bell(q0, q1);
            let r0 = mz(q0);
            let r1 = mz(q1);
            // Results should be correlated
            if r0 {
                if r1 {
                    // Both 1 - valid Bell outcome
                    x(q0);  // dummy op
                }
            }
        };

        assert!(!seq.ops.is_empty());
    }

    #[test]
    fn test_circuit_macro_repeated_ops_on_same_qubit() {
        let q = QubitId(0);

        // H-Z-H = X (up to phase)
        let seq = circuit! {
            pz(q);
            h(q);
            z(q);
            h(q);
            let _r = mz(q);
        };

        assert_eq!(seq.ops.len(), 5);
    }

    #[test]
    fn test_circuit_macro_t_gate_sequence() {
        // T^8 = I
        let q = QubitId(0);

        let seq = circuit! {
            pz(q);
            t(q);
            t(q);
            t(q);
            t(q);
            t(q);
            t(q);
            t(q);
            t(q);
            let _r = mz(q);
        };

        assert_eq!(seq.ops.len(), 10);
    }

    #[test]
    fn test_circuit_macro_if_else_with_measurements() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);

        let seq = circuit! {
            pz(q0);
            pz(q1);
            h(q0);
            let r = mz(q0);
            if r {
                h(q1);
                let _r1 = mz(q1);
            } else {
                x(q1);
                let _r2 = mz(q1);
            }
        };

        assert!(!seq.ops.is_empty());
    }

    #[test]
    fn test_circuit_macro_many_qubits() {
        // Test with a larger number of qubits
        let seq = circuit! {
            pz(QubitId(0));
            pz(QubitId(1));
            pz(QubitId(2));
            pz(QubitId(3));
            pz(QubitId(4));
            pz(QubitId(5));
            pz(QubitId(6));
            pz(QubitId(7));
            cx(QubitId(0), QubitId(1));
            cx(QubitId(2), QubitId(3));
            cx(QubitId(4), QubitId(5));
            cx(QubitId(6), QubitId(7));
        };

        assert_eq!(seq.ops.len(), 12);
    }

    #[test]
    fn test_circuit_macro_alternating_h_and_measure() {
        let q = QubitId(0);

        let seq = circuit! {
            pz(q);
            h(q);
            let _r1 = mz(q);
            pz(q);
            h(q);
            let _r2 = mz(q);
            pz(q);
            h(q);
            let _r3 = mz(q);
        };

        assert_eq!(seq.ops.len(), 9);
        assert_eq!(seq.result_count, 3);
    }

    #[test]
    fn test_circuit_macro_long_circuit() {
        // Test that recursion works for longer circuits
        let q = QubitId(0);

        let seq = circuit! {
            pz(q);
            h(q); x(q); h(q); x(q);
            h(q); x(q); h(q); x(q);
            h(q); x(q); h(q); x(q);
            h(q); x(q); h(q); x(q);
            h(q); x(q); h(q); x(q);
            let _r = mz(q);
        };

        // 1 prep + 20 gates + 1 meas = 22
        assert_eq!(seq.ops.len(), 22);
    }
}
