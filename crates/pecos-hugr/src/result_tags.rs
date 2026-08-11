//! Extract the Guppy `result(tag, ...)` -> measurement binding from a HUGR.
//!
//! This is the *sound* source of the tag<->measurement association: in the
//! compiled HUGR, a `tket.result` op's dataflow input is wired (transitively)
//! from the measurement op(s) that produced its value. That wiring is fixed at
//! compile time and is immune to any later QIS/Selene measurement reordering,
//! unlike a runtime op-stream heuristic.
//!
//! Measurement identity here is the *ordinal* of the measurement op in HUGR
//! traversal order. This module only recovers the structural binding; whether
//! that HUGR ordinal coincides with the QIS-trace `result_id`/`MeasId` order
//! is a separate property of the Guppy -> HUGR / Guppy -> trace pipelines
//! agreeing on measurement ordering. Within the narrow scope this module
//! supports (straight-line `result_bool <- read <-
//! Measure/MeasureFree`), that correspondence is **committed-test verified**
//! end-to-end by
//! `tests/qec/test_from_guppy_result_tags.py::test_result_tags_match_positional_records`
//! (a scrambled-`result()`-order Guppy program: `result_tags` DEM
//! byte-identical to the positional-records DEM). Outside that scope
//! (computed / constant / array-valued `result()`, runtime loops) the
//! correspondence is undefined and the extractor / runtime-loop guard reject
//! the case rather than relying on it.
//!
//! Note: a *runtime* loop (e.g. `for _ in range(comptime(n))`, as the surface
//! code uses for rounds) is NOT unrolled in the HUGR -- it has one static
//! measure/result op executed n times. Static extraction therefore yields
//! `tag -> static-measure-op`; expanding that to per-iteration runtime `MeasIds`
//! requires a separate static-op -> runtime-measurement correspondence.

use std::collections::{BTreeMap, HashMap};

use tket::hugr::ops::OpType;
use tket::hugr::types::Term;
use tket::hugr::{HugrView, IncomingPort, Node};

fn extension_ids(op: &OpType) -> Option<(&str, String)> {
    let ext = op.as_extension_op()?;
    Some((
        ext.extension_id().as_ref(),
        ext.unqualified_id().to_string(),
    ))
}

fn is_measurement(op: &OpType) -> bool {
    matches!(
        extension_ids(op),
        Some((ext, ref name))
            if ext == "tket.quantum" && (name == "Measure" || name == "MeasureFree")
    )
}

/// Number of *static* measurement ops in the HUGR.
///
/// For a straight-line program this equals the runtime measurement count; for
/// a program with a runtime loop it is strictly smaller (the loop body's
/// measure op is counted once). Callers use the mismatch to detect that
/// per-occurrence tag binding is not statically available.
#[must_use]
pub fn measurement_op_count<H: HugrView<Node = Node>>(hugr: &H) -> usize {
    hugr.nodes()
        .filter(|&n| is_measurement(hugr.get_optype(n)))
        .count()
}

/// Whether the HUGR contains branching, looping, or opaque control flow.
///
/// A single captured execution cannot certify a static circuit for these
/// structures. Callers may separately trust a generator-owned static layout.
///
/// Beyond `Conditional`/`TailLoop`/multi-successor blocks, two shapes are
/// rejected because they hide behavior from whole-graph analysis: a
/// `FuncDecl` is a body-less declared function (its operations are invisible
/// here, unlike a `Call` to a `FuncDefn`, whose body nodes are iterated like
/// any others), and a `CallIndirect` dispatches on a runtime function value,
/// which can select between operation sequences without any `Conditional`
/// node appearing in the graph.
#[must_use]
pub fn has_nontrivial_control_flow<H: HugrView<Node = Node>>(hugr: &H) -> bool {
    hugr.nodes().any(|node| match hugr.get_optype(node) {
        OpType::Conditional(_)
        | OpType::TailLoop(_)
        | OpType::FuncDecl(_)
        | OpType::CallIndirect(_) => true,
        OpType::DataflowBlock(block) => block.sum_rows.len() > 1,
        _ => false,
    })
}

/// Map each `result(tag, <measurement>)` to the measurement ordinal it records.
///
/// **Sound by construction, narrow by design.** Only the canonical pattern
/// `result(tag, <a single raw measurement bit>)` is recognized: a
/// `tket.result:result_bool` op whose value input is *exactly*
/// a direct measurement read. Guppy 1.0 emits `tket.measurement:Read`; older
/// HUGRs emit `tket.bool:read`. The compiled chain is verified to be precisely
/// `result_bool <- read <- Measure/MeasureFree`.
///
/// Any other shape is **deliberately represented as an unsupported occurrence**
/// (``None``) rather than guessed at -- e.g. computed values
/// (`result("x", m0 == m1)` lowers through `tket.bool:eq`), constants
/// (`result("x", True)` lowers through a `Const`), and array-valued
/// `result(...)` (`result_array_bool` lowers through `collections.borrow_arr`
/// machinery that does not cleanly expose per-element measurement provenance).
/// Resolving those structurally would silently misbind (equality is not
/// parity; an empty record set is not a detector), so they are not returned.
///
/// A tag repeated across the program accumulates one entry per call in
/// traversal order, including unsupported holes. This preserves the source
/// occurrence identity for callers that disambiguate repeated tags.
#[must_use]
pub fn extract_result_tag_measurements<H: HugrView<Node = Node>>(
    hugr: &H,
) -> BTreeMap<String, Vec<Option<usize>>> {
    // Pass 1: ordinal for every measurement op, in traversal order.
    let mut meas_ordinal: HashMap<Node, usize> = HashMap::new();
    for node in hugr.nodes() {
        if is_measurement(hugr.get_optype(node)) {
            let next = meas_ordinal.len();
            meas_ordinal.insert(node, next);
        }
    }

    // single_linked_output source op, if any.
    let src_op = |node: Node, port: usize| -> Option<Node> {
        hugr.single_linked_output(node, IncomingPort::from(port))
            .map(|(s, _)| s)
    };

    // Pass 2: accept only result_bool <- direct measurement read <- measurement.
    let mut out: BTreeMap<String, Vec<Option<usize>>> = BTreeMap::new();
    for node in hugr.nodes() {
        let op = hugr.get_optype(node);
        let Some((ext, name)) = extension_ids(op) else {
            continue;
        };
        if ext != "tket.result" {
            continue;
        }
        let Some(ext_op) = op.as_extension_op() else {
            continue;
        };
        let Some(tag) = ext_op.args().iter().find_map(|a| match a {
            Term::String(s) => Some(s.clone()),
            _ => None,
        }) else {
            continue;
        };

        let ordinal = (name == "result_bool")
            .then(|| src_op(node, 0))
            .flatten()
            .filter(|&read| {
                matches!(
                    extension_ids(hugr.get_optype(read)),
                    Some((e, ref n))
                        if (e == "tket.bool" && n == "read")
                            || (e == "tket.measurement" && n == "Read")
                )
            })
            .and_then(|read| src_op(read, 0))
            .and_then(|meas| meas_ordinal.get(&meas).copied());
        out.entry(tag).or_default().push(ordinal);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_hugr_from_bytes as read_hugr_envelope;

    // Fixtures compiled from Guppy (committed so the regression does not
    // depend on a Python toolchain at test time):
    //   scrambled: result() declared c,a,b over measures a,b,c (raw scalars)
    //   looped:    for _ in range(comptime(3)): result("synx", measure(q))
    //   computed:  result("eq", m0==m1) ; result("const", True)
    //   arr:       result("pair", measure_array(qs))   (array-valued)
    //   funcdecl:  @guppy.declare mystery(b: bool); prog calls mystery(measure(q))
    //   indirect:  g = helper; g()   (LoadFunction + CallIndirect)
    const SCRAMBLED: &[u8] = include_bytes!("../tests/fixtures/scrambled.hugr");
    const LOOPED: &[u8] = include_bytes!("../tests/fixtures/looped.hugr");
    const COMPUTED: &[u8] = include_bytes!("../tests/fixtures/computed.hugr");
    const ARR: &[u8] = include_bytes!("../tests/fixtures/arr.hugr");
    const FUNCDECL: &[u8] = include_bytes!("../tests/fixtures/funcdecl.hugr");
    const INDIRECT: &[u8] = include_bytes!("../tests/fixtures/indirect.hugr");

    /// True iff the HUGR would be rejected by the branch/loop arms alone
    /// (everything `has_nontrivial_control_flow` checked before the
    /// `FuncDecl`/`CallIndirect` arms were added).
    fn has_branch_or_loop<H: HugrView<Node = Node>>(hugr: &H) -> bool {
        hugr.nodes().any(|node| match hugr.get_optype(node) {
            OpType::Conditional(_) | OpType::TailLoop(_) => true,
            OpType::DataflowBlock(block) => block.sum_rows.len() > 1,
            _ => false,
        })
    }

    /// A body-less declared function is opaque to whole-graph analysis; a
    /// straight-line program calling one must not be certified static. The
    /// branch/loop assertion proves the `FuncDecl` arm alone is load-bearing
    /// for this fixture (reverting it would flip the second assertion).
    #[test]
    fn bodyless_declared_function_is_nontrivial_control_flow() {
        let hugr = read_hugr_envelope(FUNCDECL).unwrap();
        assert!(!has_branch_or_loop(&hugr));
        assert!(has_nontrivial_control_flow(&hugr));
    }

    /// An indirect call dispatches on a runtime function value and can select
    /// between operation sequences without any `Conditional` node. The
    /// branch/loop assertion proves the `CallIndirect` arm alone is
    /// load-bearing for this fixture.
    #[test]
    fn indirect_call_is_nontrivial_control_flow() {
        let hugr = read_hugr_envelope(INDIRECT).unwrap();
        assert!(!has_branch_or_loop(&hugr));
        assert!(has_nontrivial_control_flow(&hugr));
    }

    /// Foundation: `result()` declared in scrambled order (c, a, b) over
    /// measurements made in order (a, b, c) must still bind each tag to ITS
    /// OWN measurement. This is the exact case the prior runtime read/store
    /// heuristic got wrong (it produced `{tag_c: [0,1,2]}`); the HUGR
    /// structural binding is immune to declaration/measurement-order skew.
    #[test]
    fn scrambled_binds_each_tag_to_its_measurement() {
        let hugr = read_hugr_envelope(SCRAMBLED).unwrap();
        let map = extract_result_tag_measurements(&hugr);
        assert_eq!(
            map,
            BTreeMap::from([
                ("tag_a".to_string(), vec![Some(0)]),
                ("tag_b".to_string(), vec![Some(1)]),
                ("tag_c".to_string(), vec![Some(2)]),
            ]),
            "tag must bind to its own measurement regardless of result() order",
        );
    }

    /// Documents the known limitation: a runtime `for _ in range(comptime(n))`
    /// loop is NOT unrolled in the HUGR, so a tag emitted once per iteration
    /// has a single static measure op. Per-iteration expansion needs a
    /// separate static-op -> runtime-measurement correspondence.
    #[test]
    fn looped_tag_is_single_static_measure_op() {
        let hugr = read_hugr_envelope(LOOPED).unwrap();
        let map = extract_result_tag_measurements(&hugr);
        assert_eq!(
            map.get("synx").map(Vec::as_slice),
            Some([Some(0)].as_slice()),
            "runtime loop is not unrolled in HUGR: one static measure op",
        );
        assert!(has_nontrivial_control_flow(&hugr));
    }

    #[test]
    fn straight_line_program_has_no_nontrivial_control_flow() {
        let hugr = read_hugr_envelope(SCRAMBLED).unwrap();
        assert!(!has_nontrivial_control_flow(&hugr));
    }

    /// Soundness: a computed `result("eq", m0 == m1)` (lowers through
    /// `tket.bool:eq`) and a constant `result("const", True)` (lowers through
    /// a `Const`) must NOT be returned -- resolving them would silently
    /// misbind (equality is not parity; no measurement at all).
    #[test]
    fn computed_and_constant_tags_are_excluded() {
        let hugr = read_hugr_envelope(COMPUTED).unwrap();
        let map = extract_result_tag_measurements(&hugr);
        assert!(
            map.get("eq") == Some(&vec![None]) && map.get("const") == Some(&vec![None]),
            "computed/constant tag occurrences must be preserved as unsupported, got {map:?}",
        );
    }

    /// Soundness: an array-valued `result("pair", measure_array(qs))` lowers
    /// through `collections.borrow_arr` machinery with no clean per-element
    /// measurement provenance, so it must NOT be returned.
    #[test]
    fn array_valued_tag_is_excluded() {
        let hugr = read_hugr_envelope(ARR).unwrap();
        let map = extract_result_tag_measurements(&hugr);
        assert!(
            map.get("pair") == Some(&vec![None]),
            "array-valued result occurrence must be preserved as unsupported, got {map:?}",
        );
    }
}
