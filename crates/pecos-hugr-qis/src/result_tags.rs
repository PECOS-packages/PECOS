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
//! supports (straight-line `result_bool <- tket.bool:read <-
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

/// Map each `result(tag, <measurement>)` to the measurement ordinal it records.
///
/// **Sound by construction, narrow by design.** Only the canonical pattern
/// `result(tag, <a single raw measurement bit>)` is recognized: a
/// `tket.result:result_bool` op whose value input is *exactly*
/// `tket.bool:read` of a measurement op. The compiled chain is verified to be
/// precisely `result_bool <- tket.bool:read <- Measure/MeasureFree`.
///
/// Any other shape is **deliberately excluded** (the tag is omitted from the
/// returned map) rather than guessed at -- e.g. computed values
/// (`result("x", m0 == m1)` lowers through `tket.bool:eq`), constants
/// (`result("x", True)` lowers through a `Const`), and array-valued
/// `result(...)` (`result_array_bool` lowers through `collections.borrow_arr`
/// machinery that does not cleanly expose per-element measurement provenance).
/// Resolving those structurally would silently misbind (equality is not
/// parity; an empty record set is not a detector), so they are not returned.
///
/// A tag repeated across the program accumulates its ordinals in traversal
/// order; callers handle occurrence disambiguation / loop guarding.
#[must_use]
pub fn extract_result_tag_measurements<H: HugrView<Node = Node>>(
    hugr: &H,
) -> BTreeMap<String, Vec<usize>> {
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

    // Pass 2: accept only result_bool <- tket.bool:read <- measurement.
    let mut out: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for node in hugr.nodes() {
        let op = hugr.get_optype(node);
        let Some((ext, name)) = extension_ids(op) else {
            continue;
        };
        if ext != "tket.result" || name != "result_bool" {
            continue; // arrays / non-bool result ops: not soundly resolvable
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

        // result_bool value input (port 0) must be exactly `tket.bool:read`.
        let Some(read) = src_op(node, 0) else {
            continue;
        };
        match extension_ids(hugr.get_optype(read)) {
            Some((e, ref n)) if e == "tket.bool" && n == "read" => {}
            _ => continue, // e.g. tket.bool:eq (computed) -> exclude
        }
        // ... whose input (port 0) must be a measurement op.
        let Some(meas) = src_op(read, 0) else {
            continue;
        };
        let Some(&ord) = meas_ordinal.get(&meas) else {
            continue; // e.g. a Const -> exclude
        };
        out.entry(tag).or_default().push(ord);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read_hugr_envelope;

    // Fixtures compiled from Guppy (committed so the regression does not
    // depend on a Python toolchain at test time):
    //   scrambled: result() declared c,a,b over measures a,b,c (raw scalars)
    //   looped:    for _ in range(comptime(3)): result("synx", measure(q))
    //   computed:  result("eq", m0==m1) ; result("const", True)
    //   arr:       result("pair", measure_array(qs))   (array-valued)
    const SCRAMBLED: &[u8] = include_bytes!("../tests/fixtures/scrambled.hugr");
    const LOOPED: &[u8] = include_bytes!("../tests/fixtures/looped.hugr");
    const COMPUTED: &[u8] = include_bytes!("../tests/fixtures/computed.hugr");
    const ARR: &[u8] = include_bytes!("../tests/fixtures/arr.hugr");

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
                ("tag_a".to_string(), vec![0]),
                ("tag_b".to_string(), vec![1]),
                ("tag_c".to_string(), vec![2]),
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
            Some([0].as_slice()),
            "runtime loop is not unrolled in HUGR: one static measure op",
        );
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
            !map.contains_key("eq") && !map.contains_key("const"),
            "computed/constant tags must be excluded, got {map:?}",
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
            !map.contains_key("pair"),
            "array-valued result tag must be excluded, got {map:?}",
        );
    }
}
