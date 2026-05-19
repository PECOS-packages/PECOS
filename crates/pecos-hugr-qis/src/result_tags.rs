//! Extract the Guppy `result(tag, ...)` -> measurement binding from a HUGR.
//!
//! This is the *sound* source of the tag<->measurement association: in the
//! compiled HUGR, a `tket.result` op's dataflow input is wired (transitively)
//! from the measurement op(s) that produced its value. That wiring is fixed at
//! compile time and is immune to any later QIS/Selene measurement reordering,
//! unlike a runtime op-stream heuristic.
//!
//! Measurement identity here is the *ordinal* of the measurement op in HUGR
//! traversal order. Whether that ordinal coincides with the QIS-trace
//! `result_id`/MeasId is a separate, verified property (see the dem-polish
//! foundation tests); this module only recovers the structural binding.
//!
//! Note: a *runtime* loop (e.g. `for _ in range(comptime(n))`, as the surface
//! code uses for rounds) is NOT unrolled in the HUGR -- it has one static
//! measure/result op executed n times. Static extraction therefore yields
//! `tag -> static-measure-op`; expanding that to per-iteration runtime MeasIds
//! requires a separate static-op -> runtime-measurement correspondence.

use std::collections::{BTreeMap, HashMap, HashSet};

use tket::hugr::ops::OpType;
use tket::hugr::types::Term;
use tket::hugr::{HugrView, IncomingPort, Node};

fn extension_ids<'a>(op: &'a OpType) -> Option<(&'a str, String)> {
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

/// Map each `result(tag, ...)` to the measurement ordinals whose values it
/// recorded, in measurement-ordinal order.
///
/// A repeated tag (e.g. `result("synx", ...)` in a loop) accumulates each
/// occurrence's measurement ordinals in the order the `result` ops are
/// traversed; callers can disambiguate occurrences as needed.
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

    // Pass 2: per tket.result op, reverse-DFS over value wires to the
    // measurement ancestors feeding it.
    let mut out: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for node in hugr.nodes() {
        let op = hugr.get_optype(node);
        let Some((ext, name)) = extension_ids(op) else {
            continue;
        };
        if ext != "tket.result" || !name.starts_with("result") {
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

        // Seed ONLY from input port 0 -- the recorded value. Port 1 of a
        // result op is the linear state/order token threading result ops to
        // each other and to measurement side-effects; following it conflates
        // every measurement. From the value port, a reverse walk reaches the
        // measurement(s) via classical value ops (tket.bool:read, array
        // constructors, ...); measurements are leaves (we never descend into
        // their qubit inputs), so qubit wires are never traversed.
        let mut found: Vec<usize> = Vec::new();
        let mut seen: HashSet<Node> = HashSet::new();
        let mut stack: Vec<Node> = Vec::new();
        if let Some((src, _)) = hugr.single_linked_output(node, IncomingPort::from(0)) {
            stack.push(src);
        }
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            if let Some(&ord) = meas_ordinal.get(&n) {
                found.push(ord);
                continue; // a measurement is a leaf for value provenance
            }
            for p in 0..hugr.num_inputs(n) {
                if let Some((src, _)) = hugr.single_linked_output(n, IncomingPort::from(p)) {
                    stack.push(src);
                }
            }
        }
        found.sort_unstable();
        found.dedup();
        out.entry(tag).or_default().extend(found);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read_hugr_envelope;

    // Fixtures generated from Guppy via /tmp/gen_hugr_fixtures.py (committed so
    // the regression does not depend on a Python toolchain at test time).
    const SCRAMBLED: &[u8] = include_bytes!("../tests/fixtures/scrambled.hugr");
    const LOOPED: &[u8] = include_bytes!("../tests/fixtures/looped.hugr");

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

    /// Diagnostic: dump the looped HUGR's control-flow / region structure so
    /// the unrolled-order reconstruction can be designed against the real
    /// loop representation (TailLoop vs CFG, where the comptime bound lives,
    /// which region the measure/result ops sit in).
    #[test]
    fn dump_looped_control_flow() {
        let hugr = read_hugr_envelope(LOOPED).unwrap();
        for node in hugr.nodes() {
            let op = hugr.get_optype(node);
            let parent = hugr.get_parent(node);
            let tag = match extension_ids(op) {
                Some((e, n)) => format!("EXT {e}:{n}"),
                None => format!("{op:?}")
                    .split_whitespace()
                    .next()
                    .unwrap_or("?")
                    .to_string(),
            };
            let interesting = matches!(tag.as_str(), t if t.contains("CFG")
                || t.contains("DataflowBlock") || t.contains("TailLoop")
                || t.contains("Conditional") || t.contains("Case")
                || t.contains("ExitBlock") || t.contains("Const")
                || t.contains("FuncDefn"))
                || tag.contains("Measure")
                || tag.contains("tket.result")
                || tag.contains("LoadConstant");
            if interesting {
                eprintln!("{node:?} parent={parent:?} {tag}");
            }
        }
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
}
