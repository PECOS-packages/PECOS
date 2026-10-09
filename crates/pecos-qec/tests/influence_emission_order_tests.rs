use pecos_core::PauliString;
use pecos_qec::fault_tolerance::influence_builder::InfluenceBuilder;
use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
use pecos_quantum::{DagCircuit, F2Matrix};

fn independent_measurements() -> DagCircuit {
    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    dag.pz(&[1]);
    dag.mz(&[0]);
    dag.mz(&[1]);
    dag
}

#[test]
fn auto_detectors_and_measurements_follow_emission_order() {
    let dag = independent_measurements();
    // DFS visits the q1 chain first; keyed emission is [0, 1, 2, 3].
    assert_eq!(dag.topological_order(), [1, 3, 0, 2]);
    let map = InfluenceBuilder::new(&dag).build().unwrap();
    assert_eq!(map.measurements, [(2, 0, 0), (3, 1, 0)]);
    assert_eq!(
        map.meas_ids.iter().map(|id| id.index()).collect::<Vec<_>>(),
        [0, 1]
    );
    // Both PZ preparations give deterministic zero, hence D0={m0}, D1={m1}.
    let members: Vec<Vec<_>> = map
        .detectors
        .iter()
        .map(|det| {
            det.measurements
                .iter()
                .map(|m| (m.tick, m.qubit, m.basis))
                .collect()
        })
        .collect();
    assert_eq!(members, [vec![(2, 0, 0)], vec![(3, 1, 0)]]);
}

#[test]
fn location_order_stays_in_lockstep() {
    let dag = independent_measurements();
    assert_eq!(dag.topological_order(), [1, 3, 0, 2]);
    let replay = InfluenceBuilder::new(&dag).build().unwrap();
    let analyzer = DagFaultAnalyzer::new(&dag).build_influence_map();
    assert_eq!(replay.locations, analyzer.locations);
}

#[test]
fn correlated_measurements_preserve_detector_row_space() {
    let row_space = |order: [usize; 3]| {
        let mut dag = DagCircuit::new();
        dag.pz(&[0, 1, 2]);
        dag.h(&[0]);
        dag.cx(&[(0, 1), (0, 2)]);
        for q in order {
            dag.mz(&[q]);
        }
        let map = InfluenceBuilder::new(&dag).build().unwrap();
        // Physical columns are (q0, first measurement), (q1, first), (q2, first),
        // regardless of insertion order. All three GHZ readouts share one bit.
        let rows = map
            .detectors
            .iter()
            .map(|det| {
                let mut row = vec![0; 3];
                for m in &det.measurements {
                    row[m.qubit] ^= 1;
                }
                row
            })
            .collect();
        F2Matrix::from_rows(rows).row_reduce()
    };
    // The hand-derived parity space is spanned by q0 XOR q1 and q0 XOR q2.
    let expected = F2Matrix::from_rows(vec![vec![1, 1, 0], vec![1, 0, 1]]).row_reduce();
    assert_eq!(row_space([0, 1, 2]), expected);
    assert_eq!(row_space([2, 1, 0]), expected);
}

#[test]
fn tracked_paulis_pair_with_their_own_meta_nodes() {
    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    dag.pz(&[1]);
    dag.x(&[0]);
    let x0 = 2;
    dag.tracked_pauli(PauliString::z(0));
    dag.x(&[1]);
    let x1 = 4;
    dag.tracked_pauli(PauliString::z(1));
    let dfs = dag.topological_order();
    println!("tracked-Pauli DFS order: {dfs:?}");
    assert_eq!(dfs, [1, 4, 5, 0, 2, 3]);

    let map = InfluenceBuilder::new(&dag)
        .with_circuit_annotations()
        .unwrap()
        .build()
        .unwrap();
    // An X fault after X q0 anticommutes with Z0 at meta A, and an X
    // fault after X q1 anticommutes with Z1 at meta B. Neither crosses wires.
    for (node, output) in [(x0, 0), (x1, 1)] {
        let location = map
            .locations
            .iter()
            .position(|loc| loc.node == node && !loc.before)
            .unwrap();
        assert_eq!(map.get_tracked_pauli_indices(location, 1), [output]);
    }
}
