//! Closure-based regressions for dependent Pauli generators.

use pecos_core::clifford_rep::CliffordRep;
use pecos_core::pauli::algebra::i;
use pecos_core::pauli::*;
use pecos_core::{Pauli, PauliOperator, PauliString, QuarterPhase};
use pecos_quantum::stabilizer_group::PauliStabilizerGroupError;
use pecos_quantum::{PauliGroup, PauliSequence, PauliSet, PauliStabilizerGroup};
use std::collections::HashSet;

// Independent oracle: repeatedly multiply known elements by the generators.
// No rank, reduction, membership, order, or enumeration code under test is used.
fn closure(generators: &[PauliString]) -> HashSet<PauliString> {
    let identity = PauliString::identity();
    let mut known = HashSet::from([identity.clone()]);
    let mut pending = vec![identity];
    while let Some(element) = pending.pop() {
        for generator in generators {
            let product = element.clone() * generator;
            if known.insert(product.clone()) {
                pending.push(product);
            }
        }
    }
    known
}

fn assert_group_matches_closure(generators: Vec<PauliString>) {
    let expected = closure(&generators);
    let group = PauliGroup::new(generators).unwrap();
    for (index, generator) in group.generators().iter().enumerate() {
        assert_eq!(
            group.generator_order(index).unwrap() as usize,
            closure(std::slice::from_ref(generator)).len()
        );
    }
    assert_eq!(
        group.contains_minus_identity(),
        expected.contains(&-PauliString::identity())
    );
    assert_eq!(group.group_order().unwrap(), expected.len() as u64);
    let elements: Vec<_> = group.elements().unwrap().collect();
    assert_eq!(elements.len(), expected.len());
    assert_eq!(elements.into_iter().collect::<HashSet<_>>(), expected);
    for element in &expected {
        assert!(group.contains_with_phase(element));
    }
}

#[test]
fn scalar_dependencies_and_squares() {
    for generators in [
        vec![Xs([0, 1]), Zs([0, 1]), Ys([0, 1])],
        vec![Z(0), -Z(0)],
        vec![-PauliString::identity()],
        vec![PauliString::identity()],
        vec![i * Z(0)],
        vec![Zs([0, 1]), Zs([1, 2]), Zs([0, 2])],
    ] {
        assert_group_matches_closure(generators);
    }
}

#[test]
fn stabilizer_entry_points_reject_minus_identity() {
    for generators in [
        vec![Xs([0, 1]), Zs([0, 1]), Ys([0, 1])],
        vec![Z(0), -Z(0)],
        vec![-PauliString::identity()],
    ] {
        assert!(closure(&generators).contains(&-PauliString::identity()));
        let sequence = PauliSequence::new(generators.clone()).unwrap();
        let set = PauliSet::from_iter(generators.clone());
        for result in [
            PauliStabilizerGroup::new(generators.clone()),
            PauliStabilizerGroup::try_from(sequence),
            PauliStabilizerGroup::try_from_set(&set),
            PauliStabilizerGroup::try_from(PauliGroup::new(generators).unwrap()),
        ] {
            let error = result.unwrap_err();
            assert!(matches!(
                error,
                PauliStabilizerGroupError::GeneratesMinusIdentity { .. }
            ));
            assert!(error.to_string().contains("generate -I, which stabilizes no state (empty code space); remove contradictory generators or correct their phases"));
        }
    }
    assert!(PauliStabilizerGroup::from_strs(&["XX", "ZZ", "YY"]).is_err());
    assert!("XX\nZZ\nYY".parse::<PauliStabilizerGroup>().is_err());
    for generators in [vec![i * Z(0)], vec![i * X(0), i * X(0)]] {
        assert_eq!(
            PauliStabilizerGroup::try_from(PauliSequence::new(generators).unwrap()).unwrap_err(),
            PauliStabilizerGroupError::NonRealPhase {
                index: 0,
                phase: pecos_core::QuarterPhase::PlusI
            }
        );
    }
}

#[test]
fn failed_mutations_are_atomic() {
    let mut group = PauliStabilizerGroup::new(vec![Xs([0, 1]), Zs([0, 1])]).unwrap();
    let before = group.stabilizers().to_vec();
    for generator in [Ys([0, 1]), X(0), i * Z(0)] {
        assert!(group.add_generator(generator).is_err());
        assert_eq!(group.stabilizers(), before);
        assert_eq!(group.num_generators(), 2);
        assert!(!group.as_group().contains_minus_identity());
    }
    let mut group = PauliStabilizerGroup::new(vec![Xs([0, 1])]).unwrap();
    let before = group.stabilizers().to_vec();
    let other = PauliStabilizerGroup::new(vec![Zs([0, 1]), Ys([0, 1])]).unwrap();
    assert!(matches!(
        group.merge(&other).unwrap_err(),
        PauliStabilizerGroupError::GeneratesMinusIdentity { .. }
    ));
    assert_eq!(group.stabilizers(), before);
    assert!(!group.as_group().contains_minus_identity());
    assert!(
        group
            .merge(&PauliStabilizerGroup::new(vec![Z(0)]).unwrap())
            .is_err()
    );
    assert_eq!(group.stabilizers(), before);
}

#[test]
fn mutations_preserve_commutation_error_indices() {
    let mut group = PauliStabilizerGroup::new(vec![Z(0), Z(1)]).unwrap();
    let before = group.stabilizers().to_vec();
    assert_eq!(
        group.add_generator(X(1)).unwrap_err(),
        PauliStabilizerGroupError::NonCommuting(2, 1),
    );
    assert_eq!(group.stabilizers(), before);

    // Both new generators fail against different existing generators. The
    // first new generator takes precedence, as in PauliGroup::merge.
    let other = PauliStabilizerGroup::new(vec![X(1), X(0)]).unwrap();
    assert_eq!(
        group.merge(&other).unwrap_err(),
        PauliStabilizerGroupError::NonCommuting(2, 1),
    );
    assert_eq!(group.stabilizers(), before);

    // An imaginary generator that also anticommutes reports its phase first,
    // matching `new` on the combined list.
    assert_eq!(
        group.add_generator(i * X(0)).unwrap_err(),
        PauliStabilizerGroupError::NonRealPhase {
            index: 2,
            phase: pecos_core::QuarterPhase::PlusI
        },
    );
    assert_eq!(
        PauliStabilizerGroup::new(vec![Z(0), Z(1), i * X(0)]).unwrap_err(),
        PauliStabilizerGroupError::NonRealPhase {
            index: 2,
            phase: pecos_core::QuarterPhase::PlusI
        },
    );
    assert_eq!(group.stabilizers(), before);
}

#[test]
fn conversions_preserve_scalar_subgroup() {
    for generators in [
        vec![Xs([0, 1]), Zs([0, 1]), Ys([0, 1])],
        vec![i * X(0), X(0)],
        vec![X(0), i * X(0)],
        vec![-PauliString::identity()],
        vec![i * PauliString::identity()],
    ] {
        let expected = closure(&generators);
        let sequence = PauliSequence::new(generators.clone()).unwrap();
        let set = PauliSet::from_iter(generators);
        for group in [
            PauliGroup::try_from(sequence).unwrap(),
            PauliGroup::try_from_set(&set).unwrap(),
        ] {
            assert!(group.contains_minus_identity());
            assert_eq!(group.elements().unwrap().collect::<HashSet<_>>(), expected);
            for element in &expected {
                assert!(group.contains_with_phase(element));
            }
        }
    }
    let group =
        PauliGroup::try_from(PauliSequence::new(vec![Xs([0, 1]), Zs([0, 1]), Ys([0, 1])]).unwrap())
            .unwrap();
    assert_eq!(group.num_generators(), 3);
    assert_eq!(group.rank(), 2);
    assert!(!group.is_independent());
    assert_eq!(
        group.generators().last().unwrap(),
        &-PauliString::identity()
    );
    let group = PauliGroup::try_from(PauliSequence::new(vec![i * X(0), X(0)]).unwrap()).unwrap();
    assert_eq!(group.generators(), &[i * X(0), i * PauliString::identity()]);
}

#[test]
fn dependent_membership_and_qubit_width() {
    assert_group_matches_closure(vec![Xs([0, 1]), Zs([0, 1]), Ys([0, 1])]);
    assert_group_matches_closure(vec![X(0), -X(0)]);
    assert_group_matches_closure(vec![i * X(0), X(0)]);
    let group = PauliGroup::new(vec![Z(0)]).unwrap();
    assert!(!group.contains_with_phase(&(Z(0) & X(5))));
    assert!(!group.contains_with_phase(&X(1)));
    let group = PauliGroup::new(vec![Z(0), Z(0)]).unwrap();
    assert_eq!(group.group_order().unwrap(), 2);
    let group = PauliStabilizerGroup::new((0..40).map(|idx| Z(idx % 2)).collect()).unwrap();
    assert_eq!(group.elements().unwrap().count(), 4);
    let group = PauliStabilizerGroup::new(vec![Z(0); 65]).unwrap();
    assert_eq!(group.element(1u64 << 63).unwrap(), Z(0));
    assert_eq!(group.elements().unwrap().count(), 2);
}

#[test]
fn sequence_membership_includes_noncommuting_products() {
    for generators in [
        vec![Z(0), i * Z(0)],
        vec![X(0), Z(0)],
        vec![Z(0), X(0)],
        vec![X(0), Z(0), Y(0)],
    ] {
        let expected = closure(&generators);
        let sequence = PauliSequence::new(generators).unwrap();
        for target in all_two_qubit_paulis() {
            assert_eq!(
                sequence.contains_with_phase(&target),
                expected.contains(&target)
            );
        }
    }
    // Exhaust all one-qubit triples, including noncommuting dependencies.
    let paulis: Vec<_> = all_two_qubit_paulis()
        .into_iter()
        .filter(|p| p.qubits().iter().all(|&q| q == 0))
        .collect();
    for first in &paulis {
        for second in &paulis {
            for third in &paulis {
                let generators = vec![first.clone(), second.clone(), third.clone()];
                let expected = closure(&generators);
                let sequence = PauliSequence::new(generators).unwrap();
                for target in &paulis {
                    assert_eq!(
                        sequence.contains_with_phase(target),
                        expected.contains(target)
                    );
                }
            }
        }
    }
}

#[test]
fn clifford_preserves_minus_identity_in_both_directions() {
    for generators in [
        vec![Xs([0, 1]), Zs([0, 1]), Ys([0, 1])],
        vec![Z(0), -Z(0)],
        vec![i * X(0), X(0)],
        vec![Xs([0, 1]), Zs([0, 1]), -Ys([0, 1])],
        vec![Z(0), Z(0)],
    ] {
        let expected = closure(&generators).contains(&-PauliString::identity());
        let group = PauliGroup::new(generators).unwrap();
        for clifford in [
            CliffordRep::h(0).extended_to(2),
            CliffordRep::sz(0).extended_to(2),
            CliffordRep::cx(0, 1),
        ] {
            assert_eq!(
                group
                    .apply_clifford(&clifford)
                    .unwrap()
                    .contains_minus_identity(),
                expected
            );
        }
    }
}

#[test]
fn enumeration_uses_original_basis_and_scalar_coset_order() {
    let identity = PauliString::identity();
    let x = X(0);
    let z = Z(1);
    for (generators, expected) in [
        (
            vec![x.clone(), i * x.clone()],
            vec![
                identity.clone(),
                x.clone(),
                i * identity.clone(),
                i * x.clone(),
                -identity.clone(),
                -x.clone(),
                -i * identity.clone(),
                -i * x.clone(),
            ],
        ),
        (
            vec![i * x.clone(), x.clone()],
            vec![
                identity.clone(),
                i * x.clone(),
                -identity.clone(),
                -i * x.clone(),
                i * identity.clone(),
                -x.clone(),
                -i * identity.clone(),
                x.clone(),
            ],
        ),
        (
            vec![z.clone(), z.clone(), x.clone(), -identity.clone()],
            vec![
                identity.clone(),
                z.clone(),
                x.clone(),
                x.clone() * &z,
                -identity.clone(),
                -z.clone(),
                -x.clone(),
                -(x.clone() * &z),
            ],
        ),
        (
            vec![i * x.clone(), i * z.clone()],
            vec![
                identity.clone(),
                i * x.clone(),
                -identity.clone(),
                -i * x.clone(),
                i * z.clone(),
                -(x.clone() * &z),
                -i * z.clone(),
                x.clone() * &z,
            ],
        ),
    ] {
        assert_eq!(
            expected.iter().cloned().collect::<HashSet<_>>(),
            closure(&generators)
        );
        assert_eq!(
            PauliGroup::new(generators)
                .unwrap()
                .elements()
                .unwrap()
                .collect::<Vec<_>>(),
            expected
        );
    }
}

fn all_two_qubit_paulis() -> Vec<PauliString> {
    let mut paulis = Vec::new();
    for first in [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z] {
        for second in [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z] {
            for phase in [
                QuarterPhase::PlusOne,
                QuarterPhase::MinusOne,
                QuarterPhase::PlusI,
                QuarterPhase::MinusI,
            ] {
                paulis.push(PauliString::from_paulis_with_phase(phase, &[first, second]));
            }
        }
    }
    paulis
}

// Reimplementation of the previous mixed-radix enumeration, independent of element().
fn legacy_elements(generators: &[PauliString]) -> Vec<PauliString> {
    let orders: Vec<u64> = generators
        .iter()
        .map(|g| match g.phase() {
            QuarterPhase::PlusOne | QuarterPhase::MinusOne => 2,
            QuarterPhase::PlusI | QuarterPhase::MinusI => 4,
        })
        .collect();
    (0..orders.iter().product())
        .map(|mut index| {
            let mut product = PauliString::identity();
            for (generator, order) in generators.iter().zip(&orders) {
                for _ in 0..index % order {
                    product = product * generator;
                }
                index /= order;
            }
            product
        })
        .collect()
}

fn check_exhaustive_case(generators: &[PauliString], targets: &[PauliString]) {
    let expected = closure(generators);
    let minus_identity = expected.contains(&-PauliString::identity());
    let group = PauliGroup::new(generators.to_vec()).unwrap();
    assert_eq!(
        group.contains_minus_identity(),
        minus_identity,
        "{generators:?}"
    );
    assert_eq!(
        group.group_order().unwrap(),
        expected.len() as u64,
        "{generators:?}"
    );
    for target in targets {
        assert_eq!(
            group.contains_with_phase(target),
            expected.contains(target),
            "{generators:?}: {target:?}"
        );
    }
    let elements: Vec<_> = group.elements().unwrap().collect();
    assert_eq!(elements.len(), expected.len(), "{generators:?}");
    assert_eq!(
        elements.iter().cloned().collect::<HashSet<_>>(),
        expected,
        "{generators:?}"
    );
    let imaginary_count = generators
        .iter()
        .filter(|g| matches!(g.phase(), QuarterPhase::PlusI | QuarterPhase::MinusI))
        .count();
    if group.is_independent() && imaginary_count <= 1 {
        assert_eq!(elements, legacy_elements(generators));
    }
    assert_eq!(
        PauliStabilizerGroup::new(generators.to_vec()).is_ok(),
        imaginary_count == 0 && !minus_identity
    );
    // Both conversion paths must preserve the entire closure, including scalars.
    let reduced = PauliGroup::try_from(PauliSequence::new(generators.to_vec()).unwrap()).unwrap();
    assert_eq!(
        reduced.elements().unwrap().collect::<HashSet<_>>(),
        expected
    );
}

#[test]
fn exhaustive_commuting_lists_match_closure() {
    let start = std::time::Instant::now();
    let paulis = all_two_qubit_paulis();
    let mut counts = [0; 3];
    for first in &paulis {
        check_exhaustive_case(std::slice::from_ref(first), &paulis);
        counts[0] += 1;
        for second in &paulis {
            if !first.commutes_with(second) {
                continue;
            }
            check_exhaustive_case(&[first.clone(), second.clone()], &paulis);
            counts[1] += 1;
            for third in &paulis {
                if first.commutes_with(third) && second.commutes_with(third) {
                    check_exhaustive_case(&[first.clone(), second.clone(), third.clone()], &paulis);
                    counts[2] += 1;
                }
            }
        }
    }
    assert_eq!(counts, [64, 2176, 47104]);
    eprintln!(
        "exhaustive commuting lists: {counts:?}, elapsed {:?}",
        start.elapsed()
    );
}
