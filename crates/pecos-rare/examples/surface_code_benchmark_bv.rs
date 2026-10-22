// An example of how to perform a brute force Monte Carlo simulation for determining
// the logical error rate of a surface code

use pecos_qec::SurfaceCode;
use pecos_quantum::{AnnotationKind, Attribute, TickCircuit, TickMeasRef};
use pecos_rare::bv::{find_logical_failure_rate, gen_sequence};

/// Build a rotated, square surface-code Z-memory experiment.
///
/// Supports odd distances >= 3 and at least one syndrome round. Data qubits
/// are numbered row-major; one dedicated ancilla is allocated per check.
/// The four simultaneous CX layers use the standard N/Z extraction ordering:
///
/// ```text
///       TL -- TR       X order: TR, TL, BR, BL
///       |     |        Z order: TR, BR, TL, BL
///       BL -- BR
/// ```
/// X hooks are horizontal, perpendicular to logical X (the left column).
/// Z hooks are vertical, perpendicular to logical Z (the top row).
/// Boundary checks use the corresponding two slots of the bulk schedule.
/// This is the hook-avoiding ordering discussed by Tomita and Svore,
/// https://arxiv.org/abs/1404.3747, and used by PECOS's Python surface builder.
/// Noise is supplied separately by the Monte Carlo engine / DEM builder.
pub fn surface_code_z_memory(
    distance: usize,
    rounds: usize,
) -> Result<TickCircuit, Box<dyn std::error::Error>> {
    if distance < 3 || distance % 2 == 0 {
        return Err("surface-code distance must be odd and >= 3".into());
    }
    if rounds == 0 {
        return Err("surface-code memory requires at least one syndrome round".into());
    }

    let code = SurfaceCode::rotated(distance)?;
    let num_data = code.num_data_qubits();
    let num_x = code.num_x_stabilizers();
    let num_z = code.num_z_stabilizers();
    let data: Vec<_> = (0..num_data).collect();
    let x_ancillas: Vec<_> = (num_data..num_data + num_x).collect();
    let ancillas: Vec<_> = (num_data..num_data + num_x + num_z).collect();

    // Construct the schedule once and reuse it in every syndrome round.
    let mut layers: [Vec<(usize, usize)>; 4] = std::array::from_fn(|_| Vec::new());
    for (is_x, checks, ancilla_start) in [
        (true, code.x_stabilizers(), num_data),
        (false, code.z_stabilizers(), num_data + num_x),
    ] {
        for check in checks {
            let mut qubits = check.qubits();
            qubits.sort_unstable();
            let (slots, order): (&[usize], &[usize]) = match qubits.len() {
                4 if is_x => (&[0, 1, 2, 3], &[1, 0, 3, 2]),
                4 => (&[0, 1, 2, 3], &[1, 3, 0, 2]),
                2 if is_x && qubits[0] / distance == 0 => (&[2, 3], &[1, 0]),
                2 if is_x => (&[0, 1], &[1, 0]),
                2 if qubits[0] % distance == 0 => (&[0, 1], &[0, 1]),
                2 => (&[2, 3], &[0, 1]),
                _ => return Err("unexpected rotated surface-code check weight".into()),
            };
            let ancilla = ancilla_start + check.index;
            for (&slot, &index) in slots.iter().zip(order) {
                let q = qubits[index];
                layers[slot].push(if is_x { (ancilla, q) } else { (q, ancilla) });
            }
        }
    }

    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&data);
    let mut previous_x: Vec<TickMeasRef> = Vec::new();
    let mut previous_z: Vec<TickMeasRef> = Vec::new();
    for round in 0..rounds {
        circuit.tick().pz(&ancillas);
        circuit.tick().h(&x_ancillas);
        for layer in &layers {
            circuit.tick().cx(layer);
        }
        circuit.tick().h(&x_ancillas);
        let mut current_x = circuit.tick().mz(&ancillas);
        let current_z = current_x.split_off(num_x);

        if round == 0 {
            // Only Z checks are deterministic after product-state |0> preparation.
            for &measurement in &current_z {
                circuit.detector(&[measurement])?;
            }
        } else {
            for (&before, &after) in previous_x.iter().zip(&current_x) {
                circuit.detector(&[before, after])?;
            }
            for (&before, &after) in previous_z.iter().zip(&current_z) {
                circuit.detector(&[before, after])?;
            }
        }
        previous_x = current_x;
        previous_z = current_z;
    }

    let final_data = circuit.tick().mz(&data);
    for check in code.z_stabilizers() {
        let mut refs = vec![previous_z[check.index]];
        refs.extend(check.qubits().into_iter().map(|q| final_data[q]));
        circuit.detector(&refs)?;
    }
    let logical_z: Vec<_> = code
        .logical_z()
        .data_qubits
        .iter()
        .map(|&q| final_data[q])
        .collect();
    circuit.observable_labeled("L0", &logical_z)?;

    // The DEM builder consumes detector record metadata, while the typed
    // annotations retain the measurement references for other circuit tools.
    let detectors: Vec<_> = circuit
        .annotations()
        .iter()
        .filter_map(|annotation| match &annotation.kind {
            AnnotationKind::Detector {
                measurement_ids, ..
            } => Some(measurement_ids),
            _ => None,
        })
        .enumerate()
        .map(|(id, measurements)| {
            let ids: Vec<_> = measurements.iter().map(|m| m.index().to_string()).collect();
            format!(r#"{{"id":{id},"meas_ids":[{}]}}"#, ids.join(","))
        })
        .collect();
    circuit.set_meta(
        "detectors",
        Attribute::String(format!("[{}]", detectors.join(","))),
    );
    circuit.set_meta(
        "num_measurements",
        Attribute::String(circuit.num_measurements().to_string()),
    );
    Ok(circuit)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {

    // Goal of this script is to loop over some parameters and make sure we can converge
    // the resulting error rates with respect to all of those. 

    // Things to loop over:
    // - Number of levels
    // - distance
    // - error target
    // - flipper type
    // - burn in

    let p0 = 0.1;
    let pf = 0.00001;
    let max_brute_force_histories = 100_000_000_000_000;
    let nbootstraps_brute_force = 5000;
    let check_frequency_brute_force = 1_000;
    let max_metropolis_iterations = 100_000_000_000_000;
    let max_ratio_iterations = 1_000;
    let minimum_accepted_flips = 1_000;

    let distance = 5;
    let rounds = distance;
    let circuit = surface_code_z_memory(distance, rounds)?;
    let npts = 21;
    let prob_sequence = gen_sequence(p0, pf, npts);
    let results = find_logical_failure_rate(
        &circuit,
        prob_sequence.clone(),
        0, // seed,
        max_brute_force_histories,
        nbootstraps_brute_force,
        check_frequency_brute_force,
        0.1, // error_target,
        "basic_subregion", // flipper_type,
        0.01, // region_probability = 1.0 / ( distance / 2 ) as f64;
        max_metropolis_iterations,
        1_000, // burn_in,
        max_ratio_iterations,
        1_000, // minimum_accepted_flips,
    )?;


    // for &distance in &[3, 5, 7, 9, 11] {
    //     let region_probability = 1.0 / ( distance / 2 ) as f64;
    //     // Set up the surface code
    //     let rounds = distance;
    //     let circuit = surface_code_z_memory(distance, rounds)?;
    //     for &burn_in in &[10_000, 1_000, 100] {
    //         for &error_target in &[0.01, 0.001, 0.1, 0.0001, 0.00001] {
    //             for &npts in &[21, 51, 101, 11, 5] {
    //                 let prob_sequence = gen_sequence(p0, pf, npts);
    //                 println!("{:?}", prob_sequence);
    //                 for &flipper_type in &["basic_subregion", "weighted_subregion"] { // , "weighted", "basic"] {
    //                     for &seed in &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] {
    //                         // Specify the decoder
    //                         let results = find_logical_failure_rate(
    //                             &circuit,
    //                             prob_sequence.clone(),
    //                             seed,
    //                             max_brute_force_histories,
    //                             nbootstraps_brute_force,
    //                             check_frequency_brute_force,
    //                             error_target,
    //                             flipper_type,
    //                             region_probability,
    //                             max_metropolis_iterations,
    //                             burn_in,
    //                             max_ratio_iterations,
    //                             minimum_accepted_flips,
    //                         )?;
    //                         // Save the results
    //                         results.write_json(&format!("bv_distance{}__burnin{}__error_target{}__flipper_type{}__npts{}_seed{}_surface_code_bv_results.json", distance, burn_in, error_target, flipper_type, npts, seed), 1, Some(&prob_sequence))?;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    Ok(())
}
