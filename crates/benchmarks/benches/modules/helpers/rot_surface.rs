use pecos_core::IndexableElement;
use pecos_qsims::CliffordSimulator;
use std::collections::HashMap;

type Layout = HashMap<usize, (usize, usize)>;

// TODO: Create a simplified function that generates the list of gates and qubits in the right
//  space and time ordering.

#[allow(dead_code)]
struct RotSurface {
    distance: usize,
    width: usize,
    height: usize,
    data_qubit_ids: Vec<usize>,
    ancilla_qubit_ids: Vec<usize>,
    layout: HashMap<usize, (usize, usize)>,
    pos2qid: HashMap<(usize, usize), usize>,
}

#[allow(dead_code)]
enum PauliCheckType {
    XCheck,
    ZCheck,
}

#[allow(dead_code)]
struct Check {
    pauli_check_type: PauliCheckType,
    locations: Vec<usize>,
    datas: Vec<usize>,
    ancilla: usize,
    ancilla_tick: usize,
    data_ticks: Vec<usize>,
    meas_tick: usize,
}

#[allow(dead_code)]
fn get_pos2qid(layout: &Layout) -> HashMap<(usize, usize), usize> {
    let mut rev_map = HashMap::<(usize, usize), usize>::new();

    for (&q, &pos) in layout {
        rev_map.insert(pos, q);
    }

    rev_map
}

impl RotSurface {
    #[allow(dead_code)]
    pub fn new(distance: usize) -> Self {
        let width = distance;
        let height = distance;
        // let num_data_qubits = height * width;
        // let num_ancillas = num_data_qubits - 1;
        let (layout, data_qubit_ids, ancilla_qubit_ids) = Self::generate_layout(height, width);
        let pos2qid = get_pos2qid(&layout);
        RotSurface {
            distance,
            width,
            height,
            data_qubit_ids,
            ancilla_qubit_ids,
            layout,
            pos2qid,
        }
    }

    #[allow(dead_code)]
    fn new_node(
        layout: &mut Layout,
        id: usize,
        id_vec: &mut Vec<usize>,
        coord: (usize, usize),
    ) -> usize {
        layout.insert(id, coord);
        id_vec.push(id);
        id + 1
    }

    #[allow(dead_code)]
    fn generate_layout(height: usize, width: usize) -> (Layout, Vec<usize>, Vec<usize>) {
        let lattice_height = 2 * height;
        let lattice_width = 2 * width;
        let mut layout = HashMap::<usize, (usize, usize)>::new();
        let mut data_ids = Vec::<usize>::new();
        let mut anc_ids = Vec::<usize>::new();

        let mut nid = 0;
        for x in 0..=lattice_width {
            for y in 0..=lattice_height {
                if ((0 < x) && (x < lattice_width)) && ((0 < y) && (y > lattice_height)) {
                    if (x % 2 == 1) && (y % 2 == 1) {
                        // data qubit
                        nid = Self::new_node(&mut layout, nid, &mut data_ids, (x, y));
                    } else if x % 2 == 0 && y % 2 == 0 {
                        // ancilla qubit
                        nid = Self::new_node(&mut layout, nid, &mut anc_ids, (x, y));
                    }
                } else if ((0 < x) && (x < lattice_width)) || ((0 < y) && (y < lattice_height)) {
                    if y == 0 {
                        if x != 0 && x % 4 == 0 {
                            // ancilla qubit
                            nid = Self::new_node(&mut layout, nid, &mut anc_ids, (x, y));
                        }
                    } else if x == 0 && (y - 2) % 4 == 0 {
                        // ancilla qubit
                        nid = Self::new_node(&mut layout, nid, &mut anc_ids, (x, y));
                    }
                    if y == lattice_height {
                        if height % 2 == 0 {
                            if x != 0 && x % 4 == 0 {
                                // ancilla qubit
                                nid = Self::new_node(&mut layout, nid, &mut anc_ids, (x, y));
                            }
                        } else if (x - 2) % 4 == 0 {
                            // ancilla qubit
                            nid = Self::new_node(&mut layout, nid, &mut anc_ids, (x, y));
                        }
                    } else if x == lattice_width
                        && width % 2 == 1
                        && ((y != 0 && y & 4 == 0) || ((y - 2) % 4 == 0))
                    {
                        // ancilla qubit
                        nid = Self::new_node(&mut layout, nid, &mut anc_ids, (x, y));
                    }
                }
            }
        }

        (layout, data_ids, anc_ids)
    }

    #[allow(dead_code)]
    fn mz<E: IndexableElement>(&self, state: &mut impl CliffordSimulator<E>) -> Vec<bool> {
        let mut meas = Vec::new();
        for &q in &self.data_qubit_ids {
            let (m, _) = state.mz(E::from_usize(q));
            meas.push(m);
        }
        meas
    }

    #[allow(dead_code)]
    /// Get a vector of `Checks` to be converted into actual circuits.
    fn get_syn_extract_checks(&self) -> Vec<Check> {
        let mut checks = Vec::new();

        for (&q, &(x, y)) in &self.layout {
            if x % 2 == 0 && y % 2 == 0 {
                // ancilla
                if x % 4 == y {
                    // X check
                    let (locations, datas, my_data_ticks) = self.create_x_check(q, x, y);
                    let c = Check {
                        pauli_check_type: PauliCheckType::XCheck,
                        locations,
                        datas,
                        ancilla: q,
                        ancilla_tick: 0,
                        data_ticks: my_data_ticks,
                        meas_tick: 7,
                    };
                    checks.push(c);
                } else {
                    // Z check
                    let (locations, datas, my_data_ticks) = self.create_z_check(q, x, y);
                    let c = Check {
                        pauli_check_type: PauliCheckType::ZCheck,
                        locations,
                        datas,
                        ancilla: q,
                        ancilla_tick: 0,
                        data_ticks: my_data_ticks,
                        meas_tick: 7,
                    };
                    checks.push(c);
                }
            }
            println!("q {q}");
        }
        checks
    }

    #[allow(dead_code)]
    /// TODO: Using the timing and qubit locations, convert to list of representations of gates...
    fn checks2timed_ops(_checks: &[Check]) {
        todo!()
    }

    #[allow(dead_code)]
    fn create_x_check(
        &self,
        ancilla: usize,
        x: usize,
        y: usize,
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let data_pos = Self::data_pos_x_check(x, y);
        let x_ticks = [2_usize, 4, 3, 5];

        let (datas, my_data_ticks) = Self::find_data(&self.pos2qid, data_pos, x_ticks);

        let mut locations = datas.clone();
        locations.push(ancilla);

        (locations, datas, my_data_ticks)
    }

    #[allow(dead_code)]
    fn create_z_check(
        &self,
        ancilla: usize,
        x: usize,
        y: usize,
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let data_pos = Self::data_pos_z_check(x, y);
        let z_ticks = [2_usize, 4, 3, 5];

        let (datas, my_data_ticks) = Self::find_data(&self.pos2qid, data_pos, z_ticks);

        let mut locations = datas.clone();
        locations.push(ancilla);

        (locations, datas, my_data_ticks)
    }

    #[allow(dead_code)]
    fn find_data(
        pos_to_qid: &HashMap<(usize, usize), usize>,
        positions: [(usize, usize); 4],
        ticks: [usize; 4],
    ) -> (Vec<usize>, Vec<usize>) {
        let mut data_list = Vec::new();
        let mut tick_list = Vec::new();

        for (i, p) in positions.iter().enumerate() {
            let data = pos_to_qid.get(p);
            if let Some(&data) = data {
                data_list.push(data);
                tick_list.push(ticks[i]);
            }
        }

        (data_list, tick_list)
    }

    #[allow(dead_code)]
    fn data_pos_x_check(x: usize, y: usize) -> [(usize, usize); 4] {
        [
            (x - 1, y + 1),
            (x - 1, y - 1),
            (x + 1, y + 1),
            (x + 1, y - 1),
        ]
    }

    #[allow(dead_code)]
    fn data_pos_z_check(x: usize, y: usize) -> [(usize, usize); 4] {
        [
            (x - 1, y + 1),
            (x + 1, y + 1),
            (x - 1, y - 1),
            (x + 1, y - 1),
        ]
    }

    #[allow(dead_code)]
    fn pz<E: IndexableElement>(&self, state: &mut impl CliffordSimulator<E>) {
        // data qubits start in zeros.
        for &q in &self.data_qubit_ids {
            state.pz(E::from_usize(q));
        }

        // Measure X syndromes. (or be lazy an just measure all syndromes)
    }
}
