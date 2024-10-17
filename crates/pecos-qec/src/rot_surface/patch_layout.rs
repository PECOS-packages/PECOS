use std::collections::HashMap;

// TODO: Maybe... Add Plaquettes with a center, vertices, edges (boundary), and types. (Edges from center to vertices?)
//  maybe vertices => plaquettes...
//  maybe overall boundaries

#[derive(Debug, Clone, PartialEq)]
pub struct RotSurfLayout {
    pub id2coords: HashMap<usize, (usize, usize)>,
    pub vertices: Vec<usize>,
    pub centers: Vec<usize>,
}

#[allow(clippy::module_name_repetitions)]
#[must_use]
pub fn gen_patch_layout(height: usize, width: usize) -> RotSurfLayout {
    let lattice_height = 2 * height;
    let lattice_width = 2 * width;
    let mut id2coords = HashMap::<usize, (usize, usize)>::new();
    let mut vertices = Vec::<usize>::new();
    let mut centers = Vec::<usize>::new();

    let mut nid = 0;
    for x in 0..=lattice_width {
        for y in 0..=lattice_height {
            if ((0 < x) && (x < lattice_width)) && ((0 < y) && (y < lattice_height)) {
                if x % 2 == 1 && y % 2 == 1 {
                    // data qubit
                    nid = new_node(&mut id2coords, nid, &mut vertices, (x, y));
                } else if x % 2 == 0 && y % 2 == 0 {
                    // interior
                    nid = new_node(&mut id2coords, nid, &mut centers, (x, y));
                }
            } else if ((0 < x) && (x < lattice_width)) || ((0 < y) && (y < lattice_height)) {
                if y == 0 {
                    // Top checks
                    if x != 0 && x % 4 == 0 {
                        nid = new_node(&mut id2coords, nid, &mut centers, (x, y));
                    }
                } else if x == 0 {
                    // Left checks
                    if y % 4 == 2 {
                        nid = new_node(&mut id2coords, nid, &mut centers, (x, y));
                    }
                }
                if y == lattice_height {
                    // Bottom checks
                    if height % 2 == 0 {
                        if x != 0 && x % 4 == 0 {
                            nid = new_node(&mut id2coords, nid, &mut centers, (x, y));
                        }
                    } else if x % 4 == 2 {
                        nid = new_node(&mut id2coords, nid, &mut centers, (x, y));
                    }
                } else if x == lattice_width {
                    // Right checks
                    if width % 2 == 1 {
                        if y != 0 && y % 4 == 0 {
                            // ancilla qubit
                            nid = new_node(&mut id2coords, nid, &mut centers, (x, y));
                            println!("Right 1: {nid}");
                        }
                    } else if y % 4 == 2 {
                        // ancilla qubit
                        nid = new_node(&mut id2coords, nid, &mut centers, (x, y));
                        println!("Right 2: {nid}");
                    }
                }
            }
        }
    }

    RotSurfLayout {
        id2coords,
        vertices,
        centers,
    }
}

fn new_node(
    layout: &mut HashMap<usize, (usize, usize)>,
    id: usize,
    id_vec: &mut Vec<usize>,
    coord: (usize, usize),
) -> usize {
    layout.insert(id, coord);
    id_vec.push(id);
    id + 1
}
