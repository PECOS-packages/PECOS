use pecos_qec::rot_surface::patch_layout::gen_patch_layout;

fn main() {
    let layout = gen_patch_layout(7, 7);
    println!("Layout>> {:?}", layout.id2coords);
    let num_data = layout.vertices.len();
    let num_anc = layout.centers.len();
    println!("Data>> {:?} ({:?})", layout.vertices, num_data);
    println!("Ancillas>> {:?} ({:?})", layout.centers, num_anc);
    assert_eq!(num_data, num_anc + 1);
}
