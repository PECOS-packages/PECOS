use pyo3::prelude::*;
use nohash_hasher::{IntSet, BuildNoHashHasher};

use std::collections::HashSet;
type Set<T> = HashSet<T, BuildNoHashHasher<T>>;

#[pyfunction]
pub fn hello() {
    println!("Prop hello!");

    let mut prop = Propagator::new(2, false);

    let mut f = PauliFaults::new();
    // f.xs.insert(1);
    f.xs = HashSet::from([1, 2, 3]);
    prop.add_faults(&f);

    println!("Prop: {:?} -> {:?}", f, prop);
}

#[derive(Debug)]
pub struct PauliFaults {
    xs: HashSet<u64>,
    zs: HashSet<u64>,
    sign: u8,
    img: u64,
}

impl PauliFaults {
    pub fn new() -> PauliFaults{
        return PauliFaults {
            xs: HashSet::new(),
            zs: HashSet::new(),
            sign: 0,
            img: 0,
        }
    }
}

#[derive(Debug)]
pub struct Propagator {
    num_qubits: u64,
    faults: PauliFaults,
    track_sign: bool,
}

impl Propagator {
    pub fn new(num_qubits: u64, track_sign: bool) -> Propagator {
        return Propagator {
            num_qubits,
            faults: PauliFaults::new(),
            track_sign,
        }
    }

    pub fn set_faults(&mut self, faults: PauliFaults) -> &mut Propagator {
        self.faults = faults;
        return self
    }

    pub fn flip_sign(&mut self) -> &mut Propagator {
        self.faults.sign += 1;
        self.faults.sign %= 2;
        return self
    }

    pub fn flip_img(&mut self, num_is: u64) -> &mut Propagator {
        self.faults.img += num_is;
        self.faults.img %= 4;

        if (self.faults.img == 2) | (self.faults.img == 3) {
            self.flip_sign();
        }

        self.faults.img %= 2;

        return self
    }

    pub fn add_faults(&mut self, faults: &PauliFaults) -> &mut Propagator {

        // for i in self.xs.intersection(&faults.xs) {
        //     self.xs.
        // }

        // for x in self.faults.xs.symmetric_difference(&faults.xs) {
        //     println!("{x}");
        // }
        // self.faults.xs ^= faults.xs;

        // !!!!! replace with inplace update!
        // see: https://stackoverflow.com/questions/55975234/how-do-i-intersect-two-hashsets-while-moving-values-in-common-into-a-new-set
        self.faults.xs = &self.faults.xs ^ &faults.xs;
        self.faults.zs = &self.faults.zs ^ &faults.zs;
        return self
    }
}