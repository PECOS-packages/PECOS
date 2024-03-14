mod cointoss;

pub use cointoss::CoinToss;

/*
use std::pin::Pin;
use cxx;

#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("sparsesim.h");

        pub type State;

        // Expose the factory method
        pub fn create(num_qubits: u64, reserve_buckets: i32) -> UniquePtr<State>;
    }
}



// Example of using the State class in Rust
pub struct RustState {
    inner: cxx::UniquePtr<ffi::State>,
}

impl RustState {
    // Adjusted to use the static factory method for instantiation
    pub fn new(num_qubits: u64, reserve_buckets: i32) -> Self {
        Self {
            // Use the factory method `create` exposed via the `cxx` bridge
            inner: ffi::create(num_qubits, reserve_buckets),
        }
    }

}
*/

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
