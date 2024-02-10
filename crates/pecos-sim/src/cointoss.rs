use rand::Rng;

#[derive(Debug)]
pub struct CoinToss {
    pub num_qubits: usize,
    pub prob: f64,
}

impl CoinToss {
    pub fn new(num_qubits: usize, prob: f64) -> Self {
        Self { num_qubits, prob }
    }
}

impl Default for CoinToss {
    fn default() -> Self {
        Self {
            num_qubits: 1,
            prob: 0.5,
        }
    }
}

impl CoinToss {

    pub fn h(&mut self, _qubit: usize) -> () {

    }

    pub fn cx(&mut self, _control: usize, _target: usize) -> () {

    }

    pub fn meas(&mut self, _qubit: usize) -> bool {
        let mut rng = rand::thread_rng();
        rng.gen::<f64>() < self.prob
    }
}
