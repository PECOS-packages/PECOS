//! Frozen external oracle comparisons through the public PECOS facade builders.
use ndarray::Array1;
use pecos_decoder_core::{Decoder, DemCheckMatrix};
use pecos_relay_bp::{
    DecodingResult, MinSumBpBuilder, RelayBpBuilder, RelayBpError, StoppingCriterion,
};
use std::collections::BTreeMap;

const SHOTS: usize = 100;
const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../pecos-bp/tests/data/");

/// Deterministic gamma vector in `[-0.24, 0.66)` (no RNG dependency).
fn shared_gamma_vector(n: usize, mut state: u64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = f64::from(u32::try_from(state >> 32).unwrap()) / 4_294_967_296.0;
            -0.24 + 0.90 * u
        })
        .collect()
}

struct Header<'a>(BTreeMap<&'a str, &'a str>);

impl<'a> Header<'a> {
    fn get(&self, key: &str) -> &'a str {
        self.0
            .get(key)
            .unwrap_or_else(|| panic!("missing fixture key {key}"))
    }

    fn number<T: std::str::FromStr>(&self, key: &str) -> T {
        self.get(key)
            .parse()
            .unwrap_or_else(|_| panic!("malformed {key}={}", self.get(key)))
    }

    fn some(&self, key: &str) -> f64 {
        let value = self.get(key);
        value
            .strip_prefix("Some(")
            .and_then(|s| s.strip_suffix(')'))
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("malformed {key}={value}"))
    }

    fn build(
        &self,
        dcm: &DemCheckMatrix,
    ) -> Box<dyn Decoder<Result = DecodingResult, Error = RelayBpError>> {
        assert_eq!(
            self.get("check_matrix"),
            "crates/pecos-bp/tests/data/gross_144_12_12.dem"
        );
        assert_eq!(
            self.get("shots"),
            "crates/pecos-bp/tests/data/gross_144_12_12_shots.txt (0..99 in order)"
        );
        assert_eq!(
            self.get("error_priors"),
            "the DEM's mechanism probabilities"
        );
        let factor = self.number("alpha_iteration_scaling_factor");
        match self.get("decoder") {
            "MinSumBpBuilder" => {
                assert_eq!(self.get("alpha"), "None");
                assert_eq!(self.get("gamma0"), "None");
                assert_eq!(self.get("seed"), "not applicable (plain min-sum)");
                assert_eq!(
                    self.get("stopping_criterion"),
                    "not applicable (plain min-sum)"
                );
                Box::new(
                    MinSumBpBuilder::new(&dcm.check_matrix.view())
                        .error_priors(&dcm.error_priors)
                        .max_iter(self.number("max_iter"))
                        .alpha(None)
                        .gamma0(None)
                        .alpha_iteration_scaling_factor(factor)
                        .build()
                        .unwrap(),
                )
            }
            "RelayBpBuilder" => {
                // The external Relay builder accepted this, but never read it.
                assert_eq!(self.get("max_iter"), "200");
                let stop = match self.get("stopping_criterion") {
                    "PreIter" => StoppingCriterion::PreIter,
                    "All" => StoppingCriterion::All,
                    "NConv { stop_after: 1 }" => StoppingCriterion::NConv { stop_after: 1 },
                    other => panic!("unexpected stopping_criterion={other}"),
                };
                assert_eq!(self.get("gamma_dist_interval"), "(-0.24, 0.66)");
                let mut builder = RelayBpBuilder::new(&dcm.check_matrix.view())
                    .error_priors(&dcm.error_priors)
                    .alpha(Some(self.some("alpha")))
                    .alpha_iteration_scaling_factor(factor)
                    .gamma0(Some(self.some("gamma0")))
                    .seed(self.number("seed"))
                    .pre_iter(self.number("pre_iter"))
                    .num_sets(self.number("num_sets"))
                    .set_max_iter(self.number("set_max_iter"))
                    .gamma_dist_interval((-0.24, 0.66))
                    .stopping_criterion(stop);
                let explicit = self
                    .get("explicit_gammas")
                    .strip_suffix(" (wrapper public order: row 0 serves relay leg 1, rows cycle)")
                    .unwrap();
                match explicit {
                    "None" => assert_eq!(self.number::<usize>("num_sets"), 0),
                    "[shared_gamma_vector(8785, 0x00C0_FFEE), shared_gamma_vector(8785, 0x0BAD_F00D)]" =>
                    {
                        builder = builder.explicit_gammas(vec![
                            shared_gamma_vector(8785, 0x00C0_FFEE),
                            shared_gamma_vector(8785, 0x0BAD_F00D),
                        ]);
                    }
                    other => panic!("unexpected explicit_gammas={other}"),
                }
                Box::new(builder.build().unwrap())
            }
            other => panic!("unknown decoder={other}"),
        }
    }
}

struct OracleShot {
    converged: bool,
    correction: Vec<u8>,
}

fn parse_oracle(data: &str, mechanisms: usize) -> (Header<'_>, Vec<OracleShot>) {
    let mut header = Header(BTreeMap::new());
    let mut shots = Vec::new();
    for line in data.lines() {
        if let Some(comment) = line.strip_prefix("# ") {
            if let Some((key, value)) = comment.split_once('=') {
                assert!(header.0.insert(key, value).is_none(), "duplicate key {key}");
            } else {
                assert_eq!(
                    comment,
                    "relay-bp 0.2.2 external oracle, generated by crates/pecos-bp/tests/relay_oracle_dump.rs"
                );
            }
            continue;
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 3, "malformed shot: {line}");
        let converged = match fields[0] {
            "0" => false,
            "1" => true,
            _ => panic!("malformed convergence flag: {line}"),
        };
        let _iterations: usize = fields[1]
            .parse()
            .unwrap_or_else(|_| panic!("malformed iterations: {line}"));
        let mut correction = vec![0; mechanisms];
        if fields[2] != "-" {
            let mut previous = None;
            for index in fields[2].split(',') {
                let index: usize = index
                    .parse()
                    .unwrap_or_else(|_| panic!("malformed correction: {line}"));
                assert!(
                    index < mechanisms && previous.is_none_or(|p| p < index),
                    "invalid correction index {index}"
                );
                correction[index] = 1;
                previous = Some(index);
            }
        }
        shots.push(OracleShot {
            converged,
            correction,
        });
    }
    assert_eq!(shots.len(), SHOTS);
    (header, shots)
}

#[test]
fn public_builders_match_frozen_oracle() {
    let dem = std::fs::read_to_string(format!("{DATA}gross_144_12_12.dem")).unwrap();
    let dcm = DemCheckMatrix::from_dem_str(&dem).unwrap();
    assert_eq!((dcm.num_detectors, dcm.num_mechanisms), (1008, 8785));
    let shots = std::fs::read_to_string(format!("{DATA}gross_144_12_12_shots.txt")).unwrap();
    let shots: Vec<Array1<u8>> = shots
        .lines()
        .map(|line| {
            Array1::from(
                line.split_once(' ')
                    .unwrap()
                    .0
                    .bytes()
                    .map(|b| b - b'0')
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    assert_eq!(shots.len(), SHOTS);
    for name in [
        "c2_explicit_two_rows",
        "c5_dynamic_alpha_pre8",
        "c5_dynamic_alpha_pre16",
        "c6_preiter_run_all",
        "c8_min_sum_iter8",
        "c8_min_sum_iter16",
        "c9_preiter_pre8_relay",
        "c10_dynamic_alpha_relay",
    ] {
        let data = std::fs::read_to_string(format!("{DATA}relay_oracle/{name}.txt")).unwrap();
        let (header, oracle) = parse_oracle(&data, dcm.num_mechanisms);
        assert_eq!(header.get("name"), name);
        let mut decoder = header.build(&dcm);
        let (mut flags, mut jointly_converged, mut corrections) = (0, 0, 0);
        for (syndrome, expected) in shots.iter().zip(oracle) {
            let actual = decoder.decode(&syndrome.view()).unwrap();
            flags += usize::from(actual.converged == expected.converged);
            if actual.converged && expected.converged {
                jointly_converged += 1;
                corrections += usize::from(actual.decoding.to_vec() == expected.correction);
            }
        }
        println!("{name}: flags {flags}/100; corrections {corrections}/{jointly_converged}");
        assert!(flags >= 98, "{name}: flags {flags}");
        assert!(jointly_converged > 0, "{name}: no jointly converged shots");
        assert!(
            corrections * 20 >= jointly_converged * 19,
            "{name}: corrections {corrections}/{jointly_converged}"
        );
    }
}
