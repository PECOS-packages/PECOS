//! Compare constant evaluation with runtime generic conversion (not two runtime calls).
#![allow(
    clippy::large_const_arrays,
    clippy::cast_precision_loss,
    long_running_const_eval
)]

use pecos_core::{Angle64, NonFiniteAngleError};

const COUNT: usize = 50_000;
type Row = (
    f64,
    Result<Angle64, NonFiniteAngleError>,
    Result<Angle64, NonFiniteAngleError>,
);
const TABLE: [Row; COUNT] = {
    let mut rows = [(0.0, Ok(Angle64::ZERO), Ok(Angle64::ZERO)); COUNT];
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut i = 0;
    while i < COUNT {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let x = match i % 8 {
            0 | 1 => f64::from_bits(state),
            2 => (i as f64 - 25_000.0) * std::f64::consts::PI / 4.0,
            3 => -f64::from_bits(state & 0x000f_ffff_ffff_ffff),
            4 => (state as f64) * 1e280,
            5 => -(state as f64) * 1e280,
            6 => -(i as f64) * 1e-22,
            _ => match i % 64 {
                7 => f64::NAN,
                15 => f64::INFINITY,
                23 => f64::NEG_INFINITY,
                31 => -0.0,
                39 => f64::MAX,
                47 => -f64::MAX,
                55 => std::f64::consts::TAU,
                _ => 1.0,
            },
        };
        rows[i] = (
            x,
            Angle64::const_try_from_radians(x),
            Angle64::const_try_from_turns(x),
        );
        i += 1;
    }
    rows
};

fn assert_same(
    constant: Result<Angle64, NonFiniteAngleError>,
    runtime: Result<Angle64, NonFiniteAngleError>,
    input: f64,
) {
    match (constant, runtime) {
        (Ok(a), Ok(b)) => assert_eq!(a.fraction(), b.fraction(), "input={input:?}"),
        (Err(a), Err(b)) => {
            assert_eq!(a.value.to_bits(), b.value.to_bits());
            assert_eq!(a.unit, b.unit);
        }
        (a, b) => panic!("const/runtime disagreement for {input:?}: {a:?} vs {b:?}"),
    }
}

#[test]
fn const_evaluation_matches_generic_runtime_conversion() {
    for &(x, radians, turns) in &TABLE {
        assert_same(
            radians,
            Angle64::try_from_radians(std::hint::black_box(x)),
            x,
        );
        assert_same(turns, Angle64::try_from_turns(std::hint::black_box(x)), x);
    }
}

#[test]
fn prelude_exports_literal_macros() {
    use pecos_core::prelude::*;
    assert_eq!(radians!(0.37), pecos_core::radians!(0.37));
    assert_eq!(turns!(0.25), Angle64::QUARTER_TURN);
}

#[test]
fn migrated_exact_constants_preserve_old_float_bits() {
    use pecos_core::Angle64;
    assert_eq!(
        Angle64::try_from_radians(std::hint::black_box(f64::from_bits(0))).unwrap(),
        Angle64::ZERO
    );
    assert_eq!(
        Angle64::try_from_radians(std::hint::black_box(f64::from_bits(
            4_600_745_857_669_934_360
        )))
        .unwrap(),
        Angle64::from_turn_ratio(1, 16)
    );
    assert_eq!(
        Angle64::try_from_radians(std::hint::black_box(f64::from_bits(
            4_605_249_457_297_304_856
        )))
        .unwrap(),
        Angle64::from_turn_ratio(1, 8)
    );
    assert_eq!(
        Angle64::try_from_radians(std::hint::black_box(f64::from_bits(
            4_609_753_056_924_675_352
        )))
        .unwrap(),
        Angle64::QUARTER_TURN
    );
    assert_eq!(
        Angle64::try_from_radians(std::hint::black_box(f64::from_bits(
            4_614_256_656_552_045_848
        )))
        .unwrap(),
        Angle64::HALF_TURN
    );
    assert_eq!(
        Angle64::try_from_radians(std::hint::black_box(f64::from_bits(
            4_616_991_696_741_409_234
        )))
        .unwrap(),
        Angle64::THREE_QUARTERS_TURN
    );
    assert_eq!(
        Angle64::try_from_radians(std::hint::black_box(f64::from_bits(
            4_618_760_256_179_416_344
        )))
        .unwrap(),
        Angle64::ZERO
    );
    assert_eq!(
        Angle64::try_from_radians(std::hint::black_box(f64::from_bits(
            4_626_883_175_715_153_781
        )))
        .unwrap(),
        Angle64::HALF_TURN
    );
    assert_eq!(
        Angle64::try_from_radians(std::hint::black_box(f64::from_bits(
            13_828_621_494_152_080_664
        )))
        .unwrap(),
        Angle64::from_turn_ratio(7, 8)
    );
    assert_eq!(
        Angle64::try_from_radians(std::hint::black_box(f64::from_bits(
            13_833_125_093_779_451_160
        )))
        .unwrap(),
        Angle64::THREE_QUARTERS_TURN
    );
    assert_eq!(
        Angle64::try_from_radians(std::hint::black_box(f64::from_bits(
            13_835_860_133_968_814_546
        )))
        .unwrap(),
        Angle64::from_turn_ratio(5, 8)
    );
    assert_eq!(
        Angle64::try_from_radians(std::hint::black_box(f64::from_bits(
            13_837_628_693_406_821_656
        )))
        .unwrap(),
        Angle64::HALF_TURN
    );
    assert_eq!(
        Angle64::try_from_turns(std::hint::black_box(f64::from_bits(0))).unwrap(),
        Angle64::ZERO
    );
    assert_eq!(
        Angle64::try_from_turns(std::hint::black_box(f64::from_bits(
            4_589_168_020_290_535_424
        )))
        .unwrap(),
        Angle64::from_turn_ratio(1, 16)
    );
    assert_eq!(
        Angle64::try_from_turns(std::hint::black_box(f64::from_bits(
            4_593_671_619_917_905_920
        )))
        .unwrap(),
        Angle64::from_turn_ratio(1, 8)
    );
    assert_eq!(
        Angle64::try_from_turns(std::hint::black_box(f64::from_bits(
            4_598_175_219_545_276_416
        )))
        .unwrap(),
        Angle64::QUARTER_TURN
    );
    assert_eq!(
        Angle64::try_from_turns(std::hint::black_box(f64::from_bits(
            4_600_427_019_358_961_664
        )))
        .unwrap(),
        Angle64::from_turn_ratio(3, 8)
    );
    assert_eq!(
        Angle64::try_from_turns(std::hint::black_box(f64::from_bits(
            4_602_678_819_172_646_912
        )))
        .unwrap(),
        Angle64::HALF_TURN
    );
    assert_eq!(
        Angle64::try_from_turns(std::hint::black_box(f64::from_bits(
            4_607_182_418_800_017_408
        )))
        .unwrap(),
        Angle64::ZERO
    );
    assert_eq!(
        Angle64::try_from_turns(std::hint::black_box(f64::from_bits(
            4_618_159_942_891_732_992
        )))
        .unwrap(),
        Angle64::THREE_QUARTERS_TURN
    );
    assert_eq!(
        Angle64::try_from_turns(std::hint::black_box(f64::from_bits(
            13_817_043_656_772_681_728
        )))
        .unwrap(),
        Angle64::from_turn_ratio(7, 8)
    );
    assert_eq!(
        Angle64::try_from_turns(std::hint::black_box(f64::from_bits(
            13_821_547_256_400_052_224
        )))
        .unwrap(),
        Angle64::THREE_QUARTERS_TURN
    );
    assert_eq!(
        Angle64::try_from_turns(std::hint::black_box(f64::from_bits(
            13_830_554_455_654_793_216
        )))
        .unwrap(),
        Angle64::ZERO
    );
}
