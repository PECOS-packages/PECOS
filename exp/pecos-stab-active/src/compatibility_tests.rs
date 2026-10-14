// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.
use super::*;

#[test]
fn fixed_seed_trait_bits() {
    let mut state = StabActive::with_seed(3, 731);
    let mut records = Vec::new();
    for _ in 0..4 {
        state.h(&[QubitId(0)]).sz(&[QubitId(1)]);
        state.rx(Angle64::from_radians(0.321), &[QubitId(0), QubitId(2)]);
        state.ry(Angle64::from_radians(-0.789), &[QubitId(1)]);
        state.rzz(Angle64::from_radians(0.456), &[(QubitId(0), QubitId(1))]);
        state.cx(&[(QubitId(1), QubitId(2))]);
        records.extend(state.mx(&[QubitId(0)]).iter().map(|r| r.outcome));
        state.px(&[QubitId(2)]);
    }
    let bits: Vec<_> = state
        .amplitudes
        .iter()
        .map(|a| (a.re.to_bits(), a.im.to_bits()))
        .collect();
    assert_eq!(records, [false, false, false, false]);
    assert_eq!(
        bits,
        [
            (13_815_872_805_157_858_733, 13_830_198_988_909_340_100),
            (4_590_476_583_996_472_608, 4_597_922_288_466_730_060)
        ]
    );
}
