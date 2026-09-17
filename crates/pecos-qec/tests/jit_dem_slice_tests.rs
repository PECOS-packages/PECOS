// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

use pecos_qec::fault_tolerance::dem_builder::{
    ComposedDetectorAddress, DemBoundaryKind, DemDetectorPlacement, DemSlice, DemSliceContribution,
    DemSliceDetector, DemSliceInstance, DemSliceRoundSchedule, DemTemporalHorizon, DemWindowSpec,
    DetectorErrorModel, RelativeDetectorTarget, SliceFaultMechanism,
};
use std::sync::Arc;

#[test]
fn public_api_composes_relabelled_cached_slices() {
    let slice = Arc::new(
        DemSlice::new(
            "idle",
            vec![DemSliceDetector::new(0)],
            vec![DemSliceContribution::direct(
                SliceFaultMechanism::from_unsorted_parity(
                    [
                        RelativeDetectorTarget::new(0, 0),
                        RelativeDetectorTarget::new(0, 1),
                    ],
                    [0],
                ),
                0.01,
            )],
            DemTemporalHorizon::new(0, 1),
        )
        .unwrap(),
    );
    let instances: Vec<_> = (4..7)
        .map(|round| {
            DemSliceInstance::identity(Arc::clone(&slice), round)
                .with_detector_placement(0, DemDetectorPlacement::new(23))
                .with_dem_output_targets(0, [5])
        })
        .collect();

    let schedule = DemSliceRoundSchedule::from_instances(&DetectorErrorModel::new(), instances);
    let composed = schedule
        .compose(DemWindowSpec::new(4, 2, 1, DemBoundaryKind::Soft))
        .unwrap();

    assert_eq!(
        composed.detector_addresses,
        vec![
            ComposedDetectorAddress {
                round: 4,
                stream_id: 23,
            },
            ComposedDetectorAddress {
                round: 5,
                stream_id: 23,
            },
            ComposedDetectorAddress {
                round: 6,
                stream_id: 23,
            },
        ]
    );
    assert_eq!(
        composed.model.to_mechanisms().0,
        vec![
            (0.01, vec![0, 1], vec![5]),
            (0.01, vec![1, 2], vec![5]),
            (0.01, vec![2], vec![5]),
        ]
    );
    assert_eq!(composed.diagnostics.projected_future_contributions, 1);
}
