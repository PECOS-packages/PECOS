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

//! Structured detector-error-model windows shared by streaming decoders.

use crate::dem::dimension_count;
use crate::dem::grammar::{
    Instruction, Kind, Target, parse_line, target_indices, validate_error_components, xor_indices,
};
use crate::errors::DecoderError;
use std::fmt::Write as _;

mod commit;
pub use commit::{CommitColumn, CommitEdge, CommitMember, CommitWindow, min_buffer_rounds};

/// One graphlike or hypergraph component of an independent DEM error.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuredDemComponent {
    /// Detector targets toggled by this component.
    pub detectors: Vec<u32>,
    /// Observable targets toggled by this component.
    pub observables: Vec<u32>,
}

/// One independent DEM error, optionally split into correlated components.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuredDemError {
    /// Independent firing probability.
    pub probability: f64,
    /// Components separated by `^` in Stim DEM syntax.
    pub components: Vec<StructuredDemComponent>,
}

impl StructuredDemError {
    fn instruction(&self) -> Instruction {
        let mut targets = Vec::new();
        for (index, component) in self.components.iter().enumerate() {
            if index > 0 {
                targets.push(Target::Separator);
            }
            targets.extend(
                component
                    .detectors
                    .iter()
                    .map(|&id| Target::Detector(u64::from(id))),
            );
            targets.extend(
                component
                    .observables
                    .iter()
                    .map(|&id| Target::Observable(u64::from(id))),
            );
        }
        Instruction {
            kind: Kind::Error,
            tag: None,
            args: vec![self.probability],
            targets,
            payload: None,
            empty_repeat: false,
        }
    }

    fn validate_components(&self) -> Result<(), DecoderError> {
        let instruction = self.instruction();
        // An empty component would render as a bare separator, which no reader accepts.
        if self
            .components
            .iter()
            .any(|component| component.detectors.is_empty() && component.observables.is_empty())
        {
            return Err(DecoderError::InvalidDemSyntax(format!(
                "empty component in {instruction}; every component needs at least one target"
            )));
        }
        validate_error_components(&instruction.targets, &instruction.to_string())
    }
}

/// A flattened DEM retaining component structure, detector coordinates, and sectors.
/// Preserves written targets.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuredDem {
    /// Independent error instructions.
    pub errors: Vec<StructuredDemError>,
    /// Detector coordinates indexed by detector ID.
    pub detector_coords: Vec<Option<Vec<f64>>>,
    /// Opaque fault-sector labels indexed by detector ID; length is `num_detectors`.
    /// `None` means unknown/unset. Labels must be non-empty and are compared only
    /// for equality. These labels are not represented in DEM text.
    pub detector_fault_sectors: Vec<Option<String>>,
    /// Number of detectors, including declared but unreferenced detectors.
    pub num_detectors: usize,
    /// Number of observables, including declared but unflipped observables.
    pub num_observables: usize,
}

/// Policy for an error whose detector support crosses a requested window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemBoundaryKind {
    /// Project outside detector targets away, producing implicit boundary edges.
    Soft,
    /// Reject any independent error with support on both sides of the boundary.
    Hard,
}

/// A checked structured DEM window and its detector-ID mapping.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuredDemWindow {
    /// Window-local detector ID to source-model detector ID.
    pub local_to_global: Vec<u32>,
    /// Window-local structured model.
    pub model: StructuredDem,
}

impl StructuredDem {
    /// Parse PECOS's strict flattened Stim-DEM subset without discarding components.
    ///
    /// Accepts flat Stim instructions with tags, inline comments, and ordered
    /// components. PECOS extension targets and statements are not supported.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] for malformed targets, probabilities, coordinates,
    /// or unexpanded `repeat`/`shift_detectors` instructions.
    pub fn from_dem_str(dem: &str) -> Result<Self, DecoderError> {
        let mut errors = Vec::new();
        let mut coordinates = std::collections::BTreeMap::new();
        let mut max_detector = None;
        let mut max_observable = None;

        for line in dem.lines() {
            let Some(instruction) = parse_line(line)? else {
                continue;
            };
            instruction.require_flat("StructuredDem")?;
            let (detectors, observables) = target_indices(&instruction.targets)?;
            for &id in &detectors {
                max_detector = Some(max_detector.map_or(id, |old: u32| old.max(id)));
            }
            for id in observables {
                max_observable = Some(max_observable.map_or(id, |old: u32| old.max(id)));
            }
            match instruction.kind {
                Kind::Error => {
                    let components = instruction
                        .components()
                        .map(|targets| {
                            let (detectors, observables) = target_indices(targets)?;
                            Ok(StructuredDemComponent {
                                detectors,
                                observables,
                            })
                        })
                        .collect::<Result<Vec<_>, DecoderError>>()?;
                    errors.push(StructuredDemError {
                        probability: instruction.args[0],
                        components,
                    });
                }
                Kind::Detector if !instruction.args.is_empty() => {
                    coordinates.insert(detectors[0] as usize, instruction.args);
                }
                _ => {}
            }
        }

        let num_detectors = dimension_count(max_detector, "detector")?;
        let num_observables = dimension_count(max_observable, "observable")?;
        let mut detector_coords = vec![None; num_detectors];
        for (detector, coords) in coordinates {
            detector_coords[detector] = Some(coords);
        }
        Ok(Self {
            errors,
            detector_coords,
            detector_fault_sectors: vec![None; num_detectors],
            num_detectors,
            num_observables,
        })
    }

    /// Validate and construct a structured model supplied by another PECOS layer.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if dimensions, targets, probabilities, or
    /// coordinates or sector label counts are inconsistent, or a sector label is empty.
    pub fn try_new(
        errors: Vec<StructuredDemError>,
        detector_coords: Vec<Option<Vec<f64>>>,
        detector_fault_sectors: Vec<Option<String>>,
        num_detectors: usize,
        num_observables: usize,
    ) -> Result<Self, DecoderError> {
        if detector_coords.len() != num_detectors {
            return Err(invalid(
                "detector coordinate count does not match detector count",
            ));
        }
        validate_fault_sectors(&detector_fault_sectors, num_detectors)?;
        if detector_coords
            .iter()
            .flatten()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(invalid("detector coordinates must be finite"));
        }
        for error in &errors {
            error.validate_components()?;
            if !error.probability.is_finite() || !(0.0..=1.0).contains(&error.probability) {
                return Err(invalid("DEM probability must be finite and in [0, 1]"));
            }
            for component in &error.components {
                if component
                    .detectors
                    .iter()
                    .any(|&id| id as usize >= num_detectors)
                {
                    return Err(invalid("DEM detector target is out of range"));
                }
                if component
                    .observables
                    .iter()
                    .any(|&id| id as usize >= num_observables)
                {
                    return Err(invalid("DEM observable target is out of range"));
                }
            }
        }
        Ok(Self {
            errors,
            detector_coords,
            detector_fault_sectors,
            num_detectors,
            num_observables,
        })
    }

    /// Replace the full detector label vector and revalidate this model.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if the model or labels fail [`Self::try_new`] validation.
    pub fn with_detector_fault_sectors(
        self,
        detector_fault_sectors: Vec<Option<String>>,
    ) -> Result<Self, DecoderError> {
        Self::try_new(
            self.errors,
            self.detector_coords,
            detector_fault_sectors,
            self.num_detectors,
            self.num_observables,
        )
    }

    /// Return each detector's time coordinate, defaulting missing time to zero.
    #[must_use]
    pub fn detector_times(&self) -> Vec<f64> {
        self.detector_coords
            .iter()
            .map(|coords| {
                coords
                    .as_ref()
                    .and_then(|values| values.get(2))
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect()
    }

    /// Reject models that cannot be represented by matching decoders' `u64` masks.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] when the model has more than 64 observables.
    pub fn ensure_observables_fit_u64(&self) -> Result<(), DecoderError> {
        if self.num_observables > 64 {
            return Err(invalid(format!(
                "this matching decoder packs observables into a u64 and supports at most 64 observables, but the DEM has {}; use a wide decoder",
                self.num_observables
            )));
        }
        Ok(())
    }

    /// Extract a checked time-coordinate window.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] for invalid bounds, inconsistent sector label
    /// counts, empty sector labels, or a crossing error at a hard boundary.
    pub fn window_by_time(
        &self,
        start: f64,
        end: f64,
        boundary: DemBoundaryKind,
    ) -> Result<StructuredDemWindow, DecoderError> {
        if !start.is_finite() || !end.is_finite() || start >= end {
            return Err(invalid("DEM window bounds must be finite and increasing"));
        }
        validate_fault_sectors(&self.detector_fault_sectors, self.num_detectors)?;
        let times = self.detector_times();
        let mut global_to_local = vec![None; self.num_detectors];
        let mut local_to_global = Vec::new();
        for (global, time) in times.into_iter().enumerate() {
            if time >= start && time < end {
                let local = u32::try_from(local_to_global.len())
                    .map_err(|_| invalid("DEM window has too many detectors"))?;
                global_to_local[global] = Some(local);
                local_to_global.push(
                    u32::try_from(global)
                        .map_err(|_| invalid("global detector ID does not fit u32"))?,
                );
            }
        }

        let mut errors = Vec::new();
        for error in &self.errors {
            error.validate_components()?;
            let has_inside = error
                .components
                .iter()
                .flat_map(|part| &part.detectors)
                .any(|&id| {
                    global_to_local
                        .get(id as usize)
                        .is_some_and(Option::is_some)
                });
            let has_outside = error
                .components
                .iter()
                .flat_map(|part| &part.detectors)
                .any(|&id| global_to_local.get(id as usize).is_none_or(Option::is_none));
            if boundary == DemBoundaryKind::Hard && has_inside && has_outside {
                return Err(invalid(
                    "independent DEM error crosses a hard window boundary",
                ));
            }
            if !has_inside {
                continue;
            }
            let components = xor_indices(error.components.iter().filter_map(|component| {
                let mut detectors = component
                    .detectors
                    .iter()
                    .filter_map(|&id| global_to_local.get(id as usize).copied().flatten())
                    .collect::<Vec<_>>();
                detectors.sort_unstable();
                let mut observables = component.observables.clone();
                observables.sort_unstable();
                (!detectors.is_empty()).then_some((detectors, observables))
            }))
            .into_iter()
            .map(|(detectors, observables)| StructuredDemComponent {
                detectors,
                observables,
            })
            .collect::<Vec<_>>();
            if components.is_empty() {
                continue;
            }
            errors.push(StructuredDemError {
                probability: error.probability,
                components,
            });
        }
        let detector_coords = local_to_global
            .iter()
            .map(|&global| self.detector_coords[global as usize].clone())
            .collect();
        let detector_fault_sectors = local_to_global
            .iter()
            .map(|&global| self.detector_fault_sectors[global as usize].clone())
            .collect();
        let model = Self::try_new(
            errors,
            detector_coords,
            detector_fault_sectors,
            local_to_global.len(),
            self.num_observables,
        )?;
        Ok(StructuredDemWindow {
            local_to_global,
            model,
        })
    }

    /// Render the structured model as a flattened Stim DEM.
    #[must_use]
    pub fn to_dem_string(&self) -> String {
        let mut out = String::new();
        for error in &self.errors {
            let _ = writeln!(out, "{}", error.instruction());
        }
        for (detector, coords) in self.detector_coords.iter().enumerate() {
            if let Some(coords) = coords {
                out.push_str("detector(");
                for (index, coord) in coords.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(out, "{coord}");
                }
                let _ = writeln!(out, ") D{detector}");
            } else {
                let _ = writeln!(out, "detector D{detector}");
            }
        }
        for observable in 0..self.num_observables {
            let _ = writeln!(out, "logical_observable L{observable}");
        }
        out
    }
}

/// Check a dense detector label vector: one entry per detector, labels non-empty.
///
/// # Errors
///
/// Returns [`DecoderError`] if the length differs from `num_detectors` or a label is empty.
pub fn validate_fault_sectors(
    sectors: &[Option<String>],
    num_detectors: usize,
) -> Result<(), DecoderError> {
    if sectors.len() != num_detectors {
        return Err(invalid(format!(
            "detector sector declaration count {} does not match detector count {num_detectors}",
            sectors.len()
        )));
    }
    for (detector, sector) in sectors.iter().enumerate() {
        if sector.as_ref().is_some_and(String::is_empty) {
            return Err(invalid(format!(
                "detector D{detector} fault_sector is empty"
            )));
        }
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> DecoderError {
    DecoderError::InvalidConfiguration(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEM: &str = "\
error(0.1) D0 D2 L0 ^ D1 D3 L1\n\
error(0.2) D1 D2 D3 L1\n\
detector(4, 0, 0) D0\n\
detector(5, 0, 1) D1\n\
detector(6, 0, 2) D2\n\
detector(7, 0, 3) D3\n\
logical_observable L2\n";

    // contract: text has no sectors, including sparse and declaration-only detector IDs.
    #[test]
    fn text_parsing_leaves_all_sectors_undeclared() {
        for (text, count) in [
            ("", 0),
            ("error(0.1) D3", 4),
            ("detector D5", 6),
            ("error(0.1) D3\ndetector(1,2,3) D7", 8),
        ] {
            let model = StructuredDem::from_dem_str(text).unwrap();
            assert_eq!(model.num_detectors, count);
            assert_eq!(model.detector_fault_sectors, vec![None; count]);
        }
    }

    // contract: construction and both window boundaries reject short and oversized sectors.
    #[test]
    fn sector_counts_are_checked_at_construction_and_window_boundaries() {
        for count in [1, 3] {
            let mut model = StructuredDem::from_dem_str(
                "error(0.1) D0\nerror(0.2) D1\ndetector(0,0,0) D0\ndetector(0,0,1) D1",
            )
            .unwrap();
            model.detector_fault_sectors.resize(count, Some("X".into()));
            let expected = format!(
                "detector sector declaration count {count} does not match detector count 2"
            );
            for error in [
                StructuredDem::try_new(
                    model.errors.clone(),
                    model.detector_coords.clone(),
                    model.detector_fault_sectors.clone(),
                    2,
                    0,
                )
                .unwrap_err(),
                model
                    .window_by_time(0.0, 2.0, DemBoundaryKind::Soft)
                    .unwrap_err(),
                model.commit_window(0..2, 0..1).unwrap_err(),
            ] {
                let DecoderError::InvalidConfiguration(message) = error else {
                    panic!("wrong error: {error}");
                };
                assert_eq!(message, expected);
            }
        }
    }

    // contract: construction and both windows reject empty labels, even outside the window.
    #[test]
    fn empty_sectors_are_rejected_at_construction_and_window_boundaries() {
        let mut model =
            StructuredDem::from_dem_str("error(0.1) D0\ndetector(0,0,0) D0\ndetector(0,0,1) D1")
                .unwrap();
        model.detector_fault_sectors[1] = Some(String::new());
        for error in [
            StructuredDem::try_new(
                model.errors.clone(),
                model.detector_coords.clone(),
                model.detector_fault_sectors.clone(),
                2,
                0,
            )
            .unwrap_err(),
            model
                .window_by_time(0.0, 1.0, DemBoundaryKind::Soft)
                .unwrap_err(),
            model.commit_window(0..1, 0..1).unwrap_err(),
        ] {
            let DecoderError::InvalidConfiguration(message) = error else {
                panic!("wrong error: {error}");
            };
            assert_eq!(message, "detector D1 fault_sector is empty");
        }
    }

    // contract: both windows remap every label through their local-to-global IDs.
    #[test]
    fn windows_remap_detector_fault_sectors() {
        let mut model = StructuredDem::from_dem_str(
            "error(0.1) D0\nerror(0.1) D1\nerror(0.1) D2\nerror(0.1) D3\n\
             detector(0,0,1) D0\ndetector(0,0,0) D1\ndetector(0,0,1) D2\ndetector(0,0,1) D3",
        )
        .unwrap();
        model.detector_fault_sectors = vec![
            Some("XX".into()),
            Some("YY".into()),
            Some("ZZ".into()),
            None,
        ];
        let window = model
            .window_by_time(1.0, 2.0, DemBoundaryKind::Soft)
            .unwrap();
        let commit = model.commit_window(1..2, 1..2).unwrap();
        for (local, mapping) in [
            (&window.model, &window.local_to_global),
            (&commit.model, &commit.local_to_global_detector),
        ] {
            assert_eq!(mapping, &[0, 2, 3]);
            assert_eq!(
                local.detector_fault_sectors,
                mapping
                    .iter()
                    .map(|&id| model.detector_fault_sectors[id as usize].clone())
                    .collect::<Vec<_>>()
            );
            assert_eq!(local.detector_fault_sectors.len(), local.num_detectors);
        }
    }

    #[test]
    fn soft_window_preserves_components_hyperedges_outputs_and_coordinates() {
        let model = StructuredDem::from_dem_str(DEM).unwrap();
        let window = model
            .window_by_time(1.0, 4.0, DemBoundaryKind::Soft)
            .unwrap();

        assert_eq!(window.local_to_global, [1, 2, 3]);
        assert_eq!(window.model.num_observables, 3);
        assert_eq!(window.model.errors[0].components.len(), 2);
        assert_eq!(window.model.errors[1].components[0].detectors, [0, 1, 2]);
        assert_eq!(window.model.detector_coords[0], Some(vec![5.0, 0.0, 1.0]));
    }

    #[test]
    fn soft_window_cancels_identical_projected_components_as_sets() {
        for (targets, expected) in [
            ("D0 D1 D2 L0 ^ D1 D0 D3 L0", vec![]),
            (
                "D0 D1 D2 L0 ^ D1 D0 D3 L0 ^ D0 D1 L0",
                vec![(vec![0, 1], vec![0])],
            ),
            (
                "D0 D1 D2 L0 ^ D1 D0 D3 L1",
                vec![(vec![0, 1], vec![0]), (vec![0, 1], vec![1])],
            ),
        ] {
            let model = StructuredDem::from_dem_str(&format!(
                "error(0.1) {targets}\ndetector(0,0,0) D0\ndetector(0,0,0) D1\ndetector(0,0,1) D2\ndetector(0,0,1) D3"
            )).unwrap();
            let window = model
                .window_by_time(0.0, 1.0, DemBoundaryKind::Soft)
                .unwrap();
            let components: Vec<_> = window
                .model
                .errors
                .iter()
                .flat_map(|error| &error.components)
                .map(|component| (component.detectors.clone(), component.observables.clone()))
                .collect();
            assert_eq!(components, expected, "{targets}");
            assert_eq!(window.model.errors.len(), usize::from(!expected.is_empty()));
            assert_eq!(
                StructuredDem::from_dem_str(&window.model.to_dem_string()).unwrap(),
                window.model
            );
        }
    }

    #[test]
    fn structural_construction_and_windows_reject_duplicate_targets_and_components() {
        for components in [
            vec![StructuredDemComponent {
                detectors: vec![0, 0],
                observables: vec![],
            }],
            vec![StructuredDemComponent {
                detectors: vec![0],
                observables: vec![0, 0],
            }],
            vec![
                StructuredDemComponent {
                    detectors: vec![0, 1],
                    observables: vec![0],
                },
                StructuredDemComponent {
                    detectors: vec![1, 0],
                    observables: vec![0],
                },
            ],
        ] {
            let error = StructuredDemError {
                probability: 0.1,
                components,
            };
            let expected = parse_line(&error.instruction().to_string())
                .unwrap_err()
                .to_string();
            let coords = vec![Some(vec![0.0, 0.0, 0.0]); 2];
            let rejected = StructuredDem::try_new(
                vec![error.clone()],
                coords.clone(),
                vec![Some("X".into()); 2],
                2,
                1,
            )
            .unwrap_err();
            assert!(matches!(rejected, DecoderError::InvalidDemSyntax(_)));
            assert_eq!(rejected.to_string(), expected);
            let model = StructuredDem {
                errors: vec![error],
                detector_coords: coords,
                detector_fault_sectors: vec![Some("X".into()); 2],
                num_detectors: 2,
                num_observables: 1,
            };
            assert_eq!(model.commit_columns().unwrap_err().to_string(), expected);
            assert_eq!(
                model
                    .window_by_time(0.0, 1.0, DemBoundaryKind::Soft)
                    .unwrap_err()
                    .to_string(),
                expected
            );
        }
    }

    #[test]
    fn hard_window_rejects_crossing_independent_error() {
        let model = StructuredDem::from_dem_str(DEM).unwrap();
        let error = model
            .window_by_time(1.0, 3.0, DemBoundaryKind::Hard)
            .unwrap_err();
        assert!(error.to_string().contains("crosses a hard window boundary"));
    }

    #[test]
    fn look_behind_and_ahead_are_selected_by_explicit_bounds() {
        let model = StructuredDem::from_dem_str(DEM).unwrap();
        let window = model
            .window_by_time(0.0, 3.0, DemBoundaryKind::Soft)
            .unwrap();
        assert_eq!(window.local_to_global, [0, 1, 2]);
    }

    #[test]
    fn render_round_trip_retains_structural_dimensions() {
        let model = StructuredDem::from_dem_str(DEM).unwrap();
        let reparsed = StructuredDem::from_dem_str(&model.to_dem_string()).unwrap();
        assert_eq!(reparsed, model);
    }
}
