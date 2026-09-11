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

use crate::errors::DecoderError;
use std::fmt::Write as _;

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

/// A flattened DEM retaining component structure and detector coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuredDem {
    /// Independent error instructions.
    pub errors: Vec<StructuredDemError>,
    /// Detector coordinates indexed by detector ID.
    pub detector_coords: Vec<Option<Vec<f64>>>,
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
    /// Accepted instructions are `error(p)` with `D<n>` / `L<n>` targets and
    /// optional `^` components, `detector D<n>...` or `detector(coords) D<n>...`,
    /// `logical_observable L<n>...`, blank lines, and whole-line `#` comments.
    /// Probabilities must be finite and in `[0, 1]`; coordinates must be finite.
    /// Inline comments, `TP<n>`, unknown instructions, `repeat`, and
    /// `shift_detectors` are rejected instead of being silently ignored.
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

        for line in dem.lines().map(str::trim) {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line.starts_with("repeat") || line.starts_with("shift_detectors") {
                return Err(invalid(
                    "StructuredDem requires a flattened DEM: `repeat` / `shift_detectors` are not supported",
                ));
            }
            if let Some(rest) = line.strip_prefix("error(") {
                let close = rest
                    .find(')')
                    .ok_or_else(|| invalid("missing ) in error line"))?;
                let probability = rest[..close]
                    .parse::<f64>()
                    .map_err(|_| invalid(format!("invalid probability: {}", &rest[..close])))?;
                if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                    return Err(invalid(format!("invalid probability: {}", &rest[..close])));
                }
                let mut components = Vec::new();
                for component in rest[close + 1..].split('^') {
                    let mut detectors = Vec::new();
                    let mut observables = Vec::new();
                    for token in component.split_whitespace() {
                        if let Some(value) = token.strip_prefix('D') {
                            let detector = parse_target(value, "detector", token)?;
                            max_detector =
                                Some(max_detector.map_or(detector, |old: u32| old.max(detector)));
                            detectors.push(detector);
                        } else if let Some(value) = token.strip_prefix('L') {
                            let observable = parse_target(value, "observable", token)?;
                            max_observable = Some(
                                max_observable.map_or(observable, |old: u32| old.max(observable)),
                            );
                            observables.push(observable);
                        } else {
                            return Err(invalid(format!("invalid DEM target: {token}")));
                        }
                    }
                    components.push(StructuredDemComponent {
                        detectors,
                        observables,
                    });
                }
                errors.push(StructuredDemError {
                    probability,
                    components,
                });
                continue;
            }
            if let Some(rest) = line.strip_prefix("detector") {
                let (coords, targets) = if let Some(after) = rest.strip_prefix('(') {
                    let close = after
                        .find(')')
                        .ok_or_else(|| invalid("missing ) in detector declaration"))?;
                    let coords = after[..close]
                        .split(',')
                        .map(|value| {
                            value.trim().parse::<f64>().map_err(|_| {
                                invalid(format!("invalid detector coordinate: {value}"))
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    if coords.iter().any(|value| !value.is_finite()) {
                        return Err(invalid("detector coordinates must be finite"));
                    }
                    (Some(coords), &after[close + 1..])
                } else {
                    (None, rest)
                };
                for token in targets.split_whitespace() {
                    let value = token.strip_prefix('D').ok_or_else(|| {
                        invalid(format!("invalid detector declaration target: {token}"))
                    })?;
                    let detector = parse_target(value, "detector", token)?;
                    max_detector =
                        Some(max_detector.map_or(detector, |old: u32| old.max(detector)));
                    if let Some(coords) = &coords {
                        coordinates.insert(detector as usize, coords.clone());
                    }
                }
                continue;
            }
            if let Some(rest) = line.strip_prefix("logical_observable") {
                for token in rest.split_whitespace() {
                    let value = token.strip_prefix('L').ok_or_else(|| {
                        invalid(format!("invalid observable declaration target: {token}"))
                    })?;
                    let observable = parse_target(value, "observable", token)?;
                    max_observable =
                        Some(max_observable.map_or(observable, |old: u32| old.max(observable)));
                }
                continue;
            }
            return Err(invalid(format!("unsupported DEM instruction: {line}")));
        }

        let num_detectors = dimension(max_detector, "detector")?;
        let num_observables = dimension(max_observable, "observable")?;
        let mut detector_coords = vec![None; num_detectors];
        for (detector, coords) in coordinates {
            detector_coords[detector] = Some(coords);
        }
        Ok(Self {
            errors,
            detector_coords,
            num_detectors,
            num_observables,
        })
    }

    /// Validate and construct a structured model supplied by another PECOS layer.
    ///
    /// # Errors
    ///
    /// Returns [`DecoderError`] if dimensions, targets, probabilities, or
    /// coordinates are inconsistent.
    pub fn try_new(
        errors: Vec<StructuredDemError>,
        detector_coords: Vec<Option<Vec<f64>>>,
        num_detectors: usize,
        num_observables: usize,
    ) -> Result<Self, DecoderError> {
        if detector_coords.len() != num_detectors {
            return Err(invalid(
                "detector coordinate count does not match detector count",
            ));
        }
        if detector_coords
            .iter()
            .flatten()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(invalid("detector coordinates must be finite"));
        }
        for error in &errors {
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
            num_detectors,
            num_observables,
        })
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
    /// Returns [`DecoderError`] for invalid bounds or a crossing error at a hard boundary.
    pub fn window_by_time(
        &self,
        start: f64,
        end: f64,
        boundary: DemBoundaryKind,
    ) -> Result<StructuredDemWindow, DecoderError> {
        if !start.is_finite() || !end.is_finite() || start >= end {
            return Err(invalid("DEM window bounds must be finite and increasing"));
        }
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
            let components = error
                .components
                .iter()
                .filter_map(|component| {
                    let detectors = component
                        .detectors
                        .iter()
                        .filter_map(|&id| global_to_local.get(id as usize).copied().flatten())
                        .collect::<Vec<_>>();
                    (!detectors.is_empty()).then(|| StructuredDemComponent {
                        detectors,
                        observables: component.observables.clone(),
                    })
                })
                .collect();
            errors.push(StructuredDemError {
                probability: error.probability,
                components,
            });
        }
        let detector_coords = local_to_global
            .iter()
            .map(|&global| self.detector_coords[global as usize].clone())
            .collect();
        let model = Self::try_new(
            errors,
            detector_coords,
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
            let _ = write!(out, "error({})", error.probability);
            for (index, component) in error.components.iter().enumerate() {
                if index > 0 {
                    out.push_str(" ^");
                }
                for detector in &component.detectors {
                    let _ = write!(out, " D{detector}");
                }
                for observable in &component.observables {
                    let _ = write!(out, " L{observable}");
                }
            }
            out.push('\n');
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

fn parse_target(value: &str, kind: &str, token: &str) -> Result<u32, DecoderError> {
    value
        .parse()
        .map_err(|_| invalid(format!("invalid {kind}: {token}")))
}

fn dimension(maximum: Option<u32>, kind: &str) -> Result<usize, DecoderError> {
    maximum.map_or(Ok(0), |id| {
        usize::try_from(u64::from(id) + 1)
            .map_err(|_| invalid(format!("{kind} count does not fit usize")))
    })
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
