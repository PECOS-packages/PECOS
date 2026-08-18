use super::DecoderSpec;
use super::config::{
    BeamSearchConfig, BeliefMatchingConfig, BeliefMatchingMode, BpLsdConfig, BpOsdConfig,
    EnsembleConfig, FusionBlossomConfig, FusionBlossomSolverType, KMwpmConfig, MinSumBpConfig,
    MwpfConfig, MwpfSolverType, PecosUfPreset, PerturbedConfig, PerturbedFusionBlossomConfig,
    PyMatchingConfig, RelayBpConfig, TesseractConfig, TesseractPreset, WindowedConfig,
    WindowedMode,
};
use pecos_decoder_core::DecoderError;
use std::fmt::Display;
use std::str::FromStr;

pub(super) fn parse(type_string: &str) -> Result<DecoderSpec, DecoderError> {
    match type_string {
        "pymatching" | "pymatching_correlated" => Ok(correlated_pymatching()),
        "pymatching_uncorrelated" => Ok(DecoderSpec::PyMatching(PyMatchingConfig::default())),
        "tesseract" => Ok(DecoderSpec::Tesseract(TesseractConfig {
            preset: TesseractPreset::Fast,
            ..TesseractConfig::default()
        })),
        "k_mwpm" => Ok(DecoderSpec::KMwpm(KMwpmConfig::default())),
        "astar" => Ok(DecoderSpec::AStar),
        "astar_full" => Ok(DecoderSpec::AStarFull),
        "fusion_blossom" => Ok(DecoderSpec::FusionBlossom(FusionBlossomConfig::default())),
        "fusion_blossom_serial" => Ok(fusion_blossom(FusionBlossomSolverType::Serial, false)),
        "fusion_blossom_parallel" => Ok(fusion_blossom(FusionBlossomSolverType::Parallel, false)),
        "fusion_blossom_correlated" => Ok(fusion_blossom(FusionBlossomSolverType::Serial, true)),
        "perturbed_fb_corr" => Ok(DecoderSpec::PerturbedFusionBlossomCorrelated(
            PerturbedFusionBlossomConfig::default(),
        )),
        "bp_osd" => Ok(DecoderSpec::BpOsd(BpOsdConfig::default())),
        "bp_lsd" => Ok(DecoderSpec::BpLsd(BpLsdConfig::default())),
        "belief_find" => Ok(DecoderSpec::BeliefFind),
        "union_find" => Ok(DecoderSpec::UnionFind),
        "relay_bp" => Ok(DecoderSpec::RelayBp(RelayBpConfig::default())),
        "min_sum_bp" => Ok(DecoderSpec::MinSumBp(MinSumBpConfig::default())),
        "pecos_uf" | "pecos_uf:fast" => Ok(DecoderSpec::PecosUf(PecosUfPreset::Fast)),
        "pecos_uf:balanced" | "pecos_uf_correlated" => {
            Ok(DecoderSpec::PecosUf(PecosUfPreset::Balanced))
        }
        "pecos_uf:accurate" => Ok(DecoderSpec::PecosUf(PecosUfPreset::Accurate)),
        "pecos_uf:bp" => Ok(DecoderSpec::PecosUf(PecosUfPreset::Bp)),
        "pecos_uf:bp_serial" => Ok(DecoderSpec::PecosUf(PecosUfPreset::BpSerial)),
        "belief_matching" => Ok(belief_matching(BeliefMatchingMode::Standard)),
        "belief_matching_correlated" => Ok(belief_matching(BeliefMatchingMode::Correlated)),
        "belief_matching_mgbp" => Ok(belief_matching(BeliefMatchingMode::MatchingGraphBp)),
        "windowed" => Ok(DecoderSpec::Windowed(WindowedConfig::default())),
        "mwpf" => Ok(DecoderSpec::Mwpf(MwpfConfig::default())),
        "perturbed" => Ok(DecoderSpec::Perturbed(PerturbedConfig::default())),
        "beamsearch" => Ok(DecoderSpec::BeamSearch(BeamSearchConfig::default())),
        "logical_subgraph" => Err(logical_subgraph_error()),
        value if value.starts_with("logical_subgraph:") => Err(logical_subgraph_error()),
        value if value.starts_with("k_mwpm:") => parse_k_mwpm(&value["k_mwpm:".len()..]),
        value if value.starts_with("perturbed_fb_corr:") => {
            parse_perturbed_fb(&value["perturbed_fb_corr:".len()..])
        }
        value if value.starts_with("belief_matching_hybrid:") => {
            let full_dem = &value["belief_matching_hybrid:".len()..];
            if full_dem.is_empty() {
                return invalid("belief_matching_hybrid requires a non-empty full DEM");
            }
            Ok(DecoderSpec::BeliefMatching(BeliefMatchingConfig {
                mode: BeliefMatchingMode::Hybrid,
                embedded_full_dem: Some(full_dem.to_string()),
            }))
        }
        value if value.starts_with("windowed:") => parse_windowed(&value["windowed:".len()..]),
        value if value.starts_with("mwpf:") => parse_mwpf(&value["mwpf:".len()..]),
        value if value.starts_with("perturbed:") => parse_perturbed(&value["perturbed:".len()..]),
        value if value.starts_with("beamsearch:") => {
            parse_beamsearch(&value["beamsearch:".len()..])
        }
        value if value.starts_with("ensemble:") => parse_ensemble(&value["ensemble:".len()..]),
        _ => invalid(format!(
            "Unsupported decoder_type: {type_string}. \
             Supported: pymatching, tesseract, mwpf, pecos_uf (or \
             pecos_uf:fast/balanced/accurate), logical_subgraph, ensemble:d1,d2,..., \
             bp_osd, bp_lsd, union_find, relay_bp, min_sum_bp."
        )),
    }
}

fn correlated_pymatching() -> DecoderSpec {
    DecoderSpec::PyMatching(PyMatchingConfig {
        correlated: true,
        error_probability: None,
    })
}

fn fusion_blossom(solver_type: FusionBlossomSolverType, correlated: bool) -> DecoderSpec {
    DecoderSpec::FusionBlossom(FusionBlossomConfig {
        correlated,
        solver_type,
    })
}

fn belief_matching(mode: BeliefMatchingMode) -> DecoderSpec {
    DecoderSpec::BeliefMatching(BeliefMatchingConfig {
        mode,
        embedded_full_dem: None,
    })
}

fn parse_k_mwpm(params: &str) -> Result<DecoderSpec, DecoderError> {
    let mut config = KMwpmConfig::default();
    for (key, value) in params_iter("k_mwpm", params)? {
        match key {
            "K" | "k" => config.k = parse_number("k_mwpm", key, value)?,
            _ => return unknown_key("k_mwpm", key),
        }
    }
    Ok(DecoderSpec::KMwpm(config))
}

fn parse_perturbed_fb(params: &str) -> Result<DecoderSpec, DecoderError> {
    let mut config = PerturbedFusionBlossomConfig::default();
    for (key, value) in params_iter("perturbed_fb_corr", params)? {
        match key {
            "K" | "k" => config.k = parse_number("perturbed_fb_corr", key, value)?,
            "sigma" | "s" => config.sigma = parse_finite("perturbed_fb_corr", key, value)?,
            "seed" => config.seed = parse_number("perturbed_fb_corr", key, value)?,
            _ => return unknown_key("perturbed_fb_corr", key),
        }
    }
    Ok(DecoderSpec::PerturbedFusionBlossomCorrelated(config))
}

fn parse_windowed(params: &str) -> Result<DecoderSpec, DecoderError> {
    let (own_params, inner) = split_inner("windowed", params)?;
    let mut config = WindowedConfig::default();
    if let Some(inner) = inner {
        let inner_spec = parse(inner)?;
        config.inner = Box::new(inner_spec.clone());
        config.sandwich_phase2 = if inner == "pecos_uf" {
            Box::new(correlated_pymatching())
        } else {
            Box::new(inner_spec)
        };
    }
    for (key, value) in params_iter("windowed", own_params)? {
        match key {
            "step" => config.step_size = parse_number("windowed", key, value)?,
            "buf" | "buffer" => config.buffer_size = parse_number("windowed", key, value)?,
            "mode" => {
                config.mode = match value {
                    "sandwich" => WindowedMode::Sandwich,
                    "overlap" => WindowedMode::Overlap,
                    "nonoverlap" | "non_overlapping" => WindowedMode::NonOverlapping,
                    _ => return invalid(format!("windowed has unknown mode '{value}'")),
                };
            }
            "seam" => config.seam_half_width = parse_number("windowed", key, value)?,
            "ext" | "core_extend" => {
                config.core_extend = parse_number("windowed", key, value)?;
            }
            "wmax" | "commit_weight_max" => {
                config.commit_weight_max = parse_finite("windowed", key, value)?;
            }
            _ => return unknown_key("windowed", key),
        }
    }
    Ok(DecoderSpec::Windowed(config))
}

fn parse_mwpf(params: &str) -> Result<DecoderSpec, DecoderError> {
    let mut config = MwpfConfig::default();
    for (key, value) in params_iter("mwpf", params)? {
        match key {
            "c" | "cluster_node_limit" => {
                config.cluster_node_limit = parse_number("mwpf", key, value)?;
            }
            "t" | "timeout" => config.timeout = Some(parse_finite("mwpf", key, value)?),
            "once" | "only_solve_primal_once" => {
                config.only_solve_primal_once = parse_bool("mwpf", key, value)?;
            }
            "solver" => {
                config.solver_type = match value {
                    "uf" | "union_find" => MwpfSolverType::UnionFind,
                    "sh" | "single_hair" => MwpfSolverType::SingleHair,
                    "bp" | "bp_hybrid" => MwpfSolverType::BpHybrid,
                    "joint" | "joint_single_hair" => MwpfSolverType::JointSingleHair,
                    _ => return invalid(format!("mwpf has unknown solver '{value}'")),
                };
            }
            _ => return unknown_key("mwpf", key),
        }
    }
    Ok(DecoderSpec::Mwpf(config))
}

fn parse_perturbed(params: &str) -> Result<DecoderSpec, DecoderError> {
    let (own_params, inner) = split_inner("perturbed", params)?;
    let mut config = PerturbedConfig::default();
    if let Some(inner) = inner {
        config.inner = Box::new(parse(inner)?);
    }
    for (key, value) in params_iter("perturbed", own_params)? {
        match key {
            "K" | "k" => config.k = parse_number("perturbed", key, value)?,
            "sigma" | "s" => config.sigma = parse_finite("perturbed", key, value)?,
            "seed" => config.seed = parse_number("perturbed", key, value)?,
            _ => return unknown_key("perturbed", key),
        }
    }
    Ok(DecoderSpec::Perturbed(config))
}

fn parse_beamsearch(params: &str) -> Result<DecoderSpec, DecoderError> {
    let mut config = BeamSearchConfig::default();
    for (key, value) in params_iter("beamsearch", params)? {
        match key {
            "K" | "k" => config.beam_width = parse_number("beamsearch", key, value)?,
            "sigma" | "s" => {
                config.perturbation_sigma = parse_finite("beamsearch", key, value)?;
            }
            "seed" => config.seed = parse_number("beamsearch", key, value)?,
            "step" => config.step_size = parse_number("beamsearch", key, value)?,
            "buf" | "buffer" => {
                config.buffer_size = parse_number("beamsearch", key, value)?;
            }
            "wmax" => config.commit_weight_max = parse_finite("beamsearch", key, value)?,
            _ => return unknown_key("beamsearch", key),
        }
    }
    Ok(DecoderSpec::BeamSearch(config))
}

fn parse_ensemble(members: &str) -> Result<DecoderSpec, DecoderError> {
    if members.is_empty() {
        return invalid("ensemble needs at least one decoder");
    }
    let members = members
        .split(',')
        .map(str::trim)
        .map(|member| {
            if member.is_empty() {
                invalid("ensemble contains an empty member")
            } else {
                parse(member)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(DecoderSpec::Ensemble(EnsembleConfig { members }))
}

fn split_inner<'a>(
    family: &str,
    params: &'a str,
) -> Result<(&'a str, Option<&'a str>), DecoderError> {
    if let Some(inner) = params.strip_prefix("inner=") {
        if inner.is_empty() {
            return invalid(format!("{family} has an empty inner decoder"));
        }
        return Ok(("", Some(inner)));
    }
    if let Some(index) = params.find(",inner=") {
        let inner = &params[index + ",inner=".len()..];
        if inner.is_empty() {
            return invalid(format!("{family} has an empty inner decoder"));
        }
        return Ok((&params[..index], Some(inner)));
    }
    Ok((params, None))
}

fn params_iter<'a>(
    family: &'a str,
    params: &'a str,
) -> Result<Vec<(&'a str, &'a str)>, DecoderError> {
    if params.is_empty() {
        return Ok(Vec::new());
    }
    params
        .split(',')
        .map(|parameter| {
            let (key, value) = parameter.split_once('=').ok_or_else(|| {
                DecoderError::InvalidConfiguration(format!(
                    "{family} parameter '{parameter}' must have key=value form"
                ))
            })?;
            if key.is_empty() || value.is_empty() {
                return invalid(format!(
                    "{family} parameter '{parameter}' must have a non-empty key and value"
                ));
            }
            Ok((key, value))
        })
        .collect()
}

fn parse_number<T>(family: &str, key: &str, value: &str) -> Result<T, DecoderError>
where
    T: FromStr,
    T::Err: Display,
{
    value.parse().map_err(|error| {
        DecoderError::InvalidConfiguration(format!(
            "{family} parameter '{key}' has invalid value '{value}': {error}"
        ))
    })
}

fn parse_finite(family: &str, key: &str, value: &str) -> Result<f64, DecoderError> {
    let parsed = parse_number::<f64>(family, key, value)?;
    if !parsed.is_finite() {
        return invalid(format!(
            "{family} parameter '{key}' must be finite, got '{value}'"
        ));
    }
    Ok(parsed)
}

fn parse_bool(family: &str, key: &str, value: &str) -> Result<bool, DecoderError> {
    match value {
        "true" | "1" => Ok(true),
        "false" | "0" => Ok(false),
        _ => invalid(format!(
            "{family} parameter '{key}' must be true, false, 1, or 0; got '{value}'"
        )),
    }
}

fn unknown_key<T>(family: &str, key: &str) -> Result<T, DecoderError> {
    invalid(format!("{family} has unknown parameter '{key}'"))
}

fn logical_subgraph_error() -> DecoderError {
    DecoderError::InvalidConfiguration(
        "logical_subgraph decoder requires stab_coords. Use pecos_rslib.qec.\
         LogicalSubgraphDecoder class directly."
            .to_string(),
    )
}

fn invalid<T>(message: impl Into<String>) -> Result<T, DecoderError> {
    Err(DecoderError::InvalidConfiguration(message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{BpSchedule, DecodeModel, ExecutionTraits};

    #[test]
    fn parses_legacy_parameterized_forms() {
        assert_eq!(
            parse("k_mwpm:K=10").unwrap(),
            DecoderSpec::KMwpm(KMwpmConfig { k: 10 })
        );

        let perturbed = parse("perturbed:K=15,sigma=0.7,seed=42,inner=pymatching").unwrap();
        let DecoderSpec::Perturbed(config) = perturbed else {
            panic!("expected perturbed spec");
        };
        assert_eq!(config.k, 15);
        assert!((config.sigma - 0.7).abs() < f64::EPSILON);
        assert_eq!(config.seed, 42);
        assert_eq!(*config.inner, correlated_pymatching());

        let windowed = parse("windowed:step=5,buf=5,mode=sandwich,inner=bp_osd").unwrap();
        let DecoderSpec::Windowed(config) = windowed else {
            panic!("expected windowed spec");
        };
        assert_eq!(config.step_size, 5);
        assert_eq!(config.buffer_size, 5);
        assert_eq!(config.mode, WindowedMode::Sandwich);
        assert!(matches!(*config.inner, DecoderSpec::BpOsd(_)));
        assert!(matches!(*config.sandwich_phase2, DecoderSpec::BpOsd(_)));

        let explicit_fast = parse("windowed:step=5,buf=5,inner=pecos_uf:fast").unwrap();
        let DecoderSpec::Windowed(config) = explicit_fast else {
            panic!("expected windowed spec");
        };
        assert!(matches!(
            *config.sandwich_phase2,
            DecoderSpec::PecosUf(PecosUfPreset::Fast)
        ));

        let exact_pecos_uf = parse("windowed:step=5,buf=5,inner=pecos_uf").unwrap();
        let DecoderSpec::Windowed(config) = exact_pecos_uf else {
            panic!("expected windowed spec");
        };
        assert!(matches!(
            *config.sandwich_phase2,
            DecoderSpec::PyMatching(PyMatchingConfig {
                correlated: true,
                ..
            })
        ));

        assert_eq!(
            parse("perturbed_fb_corr:K=9,sigma=0.25,seed=7").unwrap(),
            DecoderSpec::PerturbedFusionBlossomCorrelated(PerturbedFusionBlossomConfig {
                k: 9,
                sigma: 0.25,
                seed: 7,
            })
        );

        let ensemble = parse("ensemble:pymatching,bp_osd,relay_bp").unwrap();
        let DecoderSpec::Ensemble(config) = ensemble else {
            panic!("expected ensemble spec");
        };
        assert_eq!(config.members.len(), 3);

        let mwpf = parse("mwpf:c=12,t=0.5,once=1,solver=uf").unwrap();
        assert_eq!(
            mwpf,
            DecoderSpec::Mwpf(MwpfConfig {
                solver_type: MwpfSolverType::UnionFind,
                cluster_node_limit: 12,
                timeout: Some(0.5),
                only_solve_primal_once: true,
            })
        );

        assert_eq!(
            parse("pecos_uf:accurate").unwrap(),
            DecoderSpec::PecosUf(PecosUfPreset::Accurate)
        );
        assert_eq!(
            parse("beamsearch:K=7,sigma=0.4,seed=9,step=3,buf=4,wmax=2.5").unwrap(),
            DecoderSpec::BeamSearch(BeamSearchConfig {
                beam_width: 7,
                perturbation_sigma: 0.4,
                seed: 9,
                step_size: 3,
                buffer_size: 4,
                commit_weight_max: 2.5,
                ..BeamSearchConfig::default()
            })
        );
    }

    #[test]
    fn parses_every_parameterless_legacy_form() {
        let forms = [
            "pymatching",
            "pymatching_correlated",
            "pymatching_uncorrelated",
            "tesseract",
            "k_mwpm",
            "astar",
            "astar_full",
            "fusion_blossom",
            "fusion_blossom_serial",
            "fusion_blossom_parallel",
            "fusion_blossom_correlated",
            "perturbed_fb_corr",
            "bp_osd",
            "bp_lsd",
            "belief_find",
            "union_find",
            "relay_bp",
            "min_sum_bp",
            "pecos_uf",
            "pecos_uf:fast",
            "pecos_uf:balanced",
            "pecos_uf:accurate",
            "pecos_uf:bp",
            "pecos_uf:bp_serial",
            "pecos_uf_correlated",
            "belief_matching",
            "belief_matching_correlated",
            "belief_matching_mgbp",
            "windowed",
            "mwpf",
            "perturbed",
            "beamsearch",
        ];
        for form in forms {
            assert!(parse(form).is_ok(), "failed to parse {form}");
        }
    }

    #[test]
    fn empty_parameter_suffix_uses_family_defaults() {
        for (with_suffix, bare) in [
            ("k_mwpm:", "k_mwpm"),
            ("perturbed_fb_corr:", "perturbed_fb_corr"),
            ("windowed:", "windowed"),
            ("mwpf:", "mwpf"),
            ("perturbed:", "perturbed"),
            ("beamsearch:", "beamsearch"),
        ] {
            assert_eq!(parse(with_suffix).unwrap(), parse(bare).unwrap());
        }
    }

    #[test]
    fn strict_parse_rejects_malformed_inputs() {
        for malformed in [
            "k_mwpm:K=oops",
            "perturbed:sigma=nope",
            "windowed:unknown=1",
            "mwpf:solver=unknown",
            "beamsearch:K",
            "ensemble:",
            "ensemble:pymatching,,bp_osd",
            "not_a_decoder",
            "k_mwpm_suffix",
        ] {
            assert!(parse(malformed).is_err(), "accepted {malformed}");
        }

        let error = parse("logical_subgraph:anything").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("LogicalSubgraphDecoder class directly")
        );

        let error = parse("not_a_decoder").unwrap_err();
        assert!(error.to_string().contains("Supported:"));
    }

    #[test]
    fn hybrid_parse_carries_only_the_legacy_model_fragment() {
        let spec = parse("belief_matching_hybrid:error(0.1) D0 L0").unwrap();
        assert_eq!(spec.embedded_hybrid_full_dem(), Some("error(0.1) D0 L0"));
    }

    #[test]
    fn execution_traits_propagate_through_composites() {
        let relay_ensemble = DecoderSpec::Ensemble(EnsembleConfig {
            members: vec![
                correlated_pymatching(),
                DecoderSpec::RelayBp(RelayBpConfig::default()),
            ],
        });
        assert_eq!(
            relay_ensemble.execution_traits(),
            ExecutionTraits {
                history_dependent: true,
                wall_clock_dependent: false,
            }
        );

        let timed = DecoderSpec::Perturbed(PerturbedConfig {
            inner: Box::new(DecoderSpec::Mwpf(MwpfConfig {
                timeout: Some(0.1),
                ..MwpfConfig::default()
            })),
            ..PerturbedConfig::default()
        });
        assert_eq!(
            timed.execution_traits(),
            ExecutionTraits {
                history_dependent: false,
                wall_clock_dependent: true,
            }
        );

        let serial_bp = DecoderSpec::BpOsd(super::super::BpOsdConfig {
            bp_schedule: BpSchedule::Serial,
            ..Default::default()
        });
        assert!(serial_bp.execution_traits().history_dependent);
        assert_eq!(
            correlated_pymatching().execution_traits(),
            ExecutionTraits::default()
        );
    }

    #[test]
    fn model_values_are_cloneable_data() {
        let model = DecodeModel::HybridDem {
            full: "full".to_string(),
            decomposed: "decomposed".to_string(),
        };
        assert_eq!(model.clone(), model);
    }

    #[test]
    fn decoder_spec_has_required_auto_traits() {
        fn assert_traits<T: Clone + std::fmt::Debug + Send + Sync + PartialEq>() {}
        assert_traits::<DecoderSpec>();
    }
}
