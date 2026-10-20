use super::DecoderSpec;
use super::config::{
    BeamSearchConfig, BeliefMatchingConfig, BeliefMatchingMode, BpLsdConfig, BpOsdConfig,
    EnsembleConfig, FusionBlossomConfig, FusionBlossomSolverType, KMwpmConfig, MinSumBpConfig,
    MwpfConfig, MwpfSolverType, PecosUfConfig, PecosUfPreset, PerturbedConfig,
    PerturbedFusionBlossomConfig, PyMatchingConfig, RelayBpConfig, TesseractConfig,
    TesseractPreset, TesseractTrellisConfig, WindowedConfig,
};
use pecos_decoder_core::{DecoderError, graphlike_projection::IrreduciblePolicy};
use std::fmt::Display;
use std::str::FromStr;

pub(super) fn parse(type_string: &str) -> Result<DecoderSpec, DecoderError> {
    if let Some((name, params)) = type_string.split_once(':') {
        match name {
            "pymatching"
            | "pymatching_correlated"
            | "pymatching_uncorrelated"
            | "fusion_blossom"
            | "fusion_blossom_serial"
            | "fusion_blossom_parallel"
            | "fusion_blossom_correlated"
            | "astar"
            | "belief_matching"
            | "belief_matching_correlated"
            | "belief_matching_mgbp"
            | "pecos_uf"
            | "pecos_uf_correlated" => return parse_graph_options(name, params),
            _ => {}
        }
    }
    match type_string {
        "pymatching" | "pymatching_correlated" => Ok(correlated_pymatching()),
        "pymatching_uncorrelated" => Ok(DecoderSpec::PyMatching(PyMatchingConfig::default())),
        "tesseract_trellis" => Ok(DecoderSpec::TesseractTrellis(
            TesseractTrellisConfig::default(),
        )),
        "tesseract" => Ok(DecoderSpec::Tesseract(TesseractConfig {
            preset: TesseractPreset::Fast,
            ..TesseractConfig::default()
        })),
        "k_mwpm" => Ok(DecoderSpec::KMwpm(KMwpmConfig::default())),
        "astar" => Ok(DecoderSpec::AStar(crate::spec::AStarConfig::default())),
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
        "pecos_uf" => Ok(DecoderSpec::PecosUf(PecosUfConfig::default())),
        "pecos_uf_correlated" => Ok(DecoderSpec::PecosUf(PecosUfConfig {
            preset: PecosUfPreset::Balanced,
            ..PecosUfConfig::default()
        })),
        "belief_matching" => Ok(belief_matching(BeliefMatchingMode::Standard)),
        "belief_matching_correlated" => Ok(belief_matching(BeliefMatchingMode::Correlated)),
        "belief_matching_mgbp" => Ok(belief_matching(BeliefMatchingMode::MatchingGraphBp)),
        "windowed" => invalid("windowed requires inner, buffer, and step"),
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
                irreducible: IrreduciblePolicy::default(),
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
             Supported: pymatching, tesseract, tesseract_trellis, mwpf, pecos_uf (or \
             pecos_uf:fast/balanced/accurate), logical_subgraph, ensemble:d1,d2,..., \
             bp_osd, bp_lsd, union_find, relay_bp, min_sum_bp."
        )),
    }
}

fn pecos_uf_preset(name: &str) -> Option<PecosUfPreset> {
    match name {
        "fast" => Some(PecosUfPreset::Fast),
        "balanced" => Some(PecosUfPreset::Balanced),
        "accurate" => Some(PecosUfPreset::Accurate),
        "bp" => Some(PecosUfPreset::Bp),
        "bp_serial" => Some(PecosUfPreset::BpSerial),
        _ => None,
    }
}

fn parse_graph_options(name: &str, mut params: &str) -> Result<DecoderSpec, DecoderError> {
    if name == "pecos_uf"
        && params
            .split(',')
            .skip(1)
            .any(|part| pecos_uf_preset(part).is_some())
    {
        return invalid("pecos_uf preset must come first, before key=value parameters");
    }
    let mut spec = parse(name)?;
    if name == "pecos_uf" && !params.is_empty() && !params.split(',').next().unwrap().contains('=')
    {
        let (preset, rest) = params.split_once(',').unwrap_or((params, ""));
        let Some(preset) = pecos_uf_preset(preset) else {
            return invalid(format!("pecos_uf has unknown preset '{preset}'"));
        };
        spec = DecoderSpec::PecosUf(PecosUfConfig {
            preset,
            ..Default::default()
        });
        params = rest;
    }
    for (key, value) in params_iter(name, params)? {
        if key != "irreducible" {
            return unknown_key(name, key);
        }
        let policy = parse_irreducible(value)?;
        match &mut spec {
            DecoderSpec::PyMatching(c) => c.irreducible = policy,
            DecoderSpec::FusionBlossom(c) => c.irreducible = policy,
            DecoderSpec::AStar(c) => c.irreducible = policy,
            DecoderSpec::PecosUf(c) => c.irreducible = policy,
            DecoderSpec::BeliefMatching(c) => c.irreducible = policy,
            _ => unreachable!("only graph-option families reach this parser"),
        }
    }
    Ok(spec)
}

fn parse_irreducible(value: &str) -> Result<IrreduciblePolicy, DecoderError> {
    match value {
        "refuse" => Ok(IrreduciblePolicy::Refuse),
        "drop" => Ok(IrreduciblePolicy::Drop),
        _ => invalid(format!("irreducible must be refuse|drop, got '{value}'")),
    }
}

fn correlated_pymatching() -> DecoderSpec {
    DecoderSpec::PyMatching(PyMatchingConfig {
        irreducible: IrreduciblePolicy::default(),
        correlated: true,
        error_probability: None,
    })
}

fn fusion_blossom(solver_type: FusionBlossomSolverType, correlated: bool) -> DecoderSpec {
    DecoderSpec::FusionBlossom(FusionBlossomConfig {
        irreducible: IrreduciblePolicy::default(),
        correlated,
        solver_type,
    })
}

fn belief_matching(mode: BeliefMatchingMode) -> DecoderSpec {
    DecoderSpec::BeliefMatching(BeliefMatchingConfig {
        irreducible: IrreduciblePolicy::default(),
        mode,
        embedded_full_dem: None,
    })
}

fn parse_k_mwpm(params: &str) -> Result<DecoderSpec, DecoderError> {
    let mut config = KMwpmConfig::default();
    for (key, value) in params_iter("k_mwpm", params)? {
        match key {
            "irreducible" => config.irreducible = parse_irreducible(value)?,
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
            "irreducible" => config.irreducible = parse_irreducible(value)?,
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
    let mut step_size = None;
    let mut buffer_size = None;
    for (key, value) in params_iter("windowed", own_params)? {
        match key {
            "step" => step_size = Some(parse_number("windowed", key, value)?),
            "buf" | "buffer" => buffer_size = Some(parse_number("windowed", key, value)?),
            _ => return unknown_key("windowed", key),
        }
    }
    let Some(inner) = inner else {
        return invalid("windowed requires inner");
    };
    let Some(buffer_size) = buffer_size else {
        return invalid("windowed requires buffer");
    };
    let Some(step_size) = step_size else {
        return invalid("windowed requires step");
    };
    if step_size == 0 {
        return invalid("step must be at least 1");
    }
    Ok(DecoderSpec::Windowed(WindowedConfig {
        step_size,
        buffer_size,
        inner: Box::new(parse(inner)?),
    }))
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

    // contract: all graph names accept the policy without changing their default specs.
    #[test]
    fn graph_policy_grammar() {
        for name in [
            "pymatching",
            "pymatching_correlated",
            "pymatching_uncorrelated",
            "fusion_blossom",
            "fusion_blossom_serial",
            "fusion_blossom_parallel",
            "fusion_blossom_correlated",
            "astar",
            "k_mwpm",
            "perturbed_fb_corr",
            "belief_matching",
            "belief_matching_correlated",
            "belief_matching_mgbp",
            "pecos_uf",
            "pecos_uf:fast",
            "pecos_uf:balanced",
            "pecos_uf:accurate",
            "pecos_uf:bp",
            "pecos_uf:bp_serial",
            "pecos_uf_correlated",
        ] {
            let separator = if name.contains(':') { ',' } else { ':' };
            let base = parse(name).unwrap();
            assert_eq!(base.irreducible_policy(), Some(IrreduciblePolicy::Refuse));
            assert_eq!(
                parse(&format!("{name}{separator}irreducible=refuse")).unwrap(),
                base
            );
            assert_eq!(
                parse(&format!("{name}{separator}irreducible=drop"))
                    .unwrap()
                    .irreducible_policy(),
                Some(IrreduciblePolicy::Drop)
            );
            assert!(
                parse(&format!("{name}{separator}irreducible=unknown"))
                    .unwrap_err()
                    .to_string()
                    .contains("refuse|drop")
            );
        }
        assert_eq!(
            parse("pecos_uf:irreducible=drop").unwrap(),
            parse("pecos_uf:fast,irreducible=drop").unwrap()
        );
        assert_eq!(
            parse("pecos_uf_correlated:irreducible=drop").unwrap(),
            parse("pecos_uf:balanced,irreducible=drop").unwrap()
        );
        let DecoderSpec::KMwpm(config) = parse("k_mwpm:k=3,irreducible=drop").unwrap() else {
            unreachable!()
        };
        assert_eq!(config.k, 3);
        assert_eq!(config.irreducible, IrreduciblePolicy::Drop);
        let DecoderSpec::PerturbedFusionBlossomCorrelated(config) =
            parse("perturbed_fb_corr:k=3,sigma=0.2,seed=7,irreducible=drop").unwrap()
        else {
            unreachable!()
        };
        assert_eq!(config.k, 3);
        assert_eq!(config.irreducible, IrreduciblePolicy::Drop);
    }

    // compatibility: duplicate policy keys retain the parameter parser's last-value-wins rule.
    #[test]
    fn duplicate_policy_uses_last_value() {
        for name in [
            "pymatching",
            "fusion_blossom",
            "pecos_uf",
            "belief_matching",
            "astar",
            "k_mwpm",
            "perturbed_fb_corr",
        ] {
            for (params, expected) in [
                ("drop,irreducible=refuse", IrreduciblePolicy::Refuse),
                ("refuse,irreducible=drop", IrreduciblePolicy::Drop),
            ] {
                let spec = parse(&format!("{name}:irreducible={params}")).unwrap();
                assert_eq!(spec.irreducible_policy(), Some(expected));
            }
        }
    }

    // compatibility: commas still delimit ensemble members, even after a member parameter.
    #[test]
    fn ensemble_single_parameter_member_and_hybrid_suffix() {
        let DecoderSpec::Ensemble(config) =
            parse("ensemble:pymatching:irreducible=drop,fusion_blossom").unwrap()
        else {
            unreachable!()
        };
        assert_eq!(
            config.members,
            vec![
                parse("pymatching:irreducible=drop").unwrap(),
                parse("fusion_blossom").unwrap()
            ]
        );
        let spec = parse("belief_matching_hybrid:irreducible=drop").unwrap();
        let DecoderSpec::BeliefMatching(config) = spec else {
            unreachable!()
        };
        assert_eq!(config.irreducible, IrreduciblePolicy::Refuse);
        // The entire suffix remains a DEM fragment, never a policy parameter.
        assert_eq!(
            config.embedded_full_dem.as_deref(),
            Some("irreducible=drop")
        );
    }

    #[test]
    fn parses_legacy_parameterized_forms() {
        assert_eq!(
            parse("k_mwpm:K=10").unwrap(),
            DecoderSpec::KMwpm(KMwpmConfig {
                irreducible: IrreduciblePolicy::default(),
                k: 10
            })
        );

        let perturbed = parse("perturbed:K=15,sigma=0.7,seed=42,inner=pymatching").unwrap();
        let DecoderSpec::Perturbed(config) = perturbed else {
            panic!("expected perturbed spec");
        };
        assert_eq!(config.k, 15);
        assert!((config.sigma - 0.7).abs() < f64::EPSILON);
        assert_eq!(config.seed, 42);
        assert_eq!(*config.inner, correlated_pymatching());

        let windowed = parse("windowed:step=5,buf=5,inner=bp_osd").unwrap();
        let DecoderSpec::Windowed(config) = windowed else {
            panic!("expected windowed spec");
        };
        assert_eq!(config.step_size, 5);
        assert_eq!(config.buffer_size, 5);
        assert!(matches!(*config.inner, DecoderSpec::BpOsd(_)));

        let explicit_fast = parse("windowed:step=5,buf=5,inner=pecos_uf:fast").unwrap();
        let DecoderSpec::Windowed(ref config) = explicit_fast else {
            panic!("expected windowed spec");
        };
        assert!(matches!(
            *config.inner,
            DecoderSpec::PecosUf(PecosUfConfig {
                preset: PecosUfPreset::Fast,
                ..
            })
        ));

        let exact_pecos_uf = parse("windowed:step=5,buf=5,inner=pecos_uf").unwrap();
        assert_eq!(exact_pecos_uf, explicit_fast);

        assert_eq!(
            parse("perturbed_fb_corr:K=9,sigma=0.25,seed=7").unwrap(),
            DecoderSpec::PerturbedFusionBlossomCorrelated(PerturbedFusionBlossomConfig {
                irreducible: IrreduciblePolicy::default(),
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
            DecoderSpec::PecosUf(PecosUfConfig {
                preset: PecosUfPreset::Accurate,
                ..Default::default()
            })
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
            "tesseract_trellis",
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
            "windowed:step=1,buffer=1,inner=pecos_uf",
            "mwpf",
            "perturbed",
            "beamsearch",
        ];
        for form in forms {
            assert!(parse(form).is_ok(), "failed to parse {form}");
        }
    }

    #[test]
    fn graph_options_reject_misplaced_presets() {
        let error = parse("pecos_uf:irreducible=drop,fast")
            .unwrap_err()
            .to_string();
        assert!(error.contains("preset must come first"), "{error}");
        let error = parse("pecos_uf:fast,bogus").unwrap_err().to_string();
        assert!(error.contains("key=value"), "{error}");
    }

    #[test]
    fn empty_parameter_suffix_uses_family_defaults() {
        for (with_suffix, bare) in [
            ("k_mwpm:", "k_mwpm"),
            ("perturbed_fb_corr:", "perturbed_fb_corr"),
            ("mwpf:", "mwpf"),
            ("perturbed:", "perturbed"),
            ("beamsearch:", "beamsearch"),
            ("pymatching:", "pymatching"),
            ("pecos_uf:", "pecos_uf"),
            ("astar:", "astar"),
            ("fusion_blossom:", "fusion_blossom"),
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
    #[test]
    fn test_11_windowed_spec_surface() {
        for key in [
            "mode",
            "seam",
            "ext",
            "core_extend",
            "wmax",
            "commit_weight_max",
            "sandwich_phase2",
        ] {
            let error = parse(&format!("windowed:{key}=1,buffer=1,inner=pecos_uf")).unwrap_err();
            assert!(error.to_string().contains("unknown"), "{key}: {error}");
            assert!(error.to_string().contains(key));
        }
        for text in [
            "windowed",
            "windowed:",
            "windowed:buffer=1",
            "windowed:inner=pecos_uf",
        ] {
            assert!(
                parse(text).unwrap_err().to_string().contains("requires"),
                "{text}"
            );
        }
        assert!(
            parse("windowed:buffer=1,inner=pecos_uf")
                .unwrap_err()
                .to_string()
                .contains("windowed requires step")
        );
        assert!(
            parse("windowed:step=0,buffer=1,inner=pecos_uf")
                .unwrap_err()
                .to_string()
                .contains("step must be at least 1")
        );
        assert!(parse("windowed:step=1,buffer=0,inner=pymatching_uncorrelated").is_ok());
        assert!(parse("beamsearch:step=3,buf=2,wmax=2.5,K=2,sigma=0.3,seed=7").is_ok());
    }
}
