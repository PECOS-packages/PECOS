# Copyright 2026 The PECOS Developers

"""Typed decoder-spec factory tests."""

from collections.abc import Callable

import pytest
from pecos_rslib.decoders import (
    DecoderSpec,
    astar,
    astar_full,
    beamsearch,
    belief_find,
    belief_matching,
    bp_lsd,
    bp_osd,
    ensemble,
    fusion_blossom,
    k_mwpm,
    min_sum_bp,
    mwpf,
    pecos_uf,
    perturbed,
    perturbed_fb_corr,
    pymatching,
    relay_bp,
    tesseract,
    union_find,
    windowed,
)

FactoryCase = tuple[str, Callable[[], DecoderSpec]]


FACTORY_CASES: list[FactoryCase] = [
    ("pymatching", lambda: pymatching(correlated=False, error_probability=0.1)),
    (
        "tesseract",
        lambda: tesseract(
            preset="accurate",
            det_beam=32,
            beam_climbing=True,
            verbose=False,
            no_revisit_dets=True,
            pqlimit=1000,
            det_penalty=0.25,
        ),
    ),
    (
        "bp_osd",
        lambda: bp_osd(
            error_rate=0.05,
            max_iter=50,
            bp_schedule="serial_relative",
            ms_scaling_factor=0.75,
            osd_order=2,
            random_schedule_seed=7,
        ),
    ),
    (
        "bp_lsd",
        lambda: bp_lsd(
            error_rate=0.04,
            max_iter=60,
            bp_schedule="serial",
            ms_scaling_factor=0.8,
            lsd_order=3,
            bits_per_step=2,
            random_schedule_seed=9,
        ),
    ),
    ("fusion_blossom", lambda: fusion_blossom(correlated=True, solver="serial")),
    (
        "relay_bp",
        lambda: relay_bp(
            error_rate=0.03,
            max_iter=120,
            alpha=0.9,
            alpha_iteration_scaling_factor=0.95,
            gamma0=0.6,
            pre_iter=40,
            num_sets=20,
            set_max_iter=30,
            gamma_dist_interval=(-0.1, 0.4),
            stopping_criterion=2,
            seed=11,
        ),
    ),
    ("min_sum_bp", lambda: min_sum_bp(error_rate=0.02, max_iter=80, alpha=0.7)),
    ("pecos_uf", lambda: pecos_uf(preset="balanced")),
    ("belief_matching", lambda: belief_matching(mode="matching_graph_bp")),
    (
        "windowed",
        lambda: windowed(
            step=5,
            buffer=7,
            mode="sandwich",
            seam=2,
            core_extend=1,
            commit_weight_max=2.5,
            inner=pecos_uf(preset="balanced"),
            sandwich_phase2=pymatching(correlated=False),
        ),
    ),
    (
        "mwpf",
        lambda: mwpf(
            solver="bp_hybrid",
            cluster_node_limit=75,
            timeout=0.5,
            only_solve_primal_once=True,
        ),
    ),
    (
        "perturbed",
        lambda: perturbed(inner=pecos_uf(preset="fast"), k=7, sigma=0.3, seed=12),
    ),
    (
        "beamsearch",
        lambda: beamsearch(
            beam_width=8,
            sigma=0.2,
            seed=13,
            step=3,
            buffer=4,
            commit_weight_max=1.5,
            phase2=pecos_uf(preset="balanced"),
        ),
    ),
    (
        "ensemble",
        lambda: ensemble(pymatching(correlated=True), pecos_uf(preset="fast")),
    ),
    ("k_mwpm", lambda: k_mwpm(k=4)),
    ("astar", astar),
    ("astar_full", astar_full),
    ("union_find", union_find),
    ("belief_find", belief_find),
    ("perturbed_fb_corr", lambda: perturbed_fb_corr(k=8, sigma=0.25, seed=14)),
]


@pytest.mark.parametrize(("family", "factory"), FACTORY_CASES)
def test_every_factory_is_an_equal_immutable_value(family: str, factory: Callable[[], DecoderSpec]) -> None:
    """Each factory preserves its knobs in equality and a useful representation."""
    first = factory()
    second = factory()

    assert isinstance(first, DecoderSpec)
    assert first == second
    assert family in repr(first)
    with pytest.raises(AttributeError):
        first.extra_attribute = True


def test_pymatching_requires_correlated_argument() -> None:
    """The ambiguous correlated-matching default is deliberately forbidden."""
    with pytest.raises(TypeError, match="correlated"):
        pymatching()


@pytest.mark.parametrize(
    ("factory", "parameter", "bad_value"),
    [
        (lambda: tesseract(preset="turbo"), "preset", "turbo"),
        (lambda: bp_osd(bp_schedule="random"), "bp_schedule", "random"),
        (lambda: fusion_blossom(solver="distributed"), "solver", "distributed"),
        (lambda: pecos_uf(preset="slow"), "preset", "slow"),
        (lambda: belief_matching(mode="hybrid"), "mode", "hybrid"),
        (lambda: windowed(mode="sliding"), "mode", "sliding"),
        (lambda: mwpf(solver="exact"), "solver", "exact"),
        (
            lambda: relay_bp(stopping_criterion="eventually"),
            "stopping_criterion",
            "eventually",
        ),
    ],
)
def test_enum_validation_names_parameter_and_bad_value(
    factory: Callable[[], DecoderSpec], parameter: str, bad_value: str
) -> None:
    """Enum validation errors identify both the input and its accepted domain."""
    with pytest.raises(ValueError, match="accepted") as error:
        factory()
    message = str(error.value)
    assert parameter in message
    assert bad_value in message
    assert "accepted" in message


@pytest.mark.parametrize(
    "factory",
    [
        lambda: pymatching(correlated=True, error_probability=1.5),
        lambda: k_mwpm(k=0),
        lambda: perturbed(sigma=-0.1),
        lambda: windowed(step=-1),
        lambda: relay_bp(stopping_criterion=0),
    ],
)
def test_numeric_domain_errors_are_value_errors(
    factory: Callable[[], DecoderSpec],
) -> None:
    """Correctly typed but out-of-domain values do not leak conversion errors."""
    with pytest.raises(ValueError, match="accepted|must be"):
        factory()


def test_nested_specs_require_decoder_spec_values() -> None:
    """PyO3 extraction reports wrong nested-object types as TypeError."""
    with pytest.raises(TypeError):
        windowed(inner="pecos_uf")
    with pytest.raises(TypeError):
        ensemble(pymatching(correlated=True), "relay_bp")


@pytest.mark.parametrize(
    ("legacy", "typed"),
    [
        ("pymatching", pymatching(correlated=True)),
        ("pymatching_uncorrelated", pymatching(correlated=False)),
        ("tesseract", tesseract(preset="fast")),
        ("k_mwpm:K=4", k_mwpm(k=4)),
        ("astar", astar()),
        ("astar_full", astar_full()),
        ("fusion_blossom", fusion_blossom()),
        ("fusion_blossom_serial", fusion_blossom(solver="serial")),
        ("fusion_blossom_parallel", fusion_blossom(solver="parallel")),
        (
            "fusion_blossom_correlated",
            fusion_blossom(correlated=True, solver="serial"),
        ),
        (
            "perturbed_fb_corr:K=8,sigma=0.25,seed=14",
            perturbed_fb_corr(k=8, sigma=0.25, seed=14),
        ),
        ("bp_osd", bp_osd()),
        ("bp_lsd", bp_lsd()),
        ("belief_find", belief_find()),
        ("union_find", union_find()),
        ("relay_bp", relay_bp()),
        ("min_sum_bp", min_sum_bp()),
        ("pecos_uf", pecos_uf()),
        ("pecos_uf:balanced", pecos_uf(preset="balanced")),
        ("pecos_uf:accurate", pecos_uf(preset="accurate")),
        ("pecos_uf:bp", pecos_uf(preset="bp")),
        ("pecos_uf:bp_serial", pecos_uf(preset="bp_serial")),
        ("belief_matching", belief_matching()),
        ("belief_matching_correlated", belief_matching(mode="correlated")),
        (
            "belief_matching_mgbp",
            belief_matching(mode="matching_graph_bp"),
        ),
        ("windowed", windowed()),
        (
            "windowed:step=5,buf=5,inner=pecos_uf",
            windowed(step=5, buffer=5, inner=pecos_uf()),
        ),
        ("mwpf", mwpf()),
        (
            "mwpf:c=25,t=0.5,once=true,solver=bp",
            mwpf(
                solver="bp_hybrid",
                cluster_node_limit=25,
                timeout=0.5,
                only_solve_primal_once=True,
            ),
        ),
        ("perturbed", perturbed()),
        (
            "perturbed:K=7,sigma=0.3,seed=12,inner=pecos_uf",
            perturbed(inner=pecos_uf(), k=7, sigma=0.3, seed=12),
        ),
        ("beamsearch", beamsearch()),
        (
            "beamsearch:K=8,sigma=0.2,seed=13,step=3,buf=4,wmax=1.5",
            beamsearch(
                beam_width=8,
                sigma=0.2,
                seed=13,
                step=3,
                buffer=4,
                commit_weight_max=1.5,
            ),
        ),
        (
            "ensemble:pymatching,relay_bp",
            ensemble(pymatching(correlated=True), relay_bp()),
        ),
    ],
)
def test_legacy_parse_matches_typed_factory(legacy: str, typed: DecoderSpec) -> None:
    """Factories construct the same Rust values as counterpart legacy strings."""
    assert DecoderSpec.parse(legacy) == typed


def test_parse_errors_preserve_rust_messages() -> None:
    """The static adapter exposes the strict Rust parser as ValueError."""
    with pytest.raises(ValueError, match="unknown parameter 'unknown'"):
        DecoderSpec.parse("windowed:unknown=1")


@pytest.mark.parametrize(
    ("spec", "history_dependent", "wall_clock_dependent"),
    [
        (relay_bp(), True, False),
        (mwpf(timeout=0.5), False, True),
        (mwpf(), False, False),
        (ensemble(pymatching(correlated=True), relay_bp()), True, False),
        (bp_osd(bp_schedule="serial"), True, False),
    ],
)
def test_execution_traits(spec: DecoderSpec, history_dependent: bool, wall_clock_dependent: bool) -> None:
    """Execution-trait properties reflect the recursively computed Rust values."""
    assert spec.history_dependent is history_dependent
    assert spec.wall_clock_dependent is wall_clock_dependent


def test_nested_composite_propagates_wall_clock_dependency() -> None:
    """Nested composite factories preserve transitive execution traits."""
    spec = windowed(inner=perturbed(inner=mwpf(timeout=1.0)))

    assert spec.history_dependent is False
    assert spec.wall_clock_dependent is True


def test_equal_specs_are_usable_as_dict_keys() -> None:
    first = pymatching(correlated=True)
    second = pymatching(correlated=True)
    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1
    assert {first: "value"}[second] == "value"


def test_repr_names_the_real_factory_callable() -> None:
    spec = pymatching(correlated=True)
    assert repr(spec) == "pymatching(correlated=True)"
    assert "DecoderSpec." not in repr(spec)
    nested = ensemble(pymatching(correlated=False), relay_bp())
    assert "DecoderSpec." not in repr(nested)


def test_hybrid_repr_does_not_inline_the_embedded_dem() -> None:
    dem = "error(0.1) D0 L0\n" * 200
    spec = DecoderSpec.parse(f"belief_matching_hybrid:{dem}")
    assert "error(0.1)" not in repr(spec)
    assert "bytes>" in repr(spec)


def test_seed_accepts_the_full_u64_range() -> None:
    spec = perturbed(seed=2**64 - 1)
    assert f"seed={2**64 - 1}" in repr(spec)
    with pytest.raises(OverflowError):
        relay_bp(seed=-1)


def test_stopping_criterion_rejects_bool() -> None:
    with pytest.raises(ValueError, match="stopping_criterion"):
        relay_bp(stopping_criterion=True)
