"""Python Package responsible for generating random numbers using pcg_rng."""

from .pecos_rng import (
    pcg32_boundedrand,
    pcg32_frandom,
    pcg32_random,
    pcg32_srandom,
)

__all__ = ["pcg32_boundedrand", "pcg32_frandom", "pcg32_random", "pcg32_srandom"]
