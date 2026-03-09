"""Graph-CH diffusion filter profile library.

This module centralizes reusable polynomial filters g(P) = sum_k a_k P^k used
by the Graph-CH metric. The profile families include:

- point-scale filters (delta_k)
- uniform multi-scale averages (prefix_k)
- geometrically decaying filters (resolvent / PageRank-like)
- heat-kernel inspired filters (Poisson weights)
- lazy-walk binomial filters
- Fejer / Cesaro smoothing filters
- custom user-requested filters

References for the profile families:
- Coifman & Lafon (2006), diffusion distance / diffusion maps.
- Andersen, Chung, Lang (2006), PageRank local clustering.
- Chung & Simpson (2015), heat kernel pagerank.
"""

from __future__ import annotations

from math import comb, factorial


def _profile(profile_id: str, family: str, scale: float, coeffs: dict[int, float], weighted: bool = False, epsilon: float = 1e-10) -> dict:
    return {
        "profile_id": profile_id,
        "profile_family": family,
        "profile_scale": scale,
        "graph_ch": {
            "filter_coeffs": {int(k): float(v) for k, v in coeffs.items()},
            "weighted": bool(weighted),
            "epsilon": float(epsilon),
        },
    }


def delta_profile(k: int) -> dict[int, float]:
    if k < 1:
        raise ValueError("k must be >= 1")
    return {k: 1.0}


def prefix_uniform_profile(k_max: int) -> dict[int, float]:
    if k_max < 1:
        raise ValueError("k_max must be >= 1")
    return {k: 1.0 for k in range(1, k_max + 1)}


def geometric_profile(k_max: int, rho: float) -> dict[int, float]:
    if k_max < 1:
        raise ValueError("k_max must be >= 1")
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must be in (0, 1)")
    return {k: float(rho ** (k - 1)) for k in range(1, k_max + 1)}


def heat_profile(k_max: int, tau: float) -> dict[int, float]:
    if k_max < 1:
        raise ValueError("k_max must be >= 1")
    if tau <= 0:
        raise ValueError("tau must be > 0")
    return {k: float((tau**k) / factorial(k)) for k in range(1, k_max + 1)}


def lazy_binomial_profile(m: int) -> dict[int, float]:
    if m < 1:
        raise ValueError("m must be >= 1")
    denom = 2**m
    return {k: float(comb(m, k) / denom) for k in range(1, m + 1)}


def fejer_profile(k_max: int) -> dict[int, float]:
    if k_max < 1:
        raise ValueError("k_max must be >= 1")
    return {k: float(k_max + 1 - k) for k in range(1, k_max + 1)}


def linear_increasing_profile(k_max: int) -> dict[int, float]:
    if k_max < 1:
        raise ValueError("k_max must be >= 1")
    return {k: float(k) for k in range(1, k_max + 1)}


def custom_profile_p2_over2_p3_over3() -> dict[int, float]:
    """Requested profile: P^3/3 + P^2/2."""
    return {2: 1.0 / 2.0, 3: 1.0 / 3.0}


def build_legacy_graph_ch_profiles() -> list[dict]:
    profiles: list[dict] = []

    for k in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
        profiles.append(_profile(f"delta_k{k:02d}", "delta_k", float(k), delta_profile(k)))

    for k in [2, 4, 6, 8, 10, 12]:
        profiles.append(_profile(f"prefix_k{k:02d}", "prefix_k", float(k), prefix_uniform_profile(k)))

    return profiles


def build_research_graph_ch_profiles() -> list[dict]:
    """Research-motivated profile shortlist for diffusion-distance selection."""
    return [
        _profile(
            profile_id="mix_p2_over2_p3_over3",
            family="custom_mix",
            scale=3.0,
            coeffs=custom_profile_p2_over2_p3_over3(),
        ),
        _profile(
            profile_id="geom_r085_k12",
            family="geometric",
            scale=12.0,
            coeffs=geometric_profile(k_max=12, rho=0.85),
        ),
        _profile(
            profile_id="heat_tau2_k10",
            family="heat",
            scale=10.0,
            coeffs=heat_profile(k_max=10, tau=2.0),
        ),
        _profile(
            profile_id="heat_tau4_k12",
            family="heat",
            scale=12.0,
            coeffs=heat_profile(k_max=12, tau=4.0),
        ),
        _profile(
            profile_id="lazy_binom_m6",
            family="lazy_binomial",
            scale=6.0,
            coeffs=lazy_binomial_profile(m=6),
        ),
        _profile(
            profile_id="fejer_k8",
            family="fejer",
            scale=8.0,
            coeffs=fejer_profile(k_max=8),
        ),
        _profile(
            profile_id="linear_inc_k8",
            family="linear_increasing",
            scale=8.0,
            coeffs=linear_increasing_profile(k_max=8),
        ),
        _profile(
            profile_id="band_p3_minus_p8",
            family="band_pass",
            scale=8.0,
            coeffs={3: 1.0, 8: -1.0},
        ),
    ]


def merge_profiles(*profile_lists: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for profile_list in profile_lists:
        for p in profile_list:
            pid = str(p.get("profile_id"))
            if pid in seen:
                continue
            seen.add(pid)
            merged.append(p)
    return merged
