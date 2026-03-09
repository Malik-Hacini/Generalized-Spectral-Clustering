"""Measure preprocessing versus eigensolver dominance on sparse DISBM networks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from common import (
    RESULTS_DIR,
    build_generalized_normalized_laplacian,
    build_standard_normalized_laplacian,
    build_transition_matrix,
    compute_measure_from_transition,
    compute_xi,
    ensure_dir,
    generate_core_periphery_disbm,
    json_ready,
    parse_int_list,
    run_shift_invert_eigensolver,
    symmetrize_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", default="500,1000,2000,5000,10000")
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--degree-factor", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--t", type=int, default=5)
    parser.add_argument("--p-core", type=float, default=0.10)
    parser.add_argument("--p-periphery", type=float, default=0.01)
    parser.add_argument("--p-core-periphery", type=float, default=0.05)
    parser.add_argument("--p-periphery-core", type=float, default=0.005)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def timed(callable_obj):
    start = time.perf_counter()
    value = callable_obj()
    return time.perf_counter() - start, value


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    rows: list[dict[str, object]] = []

    for n_nodes in parse_int_list(args.n_values):
        adjacency, _ = generate_core_periphery_disbm(
            n_nodes=n_nodes,
            n_clusters=args.clusters,
            p_core=args.p_core,
            p_periphery=args.p_periphery,
            p_core_periphery=args.p_core_periphery,
            p_periphery_core=args.p_periphery_core,
            degree_factor=args.degree_factor,
            seed=args.seed,
        )

        for repeat in range(args.repeats):
            t_sym, W_sym = timed(lambda: symmetrize_graph(adjacency))
            t_sc_lap, L_sc = timed(lambda: build_standard_normalized_laplacian(W_sym))
            t_sc_eig, _ = timed(
                lambda: run_shift_invert_eigensolver(
                    L_sc, n_components=args.n_components, random_state=args.seed, symmetric=True
                )
            )

            t_P, P = timed(lambda: build_transition_matrix(adjacency))
            t_measure, nu = timed(
                lambda: compute_measure_from_transition(P, alpha=args.alpha, t=args.t)
            )
            t_xi, xi = timed(lambda: compute_xi(P, nu))
            t_gsc_lap, L_gsc = timed(
                lambda: build_generalized_normalized_laplacian(P, nu, xi)
            )
            t_gsc_eig, _ = timed(
                lambda: run_shift_invert_eigensolver(
                    L_gsc, n_components=args.n_components, random_state=args.seed, symmetric=True
                )
            )

            sc_pre = t_sym + t_sc_lap
            gsc_pre = t_P + t_measure + t_xi + t_gsc_lap
            extra = gsc_pre - sc_pre
            rows.append(
                {
                    "benchmark": "network_component_share_disbm",
                    "N": n_nodes,
                    "nnz": int(adjacency.nnz),
                    "clusters": args.clusters,
                    "n_components": args.n_components,
                    "degree_factor": args.degree_factor,
                    "t": args.t,
                    "alpha": args.alpha,
                    "repeat": repeat,
                    "sc_preprocessing": sc_pre,
                    "gsc_preprocessing": gsc_pre,
                    "sc_eigensolver": t_sc_eig,
                    "gsc_eigensolver": t_gsc_eig,
                    "extra_overhead": extra,
                    "sc_pre_over_eig": sc_pre / t_sc_eig,
                    "gsc_pre_over_eig": gsc_pre / t_gsc_eig,
                    "extra_over_gsc_eig": extra / t_gsc_eig,
                }
            )

    raw_df = pd.DataFrame(rows)
    summary_df = (
        raw_df.groupby(["benchmark", "N", "nnz", "clusters", "n_components", "degree_factor", "t", "alpha"], as_index=False)
        .agg(
            sc_preprocessing_mean=("sc_preprocessing", "mean"),
            gsc_preprocessing_mean=("gsc_preprocessing", "mean"),
            sc_eigensolver_mean=("sc_eigensolver", "mean"),
            gsc_eigensolver_mean=("gsc_eigensolver", "mean"),
            extra_overhead_mean=("extra_overhead", "mean"),
            sc_pre_over_eig_mean=("sc_pre_over_eig", "mean"),
            gsc_pre_over_eig_mean=("gsc_pre_over_eig", "mean"),
            extra_over_gsc_eig_mean=("extra_over_gsc_eig", "mean"),
            sc_pre_over_eig_std=("sc_pre_over_eig", "std"),
            gsc_pre_over_eig_std=("gsc_pre_over_eig", "std"),
            extra_over_gsc_eig_std=("extra_over_gsc_eig", "std"),
            n_repeats=("repeat", "size"),
        )
        .sort_values(["nnz"])
        .reset_index(drop=True)
        .fillna(0.0)
    )

    raw_path = out_dir / "network_component_share_disbm_raw.csv"
    summary_path = out_dir / "network_component_share_disbm_summary.csv"
    json_path = out_dir / "network_component_share_disbm_summary.json"
    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    json_path.write_text(json.dumps({"config": json_ready(vars(args))}, indent=2, sort_keys=True))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
