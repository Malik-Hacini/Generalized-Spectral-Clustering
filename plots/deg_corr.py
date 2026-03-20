"""Plot AMI trends for the degree-correction DiSBM benchmark."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

if __package__ is None or __package__ == "":
	import sys

	sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.common import project_path, resolve_output_dir, validate_selection
from plots.method_style import ordered_methods, style_for_method


DEFAULT_RESULTS_DIR = "results/benchmark_deg_corr_grid_search"
DEFAULT_METHODS_TO_PLOT = ["GSC-N", "SC-N", "DSC+"]


def _parse_float_token(token: str) -> float:
	return float(token.replace("p", "."))


def _extract_ami(best_results: dict) -> float | None:
	if "graph_ch" in best_results:
		payload = best_results["graph_ch"]
		if isinstance(payload, dict):
			ami_payload = payload.get("ami")
			if isinstance(ami_payload, dict) and "mean" in ami_payload:
				return float(ami_payload["mean"])

	if "ami" in best_results:
		payload = best_results["ami"]
		if isinstance(payload, dict):
			ami_payload = payload.get("ami")
			if isinstance(ami_payload, dict) and "mean" in ami_payload:
				return float(ami_payload["mean"])

	return None


def load_deg_corr_results(results_dir: str | Path) -> pd.DataFrame:
	path = project_path(results_dir)
	if not path.exists():
		raise FileNotFoundError(f"Results directory not found: {path}")

	best_result_files = sorted(path.glob("*/*/*_best_results.json"))
	if not best_result_files:
		raise ValueError(f"No *_best_results.json files found in {path}")

	pattern = re.compile(
		r"^dcdisbm_degcorr_b([0-9-]+)_pintra([0-9p]+)_pinter([0-9p]+)"
		r"_mode(alpha|scale)_a1([0-9p]+)_s1([0-9p]+)_seed(\d+)$"
	)

	rows: list[dict] = []
	unmatched_dataset_names: list[str] = []

	for best_file in best_result_files:
		method_name = best_file.parent.name
		dataset_name = best_file.parent.parent.name

		match = pattern.match(dataset_name)
		if match is None:
			if dataset_name.startswith("dcdisbm_degcorr_"):
				unmatched_dataset_names.append(dataset_name)
			continue

		block_sizes = tuple(int(v) for v in match.group(1).split("-"))
		p_intra = _parse_float_token(match.group(2))
		p_inter = _parse_float_token(match.group(3))
		sweep_mode = match.group(4)
		alpha_high = _parse_float_token(match.group(5))
		scale_high = _parse_float_token(match.group(6))
		seed = int(match.group(7))

		best_results = json.loads(best_file.read_text())
		ami = _extract_ami(best_results)
		if ami is None:
			continue

		rows.append(
			{
				"method": method_name,
				"block_sizes": block_sizes,
				"p_intra": p_intra,
				"p_inter": p_inter,
				"sweep_mode": sweep_mode,
				"alpha_high": alpha_high,
				"scale_high": scale_high,
				"seed": seed,
				"ami": ami,
			}
		)

	if unmatched_dataset_names:
		unique_unmatched = sorted(set(unmatched_dataset_names))
		print(
			"Warning: Could not parse "
			f"{len(unique_unmatched)} degree-corrected dataset names. "
			f"First example: {unique_unmatched[0]}"
		)

	return pd.DataFrame(rows)


def _plot_mode(
	df: pd.DataFrame,
	mode: str,
	x_col: str,
	xlabel: str,
	output_file: Path,
	methods: list[str],
) -> None:
	mode_df = df[df["sweep_mode"] == mode].copy()
	if mode_df.empty:
		print(f"No rows found for mode={mode}; skipping {output_file.name}")
		return

	summary = (
		mode_df.groupby(["method", x_col])["ami"]
		.agg(ami_mean="mean", ami_std="std", n="count")
		.reset_index()
		.sort_values(x_col)
	)

	method_order = ordered_methods(methods)
	fig, ax = plt.subplots(figsize=(8.5, 6))

	for method in method_order:
		method_data = summary[summary["method"] == method]
		if method_data.empty:
			continue

		style = style_for_method(method)
		x_values = np.asarray(method_data[x_col], dtype=float)
		ami_mean = np.asarray(method_data["ami_mean"], dtype=float)
		ami_std = np.nan_to_num(np.asarray(method_data["ami_std"], dtype=float), nan=0.0)

		ax.plot(
			x_values,
			ami_mean,
			label=style.get("label", method),
			color=style.get("color", None),
			linestyle=style.get("linestyle", "-"),
			marker=style.get("marker", "o"),
			markersize=6,
			linewidth=2,
		)
		ax.fill_between(
			x_values,
			ami_mean - ami_std,
			ami_mean + ami_std,
			color=style.get("color", None),
			alpha=0.2,
		)

	ax.set_xlabel(xlabel, fontsize=12)
	ax.set_ylabel("AMI Score", fontsize=12)
	ax.grid(True, alpha=0.3, linestyle="--")
	ax.legend(loc="best", fontsize=10, framealpha=0.95)
	plt.tight_layout()

	output_file.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_file, dpi=300, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved plot: {output_file}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Plot AMI trends for degree-correction parameter sweeps."
	)
	parser.add_argument(
		"--results-dir",
		type=str,
		default=DEFAULT_RESULTS_DIR,
		help="Path to results directory (default: results/benchmark_deg_corr_grid_search).",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=None,
		help="Output directory. Defaults to plots/deg_corr/<experiment_name>/.",
	)
	parser.add_argument(
		"--methods",
		nargs="+",
		default=DEFAULT_METHODS_TO_PLOT,
		help="Methods to include in plots.",
	)
	parser.add_argument(
		"--mode",
		choices=["scale", "alpha", "both"],
		default="both",
		help="Which sweep(s) to plot.",
	)
	args = parser.parse_args()

	results_dir = project_path(args.results_dir)
	output_dir = resolve_output_dir(args.output_dir, "deg_corr", results_dir)

	print(f"Loading results from: {results_dir}")
	df = load_deg_corr_results(results_dir)
	if df.empty:
		print("No parsed entries found in results.")
		return

	available_methods = sorted(df["method"].unique().tolist())
	selected_methods = validate_selection(available_methods, args.methods, "methods")
	df = df[df["method"].isin(selected_methods)].copy()

	print(f"Loaded {len(df)} result entries")
	print(f"Methods: {sorted(df['method'].unique().tolist())}")
	print(f"Sweep modes: {sorted(df['sweep_mode'].unique().tolist())}")

	modes = ["scale", "alpha"] if args.mode == "both" else [args.mode]
	if "scale" in modes:
		scale_values = sorted(df[df["sweep_mode"] == "scale"]["scale_high"].unique().tolist())
		print(f"scale_high values: {scale_values}")
		_plot_mode(
			df=df,
			mode="scale",
			x_col="scale_high",
			xlabel=r"First-block Degree Scale ($s_1$)",
			output_file=output_dir / "deg_corr_ami_vs_scale_high.pdf",
			methods=selected_methods,
		)

	if "alpha" in modes:
		alpha_values = sorted(df[df["sweep_mode"] == "alpha"]["alpha_high"].unique().tolist())
		print(f"alpha_high values: {alpha_values}")
		_plot_mode(
			df=df,
			mode="alpha",
			x_col="alpha_high",
			xlabel=r"First-block Pareto Exponent ($\alpha_1$)",
			output_file=output_dir / "deg_corr_ami_vs_alpha_high.pdf",
			methods=selected_methods,
		)


if __name__ == "__main__":
	main()
