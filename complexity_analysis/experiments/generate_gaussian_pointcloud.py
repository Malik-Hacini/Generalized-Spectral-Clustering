"""Generate Gaussian point-cloud datasets for the complexity experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import (
    DATA_DIR,
    ensure_dir,
    generate_gaussian_point_cloud,
    log_neighbors_from_n,
    pointcloud_file_name,
    save_pointcloud_dataset,
    parse_int_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", default="500,1000,2000,5000")
    parser.add_argument("--d-values", default="2,10")
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--cluster-std", type=float, default=2.5)
    parser.add_argument("--neighbors-factor", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    rows: list[dict[str, object]] = []
    for n_samples in parse_int_list(args.n_values):
        for n_features in parse_int_list(args.d_values):
            n_neighbors = log_neighbors_from_n(n_samples, factor=args.neighbors_factor)
            X, y = generate_gaussian_point_cloud(
                n_samples=n_samples,
                n_features=n_features,
                n_clusters=args.clusters,
                cluster_std=args.cluster_std,
                seed=args.seed,
                connectivity_n_neighbors=n_neighbors,
            )
            file_name = pointcloud_file_name(n_samples, n_features, args.seed)
            path = out_dir / file_name
            metadata = {
                "n_samples": n_samples,
                "n_features": n_features,
                "n_clusters": args.clusters,
                "cluster_std": args.cluster_std,
                "connectivity_n_neighbors": n_neighbors,
                "seed": args.seed,
            }
            save_pointcloud_dataset(X, y, path, metadata)
            rows.append({"path": str(path), **metadata})

    manifest_path = out_dir / "gaussian_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    (out_dir / "gaussian_manifest.json").write_text(json.dumps(rows, indent=2))
    print(f"Saved {len(rows)} datasets to {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
