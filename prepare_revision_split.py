"""Create and validate the buffered spatial split used for the GRSL revision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import BallTree


EARTH_RADIUS_KM = 6371.0088


def normalize_point_id(value) -> str:
    """Return the filename-compatible representation of a sample ID."""
    return str(value).replace(".0", "").strip()


def available_point_ids(image_dir: Path) -> set[str]:
    return {
        normalize_point_id(path.name.split("_", 1)[0])
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    }


def nearest_distance_km(source: pd.DataFrame, reference: pd.DataFrame) -> np.ndarray:
    """Distance from each source point to its nearest reference point."""
    if source.empty:
        return np.empty(0, dtype=np.float64)
    if reference.empty:
        return np.full(len(source), np.inf, dtype=np.float64)
    source_rad = np.deg2rad(source[["Latitude", "Longitude"]].to_numpy())
    reference_rad = np.deg2rad(reference[["Latitude", "Longitude"]].to_numpy())
    tree = BallTree(reference_rad, metric="haversine")
    distances, _ = tree.query(source_rad, k=1)
    return distances[:, 0] * EARTH_RADIUS_KM


def build_spatial_split(
    samples: pd.DataFrame,
    grid_degrees: float = 3.0,
    buffer_km: float = 50.0,
    seed: int = 317,
) -> pd.DataFrame:
    """Assign geographic grid groups, then enforce pairwise spatial buffers."""
    required = {"Point_id", "Latitude", "Longitude", "SOC"}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if grid_degrees <= 0 or buffer_km < 0:
        raise ValueError("grid_degrees must be positive and buffer_km non-negative")

    result = samples.copy().reset_index(drop=True)
    result["Point_id"] = result["Point_id"].map(normalize_point_id)
    result["spatial_block"] = (
        np.floor(result["Latitude"] / grid_degrees).astype(int).astype(str)
        + "_"
        + np.floor(result["Longitude"] / grid_degrees).astype(int).astype(str)
    )
    indices = np.arange(len(result))
    groups = result["spatial_block"].to_numpy()
    clipped_soc = result["SOC"].clip(upper=60).to_numpy()

    train_idx, holdout_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed).split(
            indices, clipped_soc, groups
        )
    )
    val_rel, test_rel = next(
        GroupShuffleSplit(
            n_splits=1, test_size=0.50, random_state=seed + 10_000
        ).split(
            holdout_idx,
            clipped_soc[holdout_idx],
            groups[holdout_idx],
        )
    )
    val_idx = holdout_idx[val_rel]
    test_idx = holdout_idx[test_rel]

    result["split"] = "excluded_buffer"
    result.loc[train_idx, "split"] = "train"
    result.loc[val_idx, "split"] = "val"
    result.loc[test_idx, "split"] = "test"
    result["nearest_other_split_km"] = np.nan

    train = result.loc[train_idx]
    val = result.loc[val_idx]
    test = result.loc[test_idx]
    train_dist = nearest_distance_km(train, pd.concat([val, test], ignore_index=True))
    val_dist = nearest_distance_km(val, test)
    result.loc[train_idx, "nearest_other_split_km"] = train_dist
    result.loc[val_idx, "nearest_other_split_km"] = val_dist
    result.loc[test_idx, "nearest_other_split_km"] = nearest_distance_km(
        test, pd.concat([train, val], ignore_index=True)
    )
    result.loc[train_idx[train_dist < buffer_km], "split"] = "excluded_buffer"
    result.loc[val_idx[val_dist < buffer_km], "split"] = "excluded_buffer"

    validate_split(result, buffer_km)
    return result


def validate_split(manifest: pd.DataFrame, buffer_km: float) -> dict:
    """Fail if IDs overlap or retained subsets violate the requested buffer."""
    retained = manifest[manifest["split"].isin(["train", "val", "test"])]
    if retained["Point_id"].duplicated().any():
        duplicates = retained.loc[
            retained["Point_id"].duplicated(False), "Point_id"
        ].tolist()
        raise ValueError(f"Point IDs assigned more than once: {duplicates[:10]}")

    subsets = {name: retained[retained["split"] == name] for name in ("train", "val", "test")}
    minimums = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        distances = nearest_distance_km(subsets[left], subsets[right])
        minimum = float(distances.min()) if len(distances) else float("inf")
        minimums[f"{left}_to_{right}_min_km"] = minimum
        if minimum + 1e-6 < buffer_km:
            raise ValueError(
                f"{left}/{right} minimum distance {minimum:.3f} km is below "
                f"the {buffer_km:.3f} km buffer"
            )
    return minimums


def split_summary(manifest: pd.DataFrame, buffer_km: float) -> dict:
    retained = manifest[manifest["split"].isin(["train", "val", "test"])]
    summary = {
        "buffer_km": buffer_km,
        "counts": manifest["split"].value_counts().sort_index().to_dict(),
        "subsets": {},
        "minimum_distances_km": validate_split(manifest, buffer_km),
    }
    for name in ("train", "val", "test"):
        subset = retained[retained["split"] == name]
        summary["subsets"][name] = {
            "count": int(len(subset)),
            "soc_mean": float(subset["SOC"].mean()),
            "soc_median": float(subset["SOC"].median()),
            "soc_gt_60": int((subset["SOC"] > 60).sum()),
        }
    return summary


def plot_split(manifest: pd.DataFrame, output_path: Path) -> None:
    colors = {
        "train": "#1f77b4",
        "val": "#ff7f0e",
        "test": "#d62728",
        "excluded_buffer": "#bdbdbd",
    }
    fig, ax = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    for name in ("excluded_buffer", "train", "val", "test"):
        subset = manifest[manifest["split"] == name]
        ax.scatter(
            subset["Longitude"],
            subset["Latitude"],
            s=8 if name == "excluded_buffer" else 13,
            alpha=0.35 if name == "excluded_buffer" else 0.78,
            color=colors[name],
            label=f"{name} (n={len(subset)})",
            linewidths=0,
        )
    ax.set_xlabel("Longitude (degrees E)")
    ax.set_ylabel("Latitude (degrees N)")
    ax.set_title("Buffered spatial split for the GRSL revision")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="dataset/CN-SOC-3500_new.csv")
    parser.add_argument("--image-dir", default="dataset/l8_images_CN")
    parser.add_argument(
        "--output-dir",
        default="revision_outputs/splits/spatial_3deg_50km_seed317",
    )
    parser.add_argument("--grid-degrees", type=float, default=3.0)
    parser.add_argument("--buffer-km", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=317)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(csv_path)
    samples["Point_id"] = samples["Point_id"].map(normalize_point_id)
    image_ids = available_point_ids(image_dir)
    samples = samples[samples["Point_id"].isin(image_ids)].copy()
    if samples.empty:
        raise ValueError("No CSV samples matched image filenames")

    manifest = build_spatial_split(
        samples,
        grid_degrees=args.grid_degrees,
        buffer_km=args.buffer_km,
        seed=args.seed,
    )
    manifest_path = output_dir / "split_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    summary = split_summary(manifest, args.buffer_km)
    summary.update(
        {
            "csv": str(csv_path.resolve()),
            "image_dir": str(image_dir.resolve()),
            "available_image_ids": len(image_ids),
            "matched_samples": len(samples),
            "grid_degrees": args.grid_degrees,
            "seed": args.seed,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        }
    )
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_split(manifest, output_dir / "split_map.png")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
