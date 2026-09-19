"""Aggregate revision predictions, tail metrics, bootstrap tests, and maps."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize_point_id(value) -> str:
    return str(value).replace(".0", "").strip()


def regression_metrics(target, prediction) -> dict:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(target) & np.isfinite(prediction)
    target, prediction = target[valid], prediction[valid]
    if not len(target):
        return {name: np.nan for name in ("MAE", "RMSE", "R2", "RPIQ", "CCC")}
    error = prediction - target
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    denominator = np.sum((target - target.mean()) ** 2)
    r2 = 1.0 - np.sum(error ** 2) / denominator if denominator > 0 else np.nan
    rpiq = (np.percentile(target, 75) - np.percentile(target, 25)) / rmse if rmse > 0 else np.nan
    covariance = np.mean((target - target.mean()) * (prediction - prediction.mean()))
    ccc_denominator = target.var() + prediction.var() + (target.mean() - prediction.mean()) ** 2
    ccc = 2.0 * covariance / ccc_denominator if ccc_denominator > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "RPIQ": rpiq, "CCC": ccc}


def parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use LABEL=PATH for --method")
    label, path = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("Method label cannot be empty")
    return label.strip(), Path(path)


def prediction_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.rglob("seed_*_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No seed_*_predictions.csv files in {path}")
    return files


def seed_from_path(path: Path, fallback: int) -> int:
    match = re.search(r"seed[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else fallback


def load_prediction(path: Path, source: pd.DataFrame, oc_max: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    id_col = next((name for name in ("point_id", "Point_id", "pointid") if name in frame), None)
    if id_col is None:
        raise ValueError(f"No Point_id column in {path}")
    if "y_pred_denorm" in frame:
        prediction = frame["y_pred_denorm"]
    elif "y_pred_raw" in frame:
        prediction = frame["y_pred_raw"]
    elif "y_pred" in frame:
        prediction = frame["y_pred"] * oc_max
    else:
        raise ValueError(f"No prediction column in {path}")
    result = pd.DataFrame(
        {
            "Point_id": frame[id_col].map(normalize_point_id),
            "prediction": pd.to_numeric(prediction, errors="coerce"),
        }
    )
    return result.merge(source, on="Point_id", how="inner", validate="one_to_one")


def metric_rows(method: str, seed: int, frame: pd.DataFrame) -> list[dict]:
    subsets = {
        "clipped_all": (frame["SOC"].clip(upper=60), frame["prediction"]),
        "original_all": (frame["SOC"], frame["prediction"]),
        "upper_tail_gt60": (
            frame.loc[frame["SOC"] > 60, "SOC"],
            frame.loc[frame["SOC"] > 60, "prediction"],
        ),
    }
    rows = []
    for subset, (target, prediction) in subsets.items():
        metrics = regression_metrics(target, prediction)
        rows.append({"method": method, "seed": seed, "subset": subset, "n": len(target), **metrics})
    return rows


def paired_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    samples: int,
    seed: int,
) -> list[dict]:
    paired = left[["Point_id", "SOC", "prediction"]].merge(
        right[["Point_id", "prediction"]],
        on="Point_id",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    if len(paired) < 2:
        return []
    rng = np.random.default_rng(seed)
    target = paired["SOC"].to_numpy()
    left_prediction = paired["prediction_left"].to_numpy()
    right_prediction = paired["prediction_right"].to_numpy()
    observed_left = regression_metrics(target, left_prediction)
    observed_right = regression_metrics(target, right_prediction)
    distributions = {metric: [] for metric in ("MAE", "RMSE", "R2")}
    for _ in range(samples):
        indices = rng.integers(0, len(paired), len(paired))
        left_metrics = regression_metrics(target[indices], left_prediction[indices])
        right_metrics = regression_metrics(target[indices], right_prediction[indices])
        for metric in distributions:
            distributions[metric].append(right_metrics[metric] - left_metrics[metric])
    rows = []
    for metric, values in distributions.items():
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        p_value = min(1.0, 2.0 * min(np.mean(values <= 0), np.mean(values >= 0)))
        rows.append(
            {
                "metric": metric,
                "n": len(paired),
                "left": observed_left[metric],
                "right": observed_right[metric],
                "difference_right_minus_left": observed_right[metric] - observed_left[metric],
                "ci_low": np.percentile(values, 2.5),
                "ci_high": np.percentile(values, 97.5),
                "p_value_two_sided": p_value,
                "bootstrap_samples": samples,
            }
        )
    return rows


def plot_residual_map(method: str, frame: pd.DataFrame, output_path: Path) -> None:
    residual = frame["prediction"] - frame["SOC"]
    limit = max(1.0, float(np.nanpercentile(np.abs(residual), 95)))
    fig, ax = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    points = ax.scatter(
        frame["Longitude"],
        frame["Latitude"],
        c=residual,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        s=18,
        linewidths=0,
    )
    colorbar = fig.colorbar(points, ax=ax, shrink=0.82)
    colorbar.set_label("Prediction residual (predicted - observed, g/kg)")
    ax.set_xlabel("Longitude (degrees E)")
    ax.set_ylabel("Latitude (degrees N)")
    ax.set_title(f"Spatial residual distribution: {method}")
    ax.grid(alpha=0.2)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", type=parse_method, required=True)
    parser.add_argument("--source-csv", default="dataset/CN-SOC-3500_new.csv")
    parser.add_argument("--output-dir", default="revision_outputs/analysis")
    parser.add_argument("--oc-max", type=float, default=60.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.source_csv)
    source["Point_id"] = source["Point_id"].map(normalize_point_id)
    source = source[["Point_id", "SOC", "Latitude", "Longitude"]]

    metric_records = []
    method_frames = {}
    for method, path in args.method:
        frames = []
        for fallback, prediction_path in enumerate(prediction_files(path), start=1):
            seed = seed_from_path(prediction_path, fallback)
            frame = load_prediction(prediction_path, source, args.oc_max)
            frame["seed"] = seed
            metric_records.extend(metric_rows(method, seed, frame))
            frames.append(frame)
        method_frames[method] = pd.concat(frames, ignore_index=True)

    metrics = pd.DataFrame(metric_records)
    metrics.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    summary = (
        metrics.groupby(["method", "subset"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            n=("n", "first"),
            MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"), R2_std=("R2", "std"),
            RPIQ_mean=("RPIQ", "mean"), RPIQ_std=("RPIQ", "std"),
            CCC_mean=("CCC", "mean"), CCC_std=("CCC", "std"),
        )
    )
    summary.to_csv(output_dir / "summary_mean_std.csv", index=False)

    averaged = {}
    for method, frame in method_frames.items():
        averaged[method] = (
            frame.groupby("Point_id", as_index=False)
            .agg(SOC=("SOC", "first"), Latitude=("Latitude", "first"), Longitude=("Longitude", "first"), prediction=("prediction", "mean"))
        )
        plot_residual_map(method, averaged[method], output_dir / f"residual_map_{method}.png")

    bootstrap_records = []
    for left_name, right_name in itertools.combinations(averaged, 2):
        for row in paired_bootstrap(
            averaged[left_name],
            averaged[right_name],
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        ):
            bootstrap_records.append({"left_method": left_name, "right_method": right_name, **row})
    pd.DataFrame(bootstrap_records).to_csv(output_dir / "paired_bootstrap.csv", index=False)

    metadata = {
        "methods": {name: len(frame["seed"].unique()) for name, frame in method_frames.items()},
        "source_csv": str(Path(args.source_csv).resolve()),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
