"""Run resumable GRSL revision experiments one seed at a time."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


CORE_CONFIGS = {
    "concat": ["--revision_fusion", "concat"],
    "sap_rf_no_align": [
        "--revision_fusion", "sap_rf", "--scmrl_lambda_align", "0.0"
    ],
    "sap_rf": [
        "--revision_fusion", "sap_rf", "--scmrl_lambda_align", "0.1"
    ],
    "gated": ["--revision_fusion", "gated"],
    "cross_attention": ["--revision_fusion", "cross_attention"],
}

SENSITIVITY_CONFIGS = {
    "alpha_0": ["--revision_fusion", "sap_rf", "--scmrl_alpha_init", "0.0"],
    "alpha_0p5": ["--revision_fusion", "sap_rf", "--scmrl_alpha_init", "0.5"],
    "alpha_1": ["--revision_fusion", "sap_rf", "--scmrl_alpha_init", "1.0"],
    "lambda_0p01": ["--revision_fusion", "sap_rf", "--scmrl_lambda_align", "0.01"],
    "lambda_1": ["--revision_fusion", "sap_rf", "--scmrl_lambda_align", "1.0"],
    "tau_0p03": ["--revision_fusion", "sap_rf", "--scmrl_temperature", "0.03"],
    "tau_0p2": ["--revision_fusion", "sap_rf", "--scmrl_temperature", "0.2"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", choices=["core", "sensitivity", "all"], default="core")
    parser.add_argument("--only", nargs="*", default=None, help="Configuration names to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument(
        "--split-manifest",
        default="revision_outputs/splits/spatial_3deg_50km_seed317/split_manifest.csv",
    )
    parser.add_argument("--image-root", default="dataset/l8_images_CN")
    parser.add_argument("--output-root", default="revision_outputs")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_configs(args: argparse.Namespace) -> dict[str, list[str]]:
    configs = {}
    if args.matrix in ("core", "all"):
        configs.update(CORE_CONFIGS)
    if args.matrix in ("sensitivity", "all"):
        configs.update(SENSITIVITY_CONFIGS)
    if args.only:
        unknown = set(args.only) - set(configs)
        if unknown:
            raise ValueError(f"Unknown configurations: {sorted(unknown)}")
        configs = {name: configs[name] for name in args.only}
    return configs


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    logs_dir = output_root / "logs"
    completed_dir = output_root / "completed"
    logs_dir.mkdir(parents=True, exist_ok=True)
    completed_dir.mkdir(parents=True, exist_ok=True)

    for config_name, config_args in selected_configs(args).items():
        seeds = [args.seeds[0]] if config_name in SENSITIVITY_CONFIGS else args.seeds
        for seed in seeds:
            job_name = f"grsl_{config_name}_seed{seed}"
            marker = completed_dir / f"{job_name}.json"
            if marker.exists():
                print(f"[skip] {job_name}: verified completion marker exists")
                continue
            run_output = output_root / "runs" / config_name
            command = [
                sys.executable,
                "train.py",
                "--exp_name", job_name,
                "--dataset", "CHINA",
                "--num_workers", str(args.num_workers),
                "--train_batch_size", "32",
                "--test_batch_size", "4",
                "--learning_rate", "0.0001",
                "--num_epochs", str(args.epochs),
                "--lr_scheduler", "step",
                "--use_srtm",
                "--cnn_architecture", "ViT-CoMer",
                "--rnn_architecture", "Transformer",
                "--use_static",
                "--static_csv", "dataset/CN-SOC-3500_new.csv",
                "--seeds", str(seed),
                "--scmrl_alpha_init", "2.0",
                "--scmrl_lambda_align", "0.1",
                "--scmrl_temperature", "0.07",
                "--split_manifest", args.split_manifest,
                "--image_root", args.image_root,
                "--revision_output_dir", str(run_output),
                *config_args,
            ]
            print("[run] " + subprocess.list2cmdline(command))
            if args.dry_run:
                continue
            started = datetime.now().isoformat(timespec="seconds")
            log_path = logs_dir / f"{job_name}.log"
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parent,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if completed.returncode != 0:
                raise RuntimeError(f"{job_name} failed; inspect {log_path}")
            run_dirs = sorted(
                run_output.glob(f"{job_name}_D_*"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            verified_run_dir = next(
                (
                    path for path in run_dirs
                    if (path / "run_summary.json").is_file()
                    and (path / f"seed_{seed}_predictions.csv").is_file()
                    and (path / f"seed_{seed}_metadata.json").is_file()
                ),
                None,
            )
            if verified_run_dir is None:
                raise RuntimeError(
                    f"{job_name} exited successfully but required revision artifacts are missing; "
                    f"inspect {log_path}"
                )
            marker.write_text(
                json.dumps(
                    {
                        "job": job_name,
                        "configuration": config_name,
                        "seed": seed,
                        "started": started,
                        "finished": datetime.now().isoformat(timespec="seconds"),
                        "command": command,
                        "log": str(log_path.resolve()),
                        "run_dir": str(verified_run_dir.resolve()),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
