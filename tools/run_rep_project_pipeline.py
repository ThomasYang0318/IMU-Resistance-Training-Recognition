from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


METHOD_DIR_NAMES = {
    "labels": "labels",
    "dominant-axis": "dominant_axis",
    "short-time-energy": "short_time_energy",
    "pca-extrema": "pca_extrema",
    "pca-autocorr": "pca_autocorr",
    "pca-extrema-fft": "pca_extrema_fft",
}

DEFAULT_METHODS = [
    "labels",
    "dominant-axis",
    "short-time-energy",
    "pca-extrema",
    "pca-autocorr",
    "pca-extrema-fft",
]

DEFAULT_COMPARISON_METHODS = [
    "dominant-axis",
    "short-time-energy",
    "pca-extrema",
    "pca-autocorr",
    "pca-extrema-fft",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def method_output_dir(output_root: Path, method: str, num_classes: int, folds: int, include_other: bool) -> Path:
    displayed_classes = num_classes + 1 if include_other else num_classes
    return output_root / f"{METHOD_DIR_NAMES[method]}_{displayed_classes}class_{folds}fold"


def run_command(command: list[str], *, cwd: Path, dry_run: bool) -> None:
    print("$ " + shlex.join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def add_evaluation_args(command: list[str], args: argparse.Namespace) -> list[str]:
    command.extend(
        [
            "--data-dirs",
            *[str(path) for path in args.data_dirs],
            "--folds",
            str(args.folds),
            "--num-classes",
            str(args.num_classes),
            "--block-source",
            args.block_source,
            "--min-segment-samples",
            str(args.min_segment_samples),
            "--smooth-window",
            str(args.smooth_window),
            "--peak-prominence-scale",
            str(args.peak_prominence_scale),
            "--peak-distance-scale",
            str(args.peak_distance_scale),
            "--fft-min-period-samples",
            str(args.fft_min_period_samples),
            "--fft-max-period-fraction",
            str(args.fft_max_period_fraction),
            "--fft-peak-distance-scale",
            str(args.fft_peak_distance_scale),
            "--autocorr-min-period-samples",
            str(args.autocorr_min_period_samples),
            "--autocorr-max-period-fraction",
            str(args.autocorr_max_period_fraction),
            "--autocorr-peak-distance-scale",
            str(args.autocorr_peak_distance_scale),
            "--min-label-iou",
            str(args.min_label_iou),
            "--segmentation-iou-thresholds",
            *[str(value) for value in args.segmentation_iou_thresholds],
            "--seed",
            str(args.seed),
        ]
    )
    if args.include_other:
        command.append("--include-other")
    return command


def run_evaluations(args: argparse.Namespace, root: Path) -> None:
    for method in args.methods:
        output_dir = method_output_dir(args.output_root, method, args.num_classes, args.folds, args.include_other)
        command = [
            sys.executable,
            "tools/evaluate_rep_segmentation_classification.py",
            "--output-dir",
            str(output_dir),
            "--segment-method",
            method,
        ]
        run_command(add_evaluation_args(command, args), cwd=root, dry_run=args.dry_run)


def run_method_comparison(args: argparse.Namespace, root: Path, methods: list[str]) -> None:
    command = [
        sys.executable,
        "tools/compare_rep_segmentation_iou.py",
        "--output-dir",
        str(args.output_root / "methods_comparison"),
        "--focus-iou",
        str(args.focus_iou),
    ]
    for method in methods:
        command.extend(
            [
                "--run",
                f"{method}={method_output_dir(args.output_root, method, args.num_classes, args.folds, args.include_other)}",
            ]
        )
    run_command(command, cwd=root, dry_run=args.dry_run)


def run_waveform_comparison(args: argparse.Namespace, root: Path, methods: list[str]) -> None:
    command = [
        sys.executable,
        "tools/plot_rep_waveform_method_comparison.py",
        "--output-dir",
        str(args.output_root / "waveform_method_comparison"),
        "--window-reps",
        str(args.window_reps),
        "--min-set-reps",
        str(args.min_set_reps),
        "--set-padding-fraction",
        str(args.set_padding_fraction),
    ]
    for method in methods:
        command.extend(
            [
                "--run",
                f"{method}={method_output_dir(args.output_root, method, args.num_classes, args.folds, args.include_other)}",
            ]
        )
    if args.plot_all_sets:
        command.append("--plot-all-sets")
    if args.max_sets is not None:
        command.extend(["--max-sets", str(args.max_sets)])
    run_command(command, cwd=root, dry_run=args.dry_run)


def run_set_level_plots(args: argparse.Namespace, root: Path) -> None:
    waveform_dir = args.output_root / "waveform_method_comparison"
    command = [
        sys.executable,
        "tools/plot_set_level_method_results.py",
        "--summary",
        str(waveform_dir / "waveform_method_all_sets_summary.csv"),
        "--output-dir",
        str(waveform_dir / "set_level_results"),
    ]
    run_command(command, cwd=root, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full rep segmentation, classification, and comparison pipeline."
    )
    parser.add_argument("--data-dirs", type=Path, nargs="+", default=[Path("datasets/workout")])
    parser.add_argument("--output-root", type=Path, default=Path("artifacts_rep_classification"))
    parser.add_argument("--methods", choices=DEFAULT_METHODS, nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--comparison-methods", choices=DEFAULT_METHODS, nargs="+")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--include-other", action="store_true")
    parser.add_argument("--block-source", choices=["action-label", "active-phase-span"], default="action-label")
    parser.add_argument("--min-segment-samples", type=int, default=20)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--peak-prominence-scale", type=float, default=0.35)
    parser.add_argument("--peak-distance-scale", type=float, default=3.0)
    parser.add_argument("--fft-min-period-samples", type=int, default=25)
    parser.add_argument("--fft-max-period-fraction", type=float, default=0.8)
    parser.add_argument("--fft-peak-distance-scale", type=float, default=1.2)
    parser.add_argument("--autocorr-min-period-samples", type=int, default=25)
    parser.add_argument("--autocorr-max-period-fraction", type=float, default=0.8)
    parser.add_argument("--autocorr-peak-distance-scale", type=float, default=0.75)
    parser.add_argument("--min-label-iou", type=float, default=0.25)
    parser.add_argument("--segmentation-iou-thresholds", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--focus-iou", type=float, default=0.5)
    parser.add_argument("--window-reps", type=int, default=10)
    parser.add_argument("--plot-all-sets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-set-reps", type=int, default=1)
    parser.add_argument("--set-padding-fraction", type=float, default=0.15)
    parser.add_argument("--max-sets", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-method-comparison", action="store_true")
    parser.add_argument("--skip-waveform-comparison", action="store_true")
    parser.add_argument("--skip-set-level-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    comparison_methods = args.comparison_methods or [
        method for method in DEFAULT_COMPARISON_METHODS if method in args.methods
    ]
    needs_comparison_methods = not args.skip_method_comparison or not args.skip_waveform_comparison
    if needs_comparison_methods and not comparison_methods:
        raise ValueError("No comparison methods selected.")

    if not args.skip_evaluation:
        run_evaluations(args, root)
    if not args.skip_method_comparison:
        run_method_comparison(args, root, comparison_methods)
    if not args.skip_waveform_comparison:
        run_waveform_comparison(args, root, comparison_methods)
    if not args.skip_set_level_plots:
        run_set_level_plots(args, root)


if __name__ == "__main__":
    main()
