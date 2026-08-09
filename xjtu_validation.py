#!/usr/bin/env python3
"""External XJTU-SY validation for the TABDT manuscript.

The script reads the official XJTU-SY archive, extracts a causal vibration
health index, and evaluates three packet-use rules with leave-one-bearing-out
calibration inside each operating condition.  Network repetitions never count
as independent physical specimens.  The two tests shorter than 100 one-minute
records remain in the raw output but are excluded only from the prespecified
long-life sensitivity because a 30-record delivery deadline occupies most of
their evaluation window.  The primary aggregate uses all 15 physical bearings.

Reproduce from the full official archive:

    python3 xjtu_validation.py \
        --archive /path/to/XJTU-SY_Bearing_Datasets.zip

For development only, a previously extracted feature cache can be supplied
with --features.  The submission package intentionally excludes raw data and
feature caches.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1786050000")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tabdt-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARCHIVE_NAME = "XJTU-SY_Bearing_Datasets.zip"
ARCHIVE_SHA256 = "3cc815649a315ac7da202980c489f33db44ca2db0317bbe3bcb9dcf415375e10"
EXPECTED_RECORDS = 9216
EXPECTED_SAMPLES = 32768
BASE_SEED = 183
NETWORK_P = 0.02
DEADLINE = 30
NETWORK_REPETITIONS = 40
LONG_LIFE_MIN_RECORDS = 100
HEALTH_CAP = 1.2
EVAL_LO = 0.2
EVAL_HI = 0.9
Z90 = 1.645
BOOTSTRAP_REPETITIONS = 20000
BOOTSTRAP_SEED = 20260806

# Four bearings calibrate each held-out fold.  Predicting the random effect of
# a new bearing from their mean adds Var(b_new - mean(b_train)) =
# (1 + 1/m) sigma_b^2.  This yields 1.25 for m = 4 without fitting a multiplier
# to the held-out bearing or to packet simulations.
TRAINING_BEARINGS_PER_FOLD = 4
PREDICTIVE_BETWEEN_FACTOR = 1.0 + 1.0 / TRAINING_BEARINGS_PER_FOLD
INTERVAL_SENSITIVITY_FACTORS = (1.0, PREDICTIVE_BETWEEN_FACTOR, 1.5, 2.0)

METHODS = ("B2", "B3", "TABDT")
METHOD_LABELS = {
    "B2": "B2: stale treated as current",
    "B3": "B3: delayed packets discarded",
    "TABDT": "TABDT",
}
COLORS = {"B2": "#D62728", "B3": "#1565C0", "TABDT": "#009E49"}
MARKERS = {"B2": "s", "B3": "o", "TABDT": "D"}


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def bearing_sort_key(name: str) -> tuple[int, int]:
    nums = tuple(int(v) for v in re.findall(r"\d+", name))
    if len(nums) != 2:
        raise ValueError(f"Unexpected bearing identifier: {name}")
    return nums


def extract_features(archive: Path) -> dict[str, np.ndarray]:
    """Return one horizontal peak amplitude per one-minute CSV record."""
    archive_digest = sha256_file(archive)
    if archive_digest != ARCHIVE_SHA256:
        raise ValueError(
            f"Archive SHA-256 mismatch: {archive_digest}; expected {ARCHIVE_SHA256}"
        )

    pattern = re.compile(
        r"XJTU-SY_Bearing_Datasets/[^/]+/(Bearing\d_\d+)/(\d+)\.csv$"
    )
    grouped: dict[str, list[tuple[int, str]]] = {}
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            match = pattern.match(member)
            if match:
                grouped.setdefault(match.group(1), []).append(
                    (int(match.group(2)), member)
                )

        features: dict[str, np.ndarray] = {}
        for bearing in sorted(grouped, key=bearing_sort_key):
            rows: list[tuple[int, float]] = []
            for record_index, member in sorted(grouped[bearing]):
                payload = bundle.read(member)
                try:
                    body = payload.split(b"\n", 1)[1]
                except IndexError as exc:
                    raise ValueError(f"Missing CSV header/body in {member}") from exc
                # The official archive contains both LF and CRLF records.
                body = body.replace(b"\r\n", b",").replace(b"\n", b",")
                values = np.fromstring(body.decode("ascii"), sep=",")
                if values.size != 2 * EXPECTED_SAMPLES:
                    raise ValueError(
                        f"{member} contains {values.size} values; "
                        f"expected {2 * EXPECTED_SAMPLES}"
                    )
                signal = values.reshape(EXPECTED_SAMPLES, 2)
                horizontal_peak = float(np.max(np.abs(signal[:, 0])))
                rows.append((record_index, horizontal_peak))
            features[bearing] = np.asarray(rows, dtype=np.float64)

    count = sum(len(v) for v in features.values())
    if len(features) != 15 or count != EXPECTED_RECORDS:
        raise ValueError(
            f"Unexpected archive inventory: {len(features)} bearings, {count} records"
        )
    return features


def load_feature_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        features = {name: np.asarray(bundle[name], dtype=float) for name in bundle.files}
    count = sum(len(v) for v in features.values())
    if len(features) != 15 or count != EXPECTED_RECORDS:
        raise ValueError(
            f"Unexpected cache inventory: {len(features)} bearings, {count} records"
        )
    for bearing, values in features.items():
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError(f"Invalid feature array for {bearing}: {values.shape}")
    return features


def horizontal_peak_column(values: np.ndarray) -> np.ndarray:
    """Support the compact two-column cache and the five-column audit cache."""
    if values.shape[1] >= 4:
        return values[:, 3]
    return values[:, 1]


def health_trace(values: np.ndarray, baseline: float) -> np.ndarray:
    peaks = horizontal_peak_column(values)
    # This is a shared causal proxy, not a claim that every bearing failed at
    # exactly this transformed level.  It maps a condition baseline to one and
    # ten times that baseline to the normalized endpoint.
    health = (peaks / baseline - 1.0) / 9.0
    return np.clip(health, 0.0, HEALTH_CAP)


def calibration_terms(
    train_names: list[str], features: dict[str, np.ndarray], baseline: float
) -> tuple[float, float, float, float, float]:
    traces = [health_trace(features[name], baseline) for name in train_names]
    slopes = np.asarray([trace[-1] / (len(trace) - 1) for trace in traces])
    drift = float(np.mean(slopes))
    drift_variance = float(np.var(slopes, ddof=1))
    process_variance = float(
        np.var(np.concatenate([np.diff(trace) for trace in traces]), ddof=1)
    )

    residuals: list[np.ndarray] = []
    for trace in traces:
        lifetime = len(trace) - 1
        time = np.arange(len(trace))
        mask = (time >= EVAL_LO * lifetime) & (time <= EVAL_HI * lifetime)
        true_rul = (lifetime - time[mask]) / lifetime
        residuals.append(np.maximum(0.0, 1.0 - trace[mask]) - true_rul)
    within = float(np.mean([np.var(r, ddof=1) for r in residuals]))
    between = float(np.var([np.mean(r) for r in residuals], ddof=1))
    return drift, drift_variance, process_variance, within, between


def network_delays(rng: np.random.Generator, length: int) -> np.ndarray:
    uniforms = rng.random(length)
    delays = np.floor(
        np.log1p(-uniforms) / np.log1p(-NETWORK_P)
    ).astype(np.int32)
    return np.where(delays <= DEADLINE, delays, DEADLINE + 1)


def simulate_fold(
    bearing: str,
    features: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, object]:
    condition = bearing_sort_key(bearing)[0]
    train_names = [
        name
        for name in sorted(features, key=bearing_sort_key)
        if bearing_sort_key(name)[0] == condition and name != bearing
    ]
    if len(train_names) != 4:
        raise ValueError(f"Expected four calibration bearings for {bearing}")

    baseline = float(
        np.median(
            np.concatenate(
                [horizontal_peak_column(features[name])[:1] for name in train_names]
            )
        )
    )
    health = health_trace(features[bearing], baseline)
    drift, drift_var, process_var, residual_within, residual_between = calibration_terms(
        train_names, features, baseline
    )
    residual_variances = {
        f"{factor:g}": residual_within + factor * residual_between
        for factor in INTERVAL_SENSITIVITY_FACTORS
    }
    primary_factor_key = f"{PREDICTIVE_BETWEEN_FACTOR:g}"
    residual_var = residual_variances[primary_factor_key]

    length = len(health)
    lifetime = length - 1
    time = np.arange(length)
    evaluation = (time >= EVAL_LO * lifetime) & (time <= EVAL_HI * lifetime)
    true_rul = (lifetime - time) / lifetime

    squared_error = {method: 0.0 for method in METHODS}
    interval_hits_by_factor = {
        factor_key: 0 for factor_key in residual_variances
    }
    count = 0
    for _ in range(NETWORK_REPETITIONS):
        delays = network_delays(rng, length)
        arrivals: list[list[tuple[int, int]]] = [[] for _ in range(length)]
        for generated_at, age in enumerate(delays):
            arrives_at = generated_at + int(age)
            if age <= DEADLINE and arrives_at < length:
                arrivals[arrives_at].append((generated_at, int(age)))

        state = {method: 0.0 for method in METHODS}
        latest_generation = -1
        for now in range(length):
            if now > 0:
                state["TABDT"] = float(
                    np.clip(state["TABDT"] + drift, 0.0, HEALTH_CAP)
                )
            for generated_at, age in arrivals[now]:
                state["B2"] = float(health[generated_at])
                if age == 0:
                    state["B3"] = float(health[generated_at])
                state["TABDT"] = float(
                    np.clip(health[generated_at] + drift * age, 0.0, HEALTH_CAP)
                )
                latest_generation = generated_at

            if not evaluation[now]:
                continue
            for method in METHODS:
                estimate = max(0.0, 1.0 - state[method])
                squared_error[method] += (estimate - true_rul[now]) ** 2

            current_age = now - latest_generation if latest_generation >= 0 else now
            tab_error = abs(max(0.0, 1.0 - state["TABDT"]) - true_rul[now])
            dynamic_variance = current_age * process_var + current_age**2 * drift_var
            for factor_key, base_variance in residual_variances.items():
                variance = base_variance + dynamic_variance
                half_width = Z90 * np.sqrt(max(variance, 0.0))
                interval_hits_by_factor[factor_key] += int(tab_error <= half_width)
            count += 1

    return {
        "bearing": bearing,
        "condition": condition,
        "records": length,
        "eligible_long_life_sensitivity": length >= LONG_LIFE_MIN_RECORDS,
        "calibration_bearings": train_names,
        "baseline_horizontal_peak": baseline,
        "calibrated_health_drift": drift,
        "calibrated_drift_variance": drift_var,
        "calibrated_process_variance": process_var,
        "calibrated_residual_within_variance": residual_within,
        "calibrated_residual_between_variance": residual_between,
        "calibrated_residual_variance": residual_var,
        "predictive_between_bearing_factor": PREDICTIVE_BETWEEN_FACTOR,
        "evaluation_count": count,
        "squared_error": squared_error,
        "rmse_percentage_points": {
            method: 100.0 * np.sqrt(squared_error[method] / count)
            for method in METHODS
        },
        "tabdt_interval_hits": interval_hits_by_factor[primary_factor_key],
        "tabdt_interval_coverage": interval_hits_by_factor[primary_factor_key] / count,
        "tabdt_interval_hits_by_between_factor": interval_hits_by_factor,
        "tabdt_interval_coverage_by_between_factor": {
            factor_key: hits / count
            for factor_key, hits in interval_hits_by_factor.items()
        },
    }


def clustered_bootstrap(
    folds: list[dict[str, object]],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    counts = np.asarray([fold["evaluation_count"] for fold in folds], dtype=float)
    sse = np.asarray(
        [[fold["squared_error"][method] for method in METHODS] for fold in folds],
        dtype=float,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.empty((BOOTSTRAP_REPETITIONS, len(METHODS)))
    for index in range(BOOTSTRAP_REPETITIONS):
        sample = rng.integers(0, len(folds), len(folds))
        boot[index] = 100.0 * np.sqrt(
            np.sum(sse[sample], axis=0) / np.sum(counts[sample])
        )
    ci = {
        method: np.percentile(boot[:, j], [2.5, 97.5]).tolist()
        for j, method in enumerate(METHODS)
    }
    reductions = {
        "B2_to_TABDT": np.percentile(
            100.0 * (boot[:, 0] - boot[:, 2]) / boot[:, 0], [2.5, 97.5]
        ).tolist(),
        "B3_to_TABDT": np.percentile(
            100.0 * (boot[:, 1] - boot[:, 2]) / boot[:, 1], [2.5, 97.5]
        ).tolist(),
        "B2_minus_TABDT": np.percentile(
            boot[:, 0] - boot[:, 2], [2.5, 97.5]
        ).tolist(),
        "B3_minus_TABDT": np.percentile(
            boot[:, 1] - boot[:, 2], [2.5, 97.5]
        ).tolist(),
    }
    return ci, reductions


def summarize(
    folds: list[dict[str, object]], archive_digest: str
) -> dict[str, object]:
    primary_folds = folds
    long_life_folds = [
        fold for fold in folds if fold["eligible_long_life_sensitivity"]
    ]
    short_life_bearings = [
        fold["bearing"] for fold in folds if not fold["eligible_long_life_sensitivity"]
    ]
    total_count = sum(int(fold["evaluation_count"]) for fold in primary_folds)
    total_sse = {
        method: sum(float(fold["squared_error"][method]) for fold in primary_folds)
        for method in METHODS
    }
    rmse = {
        method: 100.0 * np.sqrt(total_sse[method] / total_count)
        for method in METHODS
    }
    primary_factor_key = f"{PREDICTIVE_BETWEEN_FACTOR:g}"
    hits_by_factor = {
        factor_key: sum(
            int(fold["tabdt_interval_hits_by_between_factor"][factor_key])
            for fold in primary_folds
        )
        for factor_key in (f"{factor:g}" for factor in INTERVAL_SENSITIVITY_FACTORS)
    }
    ci, reduction_ci = clustered_bootstrap(primary_folds)
    reduction_b2 = 100.0 * (rmse["B2"] - rmse["TABDT"]) / rmse["B2"]
    reduction_b3 = 100.0 * (rmse["B3"] - rmse["TABDT"]) / rmse["B3"]

    long_count = sum(int(fold["evaluation_count"]) for fold in long_life_folds)
    long_rmse = {
        method: 100.0
        * np.sqrt(
            sum(float(fold["squared_error"][method]) for fold in long_life_folds)
            / long_count
        )
        for method in METHODS
    }
    long_hits = sum(int(fold["tabdt_interval_hits"]) for fold in long_life_folds)
    return {
        "dataset": {
            "name": "XJTU-SY Bearing Datasets",
            "archive_name": ARCHIVE_NAME,
            "archive_sha256": archive_digest,
            "bearings": len(folds),
            "records": EXPECTED_RECORDS,
            "samples_per_record": EXPECTED_SAMPLES,
            "channels": 2,
            "feature_channel": "horizontal vibration",
            "feature": "peak absolute amplitude",
        },
        "protocol": {
            "base_seed": BASE_SEED,
            "network_probability_timely": NETWORK_P,
            "delivery_deadline_records": DEADLINE,
            "network_repetitions_per_bearing": NETWORK_REPETITIONS,
            "evaluation_life_fraction": [EVAL_LO, EVAL_HI],
            "calibration": "leave one bearing out within operating condition",
            "training_bearings_per_fold": TRAINING_BEARINGS_PER_FOLD,
            "health_definition": "tenfold condition-baseline proxy: clip((horizontal_peak/baseline)-1)/9 to [0,1.2]",
            "baseline_definition": "median first-record peak of four calibration bearings",
            "primary_bearings": "all 15 physical bearings",
            "minimum_records_long_life_sensitivity": LONG_LIFE_MIN_RECORDS,
            "predictive_between_bearing_variance_factor": PREDICTIVE_BETWEEN_FACTOR,
            "predictive_factor_derivation": "1 + 1/m for a new bearing versus the mean of m=4 calibration bearings",
            "interval_sensitivity_between_factors": list(INTERVAL_SENSITIVITY_FACTORS),
            "bootstrap_unit": "bearing",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "primary": {
            "bearings": len(primary_folds),
            "evaluation_count": total_count,
            "rmse_percentage_points": rmse,
            "rmse_cluster_bootstrap_95_ci": ci,
            "relative_reduction_percent": {
                "versus_B2": reduction_b2,
                "versus_B3": reduction_b3,
            },
            "paired_cluster_bootstrap_95_ci": reduction_ci,
            "tabdt_nominal_interval_percent": 90.0,
            "tabdt_interval_coverage_percent": 100.0
            * hits_by_factor[primary_factor_key]
            / total_count,
            "tabdt_interval_coverage_sensitivity_percent": {
                factor_key: 100.0 * hits / total_count
                for factor_key, hits in hits_by_factor.items()
            },
        },
        "long_life_sensitivity": {
            "minimum_records": LONG_LIFE_MIN_RECORDS,
            "bearings": len(long_life_folds),
            "excluded_short_bearings": short_life_bearings,
            "evaluation_count": long_count,
            "rmse_percentage_points": long_rmse,
            "tabdt_interval_coverage_percent": 100.0 * long_hits / long_count,
        },
        "all_folds": folds,
    }


def plot_primary(results: dict[str, object], output_dir: Path) -> None:
    primary = results["primary"]
    rmse = primary["rmse_percentage_points"]
    ci = primary["rmse_cluster_bootstrap_95_ci"]
    y = np.asarray([rmse[method] for method in METHODS])
    low = y - np.asarray([ci[method][0] for method in METHODS])
    high = np.asarray([ci[method][1] for method in METHODS]) - y

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.32,
            "grid.color": "#C9D2DC",
            "grid.linewidth": 0.45,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.52, 2.42))
    x = np.arange(len(METHODS))
    for j, method in enumerate(METHODS):
        ax.errorbar(
            j,
            y[j],
            yerr=np.asarray([[low[j]], [high[j]]]),
            fmt="none",
            ecolor=COLORS[method],
            elinewidth=1.1,
            capsize=3.0,
            capthick=1.0,
            zorder=2,
        )
        ax.scatter(
            j,
            y[j],
            s=34,
            marker=MARKERS[method],
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        ax.text(j, y[j] + 1.05, f"{y[j]:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(["B2", "B3", "TABDT"])
    ax.set_ylabel("normalized RUL RMSE [percentage points]")
    ax.set_ylim(34, 60)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.spines["left"].set_color("#566573")
    ax.spines["bottom"].set_color("#566573")
    ax.annotate(
        f"{primary['relative_reduction_percent']['versus_B2']:.1f}% below B2\n"
        f"{primary['relative_reduction_percent']['versus_B3']:.1f}% below B3",
        xy=(2, y[2]),
        xytext=(1.36, 38.2),
        fontsize=6.6,
        color=COLORS["TABDT"],
        ha="center",
        arrowprops={"arrowstyle": "->", "lw": 0.75, "color": COLORS["TABDT"]},
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#B8C2CC", "lw": 0.5},
    )
    fig.tight_layout(pad=0.4)

    figure_dir = output_dir / "figs"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fixed_time = datetime(2026, 8, 6, 20, 30, tzinfo=timezone.utc)
    fig.savefig(
        figure_dir / "fig6_xjtu.pdf",
        bbox_inches="tight",
        pad_inches=0.025,
        metadata={
            "Title": "XJTU-SY external RUL validation",
            "Author": "Yassir Ameen Al-Karawi and Hamed Al-Raweshidy",
            "Creator": "TABDT reproducibility package",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    fig.savefig(
        figure_dir / "fig6_xjtu.png",
        bbox_inches="tight",
        pad_inches=0.025,
        dpi=400,
        metadata={"Software": "TABDT reproducibility package"},
    )
    plt.close(fig)


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Official XJTU-SY ZIP archive")
    parser.add_argument("--features", type=Path, help="Development feature cache")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.archive is None and args.features is None:
        parser.error("provide --archive or --features")

    archive_digest = ARCHIVE_SHA256
    if args.archive is not None:
        archive_digest = sha256_file(args.archive)
        if archive_digest != ARCHIVE_SHA256:
            raise ValueError(
                f"Archive SHA-256 mismatch: {archive_digest}; expected {ARCHIVE_SHA256}"
            )
    features = (
        load_feature_cache(args.features)
        if args.features is not None
        else extract_features(args.archive)
    )
    ordered_names = sorted(features, key=bearing_sort_key)
    long_life_names = [
        name for name in ordered_names if len(features[name]) >= LONG_LIFE_MIN_RECORDS
    ]
    short_life_names = [
        name for name in ordered_names if len(features[name]) < LONG_LIFE_MIN_RECORDS
    ]
    # Separate streams preserve the established long-life random realizations while
    # making the all-bearing aggregate primary in v18.
    long_life_rng = np.random.default_rng(BASE_SEED)
    short_life_rng = np.random.default_rng(BASE_SEED + 1000000)
    folds = [simulate_fold(name, features, long_life_rng) for name in long_life_names]
    folds.extend(simulate_fold(name, features, short_life_rng) for name in short_life_names)
    folds.sort(key=lambda fold: bearing_sort_key(str(fold["bearing"])))
    results = summarize(folds, archive_digest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "xjtu_results.json"
    result_path.write_text(
        json.dumps(json_ready(results), indent=1, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_primary(results, args.output_dir)

    primary = results["primary"]
    rmse = primary["rmse_percentage_points"]
    print(
        "[XJTU] B2={:.2f}, B3={:.2f}, TABDT={:.2f} percentage points".format(
            rmse["B2"], rmse["B3"], rmse["TABDT"]
        )
    )
    print(
        "[XJTU] reductions={:.1f}%/{:.1f}%, coverage={:.1f}%".format(
            primary["relative_reduction_percent"]["versus_B2"],
            primary["relative_reduction_percent"]["versus_B3"],
            primary["tabdt_interval_coverage_percent"],
        )
    )


if __name__ == "__main__":
    main()
