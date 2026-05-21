#!/usr/bin/env python3
"""Model diagnosis for CFM trajectory checkpoints.

The script evaluates calibration, PIT, coverage, and score summaries on a
cached test split and writes the raw tables and plots needed for inspection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    # look for common repo markers up the tree
    for d in [p] + list(p.parents):
        if (
            (d / ".git").exists()
            or (d / "pyproject.toml").exists()
            or (d / "README.md").exists()
        ):
            return d
    return p.parent


ROOT = _find_repo_root(Path(__file__))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import load_model_checkpoint  # noqa: E402
from utils.inference_utils import denorm_seq_to_global, sample_many  # noqa: E402


# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------


def find_latest_key(cache_dir: Path) -> Path:
    keys = sorted(
        cache_dir.glob("*.key.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not keys:
        raise FileNotFoundError(f"No .key.json files found in {cache_dir}")
    return keys[0]


def strip_key_suffix(path: Path) -> str:
    name = path.name
    return name[: -len(".key.json")] if name.endswith(".key.json") else path.stem


def find_norm_stats(cache_dir: Path, key_id: str) -> Path:
    candidates = [
        cache_dir / f"{key_id}.norm_stats.json",
        cache_dir / f"{key_id}.key.norm_stats.json",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    hits = sorted(
        cache_dir.glob("*.norm_stats.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not hits:
        raise FileNotFoundError(f"No norm_stats json found in {cache_dir}")
    return hits[0]


def load_cache_bundle(cache_dir: Path):
    key_path = find_latest_key(cache_dir)
    key_id = strip_key_suffix(key_path)
    paths = {
        "x_te": cache_dir / f"{key_id}.X_test.npy",
        "y_te": cache_dir / f"{key_id}.Y_test.npy",
        "c_te": cache_dir / f"{key_id}.C_test.npy",
        "meta_te": cache_dir / f"{key_id}.meta_test.parquet",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")
    X_test = np.load(paths["x_te"], mmap_mode="r")
    Y_test = np.load(paths["y_te"], mmap_mode="r")
    C_test = np.load(paths["c_te"], mmap_mode="r")
    meta_test = pd.read_parquet(paths["meta_te"])
    return key_id, X_test, Y_test, C_test, meta_test


def load_norm_stats(cache_dir: Path, key_id: str):
    path = find_norm_stats(cache_dir, key_id)
    ns = json.loads(path.read_text())
    return (
        path,
        np.asarray(ns["feat_mean"], dtype=np.float32),
        np.asarray(ns["feat_std"], dtype=np.float32),
        np.asarray(ns["ctx_mean"], dtype=np.float32),
        np.asarray(ns["ctx_std"], dtype=np.float32),
    )


def safe_flight_ids(meta_test: pd.DataFrame, idxs: np.ndarray) -> np.ndarray:
    if "flight_id" in meta_test.columns:
        return meta_test.iloc[idxs]["flight_id"].astype(str).to_numpy()
    return np.asarray([str(int(i)) for i in idxs], dtype=str)


def load_key_config(cache_dir: Path, key_id: str) -> dict:
    key_path = cache_dir / f"{key_id}.key.json"
    if not key_path.exists():
        return {}
    return json.loads(key_path.read_text())


def turn_masks(meta: pd.DataFrame) -> Dict[str, np.ndarray]:
    if "has_turn_hist" not in meta.columns or "has_turn_fut" not in meta.columns:
        return {}
    hist = meta["has_turn_hist"].to_numpy(dtype=bool)
    fut = meta["has_turn_fut"].to_numpy(dtype=bool)
    return {
        "no_turn": ~(hist | fut),
        "any_turn": hist | fut,
        "hist_turn": hist,
        "fut_turn": fut,
        "both_turn": hist & fut,
        "hist_only": hist & ~fut,
        "fut_only": ~hist & fut,
    }


def select_subset_indices(
    meta_test: pd.DataFrame, n_subset: int, rng: np.random.RandomState, subset_mode: str
) -> np.ndarray:
    n_total = len(meta_test)
    n_pick = min(int(n_subset), n_total)
    if subset_mode == "random":
        return rng.choice(n_total, size=n_pick, replace=False)

    masks = turn_masks(meta_test)
    if not masks:
        raise ValueError(
            "subset_mode=turn_balanced requires has_turn_hist and has_turn_fut in meta_test"
        )

    turn_idx = np.flatnonzero(masks["any_turn"])
    no_turn_idx = np.flatnonzero(masks["no_turn"])
    n_turn = min(len(turn_idx), n_pick // 2)
    n_no_turn = min(len(no_turn_idx), n_pick - n_turn)

    chosen = []
    if n_turn:
        chosen.append(rng.choice(turn_idx, size=n_turn, replace=False))
    if n_no_turn:
        chosen.append(rng.choice(no_turn_idx, size=n_no_turn, replace=False))
    idxs = np.concatenate(chosen) if chosen else np.empty((0,), dtype=int)

    if len(idxs) < n_pick:
        remaining = np.setdiff1d(np.arange(n_total), idxs, assume_unique=False)
        extra = rng.choice(remaining, size=n_pick - len(idxs), replace=False)
        idxs = np.concatenate([idxs, extra])

    rng.shuffle(idxs)
    return idxs.astype(int, copy=False)


def write_subset_summary(out_dir: Path, meta_subset: pd.DataFrame):
    rows = [{"regime": "all", "n": int(len(meta_subset)), "fraction": 1.0}]
    for name, mask in turn_masks(meta_subset).items():
        rows.append(
            {
                "regime": name,
                "n": int(mask.sum()),
                "fraction": float(mask.mean()),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "subset_summary.csv", index=False)


def horizon_scopes(T: int, dt_seconds: float) -> List[Tuple[str, np.ndarray]]:
    if T <= 0:
        return []

    window = max(1, int(round(15.0 / max(dt_seconds, 1e-6))))
    window = min(window, T)

    scopes = [
        ("initial_horizon", np.array([0], dtype=int)),
        ("first_15_seconds", np.arange(window, dtype=int)),
    ]

    if T > window:
        start = max(0, (T - window) // 2)
        scopes.append(("middle_horizon", np.arange(start, start + window, dtype=int)))
        scopes.append(("last_15_seconds", np.arange(T - window, T, dtype=int)))
    else:
        scopes.append(("middle_horizon", np.arange(T, dtype=int)))
        scopes.append(("last_15_seconds", np.arange(T, dtype=int)))

    scopes.extend(
        [
            ("final_horizon", np.array([T - 1], dtype=int)),
            ("all_horizons", np.arange(T, dtype=int)),
        ]
    )

    seen = set()
    unique_scopes = []
    for name, idxs in scopes:
        key = tuple(int(i) for i in idxs.tolist())
        if key in seen:
            continue
        seen.add(key)
        unique_scopes.append((name, idxs))
    return unique_scopes


def transition_scopes(T: int, dt_seconds: float) -> List[Tuple[str, np.ndarray]]:
    n_transitions = max(int(T) - 1, 0)
    if n_transitions <= 0:
        return []

    window = max(1, int(round(15.0 / max(dt_seconds, 1e-6))))
    window = min(window, n_transitions)

    scopes = [
        ("initial_transition", np.array([0], dtype=int)),
        ("first_15_seconds", np.arange(window, dtype=int)),
    ]

    if n_transitions > window:
        start = max(0, (n_transitions - window) // 2)
        scopes.append(("middle_horizon", np.arange(start, start + window, dtype=int)))
        scopes.append(
            ("last_15_seconds", np.arange(n_transitions - window, n_transitions, dtype=int))
        )
    else:
        scopes.append(("middle_horizon", np.arange(n_transitions, dtype=int)))
        scopes.append(("last_15_seconds", np.arange(n_transitions, dtype=int)))

    scopes.extend(
        [
            ("final_transition", np.array([n_transitions - 1], dtype=int)),
            ("all_horizons", np.arange(n_transitions, dtype=int)),
        ]
    )

    seen = set()
    unique_scopes = []
    for name, idxs in scopes:
        key = tuple(int(i) for i in idxs.tolist())
        if key in seen:
            continue
        seen.add(key)
        unique_scopes.append((name, idxs))
    return unique_scopes


# -----------------------------------------------------------------------------
# Binning, calibration and plotting helpers
# -----------------------------------------------------------------------------


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    ids = np.searchsorted(edges, values, side="right") - 1
    return np.clip(ids, 0, max(len(edges) - 2, 0))


def bin_table(
    p: np.ndarray, y: np.ndarray, edges: np.ndarray
) -> Tuple[np.ndarray, pd.DataFrame, float]:
    bin_ids = assign_bins(p, edges)
    rows = []
    total = len(p)
    for i in range(len(edges) - 1):
        mask = bin_ids == i
        count = int(mask.sum())
        if count:
            p_bin = p[mask]
            y_bin = y[mask]
            mean_p = float(p_bin.mean())
            freq = float(y_bin.mean())
            gap = abs(mean_p - freq)
        else:
            mean_p = np.nan
            freq = np.nan
            gap = np.nan
        rows.append(
            {
                "bin_id": i,
                "bin_lower": float(edges[i]),
                "bin_upper": float(edges[i + 1]),
                "count": count,
                "mean_p": mean_p,
                "empirical_freq": freq,
                "abs_gap": float(gap) if count else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    valid = df["count"] > 0
    ece = (
        float(((df.loc[valid, "count"] / total) * df.loc[valid, "abs_gap"]).sum())
        if total
        else float("nan")
    )
    return bin_ids, df, ece


def equal_width_binning(p: np.ndarray, y: np.ndarray, nbins: int = 10):
    return bin_table(p, y, np.linspace(0.0, 1.0, nbins + 1))


def equal_count_binning(p: np.ndarray, y: np.ndarray, nbins: int = 10):
    if len(p) == 0:
        return np.empty((0,), dtype=int), pd.DataFrame(), float("nan")
    # Use quantile edges, then unique to avoid zero-width bins when p is discrete.
    q = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(p, q)
    edges = np.unique(np.concatenate(([0.0], edges[1:-1], [1.0])))
    if len(edges) < 2:
        edges = np.array([0.0, 1.0])
    return bin_table(p, y, edges)


def cumulative_diagnostics(p: np.ndarray, y: np.ndarray):
    if len(p) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            float("nan"),
            float("nan"),
            float("nan"),
        )
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    y_sorted = y[order]
    n = len(p_sorted)
    cumulative = np.cumsum((y_sorted - p_sorted) / n)
    mean_bias = float((y_sorted - p_sorted).mean())
    max_abs_dev = float(np.max(np.abs(cumulative)))
    cum_range = float(cumulative.max() - cumulative.min())
    return p_sorted, y_sorted, cumulative, mean_bias, max_abs_dev, cum_range


def ks_uniform_pvalue(ks_stat: float, n: int) -> float:
    if n <= 0 or not np.isfinite(ks_stat):
        return float("nan")
    # Kolmogorov asymptotic approximation.
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * ks_stat
    s = 0.0
    for k in range(1, 101):
        term = (-1) ** (k - 1) * math.exp(-2.0 * (k * lam) ** 2)
        s += term
        if abs(term) < 1e-12:
            break
    return float(max(0.0, min(1.0, 2.0 * s)))


def pit_summary(pit: np.ndarray) -> Dict[str, float | str]:
    pit = pit[np.isfinite(pit)]
    n = len(pit)
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "std_gap": np.nan,
            "skew": np.nan,
            "tail_mass": np.nan,
            "center_mass": np.nan,
            "ks": np.nan,
            "ks_pvalue": np.nan,
            "mode": "empty",
        }
    mean = float(pit.mean())
    std = float(pit.std(ddof=0))
    std_unif = 1.0 / math.sqrt(12.0)
    centered = pit - mean
    skew = float((centered**3).mean() / (std**3 + 1e-12))
    tail_mass = float(((pit <= 0.1) | (pit >= 0.9)).mean())
    center_mass = float(((pit >= 0.4) & (pit <= 0.6)).mean())
    xs = np.sort(pit)
    cdf = np.arange(1, n + 1) / n
    ks_plus = np.max(cdf - xs)
    ks_minus = np.max(xs - np.arange(0, n) / n)
    ks = float(max(ks_plus, ks_minus))
    mode = "mixed"
    if abs(mean - 0.5) > 0.05:
        mode = "biased_high" if mean > 0.5 else "biased_low"
    elif std < std_unif - 0.025 or center_mass > 0.26:
        mode = "overdispersed"
    elif std > std_unif + 0.025 or tail_mass > 0.24:
        mode = "underdispersed"
    return {
        "n": int(n),
        "mean": mean,
        "std": std,
        "std_gap": float(std - std_unif),
        "skew": skew,
        "tail_mass": tail_mass,
        "center_mass": center_mass,
        "ks": ks,
        "ks_pvalue": ks_uniform_pvalue(ks, n),
        "mode": mode,
    }


def plot_reliability(
    out_path: Path,
    radius: float,
    width_df: pd.DataFrame,
    count_df: pd.DataFrame,
    ece_width: float,
    ece_count: float,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, df, title in [
        (axes[0], width_df, f"Equal-width bins\nECE={ece_width:.4f}"),
        (axes[1], count_df, f"Equal-count bins\nECE={ece_count:.4f}"),
    ]:
        ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.75)
        valid = df["count"] > 0
        ax.plot(df.loc[valid, "mean_p"], df.loc[valid, "empirical_freq"], "o-", lw=1.8)
        for row in df.loc[valid].itertuples(index=False):
            ax.annotate(
                f"n={int(row.count)}",
                (row.mean_p, row.empirical_freq),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Empirical frequency")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Reliability diagram, r={int(radius)} m")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_cumulative(
    out_path: Path,
    radius: float,
    cumulative: np.ndarray,
    mean_bias: float,
    max_abs_dev: float,
    cum_range: float,
):
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(cumulative, color="black", lw=1.5)
    ax.axhline(0.0, color="gray", ls="--", lw=1.0)
    ax.set_xlabel("Sorted sample index")
    ax.set_ylabel("Cumulative sum of (y - p) / n")
    ax.set_title(
        f"Normalized cumulative calibration, r={int(radius)} m\n"
        f"mean bias={mean_bias:.5f}, max |dev|={max_abs_dev:.5f}, range={cum_range:.5f}"
    )
    ax.grid(True, alpha=0.25)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_radius_summary(out_path: Path, summary_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    r = summary_df["radius"].to_numpy()
    ax.plot(r, summary_df["mean_bias"], "o-", label="mean bias")
    ax.plot(r, summary_df["ece_equal_count"], "o-", label="ECE equal-count")
    ax.plot(r, summary_df["cumulative_range"], "o-", label="cumulative range")
    ax.axhline(0.0, color="gray", ls="--", lw=1)
    ax.set_xlabel("Radius (m)")
    ax.set_ylabel("Calibration error")
    ax.set_title("Calibration diagnostics by radius")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pit_histograms(out_path: Path, pit_by_axis: Dict[str, np.ndarray], title: str):
    axes_names = list(pit_by_axis.keys())
    fig, axes = plt.subplots(
        1, len(axes_names), figsize=(5 * len(axes_names), 4), constrained_layout=True
    )
    if len(axes_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, axes_names):
        vals = pit_by_axis[name]
        ax.hist(vals, bins=np.linspace(0, 1, 21), density=True, alpha=0.75)
        ax.axhline(1.0, color="black", ls="--", lw=1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("PIT")
        ax.set_ylabel("Density")
        ax.set_title(f"PIT {name}")
        ax.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_coverage(out_path: Path, coverage_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for scope, grp in coverage_df.groupby("scope"):
        ax.plot(grp["nominal"], grp["empirical"], "o-", label=scope)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Central interval coverage")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Regimes
# -----------------------------------------------------------------------------


def build_regime_labels(meta_subset: pd.DataFrame) -> Dict[str, np.ndarray]:
    n = len(meta_subset)
    labels: Dict[str, np.ndarray] = {"all": np.ones(n, dtype=bool)}

    labels.update(turn_masks(meta_subset))

    # Categorical regimes if the dataset already has them.
    for col in ["regime", "combined", "horizontal", "vertical"]:
        if col in meta_subset.columns:
            vals = meta_subset[col].astype(str).to_numpy()
            for val in sorted(pd.unique(vals)):
                if val and val.lower() != "nan":
                    labels[f"{col}:{val}"] = vals == val

    return labels


def run_alpha_sweep(
    out_dir: Path,
    samples_pos: np.ndarray,
    truth_pos: np.ndarray,
    radii: Sequence[float],
    nbins: int,
    alpha_values: Sequence[float],
    dt_seconds: float,
):
    alphas = sorted({float(alpha) for alpha in alpha_values if float(alpha) > 0})
    if not alphas:
        return

    nominals = np.array([0.50, 0.80, 0.90, 0.95], dtype=float)
    pit_rows = []
    coverage_rows = []
    calibration_rows = []
    summary_rows = []
    axis_names = ["x", "y", "z"][: samples_pos.shape[-1]]
    scopes = horizon_scopes(samples_pos.shape[2], dt_seconds)

    for alpha in alphas:
        center = samples_pos.mean(axis=1, keepdims=True)
        scaled = center + alpha * (samples_pos - center)

        pit_std_gap_abs = []
        pit_ks = []
        for scope, scope_idxs in scopes:
            values = scaled[:, :, scope_idxs, :]
            truth_scope = truth_pos[:, scope_idxs, :]
            for axis_i, axis_name in enumerate(axis_names):
                pit = (values[..., axis_i] <= truth_scope[:, None, :, axis_i]).mean(
                    axis=1
                )
                stats = pit_summary(pit.reshape(-1))
                pit_rows.append(
                    {"alpha": alpha, "scope": scope, "axis": axis_name, **stats}
                )
                if scope == "all_horizons":
                    if np.isfinite(stats["std_gap"]):
                        pit_std_gap_abs.append(abs(float(stats["std_gap"])))
                    if np.isfinite(stats["ks"]):
                        pit_ks.append(float(stats["ks"]))

        cov_err_all = []
        for q in nominals:
            lo = (1.0 - q) / 2.0
            hi = 1.0 - lo
            for scope, scope_idxs in scopes:
                lower = np.quantile(scaled[:, :, scope_idxs, :], lo, axis=1)
                upper = np.quantile(scaled[:, :, scope_idxs, :], hi, axis=1)
                inside = (
                    (truth_pos[:, scope_idxs, :] >= lower)
                    & (truth_pos[:, scope_idxs, :] <= upper)
                )
                emp = float(inside.mean())
                err = emp - float(q)
                coverage_rows.append(
                    {
                        "alpha": alpha,
                        "scope": f"{scope}_axes",
                        "nominal": float(q),
                        "empirical": emp,
                        "error": err,
                    }
                )
                if scope == "all_horizons":
                    cov_err_all.append(abs(err))

        final_xy = scaled[:, :, -1, :2]
        truth_final_xy = truth_pos[:, -1, :2]
        center_xy = final_xy.mean(axis=1)
        d_samples = np.linalg.norm(final_xy - center_xy[:, None, :], axis=-1)
        d_truth = np.linalg.norm(truth_final_xy - center_xy, axis=-1)
        final_spread = float(
            np.linalg.norm(final_xy - center_xy[:, None, :], axis=-1).mean()
        )

        radius_bias_abs = []
        radius_eces = []
        for radius in radii:
            p = (d_samples <= float(radius)).mean(axis=1)
            y = (d_truth <= float(radius)).astype(float)
            _, width_df, ece_width = equal_width_binning(p, y, nbins=nbins)
            _, count_df, ece_count = equal_count_binning(p, y, nbins=nbins)
            _, _, _, mean_bias, max_abs_dev, cum_range = cumulative_diagnostics(p, y)
            calibration_rows.append(
                {
                    "alpha": alpha,
                    "radius": int(round(radius)),
                    "n": int(len(p)),
                    "mean_predicted_probability": float(p.mean()),
                    "empirical_event_frequency": float(y.mean()),
                    "mean_bias": mean_bias,
                    "ece_equal_width": ece_width,
                    "ece_equal_count": ece_count,
                    "max_cumulative_deviation": max_abs_dev,
                    "cumulative_range": cum_range,
                    "number_of_non_empty_bins": int((width_df["count"] > 0).sum()),
                    "number_of_non_empty_bins_equal_count": int(
                        (count_df["count"] > 0).sum()
                    ),
                }
            )
            radius_bias_abs.append(abs(mean_bias))
            if np.isfinite(ece_count):
                radius_eces.append(float(ece_count))

        summary_rows.append(
            {
                "alpha": alpha,
                "coverage_mae_all": float(np.mean(cov_err_all)),
                "radius_bias_mae": float(np.mean(radius_bias_abs)),
                "radius_ece_mean": float(np.mean(radius_eces))
                if radius_eces
                else np.nan,
                "pit_std_gap_abs_mean": float(np.mean(pit_std_gap_abs))
                if pit_std_gap_abs
                else np.nan,
                "pit_ks_mean": float(np.mean(pit_ks)) if pit_ks else np.nan,
                "final_spread_mean_m": final_spread,
            }
        )

    alpha_summary_df = pd.DataFrame(summary_rows).sort_values("alpha")
    alpha_pit_df = pd.DataFrame(pit_rows).sort_values(["alpha", "scope", "axis"])
    alpha_coverage_df = pd.DataFrame(coverage_rows).sort_values(
        ["alpha", "scope", "nominal"]
    )
    alpha_calibration_df = pd.DataFrame(calibration_rows).sort_values(
        ["alpha", "radius"]
    )

    alpha_summary_df.to_csv(out_dir / "alpha_summary.csv", index=False)
    alpha_pit_df.to_csv(out_dir / "alpha_pit_summary.csv", index=False)
    alpha_coverage_df.to_csv(out_dir / "alpha_coverage_summary.csv", index=False)
    alpha_calibration_df.to_csv(out_dir / "alpha_calibration_summary.csv", index=False)

    print("\n=== Alpha summary ===")
    print(alpha_summary_df.to_string(index=False))


# -----------------------------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------------------------


def evaluate(
    X_test,
    Y_test,
    C_test,
    meta_test: pd.DataFrame,
    model,
    feat_mean,
    feat_std,
    ctx_mean,
    ctx_std,
    out_dir: Path,
    n_subset: int,
    n_samples: int,
    radii: Iterable[float],
    batch_size: int,
    seed: int,
    n_steps: int,
    nbins: int,
    min_regime_n: int,
    score_pair_limit: int,
    alpha_values: Sequence[float],
    dt_seconds: float,
    subset_mode: str,
):
    rng = np.random.RandomState(seed)
    idxs = select_subset_indices(meta_test, n_subset=n_subset, rng=rng, subset_mode=subset_mode)
    meta_subset = meta_test.iloc[idxs].reset_index(drop=True)
    flight_ids = safe_flight_ids(meta_test, idxs)
    regime_masks = build_regime_labels(meta_subset)
    device = next(model.parameters()).device
    radii = [float(r) for r in radii]
    radius_tags = [int(round(r)) for r in radii]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    (out_dir / "raw").mkdir(exist_ok=True)
    write_subset_summary(out_dir, meta_subset)

    raw_rows_by_radius: Dict[int, List[dict]] = {tag: [] for tag in radius_tags}
    pit_by_scope = {}
    nominals = np.array([0.50, 0.80, 0.90, 0.95], dtype=float)
    coverage_stats = {}

    energy_scores = []
    crps_values = []
    ade_values = []
    fde_values = []
    spread_final_values = []
    collect_for_alpha = any(float(alpha) > 0 for alpha in alpha_values)
    alpha_samples = [] if collect_for_alpha else None
    alpha_truth = [] if collect_for_alpha else None
    kinematic_stats = {}
    boundary_stats = {
        "truth_abs": [],
        "truth_rel": [],
        "samples_abs": [],
        "samples_rel": [],
    }

    for start in range(0, len(idxs), batch_size):
        batch_idxs = idxs[start : start + batch_size]
        B = len(batch_idxs)
        xb = torch.as_tensor(X_test[batch_idxs], dtype=torch.float32, device=device)
        yb = torch.as_tensor(Y_test[batch_idxs], dtype=torch.float32, device=device)
        cb = torch.as_tensor(C_test[batch_idxs], dtype=torch.float32, device=device)

        with torch.no_grad():
            samples = sample_many(
                model,
                xb,
                cb,
                T_out=yb.shape[1],
                n_steps=n_steps,
                n_samples=n_samples,
                chunk=n_samples,
            )

        if samples.dim() == 4:
            S, b_size, T, D = samples.shape
            samples_resh = samples
        elif samples.dim() == 3:
            b_size = xb.shape[0]
            T = samples.shape[1]
            D = samples.shape[2]
            S = samples.shape[0] // b_size
            samples_resh = samples.reshape(S, b_size, T, D)
        else:
            raise ValueError(f"Unexpected samples shape: {samples.shape}")
        if b_size != B:
            raise RuntimeError(f"Batch size mismatch: expected {B}, got {b_size}")

        samples_resh = samples_resh.permute(1, 0, 2, 3).contiguous()  # B,S,T,D
        samples_flat = samples_resh.reshape(B * S, T, D)
        cb_rep = cb.repeat_interleave(S, dim=0)
        with torch.no_grad():
            denorm_flat = denorm_seq_to_global(
                samples_flat, cb_rep, feat_mean, feat_std, ctx_mean, ctx_std
            )
            x_denorm_t = denorm_seq_to_global(
                xb, cb, feat_mean, feat_std, ctx_mean, ctx_std
            )
            y_denorm_t = denorm_seq_to_global(
                yb, cb, feat_mean, feat_std, ctx_mean, ctx_std
            )
        denorm_samples = denorm_flat.reshape(B, S, T, D).detach().cpu().numpy()
        x_denorm = x_denorm_t.detach().cpu().numpy()
        y_denorm = y_denorm_t.detach().cpu().numpy()

        pos_dims = min(3, D)
        samples_pos = denorm_samples[:, :, :, :pos_dims]
        truth_pos = y_denorm[:, :, :pos_dims]
        scope_defs = horizon_scopes(samples_pos.shape[2], dt_seconds)
        transition_defs = transition_scopes(samples_pos.shape[2], dt_seconds)
        if not pit_by_scope:
            pit_by_scope = {
                scope: {axis: [] for axis in ["x", "y", "z"][:pos_dims]}
                for scope, _ in scope_defs
            }
        if transition_defs and not kinematic_stats:
            kinematic_stats = {
                scope: {
                    "truth_abs": [],
                    "truth_rel": [],
                    "samples_abs": [],
                    "samples_rel": [],
                }
                for scope, _ in transition_defs
            }
        if collect_for_alpha:
            alpha_samples.append(samples_pos.astype(np.float32, copy=False))
            alpha_truth.append(truth_pos.astype(np.float32, copy=False))
        samples_final_xy = denorm_samples[:, :, -1, :2]
        truth_final_xy = y_denorm[:, -1, :2]
        center_xy = samples_final_xy.mean(axis=1)
        d_samples_to_center = np.linalg.norm(
            samples_final_xy - center_xy[:, None, :], axis=-1
        )
        d_truth_to_center = np.linalg.norm(truth_final_xy - center_xy, axis=-1)

        # Raw radius-event probabilities.
        for local_i, global_i in enumerate(batch_idxs):
            fid = str(flight_ids[start + local_i])
            for radius, tag in zip(radii, radius_tags):
                p_hat = float((d_samples_to_center[local_i] <= radius).mean())
                y_true = float(d_truth_to_center[local_i] <= radius)
                raw_rows_by_radius[tag].append(
                    {
                        "sample_index": int(global_i),
                        "flight_id": fid,
                        "p_hat": p_hat,
                        "y_true": y_true,
                        "d_truth_to_center_m": float(d_truth_to_center[local_i]),
                        "mean_sample_to_center_m": float(
                            d_samples_to_center[local_i].mean()
                        ),
                    }
                )

        # PIT values. Randomized tie handling is unnecessary for continuous samples.
        axis_names = ["x", "y", "z"][:pos_dims]
        for scope, scope_idxs in scope_defs:
            for axis_i, axis_name in enumerate(axis_names):
                vals = (
                    samples_pos[:, :, scope_idxs, axis_i]
                    <= truth_pos[:, None, scope_idxs, axis_i]
                ).mean(axis=1)
                pit_by_scope[scope][axis_name].append(vals.reshape(-1))

        if not coverage_stats:
            coverage_stats = {
                scope: {
                    "counts": {float(q): 0 for q in nominals},
                    "total": 0,
                }
                for scope, _ in scope_defs
            }
        for scope, scope_idxs in scope_defs:
            coverage_stats[scope]["total"] += int(
                np.prod(truth_pos[:, scope_idxs, :].shape)
            )
            for q in nominals:
                lo = (1.0 - q) / 2.0
                hi = 1.0 - lo
                lower = np.quantile(samples_pos[:, :, scope_idxs, :], lo, axis=1)
                upper = np.quantile(samples_pos[:, :, scope_idxs, :], hi, axis=1)
                inside = (
                    (truth_pos[:, scope_idxs, :] >= lower)
                    & (truth_pos[:, scope_idxs, :] <= upper)
                )
                coverage_stats[scope]["counts"][float(q)] += int(inside.sum())

        if D >= 6 and pos_dims:
            hist_last_pos = x_denorm[:, -1, :pos_dims]
            hist_last_vel = x_denorm[:, -1, 3 : 3 + pos_dims]
            truth_first_step = truth_pos[:, 0, :] - hist_last_pos
            sample_first_step = samples_pos[:, :, 0, :] - hist_last_pos[:, None, :]
            truth_boundary = truth_first_step - hist_last_vel * dt_seconds
            sample_boundary = sample_first_step - hist_last_vel[:, None, :] * dt_seconds
            boundary_stats["truth_abs"].append(
                np.linalg.norm(truth_boundary, axis=-1).reshape(-1)
            )
            boundary_stats["truth_rel"].append(
                (
                    np.linalg.norm(truth_boundary, axis=-1)
                    / (np.linalg.norm(truth_first_step, axis=-1) + 1e-6)
                ).reshape(-1)
            )
            boundary_stats["samples_abs"].append(
                np.linalg.norm(sample_boundary, axis=-1).reshape(-1)
            )
            boundary_stats["samples_rel"].append(
                (
                    np.linalg.norm(sample_boundary, axis=-1)
                    / (np.linalg.norm(sample_first_step, axis=-1) + 1e-6)
                ).reshape(-1)
            )

            truth_vel = y_denorm[:, :, 3 : 3 + pos_dims]
            sample_vel = denorm_samples[:, :, :, 3 : 3 + pos_dims]
            truth_dpos = truth_pos[:, 1:, :] - truth_pos[:, :-1, :]
            sample_dpos = samples_pos[:, :, 1:, :] - samples_pos[:, :, :-1, :]
            truth_residual = truth_dpos - truth_vel[:, :-1, :] * dt_seconds
            sample_residual = sample_dpos - sample_vel[:, :, :-1, :] * dt_seconds
            truth_abs = np.linalg.norm(truth_residual, axis=-1)
            sample_abs = np.linalg.norm(sample_residual, axis=-1)
            truth_rel = truth_abs / (np.linalg.norm(truth_dpos, axis=-1) + 1e-6)
            sample_rel = sample_abs / (np.linalg.norm(sample_dpos, axis=-1) + 1e-6)

            for scope, scope_idxs in transition_defs:
                kinematic_stats[scope]["truth_abs"].append(
                    truth_abs[:, scope_idxs].reshape(-1)
                )
                kinematic_stats[scope]["truth_rel"].append(
                    truth_rel[:, scope_idxs].reshape(-1)
                )
                kinematic_stats[scope]["samples_abs"].append(
                    sample_abs[:, :, scope_idxs].reshape(-1)
                )
                kinematic_stats[scope]["samples_rel"].append(
                    sample_rel[:, :, scope_idxs].reshape(-1)
                )

        # Accuracy and sharpness summaries.
        mean_pos = samples_pos.mean(axis=1)
        err = np.linalg.norm(mean_pos - truth_pos, axis=-1)  # B,T
        ade_values.extend(err.mean(axis=1).tolist())
        fde_values.extend(err[:, -1].tolist())
        spread_final_values.extend(
            np.linalg.norm(samples_pos[:, :, -1, :] - mean_pos[:, None, -1, :], axis=-1)
            .mean(axis=1)
            .tolist()
        )

        # Energy score and CRPS. Use exact pairwise sample term unless S is huge.
        flat_samples = samples_pos.reshape(B, S, -1)
        flat_truth = truth_pos.reshape(B, -1)
        term1 = np.linalg.norm(flat_samples - flat_truth[:, None, :], axis=-1).mean(
            axis=1
        )
        # Pairwise distances for each batch item.
        for b in range(B):
            fs = flat_samples[b]
            if S <= score_pair_limit:
                pdist = np.linalg.norm(fs[:, None, :] - fs[None, :, :], axis=-1).mean()
            else:
                pair_i = rng.randint(0, S, size=score_pair_limit)
                pair_j = rng.randint(0, S, size=score_pair_limit)
                pdist = np.linalg.norm(fs[pair_i] - fs[pair_j], axis=-1).mean()
            energy_scores.append(float(term1[b] - 0.5 * pdist))

        # CRPS aggregated over position axes and horizons.
        # Exact pair term per B,T,D; S is usually 32 or 64, so this is fine.
        abs_truth = np.abs(samples_pos - truth_pos[:, None, :, :]).mean(axis=1)  # B,T,D
        if S <= score_pair_limit:
            pair_abs = np.abs(
                samples_pos[:, :, None, :, :] - samples_pos[:, None, :, :, :]
            ).mean(axis=(1, 2))  # B,T,D
        else:
            pair_i = rng.randint(0, S, size=score_pair_limit)
            pair_j = rng.randint(0, S, size=score_pair_limit)
            pair_abs = np.abs(
                samples_pos[:, pair_i, :, :] - samples_pos[:, pair_j, :, :]
            ).mean(axis=1)
        crps = abs_truth - 0.5 * pair_abs
        crps_values.extend(crps.mean(axis=(1, 2)).tolist())

        if (start // batch_size) % 10 == 0:
            print(f"processed {min(start + B, len(idxs))}/{len(idxs)}", flush=True)

    # Radius event summaries and plots.
    summary_rows = []
    regime_rows = []
    for radius, tag in zip(radii, radius_tags):
        raw_df = pd.DataFrame(raw_rows_by_radius[tag])
        p = raw_df["p_hat"].to_numpy(dtype=float)
        y = raw_df["y_true"].to_numpy(dtype=float)
        width_ids, width_df, ece_width = equal_width_binning(p, y, nbins=nbins)
        count_ids, count_df, ece_count = equal_count_binning(p, y, nbins=nbins)
        p_sorted, y_sorted, cumulative, mean_bias, max_abs_dev, cum_range = (
            cumulative_diagnostics(p, y)
        )
        raw_df["bin_id_equal_width"] = width_ids
        raw_df["bin_id_equal_count"] = count_ids
        raw_df.to_csv(out_dir / "raw" / f"raw_r{tag}.csv", index=False)
        np.savez_compressed(
            out_dir / "raw" / f"raw_r{tag}.npz",
            p_hat=p,
            y_true=y,
            sample_index=raw_df["sample_index"].to_numpy(dtype=np.int64),
            flight_id=raw_df["flight_id"].astype(str).to_numpy(dtype=str),
            bin_id_equal_width=width_ids.astype(np.int64),
            bin_id_equal_count=count_ids.astype(np.int64),
            p_sorted=p_sorted,
            y_sorted=y_sorted,
            cumulative=cumulative,
        )
        width_df.to_csv(out_dir / f"equal_width_bins_r{tag}.csv", index=False)
        count_df.to_csv(out_dir / f"equal_count_bins_r{tag}.csv", index=False)
        pd.DataFrame(
            {
                "rank": np.arange(len(cumulative)),
                "p_sorted": p_sorted,
                "y_sorted": y_sorted,
                "cumulative": cumulative,
            }
        ).to_csv(out_dir / f"cumulative_curve_r{tag}.csv", index=False)
        plot_reliability(
            out_dir / "plots" / f"reliability_r{tag}.png",
            radius,
            width_df,
            count_df,
            ece_width,
            ece_count,
        )
        plot_cumulative(
            out_dir / "plots" / f"cumulative_r{tag}.png",
            radius,
            cumulative,
            mean_bias,
            max_abs_dev,
            cum_range,
        )

        summary_rows.append(
            {
                "radius": tag,
                "n": int(len(raw_df)),
                "mean_predicted_probability": float(p.mean()),
                "empirical_event_frequency": float(y.mean()),
                "mean_bias": mean_bias,
                "ece_equal_width": ece_width,
                "ece_equal_count": ece_count,
                "max_cumulative_deviation": max_abs_dev,
                "cumulative_range": cum_range,
                "number_of_non_empty_bins": int((width_df["count"] > 0).sum()),
                "number_of_non_empty_bins_equal_count": int(
                    (count_df["count"] > 0).sum()
                ),
            }
        )

        for regime_name, mask in regime_masks.items():
            if int(mask.sum()) < min_regime_n:
                continue
            p_reg = p[mask]
            y_reg = y[mask]
            _, wdf, ece_w = equal_width_binning(p_reg, y_reg, nbins=nbins)
            _, cdf, ece_c = equal_count_binning(p_reg, y_reg, nbins=nbins)
            _, _, cum_reg, bias_reg, max_reg, range_reg = cumulative_diagnostics(
                p_reg, y_reg
            )
            regime_rows.append(
                {
                    "radius": tag,
                    "regime": regime_name,
                    "n": int(mask.sum()),
                    "mean_predicted_probability": float(p_reg.mean()),
                    "empirical_event_frequency": float(y_reg.mean()),
                    "mean_bias": bias_reg,
                    "ece_equal_width": ece_w,
                    "ece_equal_count": ece_c,
                    "max_cumulative_deviation": max_reg,
                    "cumulative_range": range_reg,
                    "number_of_non_empty_bins": int((wdf["count"] > 0).sum()),
                    "number_of_non_empty_bins_equal_count": int(
                        (cdf["count"] > 0).sum()
                    ),
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values("radius")
    regime_df = (
        pd.DataFrame(regime_rows).sort_values(["radius", "regime"])
        if regime_rows
        else pd.DataFrame()
    )
    summary_df.to_csv(out_dir / "calibration_summary.csv", index=False)
    regime_df.to_csv(out_dir / "calibration_regime_summary.csv", index=False)
    plot_radius_summary(out_dir / "plots" / "calibration_by_radius.png", summary_df)

    # PIT outputs.
    pit_rows = []
    for scope, pit_dict in pit_by_scope.items():
        joined = {}
        for axis_name, chunks in pit_dict.items():
            if chunks:
                vals = np.concatenate(chunks).astype(float)
                joined[axis_name] = vals
                stats = pit_summary(vals)
                pit_rows.append({"scope": scope, "axis": axis_name, **stats})
        if joined:
            plot_pit_histograms(
                out_dir / "plots" / f"pit_{scope}.png",
                joined,
                f"PIT histograms ({scope})",
            )
            np.savez_compressed(out_dir / "raw" / f"pit_{scope}.npz", **joined)
    pit_df = pd.DataFrame(pit_rows)
    pit_df.to_csv(out_dir / "pit_summary.csv", index=False)

    # Coverage outputs.
    cov_rows = []
    for scope, stats in coverage_stats.items():
        total = stats["total"]
        for q in nominals:
            qf = float(q)
            empirical = stats["counts"][qf] / total if total else np.nan
            cov_rows.append(
                {
                    "scope": f"{scope}_axes",
                    "nominal": qf,
                    "empirical": empirical,
                    "error": empirical - qf,
                }
            )
    coverage_df = pd.DataFrame(cov_rows)
    coverage_df.to_csv(out_dir / "coverage_summary.csv", index=False)
    plot_coverage(out_dir / "plots" / "coverage_curve.png", coverage_df)

    # Scores and accuracy outputs.
    score_df = pd.DataFrame(
        [
            {
                "metric": "energy_score_whole_path_mean",
                "value": float(np.mean(energy_scores)),
                "std": float(np.std(energy_scores)),
                "n": len(energy_scores),
            },
            {
                "metric": "crps_position_mean",
                "value": float(np.mean(crps_values)),
                "std": float(np.std(crps_values)),
                "n": len(crps_values),
            },
            {
                "metric": "ade_mean_m",
                "value": float(np.mean(ade_values)),
                "std": float(np.std(ade_values)),
                "n": len(ade_values),
            },
            {
                "metric": "fde_mean_m",
                "value": float(np.mean(fde_values)),
                "std": float(np.std(fde_values)),
                "n": len(fde_values),
            },
            {
                "metric": "final_spread_mean_m",
                "value": float(np.mean(spread_final_values)),
                "std": float(np.std(spread_final_values)),
                "n": len(spread_final_values),
            },
        ]
    )
    score_df.to_csv(out_dir / "score_summary.csv", index=False)

    kinematic_rows = []
    if boundary_stats["truth_abs"]:
        for sequence in ["truth", "samples"]:
            abs_vals = np.concatenate(boundary_stats[f"{sequence}_abs"]).astype(float)
            rel_vals = np.concatenate(boundary_stats[f"{sequence}_rel"]).astype(float)
            kinematic_rows.append(
                {
                    "scope": "history_to_first_future",
                    "sequence": sequence,
                    "mean_abs_residual_m": float(abs_vals.mean()),
                    "median_abs_residual_m": float(np.median(abs_vals)),
                    "p90_abs_residual_m": float(np.quantile(abs_vals, 0.9)),
                    "mean_relative_residual": float(rel_vals.mean()),
                    "median_relative_residual": float(np.median(rel_vals)),
                    "p90_relative_residual": float(np.quantile(rel_vals, 0.9)),
                    "n": int(len(abs_vals)),
                }
            )
    for scope, stats in kinematic_stats.items():
        for sequence in ["truth", "samples"]:
            abs_chunks = stats[f"{sequence}_abs"]
            rel_chunks = stats[f"{sequence}_rel"]
            if not abs_chunks:
                continue
            abs_vals = np.concatenate(abs_chunks).astype(float)
            rel_vals = np.concatenate(rel_chunks).astype(float)
            kinematic_rows.append(
                {
                    "scope": scope,
                    "sequence": sequence,
                    "mean_abs_residual_m": float(abs_vals.mean()),
                    "median_abs_residual_m": float(np.median(abs_vals)),
                    "p90_abs_residual_m": float(np.quantile(abs_vals, 0.9)),
                    "mean_relative_residual": float(rel_vals.mean()),
                    "median_relative_residual": float(np.median(rel_vals)),
                    "p90_relative_residual": float(np.quantile(rel_vals, 0.9)),
                    "n": int(len(abs_vals)),
                }
            )
    kinematic_df = pd.DataFrame(kinematic_rows)
    kinematic_df.to_csv(out_dir / "kinematics_summary.csv", index=False)

    if collect_for_alpha and alpha_samples and alpha_truth:
        run_alpha_sweep(
            out_dir=out_dir,
            samples_pos=np.concatenate(alpha_samples, axis=0),
            truth_pos=np.concatenate(alpha_truth, axis=0),
            radii=radii,
            nbins=nbins,
            alpha_values=alpha_values,
            dt_seconds=dt_seconds,
        )

    print("\n=== Calibration summary ===")
    print(summary_df.to_string(index=False))
    print("\n=== PIT summary ===")
    print(pit_df.to_string(index=False))
    print("\n=== Coverage summary ===")
    print(coverage_df.to_string(index=False))
    print("\n=== Score summary ===")
    print(score_df.to_string(index=False))
    if not kinematic_df.empty:
        print("\n=== Kinematics summary ===")
        print(kinematic_df.to_string(index=False))
    if not regime_df.empty:
        print("\n=== Regime summary (filtered) ===")
        print(regime_df.to_string(index=False))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "dataset_cache"))
    ap.add_argument("--ckpt", default=str(ROOT / "models" / "cfm_base.pt"))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--n_subset", type=int, default=1000)
    ap.add_argument("--n_samples", type=int, default=32)
    ap.add_argument("--n_steps", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--radii", type=float, nargs="+", default=[50.0, 100.0, 200.0, 400.0, 800.0]
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nbins", type=int, default=10)
    ap.add_argument("--min-regime-n", type=int, default=50)
    ap.add_argument(
        "--score-pair-limit",
        type=int,
        default=128,
        help="Exact pairwise score if n_samples <= this; otherwise random pair approximation.",
    )
    ap.add_argument(
        "--subset-mode",
        choices=["random", "turn_balanced"],
        default="random",
    )
    ap.add_argument("--alpha-values", type=float, nargs="+", default=[])
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    key_id, X_test, Y_test, C_test, meta_test = load_cache_bundle(cache_dir)
    key_config = load_key_config(cache_dir, key_id)
    norm_stats_path, feat_mean, feat_std, ctx_mean, ctx_std = load_norm_stats(
        cache_dir, key_id
    )
    dt_seconds = float((key_config.get("prep") or {}).get("output_stride", 5.0))

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / f"eval_{ckpt_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"using dataset key: {key_id}")
    print(f"using norm stats: {norm_stats_path.name}")
    print(f"using checkpoint: {ckpt_path}")
    print(f"using device: {device}")
    print(f"output directory: {out_dir}")

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(
            vars(args)
            | {
                "dataset_key": key_id,
                "norm_stats": norm_stats_path.name,
                "dt_seconds": dt_seconds,
                "device": str(device),
            },
            f,
            indent=2,
        )

    model = load_model_checkpoint(str(ckpt_path), device=device)
    model.eval()

    evaluate(
        X_test=X_test,
        Y_test=Y_test,
        C_test=C_test,
        meta_test=meta_test,
        model=model,
        feat_mean=feat_mean,
        feat_std=feat_std,
        ctx_mean=ctx_mean,
        ctx_std=ctx_std,
        out_dir=out_dir,
        n_subset=int(args.n_subset),
        n_samples=int(args.n_samples),
        radii=args.radii,
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        n_steps=int(args.n_steps),
        nbins=int(args.nbins),
        min_regime_n=int(args.min_regime_n),
        score_pair_limit=int(args.score_pair_limit),
        alpha_values=args.alpha_values,
        dt_seconds=dt_seconds,
        subset_mode=str(args.subset_mode),
    )


if __name__ == "__main__":
    main()
