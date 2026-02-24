#!/usr/bin/env python3
"""
Intent classification for aircraft trajectory prediction

Extracts discrete flight-phase labels and continuous intent features from
60-second history windows. These features capture what the pilot is doing
(climbing, descending, turning, levelling-off …) so that a generative
predictor can condition on intent

All functions operate on denormalized, aircraft-centric arrays so that
physical thresholds (m/s, rad/s) are meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple

import numpy as np

# ── Feature indices in the 7-D state vector ──────────────────────────
#   0: x   1: y   2: z   3: vx   4: vy   5: vz   6: psi_rate
IDX_Z = 2
IDX_VZ = 5
IDX_PSI = 6

# ── Physical thresholds ──────────────────────────────────────────────
# vz is in m/s in aircraft-centric frame.  1 ft/min ≈ 0.00508 m/s
# 100 ft/min ≈ 0.508 m/s  –  below this we call it "level"
VZ_LEVEL_THR = 0.508          # m/s  (≈100 ft/min)
VZ_STRONG_THR = 2.54          # m/s  (≈500 ft/min)

# Turn-rate thresholds (rad/s).  0.01 rad/s ≈ 0.57 deg/s
PSI_STRAIGHT_THR = 0.005      # rad/s – below this is straight
PSI_TURN_THR = 0.01           # rad/s – above this is definitely turning
PSI_STRONG_THR = 0.03         # rad/s – strong turn

# Trend detection windows
NOW_WINDOW = 5                # last 5 s  – "what is the aircraft doing RIGHT NOW?"
PRIOR_WINDOW = 15             # 15 s window immediately BEFORE the NOW window
                              # i.e. t=40..55 for a 60s history with NOW=5
EARLY_WINDOW = 15             # first 15 s (for continuous features)

# Kept for backward compat with continuous features
TREND_WINDOW = 15

# ── Enums ────────────────────────────────────────────────────────────

class VerticalPhase(IntEnum):
    LEVEL = 0
    CLIMBING = 1
    DESCENDING = 2
    LEVELLING_OFF_FROM_CLIMB = 3
    LEVELLING_OFF_FROM_DESCENT = 4


class LateralPhase(IntEnum):
    STRAIGHT = 0
    TURNING_LEFT = 1
    TURNING_RIGHT = 2
    ROLLING_OUT_FROM_LEFT = 3   # was turning left, now straightening
    ROLLING_OUT_FROM_RIGHT = 4  # was turning right, now straightening


@dataclass
class IntentLabel:
    """Discrete intent label for a single trajectory window."""
    vertical: VerticalPhase
    lateral: LateralPhase

    def to_index(self) -> int:
        """Flat index in [0, 25) for the 5x5 grid."""
        return int(self.vertical) * 5 + int(self.lateral)

    @staticmethod
    def from_index(idx: int) -> "IntentLabel":
        v = VerticalPhase(idx // 5)
        l_ = LateralPhase(idx % 5)
        return IntentLabel(vertical=v, lateral=l_)

    @staticmethod
    def num_classes() -> int:
        return 25   # 5 vertical × 5 lateral

    def short_name(self) -> str:
        v_names = {0: "LVL", 1: "CLB", 2: "DES", 3: "LVL↑", 4: "LVL↓"}
        l_names = {0: "STR", 1: "TL", 2: "TR", 3: "RO-L", 4: "RO-R"}
        return f"{v_names[int(self.vertical)]}+{l_names[int(self.lateral)]}"

# ── Continuous intent features ───────────────────────────────────────

def compute_intent_features(hist: np.ndarray) -> np.ndarray:
    """
    Extract 12 continuous intent features from a history window.

    Args:
        hist: (T, 7) denormalized aircraft-centric history, T=60 at 1 Hz.

    Returns:
        (12,) float32 feature vector:
            0: mean_vz_late      - avg vertical rate in last TREND_WINDOW s
            1: mean_vz_early     - avg vertical rate in first EARLY_WINDOW s
            2: dvz_dt            - linear trend of vz (slope) over full window
            3: vz_last           - final vz value
            4: mean_psi_late     - avg turn rate in last TREND_WINDOW s
            5: mean_psi_early    - avg turn rate in first EARLY_WINDOW s
            6: dpsi_dt           - linear trend of psi_rate over full window
            7: psi_last          - final psi_rate value
            8: abs_vz_late       - magnitude of late vertical rate
            9: abs_psi_late      - magnitude of late turn rate
           10: vz_change         - vz_late - vz_early  (transition indicator)
           11: psi_change        - psi_late - psi_early (transition indicator)
    """
    T = hist.shape[0]
    vz = hist[:, IDX_VZ]
    psi = hist[:, IDX_PSI]

    late_sl = slice(max(0, T - TREND_WINDOW), T)
    early_sl = slice(0, min(EARLY_WINDOW, T))

    mean_vz_late = np.mean(vz[late_sl])
    mean_vz_early = np.mean(vz[early_sl])
    mean_psi_late = np.mean(psi[late_sl])
    mean_psi_early = np.mean(psi[early_sl])

    t_axis = np.arange(T, dtype=np.float64)
    dvz_dt = _linreg_slope(t_axis, vz.astype(np.float64))
    dpsi_dt = _linreg_slope(t_axis, psi.astype(np.float64))

    return np.array([
        mean_vz_late,
        mean_vz_early,
        dvz_dt,
        vz[-1],
        mean_psi_late,
        mean_psi_early,
        dpsi_dt,
        psi[-1],
        np.abs(mean_vz_late),
        np.abs(mean_psi_late),
        mean_vz_late - mean_vz_early,
        mean_psi_late - mean_psi_early,
    ], dtype=np.float32)

def compute_intent_features_batch(hist_batch: np.ndarray) -> np.ndarray:
    """
    Vectorised intent-feature extraction for a batch of windows.

    Args:
        hist_batch: (N, T, 7) denormalized aircraft-centric histories.

    Returns:
        (N, 12) float32 intent features.
    """
    N, T, _ = hist_batch.shape
    vz = hist_batch[:, :, IDX_VZ]    # (N, T)
    psi = hist_batch[:, :, IDX_PSI]  # (N, T)

    late_sl = slice(max(0, T - TREND_WINDOW), T)
    early_sl = slice(0, min(EARLY_WINDOW, T))

    mean_vz_late = vz[:, late_sl].mean(axis=1)
    mean_vz_early = vz[:, early_sl].mean(axis=1)
    mean_psi_late = psi[:, late_sl].mean(axis=1)
    mean_psi_early = psi[:, early_sl].mean(axis=1)

    t_axis = np.arange(T, dtype=np.float64)
    dvz_dt = _linreg_slope_batch(t_axis, vz.astype(np.float64))
    dpsi_dt = _linreg_slope_batch(t_axis, psi.astype(np.float64))

    feats = np.stack([
        mean_vz_late,
        mean_vz_early,
        dvz_dt,
        vz[:, -1],
        mean_psi_late,
        mean_psi_early,
        dpsi_dt,
        psi[:, -1],
        np.abs(mean_vz_late),
        np.abs(mean_psi_late),
        mean_vz_late - mean_vz_early,
        mean_psi_late - mean_psi_early,
    ], axis=1).astype(np.float32)

    return feats

# ── Discrete intent classification ──────────────────────────────────

def classify_vertical(hist: np.ndarray) -> VerticalPhase:
    """Classify the vertical phase from a single history window.

    Uses NOW_WINDOW (5s) for current state. For transition detection
    (level-off), checks the PRIOR window (15s immediately before NOW)
    so there's no overlap.
    """
    vz = hist[:, IDX_VZ]
    T = len(vz)
    now_start = max(0, T - NOW_WINDOW)
    prior_start = max(0, now_start - PRIOR_WINDOW)
    now = vz[now_start:]
    prior = vz[prior_start:now_start]

    mean_now = np.mean(now)
    abs_now = np.abs(mean_now)

    if abs_now > VZ_LEVEL_THR:
        if mean_now > 0:
            return VerticalPhase.CLIMBING
        else:
            return VerticalPhase.DESCENDING

    if len(prior) > 0:
        mean_prior = np.mean(prior)
        if np.abs(mean_prior) > VZ_STRONG_THR:
            if mean_prior > 0:
                return VerticalPhase.LEVELLING_OFF_FROM_CLIMB
            else:
                return VerticalPhase.LEVELLING_OFF_FROM_DESCENT
    return VerticalPhase.LEVEL

def classify_lateral(hist: np.ndarray) -> LateralPhase:
    """Classify the lateral phase from a single history window.

    Uses NOW_WINDOW (5s) for current state. For roll-out detection,
    checks the PRIOR window (15s immediately before NOW) so there's
    no overlap.
    """
    psi = hist[:, IDX_PSI]
    T = len(psi)
    now_start = max(0, T - NOW_WINDOW)
    prior_start = max(0, now_start - PRIOR_WINDOW)
    now = psi[now_start:]
    prior = psi[prior_start:now_start]

    mean_now = np.mean(now)
    abs_now = np.abs(mean_now)

    if abs_now > PSI_STRAIGHT_THR:
        if mean_now > 0:
            return LateralPhase.TURNING_RIGHT
        else:
            return LateralPhase.TURNING_LEFT

    if len(prior) > 0:
        mean_prior = np.mean(prior)
        if np.abs(mean_prior) > PSI_TURN_THR:
            if mean_prior > 0:
                return LateralPhase.ROLLING_OUT_FROM_RIGHT
            else:
                return LateralPhase.ROLLING_OUT_FROM_LEFT
    return LateralPhase.STRAIGHT

def classify_intent(hist: np.ndarray) -> IntentLabel:
    """Full intent classification for one history window."""
    return IntentLabel(
        vertical=classify_vertical(hist),
        lateral=classify_lateral(hist),
    )

def classify_intent_batch(hist_batch: np.ndarray) -> np.ndarray:
    """
    Vectorised intent classification for a batch.

    Args:
        hist_batch: (N, T, 7) denormalized aircraft-centric histories.

    Returns:
        (N,) int32 array of flat intent indices in [0, 25).
    """
    N, T, _ = hist_batch.shape
    vz = hist_batch[:, :, IDX_VZ]
    psi = hist_batch[:, :, IDX_PSI]

    now_start = max(0, T - NOW_WINDOW)
    prior_start = max(0, now_start - PRIOR_WINDOW)
    now_sl = slice(now_start, T)
    prior_sl = slice(prior_start, now_start)

    vz_now = vz[:, now_sl].mean(axis=1)
    psi_now = psi[:, now_sl].mean(axis=1)
    vz_prior = vz[:, prior_sl].mean(axis=1)
    psi_prior = psi[:, prior_sl].mean(axis=1)

    abs_vz_now = np.abs(vz_now)
    abs_psi_now = np.abs(psi_now)

    # Vertical: current state from NOW, transitions from PRIOR
    v = np.full(N, int(VerticalPhase.LEVEL), dtype=np.int32)
    v[vz_now > VZ_LEVEL_THR] = int(VerticalPhase.CLIMBING)
    v[vz_now < -VZ_LEVEL_THR] = int(VerticalPhase.DESCENDING)
    lvl_now = abs_vz_now < VZ_LEVEL_THR
    lvl_from_clb = lvl_now & (vz_prior > VZ_STRONG_THR)
    lvl_from_des = lvl_now & (vz_prior < -VZ_STRONG_THR)
    v[lvl_from_clb] = int(VerticalPhase.LEVELLING_OFF_FROM_CLIMB)
    v[lvl_from_des] = int(VerticalPhase.LEVELLING_OFF_FROM_DESCENT)

    # Lateral: current state from NOW, transitions from PRIOR
    l_ = np.full(N, int(LateralPhase.STRAIGHT), dtype=np.int32)
    l_[psi_now > PSI_STRAIGHT_THR] = int(LateralPhase.TURNING_RIGHT)
    l_[psi_now < -PSI_STRAIGHT_THR] = int(LateralPhase.TURNING_LEFT)
    str_now = abs_psi_now < PSI_STRAIGHT_THR
    ro_left = str_now & (psi_prior < -PSI_TURN_THR)
    ro_right = str_now & (psi_prior > PSI_TURN_THR)
    l_[ro_left] = int(LateralPhase.ROLLING_OUT_FROM_LEFT)
    l_[ro_right] = int(LateralPhase.ROLLING_OUT_FROM_RIGHT)

    return v * 5 + l_

def intent_index_to_onehot(indices: np.ndarray, n_classes: int = 25) -> np.ndarray:
    """Convert flat intent indices to one-hot encoding."""
    oh = np.zeros((len(indices), n_classes), dtype=np.float32)
    oh[np.arange(len(indices)), indices] = 1.0
    return oh

# ── Helpers ──────────────────────────────────────────────────────────

def _linreg_slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of y on x."""
    n = len(x)
    if n < 2:
        return 0.0
    xm = x.mean()
    ym = y.mean()
    num = ((x - xm) * (y - ym)).sum()
    den = ((x - xm) ** 2).sum()
    if den < 1e-12:
        return 0.0
    return float(num / den)

def _linreg_slope_batch(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """OLS slope for each row of Y against shared x.  Y: (N, T)."""
    xm = x.mean()
    xc = x - xm
    Yc = Y - Y.mean(axis=1, keepdims=True)
    num = (Yc * xc[None, :]).sum(axis=1)
    den = (xc ** 2).sum()
    return np.where(den > 1e-12, num / den, 0.0)

# ── Denormalization helpers ──────────────────────────────────────────

def denormalize_history(
    X_norm: np.ndarray,
    feat_mean: np.ndarray,
    feat_std: np.ndarray) -> np.ndarray:
    """
    Denormalize a batch of history windows from normalised aircraft-centric
    space back to physical aircraft-centric units.

    Args:
        X_norm: (N, T, 7) normalised histories.
        feat_mean, feat_std: (7,) arrays.

    Returns:
        (N, T, 7) denormalized.
    """
    return X_norm * feat_std[None, None, :] + feat_mean[None, None, :]

# ── Label names for plotting ─────────────────────────────────────────

VERTICAL_NAMES = {
    0: "Level",
    1: "Climbing",
    2: "Descending",
    3: "Level-off (↑)",
    4: "Level-off (↓)",
}

LATERAL_NAMES = {
    0: "Straight",
    1: "Turn Left",
    2: "Turn Right",
    3: "Roll-out (L)",
    4: "Roll-out (R)",
}

def intent_name(idx: int) -> str:
    v = idx // 5
    l_ = idx % 5
    return f"{VERTICAL_NAMES[v]} / {LATERAL_NAMES[l_]}"

def all_intent_names() -> List[str]:
    return [intent_name(i) for i in range(25)]