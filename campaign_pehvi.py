#!/usr/bin/env python3
"""
Active learning with closed-form pEHVI (NO true-Pareto progress):

- Acquisition: pEHVI([ST, -Density, YS, Pugh]) in per-iteration
  shifted+scaled space (shift by REF_POINT, divide by per-iter ranges).
- Selection uses acquisition = pEHVI(...) * p_bcc (p_bcc applied ONCE).

- Reporting (non-changing hypervolume):
  * Compute ONE fixed scaling at the start (per seed) from dataset min/max
    of the four objectives (maximize space). Switch scope via FIXED_RANGE_SCOPE.
  * Each iteration we report:
      - HypervolumeScaledFixedRange : HV of measured set in that fixed scale
      - Hypervolume                 : HV in raw units (monotone)

- NEW: CSV includes ChosenIsParetoMeasured = "Yes"/"No"
  "Yes" iff the just-chosen point is in the nondominated set of the
  measured set AFTER it’s added (when BCC-single). Otherwise "No".

Outputs:
- CSV: results_opt/campaign_<seed>.csv (appended per iteration)
- Plots per iteration into plots_seed_<seed>/

Author: Brent + ChatGPT
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import Tuple, Iterable, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from scipy.stats import norm
from scipy.special import expit as sigmoid
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss, log_loss
)

# ---- Quiet mode: mute warnings ----
import warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.optimize import OptimizeWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)

# =================== CONFIG / CONSTANTS ===================

RESULTS_DIR = "results_opt"
PLOTS_DIR = "results_opt"
PLOT_PREFIX = "affine_progress_opt"
PRED_PLOT_PREFIX = "affine_pred_opt"

# Feasibility thresholds (reference point for EHVI)
ST_THRESH = 2200 + 273
DENSITY_THRESH = 9.0
YS_THRESH = 700
PUGH_THRESH = 2.5

REF_ST = 0
REF_DENSITY = 30
REF_YS = 0
REF_PUGH = 0
# Reference point in the *maximize* space [ST, -Density, YS, Pugh]
REF_POINT = np.array([REF_ST, -REF_DENSITY, REF_YS, REF_PUGH], dtype=float)

# Numerical safety
EPS = 1e-12

# Elemental columns used as inputs
ELEM_COLS = ["Nb", "Mo", "Ta", "V", "W", "Cr"]

# Single-phase BCC truth flag (5 == single-phase, -5 otherwise)
BCC_SINGLE_VALUE = 5.0
VEC_THRESHOLD = 6.87

# ---- Fixed scaling (computed ONCE at start of each seed run) ----
# Choose whether to compute fixed ranges over ALL rows or only BCC-single rows.
FIXED_RANGE_SCOPE = "ALL"     # "ALL" | "BCC_ONLY"
FIXED_RANGES = None           # np.ndarray shape (4,)

DATA_CANDIDATES = ("design_space.xlsx", "design_space.csv")


def load_design_space() -> pd.DataFrame:
    """Load the design space from design_space.(xlsx|csv)."""
    for path in DATA_CANDIDATES:
        if os.path.exists(path):
            if path.lower().endswith(".xlsx"):
                return pd.read_excel(path)
            return pd.read_csv(path)
    raise FileNotFoundError(
        "Expected design_space.xlsx or design_space.csv in this directory."
    )

# =================== Hypervolume (EXACT; NOT EHVI) ===================

def _pareto_filter_non_dominated(P: np.ndarray) -> np.ndarray:
    """Keep points not dominated by any other (>= in all dims and > in at least one)."""
    if P.size == 0:
        return P
    K = P.shape[0]
    keep = np.ones(K, dtype=bool)
    for i in range(K):
        if not keep[i]:
            continue
        for j in range(K):
            if i == j or not keep[j]:
                continue
            if np.all(P[j] >= P[i]) and np.any(P[j] > P[i]):
                keep[i] = False
                break
    return P[keep]

def _hypervolume_recursive(P: np.ndarray) -> float:
    """Exact HV of union of boxes [0, p_i] in R^m via recursive slicing on the last dimension."""
    P = _pareto_filter_non_dominated(P)
    if P.size == 0:
        return 0.0
    m = P.shape[1]
    if m == 1:
        return float(np.max(P[:, 0]))
    z_vals = np.unique(P[:, -1])
    prev = 0.0
    vol = 0.0
    for z in z_vals:
        mask = P[:, -1] >= z - 1e-15
        proj = P[mask, :-1]
        area = _hypervolume_recursive(proj)
        dz = float(z - prev)
        if dz > 0.0:
            vol += area * dz
            prev = float(z)
    return float(vol)

def hypervolume_exact(points_max_space: np.ndarray, ref_point: np.ndarray) -> float:
    """Exact dominated hypervolume of `points_max_space` (maximize convention) w.r.t. `ref_point`."""
    if points_max_space.size == 0:
        return 0.0
    P = np.maximum(points_max_space - ref_point, 0.0)  # shift to ref and clamp to +orthant
    if np.all(P <= 0.0):
        return 0.0
    return _hypervolume_recursive(P)

def nondominated_mask(points: np.ndarray) -> np.ndarray:
    """General non-dominated mask in maximize space (no positivity assumption)."""
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    K = points.shape[0]
    keep = np.ones(K, dtype=bool)
    for i in range(K):
        if not keep[i]:
            continue
        for j in range(K):
            if i == j or not keep[j]:
                continue
            if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                keep[i] = False
                break
    return keep

def hypervolume_in_scaled_space(pareto_obs: np.ndarray,
                                ref_point: np.ndarray,
                                ranges: np.ndarray) -> float:
    """HV in shifted+scaled maximize space (shift by REF_POINT, divide by per-dim 'ranges')."""
    shifted = pareto_obs - ref_point
    scaled  = shifted / np.maximum(ranges, EPS)
    return hypervolume_exact(scaled, np.zeros_like(ref_point))


# =================== Data structures & helpers ===================

@dataclass
class Models:
    density: GaussianProcessRegressor
    ys: GaussianProcessRegressor
    pugh: GaussianProcessRegressor
    bcc: GaussianProcessRegressor
    st: GaussianProcessRegressor

def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makemks(path, exist_ok=True)

def build_models(seed: int) -> Models:
    base_kernel = RBF(length_scale=[1.0] * len(ELEM_COLS),
                      length_scale_bounds=(0.3, 2)) * ConstantKernel() + WhiteKernel()
    clsf_kernel = RBF(length_scale=[1.0] * len(ELEM_COLS),
                      length_scale_bounds=(0.3, 2))

    def mk_norm():
        return GaussianProcessRegressor(
            kernel=base_kernel, random_state=seed, normalize_y=True, n_restarts_optimizer=10
        )

    def mk_bcc():
        return GaussianProcessRegressor(
            kernel=clsf_kernel, random_state=seed, normalize_y=True, n_restarts_optimizer=10
        )

    return Models(
        density=mk_norm(),
        ys=mk_norm(),
        pugh=mk_norm(),
        bcc=mk_bcc(),
        st=mk_norm(),
    )

def prepare_dataframe(splice: pd.DataFrame) -> pd.DataFrame:
    """Create the working dataframe with priors and truth columns."""
    df = pd.DataFrame()
    df["YS 600C"] = splice["YS 600 C PRIOR"]
    df["YS Prior"] = 0.0

    df["Density"] = splice["PROP 25C Density (g/cm3)"]
    df["Density Prior"] = 0.0

    df["Pugh Ratio"] = splice["Pugh_Ratio_PRIOR"]
    df["Solidus Temp"] = splice["PROP ST (K)"]
    df["Melting Temp (ST Prior)"] = 0.0

    # BCC phase fraction at 600C -> truth flag: 5 (single-phase), else -5
    cols_600_bcc = [c for c in splice.columns if ("600C" in c and "BCC" in c)]
    df["600C BCC Total"] = splice[cols_600_bcc].sum(axis=1)
    df["600C BCC Total"] = np.where(df["600C BCC Total"] > 0.99, 5.0, -5.0)

    # BCC prior: default 50/50 → logit(0.5) = 0.0 everywhere
    df["VEC (BCC Prior)"] = 0.0

    # Elemental composition columns (assumed at slice 6:12)
    df = df.merge(splice.iloc[:, 6:12], left_index=True, right_index=True)

    # Clean NA and reset index
    df = df.dropna().reset_index(drop=True)
    return df

def current_observed_objectives(df_train_measured: pd.DataFrame) -> np.ndarray:
    """(K,4) measured objective matrix in maximize space."""
    if df_train_measured.shape[0] == 0:
        return np.zeros((0, 4), dtype=float)
    return np.column_stack([
        df_train_measured["Solidus Temp"].to_numpy(float),
        -df_train_measured["Density"].to_numpy(float),
        df_train_measured["YS 600C"].to_numpy(float),
        df_train_measured["Pugh Ratio"].to_numpy(float),
    ])

def gp_predict_all(models: Models, X: np.ndarray, df_rows: pd.DataFrame
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict means/stds for objectives + BCC on X matching df rows.
    Returns:
      means:  (N, 4) [ST, -Density, YS, Pugh]  (NOT shifted)
      sigmas: (N, 4) std devs (residual for prior-based ones)
      mu_bcc_logit: (N,) predicted BCC logit (residual + prior=0)
      sd_bcc: (N,) predicted std dev of BCC residual
    """
    mu_den_res, sd_den_res = models.density.predict(X, return_std=True)
    mu_den = mu_den_res + df_rows["Density Prior"].to_numpy(float)

    mu_ys_res, sd_ys_res = models.ys.predict(X, return_std=True)
    mu_ys = mu_ys_res + df_rows["YS Prior"].to_numpy(float)

    mu_pugh, sd_pugh = models.pugh.predict(X, return_std=True)

    mu_bcc_res, sd_bcc = models.bcc.predict(X, return_std=True)
    mu_bcc_logit = mu_bcc_res + df_rows["VEC (BCC Prior)"].to_numpy(float)

    mu_st_res, sd_st_res = models.st.predict(X, return_std=True)
    mu_st = mu_st_res + df_rows["Melting Temp (ST Prior)"].to_numpy(float)

    means = np.column_stack([mu_st, -mu_den, mu_ys, mu_pugh])  # -density to maximize
    sigmas = np.column_stack([sd_st_res, sd_den_res, sd_ys_res, sd_pugh])
    sigmas = np.maximum(sigmas, EPS)
    return means, sigmas, mu_bcc_logit, sd_bcc

def select_next_via_ehvi(
    acq_full: np.ndarray,
    df: pd.DataFrame,
    df_pool: pd.DataFrame,
) -> int:
    """Select argmax of acquisition over pool labels."""
    pool_labels = df_pool.index.to_numpy()
    best_rel = int(np.argmax(acq_full[pool_labels]))
    return int(pool_labels[best_rel])

# =================== pEHVI implementation ===================

def pEHVI_max_all_candidates(
    means: np.ndarray,
    sigmas: np.ndarray,
    ref: np.ndarray,
    pareto: np.ndarray
) -> np.ndarray:
    """
    Compute pointwise Expected Hypervolume Improvement (pEHVI) for all candidates
    in a vectorized manner. Inputs are in maximize-space, shifted by REF_POINT, and
    scaled by per-dimension 'ranges' (divide by smax - smin of the current iteration).
    """
    N, D = means.shape
    sig = np.maximum(sigmas, EPS)

    # Box (unconditional EI) from ref to each mean
    s_up = (means - ref) / sig
    up = (means - ref) * norm.cdf(s_up) + sig * norm.pdf(s_up)
    box = np.prod(up, axis=1)

    if pareto.size == 0:
        return np.maximum(box, 0.0)

    P = pareto.reshape(-1, D)
    pehvi = np.full(N, np.inf, dtype=float)
    for k in range(P.shape[0]):
        p = P[k]
        s_low = (means - p) / sig
        low = (means - p) * norm.cdf(s_low) + sig * norm.pdf(s_low)
        diff = np.maximum(up - low, 0.0)
        dominated_single = np.prod(diff, axis=1)
        ehvi_k = box - dominated_single
        pehvi = np.minimum(pehvi, ehvi_k)
    return np.maximum(pehvi, 0.0)

# =================== Main loop ===================

def run_campaign(seed: int = 0, iterations: int = 100) -> None:
    global FIXED_RANGES

    np.random.seed(seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Running campaign seed={seed}")

    # Load and prep
    splice = load_design_space()
    scaled_space = float(splice["PROP 25C Density (g/cm3)"].max(skipna=True)) <= 1.5
    global DENSITY_THRESH, YS_THRESH, PUGH_THRESH, ST_THRESH, VEC_THRESHOLD
    if scaled_space:
        DENSITY_THRESH = 0.218912147251372
        YS_THRESH = 0.27326687068841815
        PUGH_THRESH = 0.34208243243243236
        ST_THRESH = 0.3340611001897914
        VEC_THRESHOLD = 1.0
    else:
        DENSITY_THRESH = 9.0
        YS_THRESH = 700.0
        PUGH_THRESH = 2.5
        ST_THRESH = 2200.0 + 273.0
        VEC_THRESHOLD = 6.87
    df = prepare_dataframe(splice)

    # Ground-truth labels INCLUDING BCC requirement
    truth_pass = (
        (df["Density"] < DENSITY_THRESH) &
        (df["YS 600C"] > YS_THRESH) &
        (df["Pugh Ratio"] > PUGH_THRESH) &
        (df["Solidus Temp"] > ST_THRESH) &
        (df["600C BCC Total"] == BCC_SINGLE_VALUE)
    ).astype(int).to_numpy()

    # Accessible universe
    bcc_mask_all = (df["600C BCC Total"].to_numpy() == BCC_SINGLE_VALUE)
    if not np.any(bcc_mask_all):
        raise RuntimeError("No single-phase BCC alloys available to seed the campaign.")

    # All points in maximize-space
    all_points_max = np.column_stack([
        df["Solidus Temp"].to_numpy(float),
        -df["Density"].to_numpy(float),
        df["YS 600C"].to_numpy(float),
        df["Pugh Ratio"].to_numpy(float),
    ])

    # --------- FIXED (non-changing) ranges from dataset min/max ----------
    if FIXED_RANGE_SCOPE == "BCC_ONLY":
        pts_for_range = all_points_max[bcc_mask_all]
    else:
        pts_for_range = all_points_max

    shifted_for_range = pts_for_range - REF_POINT
    smin_fix = np.min(shifted_for_range, axis=0)
    smax_fix = np.max(shifted_for_range, axis=0)
    FIXED_RANGES = np.maximum(smax_fix - smin_fix, EPS)

    print("[Fixed ranges] scope=", FIXED_RANGE_SCOPE, " | values=", FIXED_RANGES.tolist())

    # Seed initial measured set from BCC-single
    #initial_idx = np.random.choice(np.where(bcc_mask_all)[0], 10, replace=False)
    initial_idx = np.random.choice(np.where(bcc_mask_all)[0], 1, replace=False)

    # Columns
    obj_train_cols = ELEM_COLS + [
        "Density", "Density Prior", "YS 600C", "YS Prior", "Pugh Ratio",
        "600C BCC Total", "Solidus Temp", "Melting Temp (ST Prior)", "VEC (BCC Prior)"
    ]
    phase_train_cols = ELEM_COLS + ["600C BCC Total", "VEC (BCC Prior)"]

    # Two training tables:
    df_train_measured = df.loc[initial_idx, obj_train_cols].copy().reset_index(drop=True)
    df_train_phase = df.loc[initial_idx, phase_train_cols].copy().reset_index(drop=True)

    # Pool keeps labels
    df_pool = df.drop(index=initial_idx)

    # Models
    models = build_models(seed)

    # Bookkeeping
    measured_indices = set(initial_idx.tolist())   # indices with objective measurements
    unmeasurable_indices = set()                   # chosen but non-BCC (phase-only)
    t0 = time.time()
    hv_raw_prev = 0.0

    for it in range(iterations):
        it0 = time.time()

        # --- Train OBJECTIVE GPs on measured data only ---
        X_train_obj = df_train_measured[ELEM_COLS].to_numpy(float)

        y_den_res = df_train_measured["Density"].to_numpy(float) - df_train_measured["Density Prior"].to_numpy(float)
        y_ys_res  = df_train_measured["YS 600C"].to_numpy(float) - df_train_measured["YS Prior"].to_numpy(float)
        y_pugh    = df_train_measured["Pugh Ratio"].to_numpy(float)
        y_st_res  = df_train_measured["Solidus Temp"].to_numpy(float) - df_train_measured["Melting Temp (ST Prior)"].to_numpy(float)

        models.density.fit(X_train_obj, y_den_res)
        models.ys.fit(X_train_obj, y_ys_res)
        models.pugh.fit(X_train_obj, y_pugh)
        models.st.fit(X_train_obj, y_st_res)

        # --- Train PHASE GP on ALL observed phase labels ---
        X_train_phase = df_train_phase[ELEM_COLS].to_numpy(float)
        y_bcc_res = df_train_phase["600C BCC Total"].to_numpy(float)  # ±5 regression label
        models.bcc.fit(X_train_phase, y_bcc_res)

        # --- Predictions over full design space ---
        X_full = df[ELEM_COLS].to_numpy(float)
        means_full, sigmas_full, mu_bcc_logit, sd_bcc = gp_predict_all(models, X_full, df)  # [ST, -DEN, YS, Pugh]
        p_bcc = sigmoid(mu_bcc_logit)  # (N,)
        df["p_bcc"] = np.clip(p_bcc, 0.0, 1.0)

        # ===== Expose GP prediction means (for plotting/logging) =====
        mu_st  = means_full[:, 0]
        mu_den = -(means_full[:, 1])         # convert back to +Density
        mu_ys  = means_full[:, 2]
        mu_pug = means_full[:, 3]

        sd_st  = np.maximum(sigmas_full[:, 0], EPS)
        sd_den = np.maximum(sigmas_full[:, 1], EPS)
        sd_ys  = np.maximum(sigmas_full[:, 2], EPS)
        sd_pug = np.maximum(sigmas_full[:, 3], EPS)

        df["mu_ST"]        = mu_st
        df["mu_Density"]   = mu_den
        df["mu_YS"]        = mu_ys
        df["mu_Pugh"]      = mu_pug
        df["mu_BCC_Logit"] = mu_bcc_logit
        df["sd_ST"]        = sd_st
        df["sd_Density"]   = sd_den
        df["sd_YS"]        = sd_ys
        df["sd_Pugh"]      = sd_pug
        df["sd_BCC"]       = sd_bcc

        # ---------- pEHVI (per-iteration scaling in shifted space) ----------
        shifted_full = means_full - REF_POINT                         # (N,4)
        pareto_obs_before = current_observed_objectives(df_train_measured)   # (K,4)
        shifted_pareto_before = pareto_obs_before - REF_POINT

        smin = np.min(shifted_full, axis=0)
        smax = np.max(shifted_full, axis=0)
        ranges_dyn = np.maximum(smax - smin, EPS)

        shifted_full_scaled = shifted_full / ranges_dyn
        sigmas_scaled = sigmas_full / ranges_dyn
        shifted_pareto_scaled = shifted_pareto_before / ranges_dyn

        ref_min = np.zeros((1, shifted_full_scaled.shape[1]), dtype=float)
        pehvi_full = pEHVI_max_all_candidates(
            means=shifted_full_scaled,
            sigmas=sigmas_scaled,
            ref=ref_min,
            pareto=shifted_pareto_scaled,
        )

        # --------- Acquisition = pEHVI * p_bcc ----------
        acq_full = np.maximum(pehvi_full, 0.0) * df["p_bcc"].to_numpy()
        df["Acq"] = acq_full
        df["AcqNorm"] = acq_full
        df["AcqPlotVal"] = df["AcqNorm"]
        
        # ---------- Predicted feasibility ----------
        p_den = norm.cdf((DENSITY_THRESH - mu_den) / sd_den)
        p_ys  = 1.0 - norm.cdf((YS_THRESH - mu_ys) / sd_ys)
        p_pugh = 1.0 - norm.cdf((PUGH_THRESH - mu_pug) / sd_pug)
        p_st = 1.0 - norm.cdf((ST_THRESH - mu_st) / sd_st)

        total_prob_no_bcc = np.clip(p_den * p_ys * p_pugh * p_st, 0.0, 1.0)
        total_prob_with_bcc = np.clip(total_prob_no_bcc * df["p_bcc"].to_numpy(), 0.0, 1.0)
        df["Total_Prob_With_BCC"] = total_prob_with_bcc


        pred_labels_full = ((total_prob_no_bcc > 0.5) & (df["p_bcc"].to_numpy() > 0.5)).astype(int)

        # ---- Measured-set tallies + current Pareto count (BEFORE pick) ----
        pareto_obs_before = current_observed_objectives(df_train_measured)
        if len(df_train_measured) > 0:
            den_meas_pass  = (df_train_measured["Density"].to_numpy(float)      < DENSITY_THRESH)
            ys_meas_pass   = (df_train_measured["YS 600C"].to_numpy(float)      > YS_THRESH)
            pugh_meas_pass = (df_train_measured["Pugh Ratio"].to_numpy(float)   > PUGH_THRESH)
            st_meas_pass   = (df_train_measured["Solidus Temp"].to_numpy(float) > ST_THRESH)
            bcc_meas_pass  = (df_train_measured["600C BCC Total"].to_numpy(float) == BCC_SINGLE_VALUE)

            all_no_bcc_meas   = den_meas_pass & ys_meas_pass & pugh_meas_pass & st_meas_pass
            all_with_bcc_meas = all_no_bcc_meas & bcc_meas_pass

            n_true_meas_pass_density   = int(den_meas_pass.sum())
            n_true_meas_pass_ys        = int(ys_meas_pass.sum())
            n_true_meas_pass_pugh      = int(pugh_meas_pass.sum())
            n_true_meas_pass_st        = int(st_meas_pass.sum())
            n_true_meas_pass_bcc       = int(bcc_meas_pass.sum())
            n_true_meas_pass_all_nbcc  = int(all_no_bcc_meas.sum())
            n_true_meas_pass_all_wbcc  = int(all_with_bcc_meas.sum())

            mask_nd_before = nondominated_mask(pareto_obs_before)
            true_pareto_count = int(mask_nd_before.sum())
        else:
            n_true_meas_pass_density = n_true_meas_pass_ys = n_true_meas_pass_pugh = 0
            n_true_meas_pass_st = n_true_meas_pass_bcc = 0
            n_true_meas_pass_all_nbcc = n_true_meas_pass_all_wbcc = 0
            true_pareto_count = 0

        # ---------- Hypervolume (raw, BEFORE pick) ----------
        hv_raw_before = hypervolume_exact(pareto_obs_before, REF_POINT)

        # ---------- Select next via acquisition ----------
        next_idx = select_next_via_ehvi(acq_full, df, df_pool)

        # ---------- Observe phase; conditionally observe objectives ----------
        true_is_bcc_single = (float(df.loc[next_idx, "600C BCC Total"]) == BCC_SINGLE_VALUE)

        # Always add phase label
        df_train_phase = pd.concat(
            [df_train_phase, df.loc[[next_idx], phase_train_cols]],
            ignore_index=True
        )

        # Append measured objectives if feasible, and compute chosen Pareto flag
        if true_is_bcc_single:
            next_point_obj = df.loc[next_idx, obj_train_cols].copy()
            df_train_measured = pd.concat([df_train_measured, next_point_obj.to_frame().T], ignore_index=True)
            measured_indices.add(int(next_idx))
            measured_flag = 1

            pareto_obs_after = current_observed_objectives(df_train_measured)
            mask_nd_after = nondominated_mask(pareto_obs_after)
            chosen_is_pareto_measured = bool(mask_nd_after[-1]) if len(mask_nd_after) > 0 else False
        else:
            unmeasurable_indices.add(int(next_idx))
            measured_flag = 0
            chosen_is_pareto_measured = False
            pareto_obs_after = pareto_obs_before  # no change

        # Remove from pool regardless
        df_pool = df_pool.drop(index=next_idx)

        # ---------- Fixed-range scaled HV (AFTER pick) ----------
        hv_scaled_fixed_after = hypervolume_in_scaled_space(pareto_obs_after, REF_POINT, FIXED_RANGES)

        # ---------- Log row (single CSV, updated) ----------
        # Note: HV values below are the "before-pick" status for consistency with prior logs.
        # If you prefer "after-pick" HVs, recompute using df_train_measured now and log those.
        hv_raw_gain = hv_raw_before - hv_raw_prev
        hv_raw_prev = hv_raw_before

        # ---------- Log row ----------
        row = {
            "Iteration": it,
            "ChosenIndex": int(next_idx),
            "Acquisition": float(acq_full[next_idx]),
            "Pred_ProbWithBCC": float(total_prob_with_bcc[next_idx]),
            "Pred_Class(>0.5 incl. BCC)": int(pred_labels_full[next_idx]),
            "Observed_Phase": float(df.loc[next_idx, "600C BCC Total"]),
            "MeasuredObjectives(BCC single)": int(measured_flag),
            "CumulativeMeasured": len(measured_indices),
            "CumulativePhaseOnly": len(unmeasurable_indices),
            "PoolRemaining": int(df_pool.shape[0]),

            # NEW: Pareto flag and fixed-scale HV (AFTER pick)
            "ChosenIsParetoMeasured": "Yes" if chosen_is_pareto_measured else "No",
            "HypervolumeScaledFixedRange": float(hv_scaled_fixed_after),

            # Raw HV (BEFORE pick)
            "Hypervolume": float(hv_raw_before),
            "HypervolumeRawGain": float(hv_raw_gain),
            "HypervolumeEstimate": float(hv_raw_before),
            "TrueParetoCount": true_pareto_count,

            # Metrics
            "Accuracy": accuracy_score(truth_pass, pred_labels_full),
            "Precision": precision_score(truth_pass, pred_labels_full, zero_division=0),
            "Recall": recall_score(truth_pass, pred_labels_full, zero_division=0),
            "F1": f1_score(truth_pass, pred_labels_full, zero_division=0),
            "BrierLoss": brier_score_loss(truth_pass, total_prob_with_bcc),
            "LogLoss": log_loss(truth_pass,
                                np.column_stack([1 - total_prob_with_bcc, total_prob_with_bcc]),
                                labels=[0, 1]),
            # Measured-set tallies (BEFORE pick)
            "TrueMeasPass_Density":        n_true_meas_pass_density,
            "TrueMeasPass_YS":             n_true_meas_pass_ys,
            "TrueMeasPass_Pugh":           n_true_meas_pass_pugh,
            "TrueMeasPass_ST":             n_true_meas_pass_st,
            "TrueMeasPass_BCC":            n_true_meas_pass_bcc,
            "TrueMeasPass_All_NoBCC":      n_true_meas_pass_all_nbcc,
            "TrueMeasPass_All_WithBCC":    n_true_meas_pass_all_wbcc,

            # Fixed scaling components (for reproducibility)
            "FixedScale_ST": float(FIXED_RANGES[0]),
            "FixedScale_negDensity": float(FIXED_RANGES[1]),
            "FixedScale_YS": float(FIXED_RANGES[2]),
            "FixedScale_Pugh": float(FIXED_RANGES[3]),
        }

        # GP predictions for the CHOSEN alloy (means + uncertainty)
        row.update({
            "ChosenPred_mu_ST":        float(mu_st[next_idx]),
            "ChosenPred_sd_ST":        float(sd_st[next_idx]),
            "ChosenPred_mu_Density":   float(mu_den[next_idx]),
            "ChosenPred_sd_Density":   float(sd_den[next_idx]),
            "ChosenPred_mu_YS":        float(mu_ys[next_idx]),
            "ChosenPred_sd_YS":        float(sd_ys[next_idx]),
            "ChosenPred_mu_Pugh":      float(mu_pug[next_idx]),
            "ChosenPred_sd_Pugh":      float(sd_pug[next_idx]),
            "ChosenPred_mu_BCC_Logit": float(mu_bcc_logit[next_idx]),
            "ChosenPred_sd_BCC":       float(sd_bcc[next_idx]),
            "ChosenPred_p_bcc":        float(p_bcc[next_idx]),
            "ChosenPred_TotalProbWithBCC": float(total_prob_with_bcc[next_idx]),
        })
        for c in ELEM_COLS:
            row[c] = float(df.loc[next_idx, c])

        run_path = os.path.join(RESULTS_DIR, f"campaign_{seed}.csv")
        if os.path.exists(run_path):
            prev = pd.read_csv(run_path)
            prev = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
            prev.to_csv(run_path, index=False)
        else:
            pd.DataFrame([row]).to_csv(run_path, index=False)

        it_dt = time.time() - it0
        status_str = "MEASURED objectives" if measured_flag else "PHASE-ONLY (non-BCC)"
        print(f"[Seed {seed}, Iter {it:03d}] idx={next_idx} | {status_str} | "
              f"Pareto?={'Yes' if chosen_is_pareto_measured else 'No'} | "
              f"acq={acq_full[next_idx]:.3e} | HV_fixed(before)={hv_raw_before:.3e} | "
              f"pool={df_pool.shape[0]} | {it_dt:.2f}s")

        if df_pool.empty:
            print("Pool exhausted, stopping.")
            break

    print(f"Total wall time: {time.time() - t0:.2f}s")


def _run_seed(seed: int, iterations: int = 100) -> int:
    """
    Worker: runs one campaign for a given seed.
    - Limits BLAS/OpenMP threads to 1 per process (prevents CPU oversubscription).
    - Puts plots for this seed into a unique subfolder and prefixes filenames with the seed.
    """
    # prevent thread oversubscription in each process
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # make per-seed plotting directory & prefixes
    seed_plot_dir = os.path.join(".", f"plots_seed_{seed:03d}")
    os.makedirs(seed_plot_dir, exist_ok=True)

    # Override globals (each process has its own copy)
    g = globals()
    if "PLOTS_DIR" in g:
        g["PLOTS_DIR"] = seed_plot_dir
    if "PLOT_PREFIX" in g:
        g["PLOT_PREFIX"] = f"affine_progress_s{seed:03d}"
    if "PRED_PLOT_PREFIX" in g:
        g["PRED_PLOT_PREFIX"] = f"affine_pred_s{seed:03d}"

    # ensure results dir exists
    if "RESULTS_DIR" in g:
        os.makedirs(g["RESULTS_DIR"], exist_ok=True)

    # run the campaign
    run_campaign(seed=seed, iterations=iterations)
    return seed

if __name__ == "__main__":
    # seeds 1..N, change iterations here if needed
    seeds = list(range(1, 200+1)) 
    iterations = 100

    # choose worker count (half your CPUs is a good starting point)
    #workers = min(len(seeds), max(1, mp.cpu_count() // 2))
    workers=25
    print(f"[main] Launching {len(seeds)} seeds with {workers} workers...")

    # also apply thread caps to children by setting in parent (belt & suspenders)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # run in parallel
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_run_seed, s, iterations) for s in seeds]
        for fut in as_completed(futures):
            s = fut.result()
            print(f"[main] seed {s} finished.")
    print("[main] All seeds done.")
