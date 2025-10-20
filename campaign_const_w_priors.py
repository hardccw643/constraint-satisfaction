#!/usr/bin/env python3
"""
Feasibility-first active learning (constraints-first) with priors enabled.

- Acquisition: acq_full = Total_Prob_With_BCC
- Logs fixed-range scaled HV (AFTER pick) to keep apples-to-apples with MOBO.
- Flags if the chosen point is Pareto on the measured set after adding it.
- Plots:
  * Progress plot each iter colored by p_Pugh
  * Per-objective prediction maps each iter (ST, Density, YS, Pugh, p_BCC)
  * TRUE feasible alloys overlaid as WHITE CROSSES on top

Priors (enabled as requested):
- YS Prior         = splice['YS 25C PRIOR'] + N(0, 50)
- Density Prior    = splice['Density Avg'] + N(0, 0.5)
- ST Prior (Tm)    = splice['Tm Avg']
- VEC (BCC Prior)  = ±5 with threshold 6.87 (uses 'VEC' or 'VEC Avg' if present)
- 600C BCC Total   = ±5 (sum of 600C BCC cols > 0.99 ⇒ +5, else -5)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Tuple, Iterable, Optional, List

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

from concurrent.futures import ProcessPoolExecutor, as_completed

# =================== CONFIG / CONSTANTS ===================

RESULTS_DIR = "results_const_prior"
PLOTS_DIR = "results_const_prior"           # overridden per-seed by _run_seed
PLOT_PREFIX = "affine_progress_prior"       # overridden per-seed by _run_seed
PRED_PLOT_PREFIX = "affine_pred_prior"      # overridden per-seed by _run_seed

# Feasibility thresholds
ST_THRESH = 2200 + 273  # K
DENSITY_THRESH = 9.0    # g/cm^3 (must be <)
YS_THRESH = 700         # MPa      (must be >)
PUGH_THRESH = 2.5       #          (must be >)

# Reference point r in the *maximize* space [ST, -Density, YS, Pugh]
REF_ST = 0
REF_DENSITY = 30
REF_YS = 0
REF_PUGH = 0
REF_POINT = np.array([REF_ST, -REF_DENSITY, REF_YS, REF_PUGH], dtype=float)

# Numerical safety
EPS = 1e-12

# Elemental columns used as inputs
ELEM_COLS = ["Nb", "Mo", "Ta", "V", "W", "Cr"]

# Single-phase BCC truth flag (5 == single-phase, -5 otherwise)
BCC_SINGLE_VALUE = 5.0

# Will hold dataset-fixed ranges for scaled HV
FIXED_RANGES: Optional[np.ndarray] = None  # (4,)

# =================== Hypervolume (EXACT; NOT EHVI) ===================

def _pareto_filter_non_dominated(P: np.ndarray) -> np.ndarray:
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
    if points_max_space.size == 0:
        return 0.0
    P = np.maximum(points_max_space - ref_point, 0.0)
    if np.all(P <= 0.0):
        return 0.0
    return _hypervolume_recursive(P)

def nondominated_mask(points: np.ndarray) -> np.ndarray:
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

def hypervolume_in_scaled_space(points_max_space: np.ndarray,
                                ref_point: np.ndarray,
                                ranges: np.ndarray) -> float:
    """HV in shifted+scaled maximize space (shift by REF_POINT, divide by fixed per-dim ranges)."""
    if points_max_space.size == 0:
        return 0.0
    shifted = points_max_space - ref_point
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
    """
    Enable priors as requested; attach elemental fractions; build BCC targets.
    """
    df = pd.DataFrame()

    # ---- Objectives and their priors ----
    df["YS 600C"] = splice["YS 600 C PRIOR"]
    df["YS Prior"] = splice['YS 25C PRIOR'] #+ np.random.normal(0, 50, size=splice.shape[0])

    df["Density"] = splice["PROP 25C Density (g/cm3)"]
    df["Density Prior"] = splice['Density Avg'] #+ np.random.normal(0, 0.5, size=splice.shape[0])

    df["Pugh Ratio"] = splice["Pugh_Ratio_PRIOR"]

    df["Solidus Temp"] = splice["PROP ST (K)"] #+ np.random.normal(0, 100, size=splice.shape[0])
    df["Melting Temp (ST Prior)"] =  splice['Tm Avg'] #splice['Tm Avg']

    # ---- Phase/BCC fields ----
    cols_600_bcc = [c for c in splice.columns if ("600C" in c and "BCC" in c)]
    df["600C BCC Total"] = splice[cols_600_bcc].sum(axis=1)
    df["600C BCC Total"] = np.where(df["600C BCC Total"] > 0.99, 5.0, -5.0)

    #VEC prior → latent ±5 with threshold 6.87 (use 'VEC' then fallback to 'VEC Avg')
    threshold = 6.87
    vec_col = "VEC" if "VEC" in splice.columns else ("VEC Avg" if "VEC Avg" in splice.columns else None)
    if vec_col is None:
        raise ValueError("Neither 'VEC' nor 'VEC Avg' found in splice columns.")
    df["VEC (BCC Prior)"] = np.where(splice[vec_col] >= threshold, 1.0, -1.0)

    # ---- Elemental fractions (assumed columns 6..12 in splice) ----
    df = df.merge(splice.iloc[:, 6:12], left_index=True, right_index=True)

    # Tidy
    df = df.dropna().reset_index(drop=True)

    # Ensure element columns present and numeric
    missing = [c for c in ELEM_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing elemental columns: {missing}")
    for c in ELEM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df

def current_observed_objectives(df_train_measured: pd.DataFrame) -> np.ndarray:
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
    Return:
      means:  [mu_ST, -mu_Density, mu_YS, mu_Pugh] in maximize space
      sigmas: [sd_ST_res, sd_Density_res, sd_YS_res, sd_Pugh]
      mu_bcc_logit: latent for BCC GP + VEC prior
      sd_bcc: std for BCC GP
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

def select_next_by_acq(acq_full: np.ndarray, df: pd.DataFrame, df_pool: pd.DataFrame) -> int:
    pool_labels = df_pool.index.to_numpy()
    best_rel = int(np.argmax(acq_full[pool_labels]))
    return int(pool_labels[best_rel])

# =================== Main loop ===================

def run_campaign(seed: int = 0, iterations: int = 100) -> None:
    global FIXED_RANGES

    np.random.seed(seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print(f"Running campaign seed={seed}")

    splice = pd.read_csv("EQUIL_STITCH.csv")
    df = prepare_dataframe(splice)

    truth_pass = (
        (df["Density"] < DENSITY_THRESH) &
        (df["YS 600C"] > YS_THRESH) &
        (df["Pugh Ratio"] > PUGH_THRESH) &
        (df["Solidus Temp"] > ST_THRESH) &
        (df["600C BCC Total"] == BCC_SINGLE_VALUE)
    ).astype(int).to_numpy()
    # indices of TRUE-feasible alloys (all constraints incl. BCC)
    true_idx_all_wbcc = np.flatnonzero(truth_pass == 1).tolist()
    num_true_feasible = len(true_idx_all_wbcc)
    total_points = int(df.shape[0])

    # ---- Print the requested counts ----
    print(f"[Data] Total points: {total_points}")
    print(f"[Data] True feasible (all constraints incl. BCC): {num_true_feasible}")

    # Compute fixed (non-changing) ranges ONCE from dataset min/max in maximize space
    all_points_max = np.column_stack([
        df["Solidus Temp"].to_numpy(float),
        -df["Density"].to_numpy(float),
        df["YS 600C"].to_numpy(float),
        df["Pugh Ratio"].to_numpy(float),
    ])
    shifted = all_points_max - REF_POINT
    smin_fix = np.min(shifted, axis=0)
    smax_fix = np.max(shifted, axis=0)
    FIXED_RANGES = np.maximum(smax_fix - smin_fix, EPS)
    print("[Fixed ranges] ST, -Den, YS, Pugh =", FIXED_RANGES.tolist())

    bcc_mask = (df["600C BCC Total"].to_numpy() == BCC_SINGLE_VALUE)
    if not np.any(bcc_mask):
        raise RuntimeError("No single-phase BCC alloys available to seed the campaign.")
    initial_idx = np.random.choice(np.where(bcc_mask)[0], 1, replace=False)

    obj_train_cols = ELEM_COLS + [
        "Density", "Density Prior", "YS 600C", "YS Prior", "Pugh Ratio",
        "600C BCC Total", "Solidus Temp", "Melting Temp (ST Prior)", "VEC (BCC Prior)"
    ]
    phase_train_cols = ELEM_COLS + ["600C BCC Total", "VEC (BCC Prior)"]

    df_train_measured = df.loc[initial_idx, obj_train_cols].copy().reset_index(drop=True)
    df_train_phase = df.loc[initial_idx, phase_train_cols].copy().reset_index(drop=True)
    df_pool = df.drop(index=initial_idx)

    models = build_models(seed)

    # Bookkeeping
    measured_indices = set(initial_idx.tolist())   # indices with objective measurements
    unmeasurable_indices = set()                   # chosen but non-BCC (phase-only)
    t0 = time.time()
    hv_raw_prev = 0.0

    measured_indices = set(initial_idx.tolist())
    unmeasurable_indices = set()
    t0 = time.time()

    for it in range(iterations):
        it0 = time.time()

        # --- Train OBJECTIVE GPs (measured only) ---
        X_train_obj = df_train_measured[ELEM_COLS].to_numpy(float)

        y_den_res = df_train_measured["Density"].to_numpy(float) - df_train_measured["Density Prior"].to_numpy(float)
        y_ys_res  = df_train_measured["YS 600C"].to_numpy(float) - df_train_measured["YS Prior"].to_numpy(float)
        y_pugh    = df_train_measured["Pugh Ratio"].to_numpy(float)
        y_st_res  = df_train_measured["Solidus Temp"].to_numpy(float) - df_train_measured["Melting Temp (ST Prior)"].to_numpy(float)

        models.density.fit(X_train_obj, y_den_res)
        models.ys.fit(X_train_obj, y_ys_res)
        models.pugh.fit(X_train_obj, y_pugh)
        models.st.fit(X_train_obj, y_st_res)

        # --- Train PHASE GP (all observed phase labels) ---
        X_train_phase = df_train_phase[ELEM_COLS].to_numpy(float)
        y_bcc_res = df_train_phase["600C BCC Total"].to_numpy(float)  # ±5 regression label
        models.bcc.fit(X_train_phase, y_bcc_res)

        # --- Predictions over full design space ---
        X_full = df[ELEM_COLS].to_numpy(float)
        means_full, sigmas_full, mu_bcc_logit, sd_bcc = gp_predict_all(models, X_full, df)  # [ST, -DEN, YS, Pugh]
        p_bcc = sigmoid(mu_bcc_logit)
        df["p_bcc"] = np.clip(p_bcc, 0.0, 1.0)

        # Expose means/stds (also back-transform density mean)
        mu_st  = means_full[:, 0]
        mu_den = -(means_full[:, 1])
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

        # ---------- Feasibility probabilities ----------
        p_den  = norm.cdf((DENSITY_THRESH - mu_den) / sd_den)         # P(Density < thresh)
        p_ys   = 1.0 - norm.cdf((YS_THRESH   - mu_ys)  / sd_ys)       # P(YS > thresh)
        p_pugh = 1.0 - norm.cdf((PUGH_THRESH - mu_pug) / sd_pug)      # P(Pugh > thresh)
        p_st   = 1.0 - norm.cdf((ST_THRESH   - mu_st)  / sd_st)       # P(ST > thresh)
        df["p_Density"] = np.clip(p_den, 0.0, 1.0)
        df["p_YS"]      = np.clip(p_ys, 0.0 , 1.0)
        df["p_Pugh"]    = np.clip(p_pugh, 0.0   , 1.0)
        df["p_ST"]      = np.clip(p_st, 0.0 , 1.0)

        total_prob_no_bcc   = np.clip(p_den * p_ys * p_pugh * p_st, 0.0, 1.0)
        total_prob_with_bcc = np.clip(total_prob_no_bcc * df["p_bcc"].to_numpy(), 0.0, 1.0)
        df["Prob_All_NoBCC"] = total_prob_no_bcc
        df["Total_Prob_With_BCC"] = total_prob_with_bcc

        # --------- Acquisition (feasibility-first) ----------
        acq_full = total_prob_with_bcc
        df["Acq"] = acq_full
        df["AcqNorm"] = acq_full
        df["AcqPlotVal"] = acq_full

        # Predicted labels (for metrics): require >0.5 & p_bcc>0.5
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
        next_idx = select_next_by_acq(acq_full, df, df_pool)

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

# =================== Optional: run multiple seeds in parallel ===================

def _run_seed(seed: int, iterations: int = 100) -> int:
    """
    Worker: runs one campaign for a given seed.
    - Limits BLAS/OpenMP threads to 1 per process (prevents CPU oversubscription).
    - Puts plots for this seed into a unique subfolder and prefixes filenames with the seed.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    seed_plot_dir = os.path.join(".", f"plots_seed_{seed:03d}")
    os.makedirs(seed_plot_dir, exist_ok=True)

    g = globals()
    if "PLOTS_DIR" in g:
        g["PLOTS_DIR"] = seed_plot_dir
    if "PLOT_PREFIX" in g:
        g["PLOT_PREFIX"] = f"affine_progress_s{seed:03d}"
    if "PRED_PLOT_PREFIX" in g:
        g["PRED_PLOT_PREFIX"] = f"affine_pred_s{seed:03d}"
    if "RESULTS_DIR" in g:
        os.makedirs(g["RESULTS_DIR"], exist_ok=True)

    run_campaign(seed=seed, iterations=iterations)
    return seed

if __name__ == "__main__":
    # Adjust seeds/iterations/workers as needed
    seeds = list(range(1, 200+1))   # single seed by default
    iterations = 100
    workers = 25
    print(f"[main] Launching {len(seeds)} seeds with {workers} workers...")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_run_seed, s, iterations) for s in seeds]
        for fut in as_completed(futures):
            s = fut.result()
            print(f"[main] seed {s} finished.")
    print("[main] All seeds done.")
