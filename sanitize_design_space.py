#!/usr/bin/env python3
"""
Create a shareable design-space CSV containing only the fields required by the
campaign scripts while obscuring absolute Thermo-Calc outputs.

Steps:
1. Keep the elemental fractions used by the models (Nb, Mo, Ta, V, W, Cr) and
   rename them to element_01 ... element_06.
2. Retain the objective/prior columns and 600 °C BCC phase fractions that the
   scripts rely on.
3. Min–max scale every column that begins with PROP or EQUIL, as well as the
   prior columns.
4. Apply ±1% multiplicative jitter to every numeric column for additional
   obfuscation (values are clipped to stay within sensible bounds).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

ELEMENT_COLUMNS: Sequence[str] = ("Nb", "Mo", "Ta", "V", "W", "Cr")

REQUIRED_COLUMNS: Sequence[str] = (
    "YS 600 C PRIOR",
    "YS 25C PRIOR",
    "PROP 25C Density (g/cm3)",
    "Density Avg",
    "Pugh_Ratio_PRIOR",
    "PROP ST (K)",
    "Tm Avg",
    "VEC Avg",
)

ADDITIONAL_SCALE_COLUMNS: Sequence[str] = (
    "YS 600 C PRIOR",
    "YS 25C PRIOR",
    "PROP 25C Density (g/cm3)",
    "Density Avg",
    "Pugh_Ratio_PRIOR",
    "PROP ST (K)",
    "Tm Avg",
    "VEC",
    "VEC Avg",
)

PREFIXES_TO_SCALE: Sequence[str] = ("PROP", "EQUIL")

DEFAULT_INPUT = Path("design_space.xlsx")
DEFAULT_OUTPUT = Path("design_space_sanitized.csv")


def load_design_space(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols: List[str] = []

    for col in ELEMENT_COLUMNS:
        if col in df.columns:
            keep_cols.append(col)

    for col in REQUIRED_COLUMNS:
        if col in df.columns and col not in keep_cols:
            keep_cols.append(col)

    bcc_cols = [c for c in df.columns if "600C" in c and "BCC" in c]
    keep_cols.extend([c for c in bcc_cols if c not in keep_cols])

    missing_elements = [c for c in ELEMENT_COLUMNS if c not in df.columns]
    if missing_elements:
        raise ValueError(f"Missing elemental columns: {missing_elements}")

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    if "VEC" in df.columns and "VEC" not in keep_cols:
        keep_cols.append("VEC")
    elif "VEC" not in df.columns and "VEC Avg" not in df.columns:
        raise ValueError("Neither 'VEC' nor 'VEC Avg' present in design space.")

    if not bcc_cols:
        raise ValueError("No 600C BCC columns found in design space.")

    return df.loc[:, keep_cols].copy()


def rename_element_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping: Dict[str, str] = {
        col: f"element_{idx+1:02d}" for idx, col in enumerate(ELEMENT_COLUMNS)
    }
    return df.rename(columns=mapping)


def scale_columns(
    df: pd.DataFrame,
    prefixes: Iterable[str],
    extra: Iterable[str],
) -> pd.DataFrame:
    df_scaled = df.copy()
    prefixes_lower = tuple(p.lower() for p in prefixes)
    extra_set = set(extra)

    for col in df_scaled.columns:
        lower = col.lower()
        should_scale = lower.startswith(prefixes_lower) or col in extra_set
        if not should_scale:
            continue
        series = pd.to_numeric(df_scaled[col], errors="coerce")
        col_min = series.min(skipna=True)
        col_max = series.max(skipna=True)
        if pd.isna(col_min) or pd.isna(col_max) or col_max == col_min:
            df_scaled[col] = 0.0
        else:
            df_scaled[col] = (series - col_min) / (col_max - col_min)
    return df_scaled


def apply_jitter(df: pd.DataFrame, amplitude: float = 0.01, seed: int = 2025) -> pd.DataFrame:
    df_jittered = df.copy()
    numeric_cols = df_jittered.select_dtypes(include=["number"]).columns
    if numeric_cols.empty:
        return df_jittered

    rng = np.random.default_rng(seed)
    noise = rng.uniform(-amplitude, amplitude, size=(df_jittered.shape[0], len(numeric_cols)))
    values = df_jittered[numeric_cols].to_numpy(dtype=float)
    values *= (1.0 + noise)
    values = np.clip(values, 0.0, None)

    scaled_prefixes = tuple(p.lower() for p in PREFIXES_TO_SCALE)
    for idx, col in enumerate(numeric_cols):
        lower = col.lower()
        if lower.startswith(scaled_prefixes) or col in ADDITIONAL_SCALE_COLUMNS:
            values[:, idx] = np.clip(values[:, idx], 0.0, 1.0)

    df_jittered[numeric_cols] = values
    return df_jittered


def sanitize(input_path: Path, output_path: Path) -> None:
    df = load_design_space(input_path)
    df = select_columns(df)
    df = rename_element_columns(df)
    df = scale_columns(df, PREFIXES_TO_SCALE, ADDITIONAL_SCALE_COLUMNS)
    df = apply_jitter(df)
    df.to_csv(output_path, index=False)
    print(f"Sanitized dataset saved to {output_path} ({df.shape[0]} rows, {df.shape[1]} columns).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Input design-space file (.xlsx or .csv).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output CSV path (default: design_space_sanitized.csv).")
    args = parser.parse_args()
    sanitize(args.input, args.output)


if __name__ == "__main__":
    main()
