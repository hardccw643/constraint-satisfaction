#!/usr/bin/env python3
"""
Create a shareable version of design_space.xlsx by:

1. Renaming elemental fraction columns to anonymous placeholders
   (element_01, element_02, ...).
2. Min-max scaling any column whose name starts with PROP or EQUIL
   (case-insensitive) into the [0, 1] range.

The obfuscated dataset is written to design_space_sanitized.csv by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

# Element symbols for refractory alloy work; extend if needed.
ELEMENT_COLUMNS: List[str] = [
    "Al", "Co", "Cr", "Fe", "Hf", "Mo", "Nb", "Ni", "Re", "Ta", "Ti", "V", "W", "Zr",
    "Cu", "Mn", "Mg", "Si", "Y"
]

DEFAULT_INPUT = Path("design_space.xlsx")
DEFAULT_OUTPUT = Path("design_space_sanitized.csv")


def load_design_space(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def build_element_mapping(df: pd.DataFrame) -> Dict[str, str]:
    matches: List[str] = []
    for col in df.columns:
        col_clean = col.strip()
        if col_clean in ELEMENT_COLUMNS:
            matches.append(col)
    mapping: Dict[str, str] = {}
    for idx, col in enumerate(sorted(matches)):
        mapping[col] = f"element_{idx+1:02d}"
    return mapping


def scale_columns(df: pd.DataFrame, prefixes: Iterable[str] = ("PROP", "EQUIL")) -> pd.DataFrame:
    df_scaled = df.copy()
    prefix_lower = tuple(p.lower() for p in prefixes)
    for col in df.columns:
        col_l = col.lower()
        if col_l.startswith(prefix_lower):
            series = df[col].astype(float)
            cmin = series.min(skipna=True)
            cmax = series.max(skipna=True)
            if pd.isna(cmin) or pd.isna(cmax) or cmax - cmin == 0:
                df_scaled[col] = 0.0
            else:
                df_scaled[col] = (series - cmin) / (cmax - cmin)
    return df_scaled


def sanitize(input_path: Path, output_path: Path) -> None:
    df = load_design_space(input_path)
    element_map = build_element_mapping(df)
    df = df.rename(columns=element_map)
    df = scale_columns(df)
    df.to_csv(output_path, index=False)
    print(f"Sanitized dataset written to {output_path} ({len(df)} rows).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Input design space file (xlsx or csv).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output CSV path for sanitized data.")
    args = parser.parse_args()
    sanitize(args.input, args.output)


if __name__ == "__main__":
    main()
