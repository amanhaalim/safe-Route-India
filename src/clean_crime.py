#!/usr/bin/env python3
# =============================================================================
# src/02_clean_crime.py — Clean and merge all crime datasets
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import CRIME_SEVERITY, RAW_CRIME_DIR, PROCESSED_DIR
from src.utils import (
    standardise_columns,
    standardise_place_name,
    safe_read_csv,
    normalise_series,
    ensure_dirs,
)

STATE_ALIASES = {"STATE/UT", "STATE_UT", "STATE_NAME", "ST_NAME", "STATENAME", "STATE"}
DISTRICT_ALIASES = {"DIST", "DISTRICT_NAME", "DIST_NAME", "DISTNAME", "DISTRICT"}
YEAR_ALIASES = {"YR", "YEAR_OF_CRIME", "CRIME_YEAR", "YEAR"}


def _resolve_col(df: pd.DataFrame, candidates: set):
    for col in df.columns:
        if col in candidates:
            return col
    return None


def _safe_series(df, col, default="UNKNOWN"):
    """
    Always return a pandas Series to avoid .astype errors
    """
    if col and col in df.columns:
        return df[col]
    return pd.Series([default] * len(df))


# =============================================================================
# Dataset loaders
# =============================================================================


def load_crimes_india(path: str) -> pd.DataFrame:

    df = safe_read_csv(path)
    df = standardise_columns(df)

    state_col = _resolve_col(df, STATE_ALIASES)
    dist_col = _resolve_col(df, DISTRICT_ALIASES)
    year_col = _resolve_col(df, YEAR_ALIASES)

    df["_STATE"] = standardise_place_name(_safe_series(df, state_col))
    df["_DISTRICT"] = standardise_place_name(_safe_series(df, dist_col, df["_STATE"]))

    if year_col:
        df["_YEAR"] = pd.to_numeric(df[year_col], errors="coerce")
    else:
        df["_YEAR"] = 2015

    df["_YEAR"] = df["_YEAR"].fillna(2015).astype(int)

    df["_SCORE"] = df.apply(
        lambda row: sum(
            float(row[col]) * wt
            for col, wt in CRIME_SEVERITY.items()
            if col in row.index and pd.notna(row[col])
        ),
        axis=1,
    )

    out = df[["_STATE", "_DISTRICT", "_YEAR", "_SCORE"]].copy()
    out.columns = ["STATE", "DISTRICT", "YEAR", "CRIME_SEVERITY_SCORE"]

    logger.info(f"crimes_india loaded: {len(out)} rows")

    return out


def load_district_ipc(path: str) -> pd.DataFrame:

    df = safe_read_csv(path)
    df = standardise_columns(df)

    state_col = _resolve_col(df, STATE_ALIASES)
    dist_col = _resolve_col(df, DISTRICT_ALIASES)
    year_col = _resolve_col(df, YEAR_ALIASES)

    df["_STATE"] = standardise_place_name(_safe_series(df, state_col))
    df["_DISTRICT"] = standardise_place_name(_safe_series(df, dist_col, df["_STATE"]))

    if year_col:
        df["_YEAR"] = pd.to_numeric(df[year_col], errors="coerce")
    else:
        df["_YEAR"] = 2010

    df["_YEAR"] = df["_YEAR"].fillna(2010).astype(int)

    total_col = None
    for c in df.columns:
        if "TOTAL" in c and "IPC" in c:
            total_col = c
            break

    if total_col:
        df["_SCORE"] = pd.to_numeric(df[total_col], errors="coerce").fillna(0) * 1.5
    else:
        df["_SCORE"] = df.apply(
            lambda row: sum(
                float(row[col]) * wt
                for col, wt in CRIME_SEVERITY.items()
                if col in row.index and pd.notna(row[col])
            ),
            axis=1,
        )

    out = df[["_STATE", "_DISTRICT", "_YEAR", "_SCORE"]].copy()
    out.columns = ["STATE", "DISTRICT", "YEAR", "CRIME_SEVERITY_SCORE"]

    logger.info(f"district_ipc loaded: {len(out)} rows")

    return out


def load_crime_against_women(path: str) -> pd.DataFrame:

    df = safe_read_csv(path)
    df = standardise_columns(df)

    state_col = _resolve_col(df, STATE_ALIASES)
    year_col = _resolve_col(df, YEAR_ALIASES)

    df["_STATE"] = standardise_place_name(_safe_series(df, state_col))
    df["_DISTRICT"] = df["_STATE"]

    if year_col:
        df["_YEAR"] = pd.to_numeric(df[year_col], errors="coerce")
    else:
        df["_YEAR"] = 2015

    df["_YEAR"] = df["_YEAR"].fillna(2015).astype(int)

    women_weights = {
        "RAPE": 9,
        "KIDNAPPING_ABDUCTION": 7,
        "DOWRY_DEATHS": 8,
        "ASSAULT_ON_WOMEN": 6,
        "INSULT_TO_MODESTY": 5,
        "CRUELTY_BY_HUSBAND": 5,
        "IMPORTATION_OF_GIRLS": 6,
    }

    df["_SCORE"] = (
        df.apply(
            lambda row: sum(
                float(row[col]) * wt
                for col, wt in women_weights.items()
                if col in row.index and pd.notna(row[col])
            ),
            axis=1,
        )
        * 1.2
    )

    out = df[["_STATE", "_DISTRICT", "_YEAR", "_SCORE"]].copy()
    out.columns = ["STATE", "DISTRICT", "YEAR", "CRIME_SEVERITY_SCORE"]

    logger.info(f"crime_against_women loaded: {len(out)} rows")

    return out


# =============================================================================
# Merge datasets
# =============================================================================


def merge_and_normalise(dfs):

    combined = pd.concat(dfs, ignore_index=True)

    combined["CRIME_SEVERITY_SCORE"] = pd.to_numeric(
        combined["CRIME_SEVERITY_SCORE"], errors="coerce"
    ).fillna(0)

    per_year = combined.groupby(
        ["STATE", "DISTRICT", "YEAR"], as_index=False
    )["CRIME_SEVERITY_SCORE"].mean()

    per_district = (
        per_year.groupby(["STATE", "DISTRICT"])
        .agg(
            CRIME_SEVERITY_SCORE=("CRIME_SEVERITY_SCORE", "mean"),
            CRIME_SCORE_YEARS=("CRIME_SEVERITY_SCORE", "count"),
        )
        .reset_index()
    )

    per_district["CRIME_SCORE_NORM"] = normalise_series(
        per_district["CRIME_SEVERITY_SCORE"]
    )

    per_year["CRIME_SCORE_NORM"] = normalise_series(
        per_year["CRIME_SEVERITY_SCORE"]
    )

    return per_district, per_year


# =============================================================================
# Main pipeline
# =============================================================================


def run():

    ensure_dirs(PROCESSED_DIR)

    dfs = []

    loaders = [
        ("crimes_india.csv", load_crimes_india),
        ("district_ipc_crimes.csv", load_district_ipc),
        ("crime_against_women.csv", load_crime_against_women),
    ]

    found_any = False

    for filename, loader in loaders:

        path = os.path.join(RAW_CRIME_DIR, filename)

        if os.path.exists(path):

            logger.info(f"Loading {filename}")

            try:

                df = loader(path)

                if not df.empty:
                    dfs.append(df)
                    found_any = True

            except Exception as ex:
                logger.warning(f"Skipping {filename}: {ex}")

        else:

            logger.warning(f"Missing dataset: {filename}")

    if not found_any:
        raise FileNotFoundError(f"No crime datasets found in {RAW_CRIME_DIR}")

    logger.info("Merging datasets")

    per_district, per_year = merge_and_normalise(dfs)

    out_path = os.path.join(PROCESSED_DIR, "crime_clean.csv")
    per_district.to_csv(out_path, index=False)

    year_path = os.path.join(PROCESSED_DIR, "crime_by_year.csv")
    per_year.to_csv(year_path, index=False)

    logger.success(f"Saved district crime file: {out_path}")
    logger.success(f"Saved yearly crime file: {year_path}")

    return per_district


if __name__ == "__main__":
    run()