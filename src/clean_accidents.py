#!/usr/bin/env python3
# =============================================================================
# src/03_clean_accidents.py — Clean road accident datasets
# =============================================================================
# Input:  data/raw/accidents/india_road_accidents.csv
#         data/raw/accidents/accident_blackspots.csv  (GPS-tagged)
# Output: data/processed/accidents_clean.csv
#         data/processed/blackspots_clean.csv
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_ACCIDENT_DIR, PROCESSED_DIR
from src.utils import (
    standardise_columns, standardise_place_name,
    safe_read_csv, normalise_series, ensure_dirs,
    india_bbox_filter,
)


# =============================================================================
# 1. State-level accident data
# =============================================================================

def clean_road_accidents(path: str) -> pd.DataFrame:
    """
    Input: Kaggle india-road-accident-dataset or MoRTH CSV
    Produces a normalised accident severity score per state-year.
    """
    df = safe_read_csv(path)
    df = standardise_columns(df)

    # Resolve column name variants
    renames = {
        "TOTAL_NUMBER_OF_ROAD_ACCIDENTS": "TOTAL_ACCIDENTS",
        "NO_OF_ACCIDENTS":               "TOTAL_ACCIDENTS",
        "ACCIDENTS":                     "TOTAL_ACCIDENTS",
        "PERSONS_KILLED":                "TOTAL_KILLED",
        "NO_OF_PERSONS_KILLED":          "TOTAL_KILLED",
        "KILLED":                        "TOTAL_KILLED",
        "PERSONS_INJURED":               "TOTAL_INJURED",
        "NO_OF_PERSONS_INJURED":         "TOTAL_INJURED",
        "INJURED":                       "TOTAL_INJURED",
        "STATE/UT":                      "STATE",
        "STATE_UT":                      "STATE",
    }
    for old, new in renames.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # Ensure required columns exist (fill 0 if absent)
    for col in ["TOTAL_ACCIDENTS", "TOTAL_KILLED", "TOTAL_INJURED"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "STATE" not in df.columns:
        df["STATE"] = "Unknown"
    df["STATE"] = standardise_place_name(df["STATE"])

    # -------------------------------------------------------------------------
    # FIXED YEAR HANDLING (prevents `.fillna()` on int crash)
    # -------------------------------------------------------------------------
    year_col = next((c for c in df.columns if "YEAR" in c), None)

    if year_col:
        df["YEAR"] = pd.to_numeric(df[year_col], errors="coerce")
    else:
        df["YEAR"] = 2015

    df["YEAR"] = df["YEAR"].fillna(2015).astype(int)

    # Severity: deaths matter most, then injuries, then raw counts
    df["ACCIDENT_SCORE"] = (
        df["TOTAL_KILLED"]    * 5.0 +
        df["TOTAL_INJURED"]   * 1.5 +
        df["TOTAL_ACCIDENTS"] * 0.5
    )

    # Aggregate by state-year
    agg = df.groupby(["STATE", "YEAR"], as_index=False).agg(
        TOTAL_ACCIDENTS=("TOTAL_ACCIDENTS", "sum"),
        TOTAL_KILLED=("TOTAL_KILLED",    "sum"),
        TOTAL_INJURED=("TOTAL_INJURED",   "sum"),
        ACCIDENT_SCORE=("ACCIDENT_SCORE", "sum"),
    )

    # Time-stable score per state
    per_state = agg.groupby("STATE", as_index=False).agg(
        TOTAL_ACCIDENTS=("TOTAL_ACCIDENTS", "mean"),
        TOTAL_KILLED=("TOTAL_KILLED",    "mean"),
        TOTAL_INJURED=("TOTAL_INJURED",   "mean"),
        ACCIDENT_SCORE=("ACCIDENT_SCORE", "mean"),
    )
    per_state["ACCIDENT_SCORE_NORM"] = normalise_series(per_state["ACCIDENT_SCORE"])

    # Also keep per-year frame
    agg["ACCIDENT_SCORE_NORM"] = normalise_series(agg["ACCIDENT_SCORE"])

    logger.info(f"  State-level accidents: {len(per_state)} states")
    return per_state, agg


# =============================================================================
# 2. GPS-tagged accident blackspots (highest value)
# =============================================================================

def clean_blackspots(path: str) -> pd.DataFrame:
    """
    Input: data.gov.in accident blackspots CSV (National Highways)
    Columns expected: LAT, LON (or variants), ACCIDENT_COUNT (or variant)
    Returns GPS-tagged blackspot GeoDataFrame-ready CSV.
    """
    df = safe_read_csv(path)
    df = standardise_columns(df)

    # Try to find latitude column
    lat_col = next(
        (c for c in df.columns if c in
         {"LATITUDE", "LAT", "Y", "LATTITUDE", "LATITUTE"}),
        None
    )
    lon_col = next(
        (c for c in df.columns if c in
         {"LONGITUDE", "LON", "LNG", "X", "LONG", "LONGITUTE"}),
        None
    )

    if lat_col is None or lon_col is None:
        logger.warning(
            "  Blackspot file missing LAT/LON columns — cannot create GPS blackspots. "
            "Columns found: " + str(df.columns.tolist())
        )
        return pd.DataFrame()

    df = df.rename(columns={lat_col: "LAT", lon_col: "LON"})
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    df = df.dropna(subset=["LAT", "LON"])
    df = india_bbox_filter(df)

    # Find accident count column
    count_col = next(
        (c for c in df.columns if "COUNT" in c or "ACCIDENT" in c or "NUMBER" in c),
        None,
    )
    df["ACCIDENT_COUNT"] = (
        pd.to_numeric(df[count_col], errors="coerce").fillna(1)
        if count_col else 1
    )

    df["BLACKSPOT_SCORE"] = df["ACCIDENT_COUNT"] / (df["ACCIDENT_COUNT"].max() + 1e-9)

    # Optional: location name
    name_col = next((c for c in df.columns if "NAME" in c or "LOCATION" in c), None)
    df["LOCATION"] = df[name_col] if name_col else "Unknown"

    out_cols = ["LAT", "LON", "ACCIDENT_COUNT", "BLACKSPOT_SCORE", "LOCATION"]
    result = df[out_cols].copy().reset_index(drop=True)
    logger.info(f"  Blackspots: {len(result)} GPS-tagged points")
    return result


# =============================================================================
# 3. Time-of-accident profile (from accident data)
# =============================================================================

def extract_time_profile(df_raw: pd.DataFrame) -> dict:
    """
    If the accident CSV has time-of-day columns, extract the distribution.
    Returns a dict {hour_bucket: accident_fraction} for time modifiers.
    """
    time_cols = [c for c in df_raw.columns
                 if any(t in c for t in ["6PM", "6AM", "12PM", "12AM",
                                          "TIME", "DAYTIME", "NIGHT"])]
    if not time_cols:
        return {}

    total = df_raw[time_cols].apply(pd.to_numeric, errors="coerce").sum().sum()
    if total == 0:
        return {}

    profile = {}
    for col in time_cols:
        val = pd.to_numeric(df_raw[col], errors="coerce").sum()
        profile[col] = round(val / total, 4)

    logger.info(f"  Extracted time-of-day profile: {profile}")
    return profile


# =============================================================================
# Main
# =============================================================================

def run():
    ensure_dirs(PROCESSED_DIR)

    # ── State-level accident data ──────────────────────────────────────────────
    accident_path = os.path.join(RAW_ACCIDENT_DIR, "india_road_accidents.csv")
    if os.path.exists(accident_path):
        logger.info("Cleaning road accident data...")
        df_raw = safe_read_csv(accident_path)
        per_state, per_year = clean_road_accidents(accident_path)
        per_state.to_csv(os.path.join(PROCESSED_DIR, "accidents_clean.csv"), index=False)
        per_year.to_csv(os.path.join(PROCESSED_DIR, "accidents_by_year.csv"), index=False)
        logger.success(f"Saved accidents_clean.csv — {len(per_state)} states")

        # Extract time profile
        df_raw_std = standardise_columns(safe_read_csv(accident_path))
        time_profile = extract_time_profile(df_raw_std)
        if time_profile:
            import json
            with open(os.path.join(PROCESSED_DIR, "accident_time_profile.json"), "w") as f:
                json.dump(time_profile, f, indent=2)
            logger.success("Saved accident_time_profile.json")
    else:
        logger.warning(f"Not found: {accident_path}")
        # Create empty placeholder so downstream scripts don't crash
        pd.DataFrame(columns=["STATE", "TOTAL_ACCIDENTS", "TOTAL_KILLED",
                                "TOTAL_INJURED", "ACCIDENT_SCORE", "ACCIDENT_SCORE_NORM"]
                     ).to_csv(os.path.join(PROCESSED_DIR, "accidents_clean.csv"), index=False)

    # ── GPS blackspots ─────────────────────────────────────────────────────────
    bs_path = os.path.join(RAW_ACCIDENT_DIR, "accident_blackspots.csv")
    if os.path.exists(bs_path):
        logger.info("Cleaning accident blackspot data...")
        spots = clean_blackspots(bs_path)
        if not spots.empty:
            spots.to_csv(os.path.join(PROCESSED_DIR, "blackspots_clean.csv"), index=False)
            logger.success(f"Saved blackspots_clean.csv — {len(spots)} points")
        else:
            logger.warning("  No valid blackspot data extracted.")
    else:
        logger.warning(f"  Not found: {bs_path} — blackspot layer will be skipped")


if __name__ == "__main__":
    run()