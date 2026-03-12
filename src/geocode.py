#!/usr/bin/env python3
# =============================================================================
# src/05_geocode.py — Geocode district-level risk data
# =============================================================================

import os
import sys
import pandas as pd
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DIR, RAW_MAPS_DIR
from src.utils import standardise_place_name, ensure_dirs


# =============================================================================
# Name normalization
# =============================================================================

def clean_district_name(series: pd.Series) -> pd.Series:

    return (
        series.astype(str)
        .str.lower()
        .str.replace(" district", "", regex=False)
        .str.replace(" city", "", regex=False)
        .str.replace(" urban", "", regex=False)
        .str.replace(" rural", "", regex=False)
        .str.strip()
        .str.title()
    )


# =============================================================================
# Build lookup tables
# =============================================================================

def build_district_lookup(districts_path: str):

    import geopandas as gpd

    logger.info(f"Building district centroid lookup from {districts_path}")

    gdf = gpd.read_file(districts_path)

    logger.info(f"Loaded {len(gdf)} districts")

    # detect columns
    dist_col = next(
        (c for c in gdf.columns if c.lower() in
         {"dtname", "district", "dist_name", "district_name", "name"}),
        gdf.columns[1],
    )

    state_col = next(
        (c for c in gdf.columns if c.lower() in
         {"stname", "state", "state_name", "st_name"}),
        None,
    )

    gdf["_DIST"] = clean_district_name(gdf[dist_col])

    if state_col:
        gdf["_STATE"] = standardise_place_name(gdf[state_col])
    else:
        gdf["_STATE"] = "UNKNOWN"

    # compute centroids
    gdf_m = gdf.to_crs("EPSG:32644")
    centroids = gdf_m.centroid.to_crs("EPSG:4326")

    gdf["LAT"] = centroids.y.round(5)
    gdf["LON"] = centroids.x.round(5)

    lookup = gdf[["_DIST", "_STATE", "LAT", "LON"]]

    logger.info(f"District lookup built: {len(lookup)} entries")

    return lookup


def build_state_lookup(lookup):

    return lookup.groupby("_STATE", as_index=False).agg(
        LAT=("LAT", "mean"),
        LON=("LON", "mean")
    ).rename(columns={"LAT": "STATE_LAT", "LON": "STATE_LON"})


# =============================================================================
# Generic geocoder
# =============================================================================

def geocode_dataframe(df, district_col, state_col, lookup, state_lookup):

    df = df.copy()

    df["_DIST"] = clean_district_name(df[district_col])
    df["_STATE"] = standardise_place_name(df[state_col])

    merged = df.merge(
        lookup,
        on=["_DIST", "_STATE"],
        how="left"
    )

    unmatched = merged["LAT"].isna().sum()

    if unmatched > 0:

        logger.info(f"{unmatched} rows unmatched → using state centroid")

        merged = merged.merge(state_lookup, on="_STATE", how="left")

        merged["LAT"] = merged["LAT"].fillna(merged["STATE_LAT"])
        merged["LON"] = merged["LON"].fillna(merged["STATE_LON"])

        merged.drop(columns=["STATE_LAT", "STATE_LON"], inplace=True)

    merged.drop(columns=["_DIST", "_STATE"], inplace=True)

    total = len(merged)

    matched = merged["LAT"].notna().sum()

    if total > 0:
        pct = int(100 * matched / total)
    else:
        pct = 0

    logger.info(f"Geocoding: {matched}/{total} rows ({pct}%)")

    return merged


# =============================================================================
# Dataset geocoders
# =============================================================================

def geocode_crime(lookup, state_lookup):

    path = os.path.join(PROCESSED_DIR, "crime_clean.csv")

    if not os.path.exists(path):
        logger.warning("crime_clean.csv not found")
        return

    df = pd.read_csv(path)

    logger.info(f"Geocoding crime data ({len(df)} rows)")

    result = geocode_dataframe(
        df,
        "DISTRICT",
        "STATE",
        lookup,
        state_lookup
    )

    result = result.dropna(subset=["LAT", "LON"])

    out = os.path.join(PROCESSED_DIR, "crime_geocoded.csv")

    result.to_csv(out, index=False)

    logger.success(f"Saved crime_geocoded.csv — {len(result)} rows")


def geocode_accidents(lookup, state_lookup):

    path = os.path.join(PROCESSED_DIR, "accidents_clean.csv")

    if not os.path.exists(path):
        logger.warning("accidents_clean.csv not found")
        return

    df = pd.read_csv(path)

    logger.info(f"Geocoding accident data ({len(df)} rows)")

    df["DISTRICT"] = df["STATE"]

    result = geocode_dataframe(
        df,
        "DISTRICT",
        "STATE",
        lookup,
        state_lookup
    )

    result = result.dropna(subset=["LAT", "LON"])

    out = os.path.join(PROCESSED_DIR, "accidents_geocoded.csv")

    result.to_csv(out, index=False)

    logger.success(f"Saved accidents_geocoded.csv — {len(result)} rows")


def geocode_flood(lookup, state_lookup):

    path = os.path.join(PROCESSED_DIR, "flood_risk_by_district.csv")

    if not os.path.exists(path):
        logger.warning("flood file missing")
        return

    df = pd.read_csv(path)

    logger.info(f"Geocoding flood data ({len(df)} rows)")

    df["STATE"] = df["DISTRICT_NAME"]

    result = geocode_dataframe(
        df,
        "DISTRICT_NAME",
        "STATE",
        lookup,
        state_lookup
    )

    result = result.dropna(subset=["LAT", "LON"])

    out = os.path.join(PROCESSED_DIR, "flood_geocoded.csv")

    result.to_csv(out, index=False)

    logger.success(f"Saved flood_geocoded.csv — {len(result)} rows")


# =============================================================================
# Main
# =============================================================================

def run():

    ensure_dirs(PROCESSED_DIR)

    districts_path = os.path.join(RAW_MAPS_DIR, "india_districts.geojson")

    lookup = build_district_lookup(districts_path)

    state_lookup = build_state_lookup(lookup)

    geocode_crime(lookup, state_lookup)

    geocode_accidents(lookup, state_lookup)

    geocode_flood(lookup, state_lookup)

    logger.success("Geocoding complete")


if __name__ == "__main__":
    run()