#!/usr/bin/env python3
# =============================================================================
# src/04_clean_flood.py — Process flood & disaster datasets
# =============================================================================
# Input:  data/raw/flood/*.shp  (IIT-Delhi flood inventory)
#         data/raw/flood/flood_affected_districts.csv
#         data/raw/flood/imd_rain/  (NetCDF, optional)
# Output: data/processed/flood_risk_by_district.csv
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_FLOOD_DIR, RAW_MAPS_DIR, PROCESSED_DIR
from src.utils import (
    standardise_columns, standardise_place_name,
    safe_read_csv, normalise_series, ensure_dirs,
)


# =============================================================================
# 1. IIT-Delhi Flood Inventory (.shp)
# =============================================================================

def process_flood_inventory(shp_path: str, districts_path: str) -> pd.DataFrame:
    """
    Spatial join: count flood events per district from the IIT-Delhi shapefile.
    Returns district-level flood frequency and death toll.
    """
    import geopandas as gpd

    logger.info(f"  Loading flood shapefile: {shp_path}")
    try:
        floods = gpd.read_file(shp_path)
    except Exception as ex:
        logger.error(f"  Failed to read shapefile: {ex}")
        return pd.DataFrame()

    logger.info(f"  Flood records: {len(floods)} | CRS: {floods.crs}")
    logger.info(f"  Columns: {floods.columns.tolist()}")

    logger.info(f"  Loading districts: {districts_path}")
    districts = gpd.read_file(districts_path)
    districts = districts.to_crs(floods.crs if floods.crs else "EPSG:4326")

    # Find district name column
    dist_col = next(
        (c for c in districts.columns
         if c.lower() in {"dtname", "district", "name", "dist_name", "district_name"}),
        districts.columns[1],
    )
    districts = districts.rename(columns={dist_col: "DISTRICT_NAME"})
    districts["DISTRICT_NAME"] = standardise_place_name(districts["DISTRICT_NAME"])

    # Find state name column
    state_col = next(
        (c for c in districts.columns
         if c.lower() in {"stname", "state", "state_name", "st_name"}),
        None,
    )
    if state_col:
        districts["STATE_NAME"] = standardise_place_name(districts[state_col])

    # Spatial join
    logger.info("  Running spatial join (flood polygons → districts)...")
    try:
        joined = gpd.sjoin(floods, districts[["DISTRICT_NAME", "geometry"]
                           + (["STATE_NAME"] if state_col else [])],
                           how="left", predicate="intersects")
    except Exception as ex:
        logger.error(f"  Spatial join failed: {ex}")
        return pd.DataFrame()

    # Find death / damage columns
    death_col = next(
        (c for c in joined.columns if c.upper() in
         {"DEATHS", "FATALITIES", "KILLED", "DEAD", "TOTAL_DEATHS"}),
        None,
    )
    year_col = next(
        (c for c in joined.columns if "YEAR" in c.upper()),
        None,
    )

    agg_spec = {"DISTRICT_NAME": "first"}
    if year_col:
        agg_spec["_EVENT_COUNT"] = (year_col, "count")
    else:
        joined["_event"] = 1
        agg_spec["_EVENT_COUNT"] = ("_event", "sum")

    if death_col:
        joined[death_col] = pd.to_numeric(joined[death_col], errors="coerce").fillna(0)
        agg_spec["FLOOD_DEATHS"] = (death_col, "sum")

    freq = joined.groupby("DISTRICT_NAME").agg(**{
        k: v for k, v in agg_spec.items() if k != "DISTRICT_NAME"
    }).reset_index()
    freq.rename(columns={"_EVENT_COUNT": "FLOOD_COUNT"}, inplace=True)

    if "FLOOD_DEATHS" not in freq.columns:
        freq["FLOOD_DEATHS"] = 0

    freq["FLOOD_SCORE"] = (
        freq["FLOOD_COUNT"]  * 2.0 +
        freq["FLOOD_DEATHS"] * 0.1
    )
    freq["FLOOD_SCORE_NORM"] = normalise_series(freq["FLOOD_SCORE"])

    logger.info(f"  Flood inventory: {len(freq)} districts scored")
    return freq[["DISTRICT_NAME", "FLOOD_COUNT", "FLOOD_DEATHS",
                  "FLOOD_SCORE", "FLOOD_SCORE_NORM"]]


# =============================================================================
# 2. Government flood-affected districts CSV
# =============================================================================

def process_flood_districts_csv(path: str) -> pd.DataFrame:
    """
    data.gov.in flood affected districts CSV.
    Provides a complementary district-level flood record.
    """
    df = safe_read_csv(path)
    df = standardise_columns(df)

    # Find district / state / year columns
    dist_col  = next((c for c in df.columns if "DISTRICT" in c or "DIST" in c), None)
    state_col = next((c for c in df.columns if "STATE" in c), None)
    year_col  = next((c for c in df.columns if "YEAR" in c), None)
    area_col  = next((c for c in df.columns if "AREA" in c or "AFFECTED" in c), None)

    if not dist_col and not state_col:
        logger.warning("  Flood CSV: cannot identify district/state column — skipping")
        return pd.DataFrame()

    name_col = dist_col or state_col
    df["DISTRICT_NAME"] = standardise_place_name(df[name_col])
    df["YEAR"] = pd.to_numeric(df.get(year_col, 2015), errors="coerce").fillna(2015).astype(int)

    agg = df.groupby("DISTRICT_NAME", as_index=False).agg(
        FLOOD_EVENTS=("YEAR", "count"),
        FLOOD_AREA=((area_col, "sum") if area_col else ("YEAR", "count")),
    )
    agg["FLOOD_SCORE_CSV"] = normalise_series(agg["FLOOD_EVENTS"])
    logger.info(f"  Flood CSV: {len(agg)} districts")
    return agg[["DISTRICT_NAME", "FLOOD_EVENTS", "FLOOD_SCORE_CSV"]]


# =============================================================================
# 3. IMD rainfall: compute monsoon intensity proxy per district
# =============================================================================

def compute_imd_rainfall_proxy(imd_dir: str, districts_path: str) -> pd.DataFrame:
    """
    Use imdlib NetCDF files to compute average monsoon rainfall per district.
    This is optional — returns empty DataFrame if imd_rain/ is empty.
    """
    if not os.path.exists(imd_dir) or not os.listdir(imd_dir):
        logger.info("  IMD rainfall directory empty — skipping rainfall proxy.")
        return pd.DataFrame()

    try:
        import imdlib as imd
        import geopandas as gpd
        import xarray as xr
    except ImportError:
        logger.warning("  imdlib / xarray not installed — skipping IMD rainfall step.")
        return pd.DataFrame()

    try:
        nc_files = [os.path.join(imd_dir, f) for f in os.listdir(imd_dir)
                    if f.endswith(".nc") or f.endswith(".GRD")]
        if not nc_files:
            return pd.DataFrame()

        # Load and average
        ds = xr.open_mfdataset(nc_files, combine="by_coords")
        var = list(ds.data_vars)[0]
        annual_mean = ds[var].mean(dim="time")

        # Convert to GeoDataFrame of grid points
        lat_vals = annual_mean.lat.values
        lon_vals = annual_mean.lon.values
        rain_vals = annual_mean.values

        rows = []
        for i, lat in enumerate(lat_vals):
            for j, lon in enumerate(lon_vals):
                v = rain_vals[i, j]
                if not np.isnan(v):
                    rows.append({"LAT": lat, "LON": lon, "RAIN_MM": float(v)})

        rain_df = pd.DataFrame(rows)
        rain_gdf = gpd.GeoDataFrame(
            rain_df,
            geometry=gpd.points_from_xy(rain_df["LON"], rain_df["LAT"]),
            crs="EPSG:4326",
        )

        districts = gpd.read_file(districts_path)
        dist_col = next(
            (c for c in districts.columns if "name" in c.lower()), districts.columns[1]
        )
        districts["DISTRICT_NAME"] = standardise_place_name(districts[dist_col])
        joined = gpd.sjoin(rain_gdf, districts[["DISTRICT_NAME", "geometry"]],
                           how="left", predicate="within")
        rainfall_agg = joined.groupby("DISTRICT_NAME")["RAIN_MM"].mean().reset_index()
        rainfall_agg.columns = ["DISTRICT_NAME", "AVG_RAIN_MM"]
        rainfall_agg["RAIN_RISK_NORM"] = normalise_series(rainfall_agg["AVG_RAIN_MM"])
        logger.info(f"  IMD rainfall: {len(rainfall_agg)} districts")
        return rainfall_agg

    except Exception as ex:
        logger.warning(f"  IMD processing failed: {ex}")
        return pd.DataFrame()


# =============================================================================
# 4. Merge all flood sources
# =============================================================================

def merge_flood_sources(*dfs) -> pd.DataFrame:
    """
    Merge flood inventory, CSV, and IMD data into a single district-level table.
    Missing sources are filled with 0.
    """
    base = None
    for df in dfs:
        if df is None or df.empty:
            continue
        if base is None:
            base = df.copy()
        else:
            base = base.merge(df, on="DISTRICT_NAME", how="outer")

    if base is None or base.empty:
        return pd.DataFrame()

    # Fill numeric columns
    for col in base.select_dtypes(include=[np.number]).columns:
        base[col] = base[col].fillna(0)

    # Composite flood risk
    score_cols = [c for c in base.columns if "SCORE_NORM" in c or "SCORE_CSV" in c or "RAIN_RISK" in c]
    if score_cols:
        base["FLOOD_COMPOSITE"] = base[score_cols].mean(axis=1)
        base["FLOOD_COMPOSITE_NORM"] = normalise_series(base["FLOOD_COMPOSITE"])
    elif "FLOOD_SCORE_NORM" in base.columns:
        base["FLOOD_COMPOSITE_NORM"] = base["FLOOD_SCORE_NORM"]
    else:
        base["FLOOD_COMPOSITE_NORM"] = 0.0

    return base


# =============================================================================
# Main
# =============================================================================

def run():
    ensure_dirs(PROCESSED_DIR)

    districts_path = os.path.join(RAW_MAPS_DIR, "india_districts.geojson")
    if not os.path.exists(districts_path):
        logger.error("Districts GeoJSON missing. Run: python src/01_download_data.py auto")
        raise FileNotFoundError(districts_path)

    flood_inv_df = pd.DataFrame()
    flood_csv_df = pd.DataFrame()
    imd_df       = pd.DataFrame()

    # ── 1. IIT-Delhi flood inventory shapefile ──────────────────────────────
    shp_files = [f for f in os.listdir(RAW_FLOOD_DIR) if f.endswith(".shp")]
    if shp_files:
        shp_path = os.path.join(RAW_FLOOD_DIR, shp_files[0])
        logger.info(f"Processing flood inventory: {shp_files[0]}")
        flood_inv_df = process_flood_inventory(shp_path, districts_path)
    else:
        logger.warning("No .shp flood inventory found — flood layer will be minimal")

    # ── 2. Flood districts CSV ───────────────────────────────────────────────
    csv_path = os.path.join(RAW_FLOOD_DIR, "flood_affected_districts.csv")
    if os.path.exists(csv_path):
        logger.info("Processing flood districts CSV...")
        flood_csv_df = process_flood_districts_csv(csv_path)

    # ── 3. IMD rainfall proxy ────────────────────────────────────────────────
    imd_dir = os.path.join(RAW_FLOOD_DIR, "imd_rain")
    logger.info("Processing IMD rainfall data (if available)...")
    imd_df = compute_imd_rainfall_proxy(imd_dir, districts_path)

    # ── Merge ────────────────────────────────────────────────────────────────
    merged = merge_flood_sources(flood_inv_df, flood_csv_df, imd_df)

    if merged.empty:
        # Fallback: empty table so downstream doesn't crash
        logger.warning("No flood data found — creating empty flood table.")
        merged = pd.DataFrame({
            "DISTRICT_NAME": [], "FLOOD_COMPOSITE_NORM": []
        })

    out_path = os.path.join(PROCESSED_DIR, "flood_risk_by_district.csv")
    merged.to_csv(out_path, index=False)
    logger.success(f"Saved {len(merged)} districts → {out_path}")

    if not merged.empty and "FLOOD_COMPOSITE_NORM" in merged.columns:
        top10 = merged.nlargest(10, "FLOOD_COMPOSITE_NORM")["DISTRICT_NAME"].tolist()
        logger.info(f"Top-10 flood-prone districts: {', '.join(top10)}")


if __name__ == "__main__":
    run()
