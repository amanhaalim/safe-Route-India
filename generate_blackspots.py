#!/usr/bin/env python3
# =============================================================================
# generate_blackspots.py — Derive accident blackspots from india_road_accidents.csv
# =============================================================================
# Since no free GPS-tagged blackspot dataset exists publicly for India, this
# script generates one from your existing accident data using two strategies:
#
#   Strategy A (preferred): If the CSV has LAT/LON columns → cluster nearby
#                           accidents with DBSCAN, score each cluster.
#
#   Strategy B (fallback):  If no GPS columns → geocode state/district names
#                           via Nominatim, then assign accident severity scores
#                           as the blackspot score.
#
# Output: data/raw/accidents/accident_blackspots.csv
#   Columns: LAT, LON, ACCIDENT_COUNT, BLACKSPOT_SCORE, LOCATION
#
# Usage:
#   python generate_blackspots.py
#   python generate_blackspots.py --input path/to/custom.csv
#   python generate_blackspots.py --strategy b   # force geocoding fallback
# =============================================================================

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_ACCIDENT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "accidents")
DEFAULT_INPUT    = os.path.join(RAW_ACCIDENT_DIR, "india_road_accidents.csv")
OUTPUT_PATH      = os.path.join(RAW_ACCIDENT_DIR, "accident_blackspots.csv")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DBSCAN_EPS_KM    = 0.5      # cluster radius in km (0.5 km ≈ 500 m)
DBSCAN_MIN_PTS   = 3        # minimum accidents to form a blackspot
NOMINATIM_DELAY  = 1.1      # seconds between geocoding requests (ToS limit)
MAX_GEOCODE_ROWS = 500      # cap geocoding calls to avoid rate-limit issues


# =============================================================================
# Helpers
# =============================================================================

def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.strip().str.upper()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace(r"[()]", "", regex=True)
    )
    return df


def safe_read_csv(path: str) -> pd.DataFrame:
    for enc in ["utf-8", "latin-1", "cp1252", "utf-8-sig"]:
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines="skip", low_memory=False)
            print(f"  ✅ Read {Path(path).name} ({len(df):,} rows, encoding={enc})")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {path}")


def find_latlon_cols(df: pd.DataFrame):
    LAT_NAMES = {"LATITUDE", "LAT", "Y", "LATTITUDE", "LATITUTE", "LAT_DD",
                 "ACCIDENT_LAT", "LOCATION_LAT"}
    LON_NAMES = {"LONGITUDE", "LON", "LNG", "X", "LONG", "LONGITUTE", "LON_DD",
                 "ACCIDENT_LON", "LOCATION_LON"}
    lat = next((c for c in df.columns if c in LAT_NAMES), None)
    lon = next((c for c in df.columns if c in LON_NAMES), None)
    return lat, lon


def find_location_cols(df: pd.DataFrame):
    STATE_NAMES    = {"STATE", "STATE_UT", "STATE_NAME", "STATE/UT"}
    DISTRICT_NAMES = {"DISTRICT", "DIST", "DISTRICT_NAME", "LOCATION", "CITY",
                      "PLACE", "ROAD_NAME", "ACCIDENT_LOCATION"}
    state = next((c for c in df.columns if c in STATE_NAMES), None)
    dist  = next((c for c in df.columns if c in DISTRICT_NAMES), None)
    return state, dist


def severity_score(row: pd.Series) -> float:
    killed  = float(row.get("TOTAL_KILLED",  row.get("KILLED",  0)) or 0)
    injured = float(row.get("TOTAL_INJURED", row.get("INJURED", 0)) or 0)
    count   = float(row.get("TOTAL_ACCIDENTS", row.get("ACCIDENTS", 1)) or 1)
    return killed * 5.0 + injured * 1.5 + count * 0.5


# =============================================================================
# Strategy A — DBSCAN clustering on GPS points
# =============================================================================

def strategy_a(df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    print("\n  📍 Strategy A: DBSCAN clustering on GPS coordinates")

    df = df.copy()
    df["_LAT"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["_LON"] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=["_LAT", "_LON"])

    # India bounding box filter
    df = df[(df["_LAT"] >= 6.0) & (df["_LAT"] <= 38.0) &
            (df["_LON"] >= 67.0) & (df["_LON"] <= 98.0)]

    print(f"  GPS-valid rows: {len(df):,}")

    if len(df) < DBSCAN_MIN_PTS:
        print("  ⚠️  Too few GPS rows for clustering — falling back to Strategy B")
        return pd.DataFrame()

    from sklearn.cluster import DBSCAN

    coords_rad = np.radians(df[["_LAT", "_LON"]].values)

    # haversine metric expects radians; eps in radians (0.5 km / 6371 km)
    eps_rad = DBSCAN_EPS_KM / 6371.0

    print(f"  Clustering with eps={DBSCAN_EPS_KM} km, min_pts={DBSCAN_MIN_PTS}...")
    db = DBSCAN(eps=eps_rad, min_samples=DBSCAN_MIN_PTS,
                algorithm="ball_tree", metric="haversine", n_jobs=-1)
    df["_CLUSTER"] = db.fit_predict(coords_rad)

    clustered = df[df["_CLUSTER"] >= 0]
    n_clusters = df["_CLUSTER"].nunique() - (1 if -1 in df["_CLUSTER"].values else 0)
    print(f"  Found {n_clusters} blackspot clusters from {len(clustered):,} accidents")

    # Find severity columns
    df["_SEV"] = df.apply(severity_score, axis=1)

    # Aggregate each cluster → one blackspot point
    rows = []
    for cid, grp in df[df["_CLUSTER"] >= 0].groupby("_CLUSTER"):
        lat  = grp["_LAT"].mean()
        lon  = grp["_LON"].mean()
        count = len(grp)
        sev   = grp["_SEV"].sum()

        # Try to get a location name
        loc_candidates = []
        for col in ["LOCATION", "ROAD_NAME", "PLACE", "DISTRICT", "DIST",
                    "ACCIDENT_LOCATION", "CITY", "STATE"]:
            if col in grp.columns:
                val = grp[col].dropna().mode()
                if len(val):
                    loc_candidates.append(str(val.iloc[0]))
                    break
        location = loc_candidates[0] if loc_candidates else f"Cluster_{cid}"

        rows.append({
            "LAT": round(lat, 5),
            "LON": round(lon, 5),
            "ACCIDENT_COUNT": count,
            "_RAW_SCORE": sev,
            "LOCATION": location,
        })

    result = pd.DataFrame(rows)
    max_score = result["_RAW_SCORE"].max() or 1.0
    result["BLACKSPOT_SCORE"] = (result["_RAW_SCORE"] / max_score).round(4)
    result = result.drop(columns=["_RAW_SCORE"])

    # Sort by severity descending
    result = result.sort_values("BLACKSPOT_SCORE", ascending=False).reset_index(drop=True)
    print(f"  ✅ Generated {len(result)} blackspot records")
    return result


# =============================================================================
# Strategy B — Geocode state/district names via Nominatim
# =============================================================================

def geocode_place(name: str, cache: dict) -> tuple:
    """Return (lat, lon) for a place name, with caching."""
    if name in cache:
        return cache[name]

    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="saferoute-india-blackspot-gen-v1")
        location = geolocator.geocode(f"{name}, India", timeout=10)
        if location:
            result = (round(location.latitude, 5), round(location.longitude, 5))
        else:
            result = (None, None)
    except Exception:
        result = (None, None)

    cache[name] = result
    time.sleep(NOMINATIM_DELAY)
    return result


def strategy_b(df: pd.DataFrame) -> pd.DataFrame:
    print("\n  🗺️  Strategy B: Geocoding state/district names via Nominatim")

    state_col, dist_col = find_location_cols(df)
    if not state_col and not dist_col:
        raise ValueError("No state/district/location column found in the CSV.")

    df = df.copy()
    df["_SEV"] = df.apply(severity_score, axis=1)

    # Build a location label
    if dist_col and state_col:
        df["_PLACE"] = (df[dist_col].astype(str).str.title().str.strip()
                        + ", " +
                        df[state_col].astype(str).str.title().str.strip())
    elif dist_col:
        df["_PLACE"] = df[dist_col].astype(str).str.title().str.strip()
    else:
        df["_PLACE"] = df[state_col].astype(str).str.title().str.strip()

    # Aggregate severity per unique place (avoids geocoding 1M rows)
    agg = (df.groupby("_PLACE")
             .agg(ACCIDENT_COUNT=("_SEV", "count"),
                  _RAW_SCORE=("_SEV", "sum"))
             .reset_index()
             .sort_values("_RAW_SCORE", ascending=False))

    # Cap geocoding calls
    if len(agg) > MAX_GEOCODE_ROWS:
        print(f"  Capping to top {MAX_GEOCODE_ROWS} locations by severity")
        agg = agg.head(MAX_GEOCODE_ROWS)

    print(f"  Geocoding {len(agg)} unique locations (this may take ~{len(agg)*NOMINATIM_DELAY/60:.1f} min)...")

    cache = {}
    lats, lons = [], []
    for i, place in enumerate(agg["_PLACE"], 1):
        lat, lon = geocode_place(place, cache)
        lats.append(lat)
        lons.append(lon)
        if i % 50 == 0:
            print(f"    Geocoded {i}/{len(agg)}...")

    agg["LAT"] = lats
    agg["LON"] = lons
    agg = agg.dropna(subset=["LAT", "LON"])

    max_score = agg["_RAW_SCORE"].max() or 1.0
    agg["BLACKSPOT_SCORE"] = (agg["_RAW_SCORE"] / max_score).round(4)
    agg["LOCATION"] = agg["_PLACE"]

    result = agg[["LAT", "LON", "ACCIDENT_COUNT", "BLACKSPOT_SCORE", "LOCATION"]]
    result = result.sort_values("BLACKSPOT_SCORE", ascending=False).reset_index(drop=True)

    print(f"  ✅ Generated {len(result)} blackspot records")
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    global DBSCAN_EPS_KM, DBSCAN_MIN_PTS

    parser = argparse.ArgumentParser(
        description="Generate accident_blackspots.csv from india_road_accidents.csv"
    )
    parser.add_argument("--input",    default=DEFAULT_INPUT,
                        help="Path to india_road_accidents.csv")
    parser.add_argument("--output",   default=OUTPUT_PATH,
                        help="Where to save accident_blackspots.csv")
    parser.add_argument("--strategy", choices=["auto", "a", "b"], default="auto",
                        help="'a'=DBSCAN GPS clustering, 'b'=Nominatim geocoding, "
                             "'auto'=try A then fall back to B")
    parser.add_argument("--eps",      type=float, default=DBSCAN_EPS_KM,
                        help=f"DBSCAN cluster radius in km (default {DBSCAN_EPS_KM})")
    parser.add_argument("--min-pts",  type=int,   default=DBSCAN_MIN_PTS,
                        help=f"DBSCAN min accidents per cluster (default {DBSCAN_MIN_PTS})")
    args = parser.parse_args()

    DBSCAN_EPS_KM = args.eps
    DBSCAN_MIN_PTS = args.min_pts

    print("=" * 60)
    print("  SafeRoute India — Blackspot Generator")
    print("=" * 60)

    if not os.path.exists(args.input):
        print(f"\n  ❌ Input file not found: {args.input}")
        print(f"     Place your accidents CSV there and re-run.")
        sys.exit(1)

    print(f"\n  Reading: {args.input}")
    df = safe_read_csv(args.input)
    df = standardise_columns(df)

    print(f"\n  Columns detected: {list(df.columns[:15])}" +
          ("  ..." if len(df.columns) > 15 else ""))

    result = pd.DataFrame()

    lat_col, lon_col = find_latlon_cols(df)
    has_gps = lat_col is not None and lon_col is not None

    if has_gps:
        print(f"  GPS columns found: {lat_col}, {lon_col}")
    else:
        print("  No GPS columns found.")

    if args.strategy == "a" or (args.strategy == "auto" and has_gps):
        result = strategy_a(df, lat_col, lon_col)

    if result.empty and (args.strategy in ("b", "auto")):
        result = strategy_b(df)

    if result.empty:
        print("\n  ❌ Could not generate any blackspot records.")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.to_csv(args.output, index=False)

    print(f"\n{'=' * 60}")
    print(f"  ✅ Saved {len(result)} blackspots → {args.output}")
    print(f"\n  Top 5 blackspots:")
    print(result.head().to_string(index=False))
    print(f"\n  Now re-run: python run_pipeline.py --city chennai --skip-clean")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()