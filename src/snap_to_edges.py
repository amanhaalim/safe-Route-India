#!/usr/bin/env python3
# =============================================================================
# src/06_snap_to_edges.py — Snap risk data to OSM road edges
# =============================================================================
# Input:  data/raw/maps/{city}_raw.graphml
#         data/processed/crime_geocoded.csv
#         data/processed/accidents_geocoded.csv
#         data/processed/blackspots_clean.csv
#         data/processed/flood_geocoded.csv
# Output: graphs/{city}_risk_graph.graphml
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    TARGET_CITIES, RAW_MAPS_DIR, PROCESSED_DIR, GRAPHS_DIR,
    WEIGHTS, SNAP_RADIUS_M, ROAD_CLASS_RISK, ROAD_CLASS_DEFAULT,
)
from src.utils import (
    ensure_dirs, latlon_to_gdf, get_india_metric_crs,
    progress_bar,
)


# =============================================================================
# Graph download / load
# =============================================================================

def load_or_download_graph(city_key: str):
    import osmnx as ox

    cache = os.path.join(RAW_MAPS_DIR, f"{city_key}_raw.graphml")
    if os.path.exists(cache):
        logger.info(f"Loading cached graph: {cache}")
        G = ox.load_graphml(cache)
    else:
        cfg = TARGET_CITIES[city_key]
        logger.info(f"Downloading OSM graph for {city_key}...")
        G = ox.graph_from_place(cfg["place"], network_type="drive",
                                simplify=True, retain_all=False)
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        ensure_dirs(RAW_MAPS_DIR)
        ox.save_graphml(G, cache)
        logger.success(f"  Saved {city_key} graph: "
                       f"{G.number_of_nodes()}n, {G.number_of_edges()}e")
    return G


# =============================================================================
# Load risk GeoDataFrames
# =============================================================================

def load_risk_gdfs() -> dict:
    import geopandas as gpd

    gdfs = {}

    for key, filename, score_col in [
        ("crime",    "crime_geocoded.csv",    "CRIME_SCORE_NORM"),
        ("accident", "accidents_geocoded.csv", "ACCIDENT_SCORE_NORM"),
        ("flood",    "flood_geocoded.csv",    "FLOOD_COMPOSITE_NORM"),
    ]:
        path = os.path.join(PROCESSED_DIR, filename)
        if not os.path.exists(path):
            logger.warning(f"  Missing {filename} — {key} layer will be zero")
            gdfs[key] = None
            continue

        df = pd.read_csv(path)
        if score_col not in df.columns:
            # Try to find any *_NORM column
            norm_cols = [c for c in df.columns if "NORM" in c.upper()]
            if norm_cols:
                df[score_col] = df[norm_cols[0]]
            else:
                logger.warning(f"  {filename}: no normalised score column found")
                gdfs[key] = None
                continue

        df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)
        # Keep score alongside lat/lon before filtering so alignment is preserved
        df["_SCORE"] = df[score_col]
        gdf = latlon_to_gdf(df)   # drops NaN lat/lon rows internally
        gdfs[key] = gdf
        logger.info(f"  Loaded {key}: {len(gdf)} points")

    # GPS blackspots (optional high-precision layer)
    bs_path = os.path.join(PROCESSED_DIR, "blackspots_clean.csv")
    if os.path.exists(bs_path):
        bs_df = pd.read_csv(bs_path)
        if {"LAT", "LON", "BLACKSPOT_SCORE"}.issubset(bs_df.columns):
            bs_df["_SCORE"] = bs_df["BLACKSPOT_SCORE"]
            gdfs["blackspot"] = latlon_to_gdf(bs_df)
            logger.info(f"  Loaded blackspots: {len(gdfs['blackspot'])} GPS points")
        else:
            gdfs["blackspot"] = None
    else:
        gdfs["blackspot"] = None

    return gdfs


# =============================================================================
# Core scoring: buffer join
# =============================================================================

def score_edges(G, gdfs: dict, radius_m: int) -> pd.DataFrame:
    """
    For every edge in G, buffer by radius_m metres and aggregate
    nearby risk point scores. Returns a DataFrame indexed by (u, v, key).
    """
    import osmnx as ox
    import geopandas as gpd

    METRIC_CRS = get_india_metric_crs()

    # Convert graph to GeoDataFrame of edges
    _, edges = ox.graph_to_gdfs(G)
    edges = edges.reset_index()   # columns: u, v, key + osmid, length, geometry, ...
    n_edges = len(edges)
    logger.info(f"  Scoring {n_edges} edges (radius={radius_m}m)...")

    edges_m = edges.to_crs(METRIC_CRS)

    # Project all risk GDFs to metric CRS once
    projected = {}
    for key, gdf in gdfs.items():
        if gdf is not None and not gdf.empty:
            projected[key] = gdf.to_crs(METRIC_CRS)

    crime_scores, accident_scores, flood_scores, road_scores = [], [], [], []

    for i, row in progress_bar(edges_m.iterrows(), desc="Scoring edges", total=n_edges):
        geom = row["geometry"]
        buf  = geom.buffer(radius_m)

        # ── Crime ─────────────────────────────────────────────────────────
        c_gdf = projected.get("crime")
        if c_gdf is not None and not c_gdf.empty:
            mask = c_gdf.geometry.intersects(buf)
            c_score = float(c_gdf.loc[mask, "_SCORE"].mean()) if mask.any() else 0.0
        else:
            c_score = 0.0

        # ── Accident (state-level + GPS blackspots) ────────────────────────
        a_gdf = projected.get("accident")
        a_score = 0.0
        if a_gdf is not None and not a_gdf.empty:
            mask = a_gdf.geometry.intersects(buf)
            a_score = float(a_gdf.loc[mask, "_SCORE"].mean()) if mask.any() else 0.0

        b_gdf = projected.get("blackspot")
        if b_gdf is not None and not b_gdf.empty:
            mask = b_gdf.geometry.intersects(buf)
            if mask.any():
                bs_score = float(b_gdf.loc[mask, "_SCORE"].max())
                a_score  = max(a_score, bs_score)  # take worst-case from blackspot

        # ── Flood ──────────────────────────────────────────────────────────
        f_gdf = projected.get("flood")
        if f_gdf is not None and not f_gdf.empty:
            mask = f_gdf.geometry.intersects(buf)
            f_score = float(f_gdf.loc[mask, "_SCORE"].mean()) if mask.any() else 0.0
        else:
            f_score = 0.0

        # ── Road class (infra) ─────────────────────────────────────────────
        hw = row.get("highway", "unclassified")
        if isinstance(hw, list):
            hw = hw[0]
        r_score = ROAD_CLASS_RISK.get(str(hw).lower(), ROAD_CLASS_DEFAULT)

        crime_scores.append(round(c_score, 4))
        accident_scores.append(round(a_score, 4))
        flood_scores.append(round(f_score, 4))
        road_scores.append(round(r_score, 4))

    edges["crime_score"]    = crime_scores
    edges["accident_score"] = accident_scores
    edges["flood_score"]    = flood_scores
    edges["road_score"]     = road_scores

    # Composite weighted risk
    edges["composite_risk"] = (
        WEIGHTS["crime"]    * edges["crime_score"]    +
        WEIGHTS["accident"] * edges["accident_score"] +
        WEIGHTS["flood"]    * edges["flood_score"]    +
        WEIGHTS["infra"]    * edges["road_score"]
    ).clip(0, 1)

    return edges


# =============================================================================
# Inject scores into graph
# =============================================================================

def inject_scores_into_graph(G, scored_edges: pd.DataFrame):
    """Write edge scores from scored_edges DataFrame back into G."""
    # Build fast lookup: (u, v, key) → row
    idx = scored_edges.set_index(["u", "v", "key"])

    for u, v, key, data in G.edges(data=True, keys=True):
        try:
            row = idx.loc[(u, v, key)]
            data["crime_score"]    = float(row["crime_score"])
            data["accident_score"] = float(row["accident_score"])
            data["flood_score"]    = float(row["flood_score"])
            data["road_score"]     = float(row["road_score"])
            data["composite_risk"] = float(row["composite_risk"])
        except KeyError:
            # Edge not in scored table (e.g. simplified away) — use defaults
            data.setdefault("crime_score",    0.3)
            data.setdefault("accident_score", 0.3)
            data.setdefault("flood_score",    0.1)
            data.setdefault("road_score",     0.5)
            data["composite_risk"] = (
                WEIGHTS["crime"]    * data["crime_score"]    +
                WEIGHTS["accident"] * data["accident_score"] +
                WEIGHTS["flood"]    * data["flood_score"]    +
                WEIGHTS["infra"]    * data["road_score"]
            )
    return G


# =============================================================================
# Main
# =============================================================================

def build_risk_graph(city_key: str):
    import osmnx as ox

    ensure_dirs(GRAPHS_DIR)

    logger.info(f"═══ Building risk graph for: {city_key} ═══")
    G   = load_or_download_graph(city_key)
    gdfs = load_risk_gdfs()

    scored = score_edges(G, gdfs, SNAP_RADIUS_M)
    G      = inject_scores_into_graph(G, scored)

    out = os.path.join(GRAPHS_DIR, f"{city_key}_risk_graph.graphml")
    ox.save_graphml(G, out)
    logger.success(f"Risk graph saved → {out}")

    # Quick stats
    risks = [d["composite_risk"] for _, _, d in G.edges(data=True)]
    logger.info(f"  Composite risk — mean: {np.mean(risks):.3f}  "
                f"max: {np.max(risks):.3f}  "
                f"high (>0.08): {sum(r > 0.08 for r in risks)} edges")
    return G


if __name__ == "__main__":
    import click

    @click.command()
    @click.argument("cities", nargs=-1)
    def main(cities):
        """Build risk graphs for given city keys (default: all in config)."""
        targets = list(cities) if cities else list(TARGET_CITIES.keys())
        for city in targets:
            if city not in TARGET_CITIES:
                logger.warning(f"Unknown city: {city}")
                continue
            build_risk_graph(city)

    main()
    