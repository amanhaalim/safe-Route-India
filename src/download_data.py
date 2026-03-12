#!/usr/bin/env python3
# =============================================================================
# src/01_download_data.py — Data download helpers (no API keys)
# =============================================================================
# Handles:
#   • IMD rainfall data via imdlib (no login needed)
#   • OSM road network via osmnx  (no login needed)
#   • India district / state GeoJSON from public GitHub mirrors
#   • Prints download instructions for Kaggle / data.gov.in datasets
#     that require a browser click
# =============================================================================

import os
import sys
import time
import requests
from pathlib import Path
from loguru import logger

# ── path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (TARGET_CITIES, RAW_MAPS_DIR, RAW_FLOOD_DIR,
                    RAW_CRIME_DIR, RAW_ACCIDENT_DIR)
from src.utils import ensure_dirs, progress_bar


# =============================================================================
# 1. Print manual download instructions
# =============================================================================

MANUAL_DATASETS = [
    {
        "name":  "Indian Crimes Dataset (Primary)",
        "url":   "https://www.kaggle.com/datasets/sudhanvahg/indian-crimes-dataset",
        "save":  "data/raw/crime/crimes_india.csv",
        "note":  "Click 'Download' → extract ZIP → rename main CSV",
    },
    {
        "name":  "NCRB District IPC Crimes",
        "url":   "https://www.kaggle.com/datasets/rajanand/crime-in-india",
        "save":  "data/raw/crime/district_ipc_crimes.csv",
        "note":  "Download ZIP → use '01_District_wise_crimes_committed_IPC_2001_2012.csv'",
    },
    {
        "name":  "Crime Against Women",
        "url":   "https://data.gov.in/catalog/crime-against-women",
        "save":  "data/raw/crime/crime_against_women.csv",
        "note":  "Click 'Download CSV' on the page",
    },
    {
        "name":  "India Road Accidents (Kaggle)",
        "url":   "https://www.kaggle.com/datasets/data125661/india-road-accident-dataset",
        "save":  "data/raw/accidents/india_road_accidents.csv",
        "note":  "Click 'Download' on Kaggle",
    },
    {
        "name":  "NH Accident Blackspots",
        "url":   "https://data.gov.in/catalog/accident-black-spots",
        "save":  "data/raw/accidents/accident_blackspots.csv",
        "note":  "Download CSV / Excel. If Excel: Save As CSV first.",
    },
    {
        "name":  "Flood Affected Districts",
        "url":   "https://data.gov.in/catalog/flood-affected-districts",
        "save":  "data/raw/flood/flood_affected_districts.csv",
        "note":  "Click 'Download CSV'",
    },
    {
        "name":  "India Flood Inventory (IIT-Delhi)",
        "url":   "https://github.com/hydrosenselab/India-Flood-Inventory",
        "save":  "data/raw/flood/  (all .shp .dbf .prj .shx files)",
        "note":  "Click 'Code' → 'Download ZIP' → extract ALL files to data/raw/flood/",
    },
]


def print_manual_instructions():
    print("\n" + "=" * 65)
    print("  MANUAL DOWNLOAD REQUIRED FOR THESE DATASETS")
    print("=" * 65)
    for i, ds in enumerate(MANUAL_DATASETS, 1):
        status = "✅" if os.path.exists(os.path.join(PROJECT_ROOT, ds["save"].split("(")[0].strip())) else "❌"
        print(f"\n  [{i}] {status} {ds['name']}")
        print(f"       URL:  {ds['url']}")
        print(f"       Save: {ds['save']}")
        print(f"       Note: {ds['note']}")
    print("\n" + "=" * 65 + "\n")


# =============================================================================
# 2. Auto-download: India district & state boundaries (public GitHub mirrors)
# =============================================================================

GEOJSON_SOURCES = [
    {
        "name": "India Districts GeoJSON",
        "urls": [
            "https://raw.githubusercontent.com/datameet/maps/master/Districts/India_Districts.geojson",
            "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson",
        ],
        "save": os.path.join(RAW_MAPS_DIR, "india_districts.geojson"),
    },
    {
        "name": "India States GeoJSON",
        "urls": [
            "https://raw.githubusercontent.com/datameet/maps/master/States/Admin2.geojson",
            "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson",
        ],
        "save": os.path.join(RAW_MAPS_DIR, "india_states.geojson"),
    },
]


def download_geojson_files():
    """Try each mirror URL until one succeeds."""
    ensure_dirs(RAW_MAPS_DIR)
    for item in GEOJSON_SOURCES:
        if os.path.exists(item["save"]):
            logger.info(f"Already exists: {item['name']}")
            continue
        logger.info(f"Downloading {item['name']}...")
        success = False
        for url in item["urls"]:
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    with open(item["save"], "wb") as f:
                        f.write(resp.content)
                    logger.success(f"  Saved → {item['save']}")
                    success = True
                    break
            except Exception as e:
                logger.warning(f"  Mirror {url} failed: {e}")
        if not success:
            logger.error(f"  Could not download {item['name']}. "
                         "Please download manually from datameet/maps on GitHub.")


# =============================================================================
# 3. Auto-download: City road graphs via osmnx
# =============================================================================

def download_city_graphs(cities: list = None):
    """
    Download OSM road graphs for the specified cities using osmnx.
    Graphs are cached as .graphml files so subsequent runs are instant.
    """
    import osmnx as ox
    ensure_dirs(RAW_MAPS_DIR)

    cities = cities or list(TARGET_CITIES.keys())
    for city_key in cities:
        if city_key not in TARGET_CITIES:
            logger.warning(f"Unknown city: {city_key}")
            continue

        cache_path = os.path.join(RAW_MAPS_DIR, f"{city_key}_raw.graphml")
        if os.path.exists(cache_path):
            logger.info(f"Graph already cached: {city_key}")
            continue

        cfg = TARGET_CITIES[city_key]
        logger.info(f"Downloading OSM graph for {city_key} ({cfg['place']})...")
        try:
            G = ox.graph_from_place(cfg["place"], network_type="drive",
                                    simplify=True, retain_all=False)
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)
            ox.save_graphml(G, cache_path)
            n, e = G.number_of_nodes(), G.number_of_edges()
            logger.success(f"  Saved {city_key}: {n} nodes, {e} edges → {cache_path}")
        except Exception as ex:
            logger.error(f"  Failed to download {city_key}: {ex}")


# =============================================================================
# 4. Auto-download: IMD rainfall data via imdlib
# =============================================================================

def download_imd_rainfall(start_year: int = 2010, end_year: int = 2022):
    """Download IMD gridded rainfall via imdlib (no API key needed)."""
    try:
        import imdlib as imd
    except ImportError:
        logger.error("imdlib not installed. Run: pip install imdlib")
        return

    save_dir = os.path.join(RAW_FLOOD_DIR, "imd_rain")
    ensure_dirs(save_dir)

    if len(os.listdir(save_dir)) > 0:
        logger.info("IMD rainfall data already present.")
        return

    logger.info(f"Downloading IMD gridded rainfall {start_year}–{end_year}...")
    logger.info("  This downloads ~400–600 MB and takes 15–30 minutes.")
    try:
        imd.get_data(
            variable="rain",
            start_yr=start_year,
            end_yr=end_year,
            fn_format="yearwise",
            file_dir=save_dir,
        )
        logger.success(f"  IMD rainfall saved → {save_dir}")
    except Exception as ex:
        logger.error(f"  IMD download failed: {ex}")


# =============================================================================
# 5. Verification check
# =============================================================================

REQUIRED_FILES = [
    ("data/raw/crime/crimes_india.csv",              "❌ MISSING — see manual download above"),
    ("data/raw/crime/district_ipc_crimes.csv",        "❌ MISSING — see manual download above"),
    ("data/raw/accidents/india_road_accidents.csv",   "❌ MISSING — see manual download above"),
    ("data/raw/maps/india_districts.geojson",         "❌ MISSING — run: python src/01_download_data.py auto"),
]

OPTIONAL_FILES = [
    ("data/raw/crime/crime_against_women.csv",        "optional but recommended"),
    ("data/raw/accidents/accident_blackspots.csv",    "optional but recommended — GPS blackspots"),
    ("data/raw/flood/flood_affected_districts.csv",   "optional"),
]


def verify_downloads():
    print("\n── Required Files ──")
    all_ok = True
    for rel_path, msg in REQUIRED_FILES:
        full = os.path.join(PROJECT_ROOT, rel_path)
        ok = os.path.exists(full)
        if not ok:
            all_ok = False
        print(f"  {'✅' if ok else msg:50s}  {rel_path}")

    print("\n── Optional Files ──")
    for rel_path, note in OPTIONAL_FILES:
        full = os.path.join(PROJECT_ROOT, rel_path)
        ok = os.path.exists(full)
        print(f"  {'✅' if ok else '⚠️  ' + note:50s}  {rel_path}")

    # Check at least one .shp in flood dir
    shp_files = list_shp_files()
    print(f"\n  {'✅' if shp_files else '⚠️  MISSING — download flood inventory from GitHub':50s}  "
          f"data/raw/flood/*.shp")

    # Check at least one .osm.pbf or cached .graphml
    osm_present = any(
        f.endswith((".osm.pbf", "_raw.graphml"))
        for f in os.listdir(os.path.join(PROJECT_ROOT, "data/raw/maps"))
    )
    print(f"  {'✅' if osm_present else '⚠️  Run: python src/01_download_data.py cities chennai':50s}  "
          f"data/raw/maps/*.osm.pbf or *_raw.graphml\n")
    return all_ok


def list_shp_files():
    flood_dir = os.path.join(PROJECT_ROOT, RAW_FLOOD_DIR)
    if not os.path.exists(flood_dir):
        return []
    return [f for f in os.listdir(flood_dir) if f.endswith(".shp")]


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    import click

    @click.command()
    @click.argument("action", type=click.Choice(
        ["all", "auto", "cities", "imd", "verify", "instructions"],
        case_sensitive=False,
    ), default="instructions")
    @click.option("--cities", "-c", default="chennai",
                  help="Comma-separated city keys, e.g. chennai,mumbai")
    @click.option("--imd-start", default=2010, type=int)
    @click.option("--imd-end",   default=2022, type=int)
    def main(action, cities, imd_start, imd_end):
        """
        Download data for SafeRoute India.

        \b
        Actions:
          instructions  Print manual download steps (default)
          auto          Download all auto-downloadable data
          cities        Download OSM graphs for specified cities
          imd           Download IMD rainfall data
          verify        Check which files are present
          all           Run auto + cities + imd
        """
        ensure_dirs(
            RAW_CRIME_DIR, RAW_ACCIDENT_DIR,
            RAW_FLOOD_DIR, RAW_MAPS_DIR,
            os.path.join(PROJECT_ROOT, "logs"),
        )

        if action in ("instructions", "all"):
            print_manual_instructions()

        if action in ("auto", "all"):
            download_geojson_files()

        if action in ("cities", "all"):
            city_list = [c.strip() for c in cities.split(",")]
            download_city_graphs(city_list)

        if action in ("imd", "all"):
            download_imd_rainfall(imd_start, imd_end)

        if action in ("verify",):
            verify_downloads()

        if action not in ("verify", "instructions"):
            print("\n── Download summary ──")
            verify_downloads()

    main()
