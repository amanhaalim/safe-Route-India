#!/usr/bin/env python3
# =============================================================================
# run_pipeline.py — Master pipeline runner for SafeRoute India
# =============================================================================
# Runs the full preprocessing → training → scoring pipeline in one command.
#
# Usage:
#   python run_pipeline.py --city chennai
#   python run_pipeline.py --city chennai mumbai --skip-clean
#   python run_pipeline.py --city all
# =============================================================================

import os
import sys
import time
import argparse
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import TARGET_CITIES, MODELS_DIR, GRAPHS_DIR


def step(title: str):
    print(f"\n{'─'*60}")
    print(f"  🔧  {title}")
    print(f"{'─'*60}")


def run_pipeline(cities: list, skip_clean: bool = False,
                 skip_train: bool = False, skip_score: bool = False):
    total_start = time.time()

    # ── 0. Auto-download GeoJSON boundaries ───────────────────────────────────
    step("Downloading auto-downloadable data (district boundaries, etc.)")
    from src.download_data import download_geojson_files
    download_geojson_files()

    # ── 1. Clean all datasets ─────────────────────────────────────────────────
    if not skip_clean:
        step("Cleaning crime data")
        from src.clean_crime import run as clean_crime
        try:
            clean_crime()
        except FileNotFoundError as e:
            logger.warning(f"  Skipped crime cleaning: {e}")

        step("Cleaning accident data")
        from src.clean_accidents import run as clean_accidents
        clean_accidents()

        step("Cleaning flood data")
        from src.clean_flood import run as clean_flood
        try:
            clean_flood()
        except Exception as e:
            logger.warning(f"  Skipped flood cleaning: {e}")

        step("Geocoding all datasets")
        from src.geocode import run as geocode
        geocode()
    else:
        logger.info("Skipping data cleaning (--skip-clean)")

    # ── 2. Build risk graphs for each city ────────────────────────────────────
    from src.snap_to_edges import build_risk_graph
    for city in cities:
        step(f"Building risk graph: {city}")
        build_risk_graph(city)

    # ── 3. Train model ────────────────────────────────────────────────────────
    if not skip_train:
        step("Training risk classifier")
        from src.train_model import run as train
        train()
    else:
        logger.info("Skipping model training (--skip-train)")

    # ── 4. Apply model to graphs ──────────────────────────────────────────────
    if not skip_score:
        from src.score_graph import apply_model_to_graph
        for city in cities:
            step(f"Scoring final graph: {city}")
            apply_model_to_graph(city)
    else:
        logger.info("Skipping ML scoring (--skip-score)")

    elapsed = round(time.time() - total_start, 1)
    print(f"\n{'='*60}")
    print(f"  ✅  Pipeline complete in {elapsed}s")
    print(f"  Cities processed: {', '.join(cities)}")
    print(f"  Start the API: uvicorn api.main:app --port 8000 --reload")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeRoute India pipeline runner")
    parser.add_argument("--city", nargs="+", default=["chennai"],
                        help="City key(s) or 'all'")
    parser.add_argument("--skip-clean", action="store_true",
                        help="Skip data cleaning (use existing processed files)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip model training (use existing model)")
    parser.add_argument("--skip-score", action="store_true",
                        help="Skip final graph scoring")

    args = parser.parse_args()

    # Resolve city list
    if "all" in args.city:
        cities = list(TARGET_CITIES.keys())
    else:
        cities = []
        for c in args.city:
            if c not in TARGET_CITIES:
                print(f"Unknown city: '{c}'. Options: {list(TARGET_CITIES.keys())}")
                sys.exit(1)
            cities.append(c)

    run_pipeline(
        cities=cities,
        skip_clean=args.skip_clean,
        skip_train=args.skip_train,
        skip_score=args.skip_score,
    )
