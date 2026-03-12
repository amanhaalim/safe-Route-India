#!/usr/bin/env python3
"""
validation/validate_model.py
══════════════════════════════════════════════════════════════════
SafeRoute India — Full Model Validation Suite
══════════════════════════════════════════════════════════════════
Run after completing the full pipeline to verify everything works.

Usage:
    python validation/validate_model.py --city chennai
    python validation/validate_model.py --city all
    python validation/validate_model.py --city chennai --verbose
    python validation/validate_model.py --offline   # skip geocoding tests
"""

import sys, os, argparse, logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import TARGET_CITIES, TIME_MODIFIERS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("saferoute.validation")

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

# ── Known test route pairs ────────────────────────────────────────────────────
TEST_PAIRS = {
    "chennai":   [
        {"origin": "Chennai Central Railway Station",
         "destination": "T. Nagar, Chennai",
         "label": "Central → T.Nagar"},
        {"origin": "Anna Salai, Chennai",
         "destination": "Adyar, Chennai",
         "label": "Anna Salai → Adyar"},
    ],
    "mumbai":    [
        {"origin": "Chhatrapati Shivaji Terminus, Mumbai",
         "destination": "Bandra West, Mumbai",
         "label": "CST → Bandra"},
    ],
    "delhi":     [
        {"origin": "Connaught Place, Delhi",
         "destination": "Hauz Khas, Delhi",
         "label": "CP → Hauz Khas"},
    ],
    "bengaluru": [
        {"origin": "Kempegowda Bus Station, Bengaluru",
         "destination": "Koramangala, Bengaluru",
         "label": "Majestic → Koramangala"},
    ],
    "hyderabad": [
        {"origin": "Secunderabad Railway Station",
         "destination": "Hitech City, Hyderabad",
         "label": "Secunderabad → Hitech City"},
    ],
}


class Results:
    def __init__(self):
        self.passed = self.failed = self.warned = 0
        self.rows   = []

    def ok(self, check, detail=""):
        self.passed += 1
        self.rows.append((PASS, check, detail))

    def fail(self, check, detail=""):
        self.failed += 1
        self.rows.append((FAIL, check, detail))

    def warn(self, check, detail=""):
        self.warned += 1
        self.rows.append((WARN, check, detail))

    def print_summary(self):
        total = self.passed + self.failed + self.warned
        print(f"\n{'═'*65}")
        print(f"  Validation: {total} checks | "
              f"{self.passed} passed | {self.failed} failed | {self.warned} warnings")
        print(f"{'═'*65}")
        for status, check, detail in self.rows:
            print(f"  {status}  {check}")
            if detail:
                print(f"             {detail}")
        print(f"{'═'*65}\n")
        return self.failed == 0


# ── Check functions ───────────────────────────────────────────────────────────

def check_files(R: Results):
    """Verify all expected output files exist from the pipeline."""
    required = {
        "models/risk_classifier.pkl": "Run: python src/07_train_model.py",
        "models/scaler.pkl":          "Run: python src/07_train_model.py",
        "data/processed/crime_clean.csv":      "Run: python src/02_clean_crime.py",
        "data/processed/crime_geocoded.csv":   "Run: python src/05_geocode.py",
        "data/processed/accidents_clean.csv":  "Run: python src/03_clean_accidents.py",
        "data/processed/flood_risk_by_district.csv": "Run: python src/04_clean_flood.py",
    }
    for path, hint in required.items():
        p = Path(path)
        if p.exists():
            kb = p.stat().st_size / 1024
            R.ok(f"File exists: {path} ({kb:.0f} KB)")
        else:
            R.fail(f"Missing: {path}", hint)


def check_processed_data(R: Results):
    """Validate processed CSV files have expected columns and row counts."""
    specs = {
        "data/processed/crime_clean.csv": {
            "required": ["STATE", "DISTRICT", "CRIME_SCORE_NORM"],
            "min_rows": 100,
        },
        "data/processed/crime_geocoded.csv": {
            "required": ["LAT", "LON", "CRIME_SCORE_NORM"],
            "min_rows": 100,
            "lat_range": (6.0, 37.5),
            "lon_range": (67.0, 98.0),
        },
        "data/processed/accidents_clean.csv": {
            "required": ["STATE", "ACCIDENT_SCORE"],
            "min_rows": 10,
        },
        "data/processed/flood_risk_by_district.csv": {
            "required": ["DISTRICT_NAME", "FLOOD_SCORE_NORM"],
            "min_rows": 50,
        },
    }
    for path, spec in specs.items():
        if not Path(path).exists():
            continue  # already caught by check_files
        df = pd.read_csv(path)
        missing = [c for c in spec["required"] if c not in df.columns]
        if missing:
            R.warn(f"{Path(path).name}: missing columns {missing}")
        elif len(df) < spec["min_rows"]:
            R.warn(f"{Path(path).name}: only {len(df)} rows (expected ≥{spec['min_rows']})")
        else:
            R.ok(f"{Path(path).name}: {len(df):,} rows, all columns present")

        # Coordinate sanity for geocoded data
        if "lat_range" in spec and "LAT" in df.columns:
            lat_min, lat_max = spec["lat_range"]
            lon_min, lon_max = spec["lon_range"]
            bad = df[~df["LAT"].between(lat_min, lat_max) | ~df["LON"].between(lon_min, lon_max)]
            if len(bad) > 0:
                pct = 100 * len(bad) / len(df)
                R.warn(f"{Path(path).name}: {len(bad)} rows ({pct:.1f}%) have coords outside India")
            else:
                R.ok(f"{Path(path).name}: all lat/lon values within India bounding box")


def check_graph(G, city_key: str, R: Results):
    """Validate graph attribute coverage and value ranges."""
    n_edges = G.number_of_edges()
    n_nodes = G.number_of_nodes()

    if n_nodes < 100 or n_edges < 100:
        R.fail(f"[{city_key}] Graph too small: {n_nodes} nodes, {n_edges} edges")
        return

    R.ok(f"[{city_key}] Graph size: {n_nodes:,} nodes, {n_edges:,} edges")

    # Check required attributes
    required = ["crime_score", "accident_score", "flood_score",
                "composite_risk", "effective_weight"]
    for attr in required:
        missing = sum(1 for _, _, d in G.edges(data=True) if attr not in d)
        pct_ok  = 100 * (1 - missing / n_edges)
        if pct_ok >= 99:
            R.ok(f"[{city_key}] {attr}: {pct_ok:.1f}% coverage")
        elif pct_ok >= 90:
            R.warn(f"[{city_key}] {attr}: {pct_ok:.1f}% coverage ({missing} edges missing)")
        else:
            R.fail(f"[{city_key}] {attr}: only {pct_ok:.1f}% coverage — check pipeline")

    # Value range check
    for attr in ["crime_score", "accident_score", "flood_score", "composite_risk"]:
        vals = [float(d.get(attr, 0)) for _, _, d in G.edges(data=True)]
        mn, mx, std = min(vals), max(vals), float(np.std(vals))
        if 0 <= mn and mx <= 1:
            if std < 0.01:
                R.warn(f"[{city_key}] {attr}: low variance (std={std:.4f}) — all roads similar risk")
            else:
                R.ok(f"[{city_key}] {attr} range OK: [{mn:.3f},{mx:.3f}] std={std:.3f}")
        else:
            R.fail(f"[{city_key}] {attr} out of [0,1]: min={mn:.3f} max={mx:.3f}")


def check_time_modifiers_effect(G, city_key: str, R: Results):
    """Verify night weights are higher than daytime weights."""
    import copy
    from src.routing import apply_time_modifiers_to_graph

    G_night = copy.deepcopy(G)
    G_day   = copy.deepcopy(G)
    apply_time_modifiers_to_graph(G_night, hour=2)
    apply_time_modifiers_to_graph(G_day,   hour=11)

    night_mean = np.mean([float(d.get("time_adjusted_weight", 0)) for _,_,d in G_night.edges(data=True)])
    day_mean   = np.mean([float(d.get("time_adjusted_weight", 0)) for _,_,d in G_day.edges(data=True)])
    ratio      = night_mean / day_mean if day_mean > 0 else 1.0

    if ratio > 1.1:
        R.ok(f"[{city_key}] Time modifier: night/day ratio = {ratio:.2f}× (night weights higher ✓)")
    else:
        R.fail(f"[{city_key}] Time modifier not working: ratio={ratio:.2f} (should be >1.1)")


def check_routing(city_key: str, pairs: list, R: Results, verbose: bool):
    """Run actual route requests and verify safest ≤ risk of fastest."""
    from src.routing import find_safe_routes

    for pair in pairs:
        label = pair["label"]
        try:
            routes, orig, dest = find_safe_routes(
                pair["origin"], pair["destination"],
                city_key, travel_hour=14,
            )
        except Exception as e:
            R.warn(f"[{city_key}] {label}: routing error — {e}")
            continue

        safest  = routes.get("safest")
        fastest = routes.get("fastest")

        if not safest or not fastest:
            R.warn(f"[{city_key}] {label}: one route is None (no path found)")
            continue

        s_risk = safest["summary"]["avg_risk_score"]
        f_risk = fastest["summary"]["avg_risk_score"]
        s_dist = safest["summary"]["distance_km"]
        f_dist = fastest["summary"]["distance_km"]

        if verbose:
            print(f"\n    Route: {label}")
            print(f"    Safest : {s_dist:.1f} km | risk {s_risk:.3f} | "
                  f"{safest['summary']['risk_label']}")
            print(f"    Fastest: {f_dist:.1f} km | risk {f_risk:.3f} | "
                  f"{fastest['summary']['risk_label']}")

        # Core assertion: safest route should have ≤ risk than fastest
        if s_risk <= f_risk + 0.02:
            R.ok(f"[{city_key}] {label}: safest({s_risk:.3f}) ≤ fastest({f_risk:.3f}) ✓")
        else:
            R.fail(f"[{city_key}] {label}: safest route MORE risky than fastest",
                   f"safest={s_risk:.3f}  fastest={f_risk:.3f} — check RISK_MULTIPLIER in config")

        # Fastest should be physically shorter (or same)
        if f_dist <= s_dist + 1.0:
            R.ok(f"[{city_key}] {label}: fastest({f_dist:.1f}km) ≤ safest({s_dist:.1f}km) ✓")
        else:
            R.warn(f"[{city_key}] {label}: fastest route longer than safest",
                   "Possible disconnected graph; try different addresses")

        # All summaries should have valid structure
        for name, route in routes.items():
            if not route:
                continue
            s = route["summary"]
            assert "distance_km" in s
            assert "safety_pct" in s
            assert 0 <= s["avg_risk_score"] <= 1, f"{name} risk out of range"
            assert 0 <= s["safety_pct"] <= 100,   f"{name} safety_pct out of range"
        R.ok(f"[{city_key}] {label}: all route summaries structurally valid")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--city",    "-c", default="chennai",
                    help="City key, or 'all'")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--offline",       action="store_true",
                    help="Skip live geocoding/routing tests")
    args = ap.parse_args()

    R = Results()

    print(f"\n🛡️  SafeRoute India — Validation Suite")
    print(f"{'═'*65}")

    check_files(R)
    check_processed_data(R)

    cities = list(TARGET_CITIES.keys()) if args.city == "all" else [args.city]

    for city_key in cities:
        final_path = Path(f"graphs/{city_key}_final_graph.graphml")
        risk_path  = Path(f"graphs/{city_key}_risk_graph.graphml")

        if final_path.exists():
            graph_path = str(final_path)
        elif risk_path.exists():
            graph_path = str(risk_path)
            R.warn(f"[{city_key}] Only partial graph (risk, not final)",
                   "Run: python src/08_score_graph.py " + city_key)
        else:
            R.fail(f"[{city_key}] No graph found",
                   "Run: python src/06_snap_to_edges.py " + city_key)
            continue

        import osmnx as ox
        logger.info(f"Loading graph for validation: {graph_path}")
        G = ox.load_graphml(graph_path)

        check_graph(G, city_key, R)
        check_time_modifiers_effect(G, city_key, R)

        if not args.offline:
            pairs = TEST_PAIRS.get(city_key, [])
            if pairs:
                check_routing(city_key, pairs, R, args.verbose)
            else:
                R.warn(f"[{city_key}] No test pairs defined for this city")

    ok = R.print_summary()
    sys.exit(0 if ok else 1)
