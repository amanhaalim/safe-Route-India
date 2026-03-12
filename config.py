# =============================================================================
# config.py — SafeRoute India Central Configuration
# =============================================================================
# All tunable parameters live here. Change weights, cities, thresholds,
# and file paths from this single file without touching any source code.
# =============================================================================

import os

# ── Cities supported by the MVP ──────────────────────────────────────────────
TARGET_CITIES = {
    "chennai": {
        "place":  "Chennai, Tamil Nadu, India",
        "center": (13.0827, 80.2707),
        "state":  "Tamil Nadu",
        "bbox":   (12.80, 80.10, 13.25, 80.45),  # (south, west, north, east)
    },
    "mumbai": {
        "place":  "Mumbai, Maharashtra, India",
        "center": (19.0760, 72.8777),
        "state":  "Maharashtra",
        "bbox":   (18.85, 72.75, 19.35, 73.00),
    },
    "delhi": {
        "place":  "Delhi, India",
        "center": (28.6139, 77.2090),
        "state":  "Delhi",
        "bbox":   (28.40, 76.84, 28.88, 77.35),
    },
    "bengaluru": {
        "place":  "Bengaluru, Karnataka, India",
        "center": (12.9716, 77.5946),
        "state":  "Karnataka",
        "bbox":   (12.82, 77.45, 13.15, 77.78),
    },
    "hyderabad": {
        "place":  "Hyderabad, Telangana, India",
        "center": (17.3850, 78.4867),
        "state":  "Telangana",
        "bbox":   (17.20, 78.30, 17.60, 78.65),
    },
}

# ── Risk Factor Weights (must sum to 1.0) ────────────────────────────────────
WEIGHTS = {
    "crime":    0.40,   # crime severity score per district
    "accident": 0.30,   # road accident + blackspot score
    "flood":    0.15,   # historical flood frequency + IMD rainfall
    "infra":    0.15,   # road class, surface quality (from OSM)
}

# ── Crime Severity Weights ────────────────────────────────────────────────────
# Higher = more dangerous for a route traveller
CRIME_SEVERITY = {
    "MURDER":           10,
    "RAPE":              9,
    "DACOITY":           8,
    "ROBBERY":           7,
    "KIDNAPPING":        7,
    "RIOTS":             6,
    "ASSAULT":           6,
    "HURT":              4,
    "BURGLARY":          5,
    "THEFT":             3,
    "OTHER_IPC":         2,
    "CULPABLE_HOMICIDE": 9,
    "ABDUCTION":         6,
    "ARSON":             5,
}

# ── OSM Road Class Risk (lower highway class = more exposure) ─────────────────
ROAD_CLASS_RISK = {
    "motorway":       0.10,
    "motorway_link":  0.12,
    "trunk":          0.18,
    "trunk_link":     0.18,
    "primary":        0.28,
    "primary_link":   0.28,
    "secondary":      0.38,
    "secondary_link": 0.38,
    "tertiary":       0.48,
    "tertiary_link":  0.48,
    "residential":    0.58,
    "living_street":  0.52,
    "unclassified":   0.65,
    "service":        0.68,
    "track":          0.82,
    "path":           0.88,
    "footway":        0.78,
    "cycleway":       0.60,
    "steps":          0.75,
}
ROAD_CLASS_DEFAULT = 0.65

# ── Time-of-Day Modifiers ─────────────────────────────────────────────────────
# Multiplied onto crime/accident base scores based on travel hour
TIME_MODIFIERS = {
    (0,  6):  {"crime": 2.0, "accident": 0.9},   # midnight–dawn: highest crime
    (6,  9):  {"crime": 0.6, "accident": 1.2},   # morning rush
    (9,  18): {"crime": 0.7, "accident": 0.8},   # daytime: safest window
    (18, 21): {"crime": 1.2, "accident": 1.3},   # evening rush
    (21, 24): {"crime": 1.5, "accident": 1.1},   # night: elevated crime
}

# ── Routing Engine Parameters ─────────────────────────────────────────────────
RISK_MULTIPLIER   = 3.0    # effective_weight = length * (1 + risk * RISK_MULTIPLIER)
SNAP_RADIUS_M     = 1000   # metres: crime/accident points influence road edges within this radius
BALANCED_ALPHA    = 0.5    # balanced_weight = alpha*length + (1-alpha)*effective_weight

# ── User Report Recency Decay ─────────────────────────────────────────────────
REPORT_DECAY_RATE = 0.05   # exponential decay per day: score * exp(-rate * days_old)
REPORT_MAX_AGE_DAYS = 180  # ignore reports older than this

# ── User Safety Profiles ──────────────────────────────────────────────────────
SAFETY_PROFILES = {
    "default": {"crime": 0.40, "accident": 0.30, "flood": 0.15, "infra": 0.15},
    "women":   {"crime": 0.60, "accident": 0.20, "flood": 0.10, "infra": 0.10},
    "cyclist": {"crime": 0.30, "accident": 0.40, "flood": 0.10, "infra": 0.20},
    "night":   {"crime": 0.55, "accident": 0.25, "flood": 0.10, "infra": 0.10},
}

# ── File / Directory Paths ────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
RAW_CRIME_DIR     = os.path.join(BASE_DIR, "data", "raw", "crime")
RAW_ACCIDENT_DIR  = os.path.join(BASE_DIR, "data", "raw", "accidents")
RAW_FLOOD_DIR     = os.path.join(BASE_DIR, "data", "raw", "flood")
RAW_MAPS_DIR      = os.path.join(BASE_DIR, "data", "raw", "maps")
PROCESSED_DIR     = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR        = os.path.join(BASE_DIR, "models")
GRAPHS_DIR        = os.path.join(BASE_DIR, "graphs")

# ── Nominatim Geocoder Settings ───────────────────────────────────────────────
NOMINATIM_USER_AGENT  = "saferoute-india-mvp-v1"
NOMINATIM_RATE_LIMIT  = 1.1   # seconds between requests (Nominatim ToS: max 1/sec)

# ── API Server Settings ───────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Model Training Parameters ─────────────────────────────────────────────────
MODEL_FEATURES    = ["crime_score", "accident_score", "flood_score",
                     "road_score",  "length",         "composite_risk"]
MODEL_TARGET      = "risk_tier"
RISK_TIERS        = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
RISK_THRESHOLDS       = {"LOW": 0.088, "MEDIUM": 0.093}  # auto-calibrated from data percentiles
RISK_TIER_PERCENTILES = {"LOW": 50, "MEDIUM": 80}        # target: ~50% LOW, ~30% MEDIUM, ~20% HIGH

RF_PARAMS = {
    "n_estimators":     300,
    "max_depth":        15,
    "min_samples_split": 4,
    "min_samples_leaf":  2,
    "max_features":    "sqrt",
    "class_weight":    "balanced",
    "random_state":     42,
    "n_jobs":           -1,
}
