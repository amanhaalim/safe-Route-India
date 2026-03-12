# =============================================================================
# src/utils.py — Shared helper utilities for SafeRoute India
# =============================================================================

import os
import sys
import time
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Optional, Dict, Any


# ── Add project root to path ──────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Logging setup
# =============================================================================

def setup_logger(name: str = "saferoute", level: str = "INFO") -> None:
    """Configure loguru with console + file logging."""

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | "
        "<cyan>{name}</cyan> — {message}",
        level=level,
        colorize=True,
    )

    logger.add(
        logs_dir / f"{name}.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
    )


# =============================================================================
# Directory helpers
# =============================================================================

def ensure_dirs(*paths) -> None:
    """Create directories if they do not exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def find_file(directory: str, extensions: list) -> Optional[str]:
    """Return the first file found in directory with given extensions."""

    if not os.path.exists(directory):
        return None

    for f in os.listdir(directory):
        if any(f.lower().endswith(ext) for ext in extensions):
            return os.path.join(directory, f)

    return None


def list_files(directory: str, extension: str) -> list:
    """Return all files in directory with the given extension."""

    if not os.path.exists(directory):
        return []

    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.lower().endswith(extension)
    ]


# =============================================================================
# DataFrame helpers
# =============================================================================

def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names."""

    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )

    return df


def standardise_place_name(series: pd.Series) -> pd.Series:
    """Title-case and strip place names."""
    return series.astype(str).str.title().str.strip()


def safe_read_csv(path: str, **kwargs) -> pd.DataFrame:
    """Read CSV using multiple encodings."""

    encodings = ["utf-8", "latin-1", "cp1252", "utf-8-sig"]

    for enc in encodings:
        try:

            df = pd.read_csv(
                path,
                encoding=enc,
                on_bad_lines="skip",
                low_memory=False,
                **kwargs,
            )

            logger.debug(f"Read {path} with encoding={enc} — {len(df)} rows")

            return df

        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not read {path} with common encodings.")


def normalise_series(series: pd.Series, epsilon: float = 1e-9) -> pd.Series:
    """Min-max normalize numeric series to [0,1]."""

    mn = series.min()
    mx = series.max()

    if mx - mn < epsilon:
        return pd.Series(np.zeros(len(series)), index=series.index)

    return (series - mn) / (mx - mn)


# =============================================================================
# Spatial helpers
# =============================================================================

def india_bbox_filter(
    df: pd.DataFrame,
    lat_col: str = "LAT",
    lon_col: str = "LON",
) -> pd.DataFrame:
    """Filter rows inside India's geographic bounding box."""

    return df[
        (df[lat_col] >= 6.0)
        & (df[lat_col] <= 38.0)
        & (df[lon_col] >= 67.0)
        & (df[lon_col] <= 98.0)
    ].copy()


def latlon_to_gdf(
    df: pd.DataFrame,
    lat_col: str = "LAT",
    lon_col: str = "LON",
    crs: str = "EPSG:4326",
):
    """Convert DataFrame with LAT/LON to GeoDataFrame."""

    import geopandas as gpd

    df = df.dropna(subset=[lat_col, lon_col]).copy()

    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

    df = df.dropna(subset=[lat_col, lon_col])

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=crs,
    )

    return gdf


def get_india_metric_crs() -> str:
    """Metric CRS for India."""
    return "EPSG:32644"


# =============================================================================
# Model persistence
# =============================================================================

def save_model(obj: Any, path: str) -> None:

    ensure_dirs(os.path.dirname(path))

    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Saved → {path}")


def load_model(path: str) -> Any:

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# Risk helpers
# =============================================================================

def compute_risk_tier(
    score: float,
    low_threshold: float = 0.25,
    med_threshold: float = 0.55,
) -> int:
    """Convert normalized risk score → class label."""

    if score < low_threshold:
        return 0
    elif score < med_threshold:
        return 1
    else:
        return 2


def get_time_modifier(hour: int, time_modifiers: dict) -> Dict[str, float]:

    for (start, end), mods in time_modifiers.items():

        if start <= hour < end:
            return mods

    return {"crime": 1.0, "accident": 1.0}


def composite_time_multiplier(hour: int, time_modifiers: dict) -> float:

    mods = get_time_modifier(hour, time_modifiers)

    return 0.4 * mods["crime"] + 0.3 * mods["accident"] + 0.3


# =============================================================================
# Misc utilities
# =============================================================================

def progress_bar(iterable, desc: str = "", total: int = None):

    from tqdm import tqdm

    return tqdm(
        iterable,
        desc=desc,
        total=total,
        ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
    )


def file_hash(path: str) -> str:
    """Compute MD5 hash of file."""

    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()


def rate_limited_sleep(last_call_time: float, min_interval: float) -> float:
    """Sleep if API rate limit requires it."""

    elapsed = time.time() - last_call_time

    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

    return time.time()