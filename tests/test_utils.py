#!/usr/bin/env python3
# =============================================================================
# tests/test_utils.py — Unit tests for utility functions
# Run: python -m pytest tests/ -v
# =============================================================================

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    standardise_columns, standardise_place_name,
    normalise_series, india_bbox_filter,
    compute_risk_tier, get_time_modifier,
    composite_time_multiplier,
)
from config import TIME_MODIFIERS


# ── standardise_columns ───────────────────────────────────────────────────────
def test_standardise_columns_strips_spaces():
    df = pd.DataFrame({" State Name ": [1], "Crime Count": [2]})
    df = standardise_columns(df)
    assert "STATE_NAME" in df.columns
    assert "CRIME_COUNT" in df.columns


def test_standardise_columns_removes_parens():
    df = pd.DataFrame({"Total (IPC)": [1]})
    df = standardise_columns(df)
    assert "TOTAL_IPC" in df.columns


# ── normalise_series ──────────────────────────────────────────────────────────
def test_normalise_series_bounds():
    s      = pd.Series([0, 5, 10, 15, 20])
    normed = normalise_series(s)
    assert abs(normed.min()) < 1e-6
    assert abs(normed.max() - 1.0) < 1e-6


def test_normalise_series_constant():
    s      = pd.Series([7, 7, 7])
    normed = normalise_series(s)
    assert (normed == 0).all()


# ── india_bbox_filter ─────────────────────────────────────────────────────────
def test_bbox_filter_keeps_india():
    df = pd.DataFrame({
        "LAT": [13.08, 28.61, 0.0,  90.0],
        "LON": [80.27, 77.21, 50.0, 50.0],
    })
    filtered = india_bbox_filter(df)
    assert len(filtered) == 2   # only Chennai and Delhi


# ── compute_risk_tier ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("score,expected", [
    (0.00, 0),  # LOW
    (0.24, 0),
    (0.25, 1),  # MEDIUM
    (0.54, 1),
    (0.55, 2),  # HIGH
    (1.00, 2),
])
def test_risk_tiers(score, expected):
    assert compute_risk_tier(score) == expected


# ── time modifiers ────────────────────────────────────────────────────────────
def test_time_modifier_night():
    mods = get_time_modifier(1, TIME_MODIFIERS)
    assert mods["crime"] >= 1.5    # midnight should have high crime modifier


def test_time_modifier_daytime():
    mods = get_time_modifier(11, TIME_MODIFIERS)
    assert mods["crime"] < 1.0     # daytime should reduce crime weight


def test_composite_multiplier_range():
    for hour in range(24):
        mult = composite_time_multiplier(hour, TIME_MODIFIERS)
        assert 0.5 <= mult <= 2.5, f"Multiplier out of range at hour {hour}: {mult}"


# ── standardise_place_name ────────────────────────────────────────────────────
def test_place_name_title_case():
    s = pd.Series(["  CHENNAI  ", "new delhi", "MUMBAI"])
    result = standardise_place_name(s)
    assert result.tolist() == ["Chennai", "New Delhi", "Mumbai"]
