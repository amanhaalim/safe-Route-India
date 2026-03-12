#!/usr/bin/env python3
# =============================================================================
# tests/test_routing.py — Unit tests for routing logic (no live data needed)
# =============================================================================

import sys
import numpy as np
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.routing import compute_route_summary, get_route_coordinates, apply_time_weights
from config import TIME_MODIFIERS


# ── Build a tiny mock graph for testing ──────────────────────────────────────
def make_mock_graph():
    """Build a simple 4-node directed graph with risk attributes."""
    import networkx as nx

    G = nx.MultiDiGraph()
    # nodes: id → (y=lat, x=lon)
    nodes = {
        1: {"y": 13.0, "x": 80.0},
        2: {"y": 13.1, "x": 80.1},
        3: {"y": 13.2, "x": 80.2},
        4: {"y": 13.3, "x": 80.3},
    }
    for nid, attrs in nodes.items():
        G.add_node(nid, **attrs)

    # edges: (u, v) → attributes
    edges = [
        (1, 2, {"length": 1000, "crime_score": 0.8, "accident_score": 0.7,
                 "flood_score": 0.2, "road_score": 0.5, "composite_risk": 0.68,
                 "risk_prob_high": 0.8, "predicted_risk_tier": 2,
                 "effective_weight": 3400, "balanced_weight": 2200}),
        (2, 3, {"length": 1200, "crime_score": 0.1, "accident_score": 0.1,
                 "flood_score": 0.05, "road_score": 0.3, "composite_risk": 0.12,
                 "risk_prob_high": 0.05, "predicted_risk_tier": 0,
                 "effective_weight": 1380, "balanced_weight": 1290}),
        (3, 4, {"length": 800, "crime_score": 0.3, "accident_score": 0.2,
                 "flood_score": 0.1, "road_score": 0.4, "composite_risk": 0.27,
                 "risk_prob_high": 0.2, "predicted_risk_tier": 1,
                 "effective_weight": 1280, "balanced_weight": 1040}),
        # High-risk direct shortcut 1→4
        (1, 4, {"length": 500, "crime_score": 0.9, "accident_score": 0.85,
                 "flood_score": 0.5, "road_score": 0.8, "composite_risk": 0.87,
                 "risk_prob_high": 0.9, "predicted_risk_tier": 2,
                 "effective_weight": 1850, "balanced_weight": 1175}),
    ]
    for u, v, attrs in edges:
        G.add_edge(u, v, **attrs)

    return G


# ── compute_route_summary ─────────────────────────────────────────────────────
def test_route_summary_values():
    G     = make_mock_graph()
    route = [1, 2, 3, 4]          # 3 edges: 1→2, 2→3, 3→4
    s     = compute_route_summary(G, route)

    assert s["distance_km"] == pytest.approx(3.0, abs=0.01)   # 1000+1200+800 = 3000m
    assert 0 <= s["avg_risk_score"] <= 1
    assert 0 <= s["safety_pct"]    <= 100
    assert s["total_segments"] == 3
    assert s["high_risk_segments"] == 1   # only edge 1→2 has tier 2


def test_route_summary_risk_label_low():
    G     = make_mock_graph()
    route = [2, 3, 4]             # low risk path
    s     = compute_route_summary(G, route)
    assert s["risk_label"] in ("LOW", "MEDIUM")


def test_route_summary_empty():
    G = make_mock_graph()
    assert compute_route_summary(G, []) == {}
    assert compute_route_summary(G, [1]) == {}


# ── get_route_coordinates ─────────────────────────────────────────────────────
def test_route_coordinates_shape():
    G     = make_mock_graph()
    route = [1, 2, 3, 4]
    coords = get_route_coordinates(G, route)
    assert len(coords) == 4
    for c in coords:
        assert len(c) == 2         # [lat, lon]
        assert isinstance(c[0], float)
        assert isinstance(c[1], float)


# ── apply_time_weights ────────────────────────────────────────────────────────
def test_time_weights_night_higher_than_day():
    import copy
    G_night = make_mock_graph()
    G_day   = make_mock_graph()
    apply_time_weights(G_night, hour=2,  profile="default")
    apply_time_weights(G_day,   hour=11, profile="default")

    night_w = [d.get("time_adjusted_weight", 0) for _, _, d in G_night.edges(data=True)]
    day_w   = [d.get("time_adjusted_weight", 0) for _, _, d in G_day.edges(data=True)]

    # Average night weight should be higher than average day weight
    assert np.mean(night_w) > np.mean(day_w)


def test_time_weights_low_risk_edge_unchanged_relative():
    """Low-risk edges should have smaller absolute adjustment than high-risk edges."""
    G = make_mock_graph()
    apply_time_weights(G, hour=2, profile="default")

    weights = {}
    for u, v, key, d in G.edges(data=True, keys=True):
        weights[(u,v,key)] = d.get("time_adjusted_weight", 0)

    # Edge 1→2 (high risk) should have much higher weight than edge 2→3 (low risk)
    w_high = weights.get((1, 2, 0), 0)
    w_low  = weights.get((2, 3, 0), 0)
    assert w_high > w_low * 1.5
