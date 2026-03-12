"""
src/09_routing.py
═══════════════════════════════════════════════════════════════════
SafeRoute India — Core Routing Engine
═══════════════════════════════════════════════════════════════════
This module is the heart of the system. It:
  1. Loads pre-built risk-weighted road graphs
  2. Geocodes text addresses → GPS coordinates via Nominatim (no key)
  3. Applies time-of-day risk multipliers at query time
  4. Runs three Dijkstra variants: Safest / Balanced / Fastest
  5. Returns structured route objects with full risk breakdowns

Used by:
  - api/main.py       (REST API endpoint)
  - quick_test.py     (CLI testing)
  - notebooks/        (exploration)
"""

import os, sys, time, math, logging
from typing import Optional, Tuple, Dict, List

import numpy as np
import networkx as nx
import osmnx as ox
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TARGET_CITIES, TIME_MODIFIERS, RISK_MULTIPLIER

logger = logging.getLogger("saferoute.routing")

# ── In-memory graph cache (avoid reloading on every request) ──────
_GRAPH_CACHE: Dict[str, nx.MultiDiGraph] = {}

_GEOLOCATOR = Nominatim(
    user_agent="saferoute-india-mvp-v1.0"
)

# ═══════════════════════════════════════════════════════════════
# GRAPH LOADING
# ═══════════════════════════════════════════════════════════════

def load_graph(city_key: str) -> nx.MultiDiGraph:
    """
    Load the final risk-weighted graph for a city from disk.
    Caches in memory after first load.
    """
    if city_key not in TARGET_CITIES:
        raise KeyError(f"Unknown city '{city_key}'. Available: {list(TARGET_CITIES.keys())}")

    if city_key in _GRAPH_CACHE:
        return _GRAPH_CACHE[city_key]

    candidates = [
        f"graphs/{city_key}_final_graph.graphml",
        f"graphs/{city_key}_risk_graph.graphml",
    ]
    graph_path = next((p for p in candidates if os.path.exists(p)), None)

    if graph_path is None:
        raise FileNotFoundError(
            f"No graph found for '{city_key}'.\n"
            f"Build it first:\n"
            f"  python src/06_snap_to_edges.py {city_key}\n"
            f"  python src/07_train_model.py\n"
            f"  python src/08_score_graph.py {city_key}"
        )

    logger.info(f"Loading graph from {graph_path} ...")
    G = ox.load_graphml(graph_path)
    _fill_missing_edge_attributes(G)
    _GRAPH_CACHE[city_key] = G
    logger.info(f"Graph ready: {city_key} | {G.number_of_nodes():,} nodes | {G.number_of_edges():,} edges")
    return G


def _fill_missing_edge_attributes(G: nx.MultiDiGraph) -> None:
    """Ensure every edge has required risk attributes; fill with safe defaults."""
    D = 0.35  # default risk for unscored edges
    for u, v, key, data in G.edges(data=True, keys=True):
        data.setdefault("crime_score",         D)
        data.setdefault("accident_score",      D)
        data.setdefault("flood_score",         0.20)
        data.setdefault("road_score",          0.50)
        data.setdefault("composite_risk",      D)
        data.setdefault("risk_prob_high",      D)
        data.setdefault("predicted_risk_tier", 1)
        L = float(data.get("length", 50))
        R = float(data.get("composite_risk", D))
        data.setdefault("effective_weight", L * (1 + R * RISK_MULTIPLIER))
        data.setdefault("balanced_weight",  0.5 * L + 0.5 * data["effective_weight"])


def clear_graph_cache(city_key: Optional[str] = None) -> None:
    if city_key:
        _GRAPH_CACHE.pop(city_key, None)
    else:
        _GRAPH_CACHE.clear()


# ═══════════════════════════════════════════════════════════════
# GEOCODING
# ═══════════════════════════════════════════════════════════════

def geocode_address(address: str, city: str, max_retries: int = 3) -> Tuple[float, float]:
    """
    Convert a text address to (lat, lon) using Nominatim (free, no API key).
    Tries progressively broader queries if specific ones fail.
    """
    queries = [
        f"{address}, {city}, India",
        f"{address}, India",
        f"{city}, India",
    ]
    last_error = None
    for attempt, query in enumerate(queries):
        for retry in range(max_retries):
            try:
                time.sleep(1.1)  # Nominatim rate limit: 1 req/sec
                loc = _GEOLOCATOR.geocode(query, country_codes="in", timeout=10)
                if loc:
                    logger.info(f"Geocoded '{address}' → ({loc.latitude:.4f}, {loc.longitude:.4f})")
                    return (loc.latitude, loc.longitude)
            except GeocoderTimedOut:
                last_error = "timeout"
                time.sleep(2)
            except GeocoderServiceError as e:
                last_error = str(e)
                time.sleep(3)
    raise ValueError(
        f"Could not geocode '{address}' in {city}. "
        f"Try a more specific address. Last error: {last_error}"
    )


def reverse_geocode(lat: float, lon: float) -> str:
    try:
        time.sleep(1.1)
        loc = _GEOLOCATOR.reverse((lat, lon), language="en", timeout=10)
        return loc.address if loc else f"{lat:.4f}, {lon:.4f}"
    except Exception:
        return f"{lat:.4f}, {lon:.4f}"


# ═══════════════════════════════════════════════════════════════
# TIME-OF-DAY MODIFIERS
# ═══════════════════════════════════════════════════════════════

def get_time_weight_modifier(hour: int) -> float:
    """
    Return a composite multiplier for the given hour.
    Amplifies crime + accident risk based on time of day.
    Night (0-6AM) = up to 2.0×; Daytime = ~0.75×
    """
    hour = hour % 24
    for (start, end), mods in TIME_MODIFIERS.items():
        if start <= hour < end:
            return round(0.40 * mods["crime"] + 0.30 * mods["accident"] + 0.30, 3)
    return 1.0


def apply_time_modifiers_to_graph(G: nx.MultiDiGraph, hour: int) -> None:
    """
    Inject time-aware weights onto every edge (in-place, at query time).
    Recalculated fresh for every route request — no graph rebuild needed.
    """
    hour = hour % 24
    mods = {}
    for (start, end), m in TIME_MODIFIERS.items():
        if start <= hour < end:
            mods = m
            break
    crime_mult    = mods.get("crime",    1.0)
    accident_mult = mods.get("accident", 1.0)

    for u, v, key, data in G.edges(data=True, keys=True):
        L = float(data.get("length", 50))
        time_risk = min(
            0.40 * float(data.get("crime_score",    0.3)) * crime_mult +
            0.30 * float(data.get("accident_score", 0.3)) * accident_mult +
            0.15 * float(data.get("flood_score",    0.2)) +
            0.15 * float(data.get("road_score",     0.5)),
            1.0
        )
        data["time_adjusted_weight"] = L * (1 + time_risk * RISK_MULTIPLIER)


# ═══════════════════════════════════════════════════════════════
# ROUTE COMPUTATION
# ═══════════════════════════════════════════════════════════════

def _run_dijkstra(G, orig_node, dest_node, weight_key) -> Optional[List[int]]:
    """Dijkstra with undirected fallback if no directed path exists."""
    try:
        return nx.shortest_path(G, orig_node, dest_node, weight=weight_key)
    except nx.NetworkXNoPath:
        try:
            return nx.shortest_path(G.to_undirected(), orig_node, dest_node, weight=weight_key)
        except Exception:
            return None
    except Exception as e:
        logger.error(f"Routing error ({weight_key}): {e}")
        return None


def compute_route_summary(G: nx.MultiDiGraph, route_nodes: List[int]) -> Dict:
    """Build a complete risk breakdown dict for a route."""
    if not route_nodes or len(route_nodes) < 2:
        return {}

    total_length = 0.0
    crime, accident, flood, road, composite = [], [], [], [], []
    high_risk = med_risk = total_edges = 0

    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        if not G.has_edge(u, v):
            continue
        e = min(G[u][v].values(), key=lambda d: d.get("effective_weight", float("inf")))
        total_length += float(e.get("length", 0))
        crime.append(   float(e.get("crime_score",    0.3)))
        accident.append(float(e.get("accident_score", 0.3)))
        flood.append(   float(e.get("flood_score",    0.2)))
        road.append(    float(e.get("road_score",     0.5)))
        composite.append(float(e.get("composite_risk", 0.3)))
        tier = int(e.get("predicted_risk_tier", 1))
        if tier == 2: high_risk += 1
        elif tier == 1: med_risk += 1
        total_edges += 1

    def sm(lst): return round(float(np.mean(lst)), 4) if lst else 0.0
    def sx(lst): return round(float(np.max(lst)),  4) if lst else 0.0

    avg_risk = sm(composite)
    return {
        "distance_km":          round(total_length / 1000, 2),
        "estimated_minutes":    round((total_length / 1000) / 30 * 60, 1),
        "avg_risk_score":       avg_risk,
        "max_risk_score":       sx(composite),
        "safety_pct":           round(100 * (1 - avg_risk), 1),
        "avg_crime_score":      sm(crime),
        "max_crime_score":      sx(crime),
        "avg_accident_score":   sm(accident),
        "max_accident_score":   sx(accident),
        "avg_flood_score":      sm(flood),
        "avg_road_score":       sm(road),
        "total_segments":       total_edges,
        "high_risk_segments":   high_risk,
        "medium_risk_segments": med_risk,
        "low_risk_segments":    total_edges - high_risk - med_risk,
        "pct_high_risk":        round(100 * high_risk / max(total_edges, 1), 1),
        "risk_label":           _risk_label(avg_risk),
        "risk_color":           _risk_color(avg_risk),
    }


def _risk_label(s): 
    return (
        "Very Safe" if s < 0.20 else
        "Safe" if s < 0.35 else
        "Moderate Risk" if s < 0.50 else
        "High Risk" if s < 0.65 else
        "Very High Risk"
    )

def _risk_color(s):
    return (
        "#16a34a" if s < 0.20 else
        "#22c55e" if s < 0.35 else
        "#f59e0b" if s < 0.50 else
        "#ef4444" if s < 0.65 else
        "#991b1b"
    )


def get_route_coordinates(G, route_nodes) -> List[List[float]]:
    """Extract [[lat, lon], ...] with edge geometry interpolation for smooth polylines."""
    coords = []
    for i, node in enumerate(route_nodes):
        nd = G.nodes[node]
        coords.append([float(nd.get("y", 0)), float(nd.get("x", 0))])
        if i < len(route_nodes) - 1:
            nxt = route_nodes[i + 1]
            if G.has_edge(node, nxt):
                e = min(G[node][nxt].values(), key=lambda d: d.get("effective_weight", float("inf")))
                geom = e.get("geometry")
                if geom is not None:
                    try:
                        for lon_g, lat_g in list(geom.coords)[1:-1]:
                            coords.append([float(lat_g), float(lon_g)])
                    except Exception:
                        pass
    return coords


def get_segment_risk_colors(G, route_nodes) -> List[Dict]:
    """Return per-segment risk colors for gradient heatmap rendering."""
    segments = []
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        if not G.has_edge(u, v):
            continue
        e  = min(G[u][v].values(), key=lambda d: d.get("effective_weight", float("inf")))
        r  = float(e.get("composite_risk", 0.3))
        ud, vd = G.nodes[u], G.nodes[v]
        segments.append({
            "coords": [
                [float(ud.get("y", 0)), float(ud.get("x", 0))],
                [float(vd.get("y", 0)), float(vd.get("x", 0))],
            ],
            "color":          _risk_color(r),
            "risk":           round(r, 3),
            "crime_score":    round(float(e.get("crime_score",    0)), 3),
            "accident_score": round(float(e.get("accident_score", 0)), 3),
            "flood_score":    round(float(e.get("flood_score",    0)), 3),
            "highway":        str(e.get("highway", "unknown")),
        })
    return segments


def list_available_cities() -> List[Dict]:
    cities = []
    for key, cfg in TARGET_CITIES.items():
        final   = os.path.exists(f"graphs/{key}_final_graph.graphml")
        partial = os.path.exists(f"graphs/{key}_risk_graph.graphml")
        cities.append({
            "key":    key,
            "name":   cfg["place"].split(",")[0].strip(),
            "center": list(cfg["center"]),
            "ready":  final,
            "status": "ready" if final else ("partial" if partial else "not_built"),
        })
    return cities


def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


# ═══════════════════════════════════════════════════════════════
# PRIMARY PUBLIC FUNCTION
# ═══════════════════════════════════════════════════════════════

def find_safe_routes(
    origin_address: str,
    destination_address: str,
    city_key: str,
    travel_hour: int = 12,
    include_segments: bool = False,
) -> Tuple[Dict, Tuple[float, float], Tuple[float, float]]:
    """
    Given two text addresses in an Indian city, return three route options
    with full risk breakdowns (Safest / Balanced / Fastest).

    Args:
        origin_address:      e.g. "Chennai Central Station"
        destination_address: e.g. "T. Nagar Bus Terminus"
        city_key:            e.g. "chennai"
        travel_hour:         0–23 (departure hour; affects risk weights)
        include_segments:    if True, add per-segment risk colors to response

    Returns:
        (routes_dict, origin_coords, dest_coords)
        routes_dict keys: "safest", "balanced", "fastest"
        Each route: { coordinates, summary, [segments] }
    """
    city_name = TARGET_CITIES[city_key]["place"].split(",")[0].strip()

    G = load_graph(city_key)

    orig_coords = geocode_address(origin_address,      city_name)
    dest_coords = geocode_address(destination_address, city_name)

    if haversine_distance(*orig_coords, *dest_coords) < 0.05:
        raise ValueError("Origin and destination are too close (< 50 m).")

    orig_node = ox.nearest_nodes(G, orig_coords[1], orig_coords[0])
    dest_node = ox.nearest_nodes(G, dest_coords[1], dest_coords[0])

    logger.info(f"Route request: {orig_node} → {dest_node} | {city_key} | hour={travel_hour}")

    apply_time_modifiers_to_graph(G, travel_hour)

    routing_configs = [
        ("safest",   "time_adjusted_weight"),
        ("balanced", "balanced_weight"),
        ("fastest",  "length"),
    ]

    results = {}
    for name, weight_key in routing_configs:
        nodes = _run_dijkstra(G, orig_node, dest_node, weight_key)
        if nodes is None:
            results[name] = None
            continue
        route_obj = {
            "coordinates": get_route_coordinates(G, nodes),
            "summary":     compute_route_summary(G, nodes),
            "node_count":  len(nodes),
        }
        if include_segments:
            route_obj["segments"] = get_segment_risk_colors(G, nodes)
        results[name] = route_obj
        s = route_obj["summary"]
        logger.info(f"  {name}: {s['distance_km']} km | safety={s['safety_pct']}% | {s['risk_label']}")

    if all(v is None for v in results.values()):
        raise ValueError(f"No route found between '{origin_address}' and '{destination_address}' in {city_name}.")

    return results, orig_coords, dest_coords


# ── CLI quick-test ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%H:%M:%S")

    ap = argparse.ArgumentParser(description="SafeRoute India — CLI route test")
    ap.add_argument("--city",        default="chennai")
    ap.add_argument("--origin",      default="Chennai Central Railway Station")
    ap.add_argument("--destination", default="T. Nagar Bus Terminus")
    ap.add_argument("--hour",        type=int, default=12)
    args = ap.parse_args()

    print(f"\n{'═'*58}")
    print(f"  SafeRoute India | {args.city} | {args.hour:02d}:00")
    print(f"  From: {args.origin}")
    print(f"  To:   {args.destination}")
    print(f"{'═'*58}\n")

    try:
        routes, orig, dest = find_safe_routes(
            args.origin, args.destination, args.city, args.hour
        )
        print(f"Origin:      ({orig[0]:.4f}, {orig[1]:.4f})")
        print(f"Destination: ({dest[0]:.4f}, {dest[1]:.4f})\n")

        icons = {"safest": "🟢", "balanced": "🟡", "fastest": "🔴"}
        for name, route in routes.items():
            if not route:
                print(f"{icons[name]} {name.upper():<10} ✗ No path")
                continue
            s = route["summary"]
            print(f"{icons[name]} {name.upper():<10} "
                  f"{s['distance_km']:5.1f} km  "
                  f"safety={s['safety_pct']:5.1f}%  "
                  f"risk={s['avg_risk_score']:.3f}  "
                  f"[{s['risk_label']}]  "
                  f"~{s['estimated_minutes']} min")

        if routes.get("safest") and routes.get("fastest"):
            d = routes["safest"]["summary"]["distance_km"] - routes["fastest"]["summary"]["distance_km"]
            r = routes["fastest"]["summary"]["avg_risk_score"] - routes["safest"]["summary"]["avg_risk_score"]
            print(f"\n  ✅ Safest adds {d:+.1f} km but cuts risk by {r:.3f}")
        print(f"\n{'═'*58}\n")

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\n❌ {e}\n")
        sys.exit(1)


# ── Compatibility aliases (used by tests and api/main.py) ──────────────────
def apply_time_weights(G, hour: int, profile: str = "default") -> None:
    """Alias for apply_time_modifiers_to_graph. Supports optional profile arg."""
    apply_time_modifiers_to_graph(G, hour)

def get_graph(city_key: str):
    """Alias for load_graph used by api/main.py."""
    return load_graph(city_key)

def clear_cache(city_key=None):
    """Alias for clear_graph_cache used by api/main.py."""
    clear_graph_cache(city_key)
