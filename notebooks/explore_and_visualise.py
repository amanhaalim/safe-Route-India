#!/usr/bin/env python3
"""
notebooks/explore_and_visualise.py
══════════════════════════════════════════════════════════════════
SafeRoute India — Interactive Exploration Script
══════════════════════════════════════════════════════════════════
Run this after building a city graph to explore risk scores,
generate heatmaps, and test routes interactively.

Designed to be run cell-by-cell in Jupyter OR as a plain script.
If running in Jupyter, use: jupyter notebook (then open this file)
If running as a script:  python notebooks/explore_and_visualise.py

Outputs:
    outputs/risk_heatmap_{city}.html      — crime density map
    outputs/route_comparison_{city}.html  — safest vs fastest route
    outputs/risk_distribution.png         — histogram of edge risk scores
"""

import sys, os, logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
from folium.plugins import HeatMap, MiniMap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import TARGET_CITIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# CELL 1 — Configuration
# ─────────────────────────────────────────────────────────────────
CITY_KEY = "chennai"           # ← change this to your city
CITY_CFG = TARGET_CITIES[CITY_KEY]
CITY_CENTER = CITY_CFG["center"]

print(f"City: {CITY_KEY}  |  Center: {CITY_CENTER}")


# ─────────────────────────────────────────────────────────────────
# CELL 2 — Load graph and extract edge data
# ─────────────────────────────────────────────────────────────────
def load_graph_and_extract_edges(city_key: str) -> pd.DataFrame:
    import osmnx as ox

    graph_path = (
        f"graphs/{city_key}_final_graph.graphml"
        if Path(f"graphs/{city_key}_final_graph.graphml").exists()
        else f"graphs/{city_key}_risk_graph.graphml"
    )

    if not Path(graph_path).exists():
        raise FileNotFoundError(
            f"No graph for {city_key}. Run the pipeline first."
        )

    logger.info(f"Loading {graph_path}...")
    G = ox.load_graphml(graph_path)
    nodes, edges_gdf = ox.graph_to_gdfs(G)

    # Pull risk attributes into the GeoDataFrame
    edge_attrs = []
    for (u, v, key), row in edges_gdf.iterrows():
        data = G[u][v][key]
        edge_attrs.append({
            "u": u, "v": v,
            "crime_score":     float(data.get("crime_score",    0.3)),
            "accident_score":  float(data.get("accident_score", 0.3)),
            "flood_score":     float(data.get("flood_score",    0.2)),
            "composite_risk":  float(data.get("composite_risk", 0.3)),
            "risk_tier":       int(data.get("predicted_risk_tier", 1)),
            "length":          float(data.get("length", 50)),
            "highway":         str(data.get("highway", "unknown")),
        })

    attrs_df = pd.DataFrame(edge_attrs)
    logger.info(f"Extracted {len(attrs_df)} edge records")
    return G, edges_gdf, attrs_df


G, edges_gdf, attrs_df = load_graph_and_extract_edges(CITY_KEY)
print(f"\nEdge statistics:")
print(attrs_df[["crime_score","accident_score","flood_score","composite_risk"]].describe().round(3))


# ─────────────────────────────────────────────────────────────────
# CELL 3 — Risk distribution histogram
# ─────────────────────────────────────────────────────────────────
def plot_risk_distribution(attrs_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Risk Score Distributions — {CITY_KEY.title()}", fontsize=14, fontweight="bold")

    plot_config = [
        ("crime_score",    "#ef4444", "Crime Risk Score"),
        ("accident_score", "#f59e0b", "Accident Risk Score"),
        ("flood_score",    "#3b82f6", "Flood Risk Score"),
        ("composite_risk", "#6366f1", "Composite Risk Score"),
    ]

    for ax, (col, color, title) in zip(axes.flat, plot_config):
        vals = attrs_df[col].dropna()
        ax.hist(vals, bins=50, color=color, alpha=0.75, edgecolor="white", linewidth=0.3)
        ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=1.5,
                   label=f"Mean={vals.mean():.3f}")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Risk Score (0=safe, 1=dangerous)")
        ax.set_ylabel("Number of Road Segments")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"risk_distribution_{CITY_KEY}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {out_path}")
    plt.show()
    return out_path


plot_risk_distribution(attrs_df)


# ─────────────────────────────────────────────────────────────────
# CELL 4 — Risk tier breakdown by road type
# ─────────────────────────────────────────────────────────────────
def risk_by_highway_type(attrs_df: pd.DataFrame):
    # Normalise highway column (sometimes comes as list)
    attrs_df = attrs_df.copy()
    attrs_df["highway_clean"] = (
        attrs_df["highway"]
        .str.replace(r"[\[\]']", "", regex=True)
        .str.split(",")
        .apply(lambda x: x[0].strip() if isinstance(x, list) else str(x).strip())
    )

    summary = (
        attrs_df.groupby("highway_clean")
        .agg(
            count=("composite_risk", "count"),
            mean_risk=("composite_risk", "mean"),
            pct_high=("risk_tier", lambda x: 100 * (x == 2).sum() / len(x)),
        )
        .sort_values("mean_risk", ascending=False)
        .reset_index()
    )

    print("\nRisk by Road Type (top 12):")
    print(f"{'Road Type':<20} {'Count':>8} {'Mean Risk':>10} {'% High Risk':>12}")
    print("─" * 55)
    for _, row in summary.head(12).iterrows():
        bar = "█" * int(row["mean_risk"] * 20)
        print(f"{row['highway_clean']:<20} {int(row['count']):>8} "
              f"{row['mean_risk']:>10.3f} {row['pct_high']:>11.1f}%  {bar}")

    return summary


risk_by_highway_type(attrs_df)


# ─────────────────────────────────────────────────────────────────
# CELL 5 — Crime heatmap (Folium)
# ─────────────────────────────────────────────────────────────────
def generate_crime_heatmap(city_key: str) -> str:
    crime_path = "data/processed/crime_geocoded.csv"
    if not Path(crime_path).exists():
        logger.warning("crime_geocoded.csv not found — run src/05_geocode.py first")
        return None

    df = pd.read_csv(crime_path)
    score_col = next((c for c in df.columns if "NORM" in c.upper()), None)
    if not score_col:
        logger.warning("No normalised score column found in crime_geocoded.csv")
        return None

    df = df.dropna(subset=["LAT", "LON", score_col])

    # Filter to city bounding box
    bbox = TARGET_CITIES[city_key]["bbox"]  # (south, west, north, east)
    df = df[
        df["LAT"].between(bbox[0], bbox[2]) &
        df["LON"].between(bbox[1], bbox[3])
    ]

    if df.empty:
        logger.warning(f"No crime data found within {city_key} bounding box")
        return None

    logger.info(f"Plotting {len(df)} crime data points for {city_key}")

    m = folium.Map(
        location=list(CITY_CENTER),
        zoom_start=12,
        tiles="CartoDB dark_matter",
    )

    heat_data = df[["LAT", "LON", score_col]].values.tolist()
    HeatMap(
        heat_data,
        name="Crime Density",
        radius=18,
        blur=12,
        max_zoom=16,
        gradient={0.2: "blue", 0.4: "green", 0.6: "yellow", 0.8: "orange", 1.0: "red"},
    ).add_to(m)

    MiniMap().add_to(m)

    # Legend
    legend_html = """
    <div style='position:fixed;bottom:30px;right:10px;z-index:1000;
                background:rgba(0,0,0,0.8);color:white;padding:12px;
                border-radius:10px;font-size:12px;font-family:Arial;'>
      <b>Crime Density</b><br>
      <span style='color:#ef4444'>■</span> High<br>
      <span style='color:#f59e0b'>■</span> Medium<br>
      <span style='color:#22c55e'>■</span> Low
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    out_path = str(OUTPUT_DIR / f"crime_heatmap_{city_key}.html")
    m.save(out_path)
    logger.info(f"Crime heatmap saved → {out_path}")
    return out_path


generate_crime_heatmap(CITY_KEY)


# ─────────────────────────────────────────────────────────────────
# CELL 6 — Route comparison map
# ─────────────────────────────────────────────────────────────────
def generate_route_comparison_map(
    origin:      str,
    destination: str,
    city_key:    str,
    travel_hour: int = 14,
) -> str:
    from src.routing import find_safe_routes

    logger.info(f"Computing routes: {origin} → {destination}")
    try:
        routes, orig_coords, dest_coords = find_safe_routes(
            origin_address=origin,
            destination_address=destination,
            city_key=city_key,
            travel_hour=travel_hour,
            include_segments=True,
        )
    except Exception as e:
        logger.error(f"Routing failed: {e}")
        return None

    m = folium.Map(location=list(orig_coords), zoom_start=13,
                   tiles="CartoDB positron")

    # Draw gradient-colored routes using per-segment risk colors
    ROUTE_STYLE = {
        "safest":   {"weight": 7,  "opacity": 0.90, "default_color": "#22c55e"},
        "balanced": {"weight": 5,  "opacity": 0.75, "default_color": "#f59e0b"},
        "fastest":  {"weight": 4,  "opacity": 0.60, "default_color": "#ef4444"},
    }

    for name, route in routes.items():
        if not route:
            continue
        style = ROUTE_STYLE[name]
        fg = folium.FeatureGroup(name=f"{name.title()} Route")

        segments = route.get("segments", [])
        if segments:
            # Draw per-segment risk heatmap
            for seg in segments:
                folium.PolyLine(
                    seg["coords"],
                    color=seg["color"],
                    weight=style["weight"],
                    opacity=style["opacity"],
                    tooltip=(f"{name.title()}: risk={seg['risk']:.3f} | "
                             f"crime={seg['crime_score']:.3f} | "
                             f"accident={seg['accident_score']:.3f}"),
                ).add_to(fg)
        else:
            # Fallback: solid color
            folium.PolyLine(
                route["coordinates"],
                color=style["default_color"],
                weight=style["weight"],
                opacity=style["opacity"],
            ).add_to(fg)

        fg.add_to(m)

        # Add summary popup at midpoint
        s   = route["summary"]
        mid = route["coordinates"][len(route["coordinates"]) // 2]
        folium.Marker(
            mid,
            icon=folium.DivIcon(html=(
                f"<div style='background:rgba(0,0,0,0.8);color:white;"
                f"padding:6px 10px;border-radius:6px;font-size:11px;"
                f"font-family:Arial;white-space:nowrap;'>"
                f"<b>{name.upper()}</b><br>"
                f"{s['distance_km']} km · safety {s['safety_pct']}%<br>"
                f"risk {s['avg_risk_score']:.3f} · [{s['risk_label']}]"
                f"</div>"
            )),
        ).add_to(m)

    # Origin / Destination markers
    def icon_html(emoji, label, color):
        return folium.DivIcon(html=(
            f"<div style='background:{color};color:white;padding:4px 8px;"
            f"border-radius:20px;font-size:12px;font-weight:bold;'>{emoji} {label}</div>"
        ))

    folium.Marker(list(orig_coords),  icon=icon_html("🟢", "Start", "#1d4ed8")).add_to(m)
    folium.Marker(list(dest_coords),  icon=icon_html("🔴", "End",   "#dc2626")).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    out_path = str(OUTPUT_DIR / f"route_comparison_{city_key}.html")
    m.save(out_path)
    logger.info(f"Route comparison map saved → {out_path}")

    # Print summary table
    print(f"\n{'Route':<12} {'Distance':>10} {'Safety%':>9} {'Risk':>7} {'Label'}")
    print("─" * 58)
    for name, route in routes.items():
        if route:
            s = route["summary"]
            print(f"{name.upper():<12} {s['distance_km']:>8.1f} km "
                  f"{s['safety_pct']:>8.1f}% "
                  f"{s['avg_risk_score']:>7.3f}  {s['risk_label']}")

    return out_path


generate_route_comparison_map(
    origin="Chennai Central Railway Station",
    destination="T. Nagar, Chennai",
    city_key=CITY_KEY,
    travel_hour=21,   # 9 PM — elevated risk
)


# ─────────────────────────────────────────────────────────────────
# CELL 7 — Time-of-day risk sensitivity analysis
# ─────────────────────────────────────────────────────────────────
def analyse_time_sensitivity(city_key: str):
    import copy
    from src.routing import load_graph, apply_time_modifiers_to_graph

    G = load_graph(city_key)
    hours = list(range(0, 24, 2))
    mean_weights = []

    for h in hours:
        G_copy = copy.deepcopy(G)
        apply_time_modifiers_to_graph(G_copy, h)
        w = [float(d.get("time_adjusted_weight", 0)) for _, _, d in G_copy.edges(data=True)]
        mean_weights.append(np.mean(w))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hours, mean_weights, "o-", color="#6366f1", linewidth=2, markersize=6)
    ax.fill_between(hours, mean_weights, min(mean_weights), alpha=0.15, color="#6366f1")
    ax.set_xlabel("Hour of Day (24h)", fontsize=11)
    ax.set_ylabel("Mean Effective Edge Weight", fontsize=11)
    ax.set_title(f"Time-of-Day Risk Sensitivity — {city_key.title()}", fontsize=12, fontweight="bold")
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Shade night hours
    ax.axvspan(0, 6,   alpha=0.08, color="red",   label="High crime (0–6AM)")
    ax.axvspan(18, 21, alpha=0.06, color="orange", label="Evening rush (6–9PM)")
    ax.legend(fontsize=9, loc="upper center")

    out_path = OUTPUT_DIR / f"time_sensitivity_{city_key}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {out_path}")
    plt.show()


analyse_time_sensitivity(CITY_KEY)

print("\n✅ All visualisations complete.")
print(f"   Output files saved to: {OUTPUT_DIR}/")
print("   Open the HTML files in your browser to view interactive maps.")
