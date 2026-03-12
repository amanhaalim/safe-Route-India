#!/usr/bin/env python3
# =============================================================================
# src/08_score_graph.py — Apply trained ML model to produce final routing graph
# =============================================================================

import os
import sys
import numpy as np
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    TARGET_CITIES,
    GRAPHS_DIR,
    MODELS_DIR,
    MODEL_FEATURES,
    RISK_MULTIPLIER,
    BALANCED_ALPHA,
)

from src.utils import load_model, ensure_dirs


# =============================================================================
# Apply ML model
# =============================================================================

def apply_model_to_graph(city_key: str):

    import osmnx as ox

    risk_graph_path = os.path.join(GRAPHS_DIR, f"{city_key}_risk_graph.graphml")

    if not os.path.exists(risk_graph_path):
        raise FileNotFoundError(
            f"Risk graph not found: {risk_graph_path}\n"
            "Run: python src/06_snap_to_edges.py <city>"
        )

    logger.info(f"Loading model...")

    model = load_model(os.path.join(MODELS_DIR, "risk_classifier.pkl"))
    scaler = load_model(os.path.join(MODELS_DIR, "scaler.pkl"))

    logger.info(f"Loading risk graph: {city_key}")

    G = ox.load_graphml(risk_graph_path)

    edge_list = list(G.edges(data=True, keys=True))
    n_edges = len(edge_list)

    logger.info(f"Applying ML scores to {n_edges} edges")

    # -----------------------------------------------------------------
    # Build feature matrix
    # -----------------------------------------------------------------

    feature_matrix = np.array([
        [float(data.get(f, 0.0)) for f in MODEL_FEATURES]
        for u, v, key, data in edge_list
    ])

    X_scaled = scaler.transform(feature_matrix)

    risk_probs = model.predict_proba(X_scaled)
    risk_labels = model.predict(X_scaled)

    # Map probability columns safely.
    # model.classes_ holds integers (0, 1, 2) not strings — translate them.
    INT_TO_LABEL = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    class_order = [INT_TO_LABEL.get(int(c), str(c)) for c in model.classes_]

    def _idx(label: str) -> int:
        """Return column index for a class label, or -1 if not present."""
        return class_order.index(label) if label in class_order else -1

    idx_low  = _idx("LOW")
    idx_med  = _idx("MEDIUM")
    idx_high = _idx("HIGH")

    # -----------------------------------------------------------------
    # Apply predictions
    # -----------------------------------------------------------------

    for i, (u, v, key, data) in enumerate(edge_list):

        # Safely pull probabilities; default sensibly if class is absent
        p_low  = float(risk_probs[i, idx_low])  if idx_low  >= 0 else 1.0
        p_med  = float(risk_probs[i, idx_med])  if idx_med  >= 0 else 0.0
        p_high = float(risk_probs[i, idx_high]) if idx_high >= 0 else 0.0

        # Normalise so probabilities always sum to 1
        total = p_low + p_med + p_high or 1.0
        p_low, p_med, p_high = p_low / total, p_med / total, p_high / total

        raw_label = risk_labels[i]
        tier_label = INT_TO_LABEL.get(int(raw_label), "LOW")

        tier_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        tier = tier_map.get(tier_label, 0)

        data["risk_prob_low"] = round(p_low, 4)
        data["risk_prob_medium"] = round(p_med, 4)
        data["risk_prob_high"] = round(p_high, 4)

        data["predicted_risk_tier"] = tier
        data["predicted_risk_label"] = tier_label

        length = float(data.get("length", 50.0))

        # -------------------------------------------------------------
        # Routing weights
        # -------------------------------------------------------------

        data["effective_weight"] = length * (1.0 + p_high * RISK_MULTIPLIER)

        data["balanced_weight"] = (
            BALANCED_ALPHA * length +
            (1 - BALANCED_ALPHA) * data["effective_weight"]
        )

    # -----------------------------------------------------------------
    # Save graph
    # -----------------------------------------------------------------

    ensure_dirs(GRAPHS_DIR)

    out_path = os.path.join(GRAPHS_DIR, f"{city_key}_final_graph.graphml")

    ox.save_graphml(G, out_path)

    # -----------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------

    p_highs = [data.get("risk_prob_high", 0) for _, _, data in G.edges(data=True)]
    tiers = [data.get("predicted_risk_tier", 0) for _, _, data in G.edges(data=True)]

    high_ct = sum(1 for t in tiers if t == 2)

    logger.success(
        f"{city_key} final graph saved → {out_path}\n"
        f"Avg P(high): {np.mean(p_highs):.3f} | "
        f"High-risk edges: {high_ct}/{n_edges} "
        f"({round(100*high_ct/n_edges,2)}%)"
    )

    return G


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":

    import click

    @click.command()
    @click.argument("cities", nargs=-1)
    def main(cities):

        model_path = os.path.join(MODELS_DIR, "risk_classifier.pkl")

        if not os.path.exists(model_path):
            logger.error("Model not found. Run: python src/07_train_model.py")
            sys.exit(1)

        targets = list(cities) if cities else [
            k for k in TARGET_CITIES
            if os.path.exists(os.path.join(GRAPHS_DIR, f"{k}_risk_graph.graphml"))
        ]

        if not targets:
            logger.error(
                "No risk graphs found.\n"
                "Run: python src/06_snap_to_edges.py <city>"
            )
            sys.exit(1)

        for city in targets:

            try:

                apply_model_to_graph(city)

            except Exception as ex:

                logger.error(f"{city} failed: {ex}")

    main()