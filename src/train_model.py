#!/usr/bin/env python3
# =============================================================================
# src/07_train_model.py — Train the risk classification model
# =============================================================================

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    TARGET_CITIES, GRAPHS_DIR, MODELS_DIR,
    MODEL_FEATURES, MODEL_TARGET,
    RISK_THRESHOLDS, RISK_TIER_PERCENTILES, RF_PARAMS,
)
from src.utils import ensure_dirs, save_model, compute_risk_tier


# =============================================================================
# 1. Extract training data from scored graphs
# =============================================================================

def extract_features(city_key: str) -> pd.DataFrame:

    import osmnx as ox

    path = os.path.join(GRAPHS_DIR, f"{city_key}_risk_graph.graphml")

    if not os.path.exists(path):
        logger.warning(f"Graph not found: {path}")
        return pd.DataFrame()

    G = ox.load_graphml(path)

    rows = []

    for u, v, key, data in G.edges(data=True, keys=True):

        row = {f: float(data.get(f, 0.3)) for f in MODEL_FEATURES}

        row["city"] = city_key

        rows.append(row)

    df = pd.DataFrame(rows)

    logger.info(f"{city_key}: {len(df)} edges")

    return df


def build_training_dataframe(cities: list) -> pd.DataFrame:

    frames = []

    for city in cities:

        df = extract_features(city)

        if not df.empty:
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No risk graphs found. Run src/06_snap_to_edges.py first."
        )

    combined = pd.concat(frames, ignore_index=True)

    # ── Auto-calibrate thresholds from data percentiles ──────────────────
    scores = combined["composite_risk"].dropna().values
    low_pct = RISK_TIER_PERCENTILES.get("LOW",  50)
    med_pct = RISK_TIER_PERCENTILES.get("MEDIUM", 80)
    low_threshold = float(np.percentile(scores, low_pct))
    med_threshold = float(np.percentile(scores, med_pct))

    # Safety guard: ensure thresholds are distinct
    if med_threshold <= low_threshold:
        med_threshold = low_threshold + 1e-6

    logger.info(
        f"Auto-calibrated thresholds → "
        f"LOW < {low_threshold:.5f} (p{low_pct}), "
        f"MEDIUM < {med_threshold:.5f} (p{med_pct})"
    )

    combined[MODEL_TARGET] = combined["composite_risk"].apply(
        lambda s: compute_risk_tier(
            s,
            low_threshold=low_threshold,
            med_threshold=med_threshold,
        )
    )

    combined = combined.dropna(subset=MODEL_FEATURES + [MODEL_TARGET])

    logger.info(f"Total training samples: {len(combined)}")

    class_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

    logger.info(
        "Class distribution:\n"
        + combined[MODEL_TARGET].map(class_map).value_counts().to_string()
    )

    return combined, low_threshold, med_threshold


# =============================================================================
# 2. Train
# =============================================================================

def train_model(df: pd.DataFrame):

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.utils.class_weight import compute_class_weight

    X = df[MODEL_FEATURES].values
    y = df[MODEL_TARGET].values.astype(int)

    unique_classes = np.unique(y)

    # fallback if only one class exists
    stratify_param = y if len(unique_classes) > 1 else None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=stratify_param,
        random_state=42,
    )

    scaler = StandardScaler()

    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # compute class weights safely
    if len(unique_classes) > 1:

        cw = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=y_tr
        )

        class_weight = {cls: w for cls, w in zip(unique_classes, cw)}

    else:

        class_weight = None

    params = {**RF_PARAMS, "class_weight": class_weight}

    logger.info("Training Random Forest classifier...")

    model = RandomForestClassifier(**params)

    model.fit(X_tr_s, y_tr)

    y_pred = model.predict(X_te_s)

    class_names = ["LOW", "MEDIUM", "HIGH"]

    labels = sorted(np.unique(np.concatenate([y_te, y_pred])))

    report = classification_report(
        y_te,
        y_pred,
        labels=labels,
        target_names=[class_names[i] for i in labels],
        zero_division=0
    )

    cm = confusion_matrix(y_te, y_pred, labels=labels)

    logger.info(f"\nClassification Report:\n{report}")

    logger.info(f"Confusion Matrix:\n{cm}")

    # cross validation
    if len(unique_classes) > 1:

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        cv_scores = cross_val_score(
            model,
            X_tr_s,
            y_tr,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1
        )

        logger.info(
            f"CV F1 (weighted): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
        )

    else:

        cv_scores = np.array([0.0])

        logger.warning("Cross-validation skipped (only one class present).")

    importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    logger.info(
        f"\nFeature Importance:\n{importance.to_string(index=False)}"
    )

    return model, scaler, report, importance, cv_scores


# =============================================================================
# 3. Save artifacts
# =============================================================================

def save_artifacts(model, scaler, report, importance, cv_scores, low_threshold, med_threshold):

    ensure_dirs(MODELS_DIR)

    save_model(model, os.path.join(MODELS_DIR, "risk_classifier.pkl"))
    save_model(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    importance.to_csv(
        os.path.join(MODELS_DIR, "feature_importance.csv"),
        index=False
    )

    report_path = os.path.join(MODELS_DIR, "training_report.txt")

    with open(report_path, "w") as f:

        f.write("SafeRoute India — Risk Classifier Training Report\n")
        f.write("=" * 50 + "\n\n")

        f.write(report + "\n")

        f.write(
            f"CV F1 (weighted): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}\n"
        )

        f.write(
            f"\nFeature Importance:\n{importance.to_string(index=False)}\n"
        )

    meta = {
        "features": MODEL_FEATURES,
        "target": MODEL_TARGET,
        "n_estimators": RF_PARAMS["n_estimators"],
        "cv_f1_mean": round(float(cv_scores.mean()), 3),
        "cv_f1_std": round(float(cv_scores.std()), 3),
        "risk_thresholds": {
            "LOW":    round(low_threshold, 6),
            "MEDIUM": round(med_threshold, 6),
        },
    }

    with open(os.path.join(MODELS_DIR, "model_meta.json"), "w") as f:

        json.dump(meta, f, indent=2)

    logger.success("All model artifacts saved → models/")


# =============================================================================
# Main
# =============================================================================

def run():

    available = [
        k for k in TARGET_CITIES
        if os.path.exists(os.path.join(GRAPHS_DIR, f"{k}_risk_graph.graphml"))
    ]

    if not available:

        raise FileNotFoundError(
            "No risk graphs found. Run: python src/06_snap_to_edges.py <city>"
        )

    logger.info(f"Training on data from: {available}")

    df, low_thr, med_thr = build_training_dataframe(available)

    model, scaler, report, importance, cv = train_model(df)

    save_artifacts(model, scaler, report, importance, cv, low_thr, med_thr)


if __name__ == "__main__":
    run()