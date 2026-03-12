#!/usr/bin/env python3
# =============================================================================
# api/main.py — SafeRoute India FastAPI backend
# =============================================================================
# Run: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# =============================================================================

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, validator
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import TARGET_CITIES, SAFETY_PROFILES, API_HOST, API_PORT, GRAPHS_DIR
from src.routing import find_safe_routes, get_graph, clear_cache


# =============================================================================
# App initialisation
# =============================================================================

app = FastAPI(
    title="SafeRoute India API",
    description=(
        "Risk-aware route recommendation for Indian cities. "
        "Analyzes crime, accidents, and flood data to find the safest path."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
frontend_dir = PROJECT_ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# =============================================================================
# Pydantic models
# =============================================================================

class RouteRequest(BaseModel):
    origin:      str = Field(..., min_length=3, example="Chennai Central Railway Station")
    destination: str = Field(..., min_length=3, example="T. Nagar Bus Stand")
    city:        str = Field("chennai", example="chennai")
    travel_hour: Optional[int] = Field(12, ge=0, le=23,
                                        description="Hour of travel in 24h format")
    profile:     Optional[str] = Field("default",
                                        description="Safety profile: default/women/cyclist/night")

    @validator("city")
    def city_must_be_supported(cls, v):
        if v not in TARGET_CITIES:
            raise ValueError(f"Unsupported city '{v}'. Options: {list(TARGET_CITIES.keys())}")
        return v

    @validator("profile")
    def profile_must_be_valid(cls, v):
        if v and v not in SAFETY_PROFILES:
            raise ValueError(f"Unknown profile '{v}'. Options: {list(SAFETY_PROFILES.keys())}")
        return v or "default"


class RouteSummary(BaseModel):
    distance_km:          float
    avg_risk_score:       float
    avg_crime_score:      float
    avg_accident_score:   float
    avg_flood_score:      float
    high_risk_segments:   int
    total_segments:       int
    safety_pct:           float
    risk_label:           str


class RouteResult(BaseModel):
    coordinates: List[List[float]]
    summary:     RouteSummary


class RouteResponse(BaseModel):
    origin_coords:      List[float]
    destination_coords: List[float]
    city:               str
    travel_hour:        int
    profile:            str
    routes:             Dict[str, Optional[RouteResult]]
    computation_ms:     float


class UserReport(BaseModel):
    lat:           float = Field(..., ge=6.0,  le=38.0)
    lon:           float = Field(..., ge=67.0, le=98.0)
    incident_type: str   = Field(..., example="Theft")
    severity:      int   = Field(..., ge=1, le=10)
    description:   Optional[str] = None


# =============================================================================
# Startup: pre-load graphs for available cities
# =============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("SafeRoute India API starting...")
    os.makedirs(PROJECT_ROOT / "logs", exist_ok=True)

    # Pre-load graphs that exist
    for city_key in TARGET_CITIES:
        final = PROJECT_ROOT / GRAPHS_DIR / f"{city_key}_final_graph.graphml"
        risk  = PROJECT_ROOT / GRAPHS_DIR / f"{city_key}_risk_graph.graphml"
        if final.exists() or risk.exists():
            try:
                get_graph(city_key)
                logger.success(f"  Pre-loaded graph: {city_key}")
            except Exception as ex:
                logger.warning(f"  Could not pre-load {city_key}: {ex}")

    logger.success("API ready.")


# =============================================================================
# Routes
# =============================================================================

@app.get("/", response_class=FileResponse)
def serve_index():
    """Serve the frontend map interface."""
    index = frontend_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "SafeRoute India API", "docs": "/docs"})


@app.get("/health")
def health_check():
    """Health check endpoint."""
    loaded = list(_get_loaded_cities())
    return {
        "status":        "healthy",
        "loaded_cities": loaded,
        "supported":     list(TARGET_CITIES.keys()),
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Suppress favicon 404."""
    favicon_path = frontend_dir / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(str(favicon_path))
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/cities")
def list_cities():
    """Return all supported cities with their metadata."""
    return {
        "cities": {
            k: {"place": v["place"], "center": list(v["center"])}
            for k, v in TARGET_CITIES.items()
        }
    }


@app.get("/profiles")
def list_profiles():
    """Return available safety profiles."""
    return {"profiles": SAFETY_PROFILES}


@app.post("/route", response_model=RouteResponse)
def get_routes(req: RouteRequest):
    """
    Find safe routes between two addresses.

    Returns three route options:
    - **safest**: minimises composite risk score (crime + accident + flood)
    - **balanced**: balances safety and distance
    - **fastest**: minimises physical distance only
    """
    t_start = time.time()

    import inspect
    routing_sig = inspect.signature(find_safe_routes)
    routing_kwargs = dict(
        origin_address=req.origin,
        destination_address=req.destination,
        city_key=req.city,
        travel_hour=req.travel_hour or 12,
    )
    # Only pass profile if routing.py supports it
    if "profile" in routing_sig.parameters:
        routing_kwargs["profile"] = req.profile or "default"
    if "include_segments" in routing_sig.parameters:
        routing_kwargs["include_segments"] = True

    try:
        routes, orig, dest = find_safe_routes(**routing_kwargs)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Graph not ready for '{req.city}'. {str(e)}"
        )
    except Exception as e:
        logger.error(f"Routing error: {e}")
        raise HTTPException(status_code=500, detail=f"Routing failed: {str(e)}")

    # Strip node_path (not JSON-serialisable) and validate shapes
    clean_routes = {}
    for name, route in routes.items():
        if route:
            clean_routes[name] = RouteResult(
                coordinates=route["coordinates"],
                summary=RouteSummary(**route["summary"]),
            )
        else:
            clean_routes[name] = None

    return RouteResponse(
        origin_coords=list(orig),
        destination_coords=list(dest),
        city=req.city,
        travel_hour=req.travel_hour or 12,
        profile=req.profile or "default",
        routes=clean_routes,
        computation_ms=round((time.time() - t_start) * 1000, 1),
    )


@app.post("/report")
def report_incident(report: UserReport):
    """
    Submit a user-reported unsafe incident.
    These are stored and periodically incorporated into the risk model.
    """
    import json
    from datetime import datetime

    reports_path = PROJECT_ROOT / "data" / "raw" / "crime" / "user_reports.jsonl"
    os.makedirs(reports_path.parent, exist_ok=True)

    record = report.dict()
    record["timestamp"] = datetime.utcnow().isoformat()

    with open(reports_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    return {"status": "received", "message": "Thank you for your report."}


@app.get("/heatmap/{city}")
def get_crime_heatmap(city: str):
    """Return crime data as a lat/lon/weight array for heatmap display."""
    import pandas as pd

    if city not in TARGET_CITIES:
        raise HTTPException(status_code=404, detail=f"City not found: {city}")

    crime_path = PROJECT_ROOT / "data" / "processed" / "crime_geocoded.csv"
    if not crime_path.exists():
        raise HTTPException(status_code=503, detail="Crime data not yet processed.")

    df = pd.read_csv(crime_path)
    score_col = next((c for c in df.columns if "NORM" in c.upper()), None)
    if not score_col:
        raise HTTPException(status_code=503, detail="No normalised score column found.")

    df = df.dropna(subset=["LAT", "LON", score_col])
    # Filter to city bounding box
    bbox = TARGET_CITIES[city]["bbox"]  # (south, west, north, east)
    df = df[
        (df["LAT"] >= bbox[0]) & (df["LAT"] <= bbox[2]) &
        (df["LON"] >= bbox[1]) & (df["LON"] <= bbox[3])
    ]

    points = df[["LAT", "LON", score_col]].values.tolist()
    return {"city": city, "points": points, "count": len(points)}


# =============================================================================
# Helpers
# =============================================================================

def _get_loaded_cities():
    from src.routing import _GRAPH_CACHE
    return _GRAPH_CACHE.keys()


# =============================================================================
# Dev server entry point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)