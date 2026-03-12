# 🛡️ SafeRoute India

**Risk-aware route recommendation for Indian cities**  
Recommends the *safest* road route between two points by analysing crime, road accidents, flood zones, and road infrastructure — not just shortest distance.

---

## What it does

Instead of optimising for time or distance, SafeRoute weights every road segment by:

| Factor | Weight | Source |
|--------|--------|--------|
| Crime severity | 40% | NCRB district-level IPC data |
| Road accidents | 30% | MoRTH state/district + GPS blackspots |
| Flood risk | 15% | IIT-Delhi flood inventory 1985–2016 |
| Road quality | 15% | OSM highway classification |

Time-of-day multipliers amplify risk at night (2 AM crime weight = 2×) and during rush hours.

---

## Supported Cities (MVP)

Chennai · Mumbai · Delhi · Bengaluru · Hyderabad

---

## Project Structure

```
safe-route-india/
├── README.md
├── config.py                        ← All weights, cities, thresholds
├── requirements.txt
│
├── src/
│   ├── 01_download_data.py          ← IMD rainfall + OSM graph download
│   ├── 02_clean_crime.py            ← NCRB CSV → normalised crime scores
│   ├── 03_clean_accidents.py        ← MoRTH data + blackspot GPS cleaning
│   ├── 04_clean_flood.py            ← IIT-Delhi shapefile → district risk
│   ├── 05_geocode.py                ← District names → lat/lon coordinates
│   ├── 06_snap_to_edges.py          ← Snap risk data to OSM road segments
│   ├── 07_train_model.py            ← Train Random Forest risk classifier
│   ├── 08_score_graph.py            ← Apply model → final weighted graph
│   ├── 09_routing.py                ← Route engine: Dijkstra × 3 strategies
│   └── utils.py                     ← Shared helpers
│
├── api/
│   └── main.py                      ← FastAPI: /route /cities /heatmap /report
│
├── frontend/
│   ├── index.html                   ← Map UI (Leaflet.js, dark theme)
│   └── map.js                       ← Route drawing, risk panel, heatmap
│
├── notebooks/
│   ├── 01_explore_crime_data.ipynb
│   ├── 02_explore_accident_data.ipynb
│   ├── 03_build_risk_graph.ipynb
│   └── 04_validate_model.ipynb
│
├── data/
│   ├── raw/{crime,accidents,flood,maps}/
│   └── processed/
│
├── models/
│   ├── risk_classifier.pkl
│   └── scaler.pkl
│
├── graphs/
│   └── {city}_final_graph.graphml
│
├── validation/
│   └── validate_model.py
└── tests/
    ├── test_routing.py
    └── test_utils.py
```

---

## Quick Start

### 1. Setup environment

```bash
git clone <repo>
cd safe-route-india
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download datasets

Manually download these free datasets (no API keys needed):

**Crime data** — pick any:
- https://www.kaggle.com/datasets/sudhanvahg/indian-crimes-dataset
- https://data.gov.in/catalog/crime-against-women

**Accident data:**
- https://data.gov.in/catalog/accident-black-spots  ← GPS coordinates
- https://www.kaggle.com/datasets/data125661/india-road-accident-dataset

**Flood data:**
- https://github.com/hydrosenselab/India-Flood-Inventory  ← Shapefiles

Place files in `data/raw/crime/`, `data/raw/accidents/`, `data/raw/flood/`.

OSM map is auto-downloaded in the pipeline.

### 3. Run the full pipeline

```bash
python run_pipeline.py --city chennai
```

Or step by step:
```bash
python src/02_clean_crime.py
python src/03_clean_accidents.py
python src/04_clean_flood.py
python src/05_geocode.py
python src/06_snap_to_edges.py chennai
python src/07_train_model.py
python src/08_score_graph.py chennai
```

### 4. Test a route

```bash
python quick_test.py --city chennai --origin "Chennai Central" --dest "T. Nagar" --hour 21
```

Expected output:
```
🟢 SAFEST     12.4 km  safety= 78.2%  risk=0.218  [Safe]         ~24 min
🟡 BALANCED   10.1 km  safety= 65.3%  risk=0.347  [Moderate Risk] ~20 min
🔴 FASTEST     8.9 km  safety= 51.0%  risk=0.490  [Moderate Risk] ~18 min
```

### 5. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 to use the map interface.

API docs: http://localhost:8000/docs

---

## Risk Score Formula

```
ERS = 0.40×crime + 0.30×accident + 0.15×flood + 0.15×road_quality

effective_weight = edge_length × (1 + ERS × 3.0)
```

Routing uses NetworkX Dijkstra with `effective_weight` as the cost metric.  
Three route variants are returned: **Safest** (minimise risk), **Balanced** (50/50 risk+distance), **Fastest** (minimise distance).

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/route` | Compute 3 routes between addresses |
| GET | `/cities` | List supported cities and their status |
| GET | `/heatmap/{city}` | Crime heatmap points for a city |
| POST | `/report` | Submit a user incident report |

### Example `/route` request

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{
    "origin": "Chennai Central Railway Station",
    "destination": "T. Nagar",
    "city": "chennai",
    "travel_hour": 21,
    "profile": "default",
    "include_segments": true
  }'
```

---

## Safety Profiles

| Profile | Crime | Accident | Flood | Road |
|---------|-------|----------|-------|------|
| `default` | 40% | 30% | 15% | 15% |
| `women` | 60% | 20% | 10% | 10% |
| `cyclist` | 30% | 40% | 10% | 20% |
| `night` | 55% | 25% | 10% | 10% |

---

## Time-of-Day Risk Multipliers

| Time Window | Crime × | Accident × |
|-------------|---------|------------|
| 00:00–06:00 | 2.0 | 0.9 |
| 06:00–09:00 | 0.6 | 1.2 |
| 09:00–18:00 | 0.7 | 0.8 |
| 18:00–21:00 | 1.2 | 1.3 |
| 21:00–24:00 | 1.5 | 1.1 |

---

## Validation

```bash
python validation/validate_model.py --city chennai
python validation/validate_model.py --city all --offline   # skip live geocoding
```

Run unit tests:
```bash
pytest tests/ -v
```

---

## Extending to a New City

```bash
# 1. Add city to config.py TARGET_CITIES dict
# 2. Build graph
python src/06_snap_to_edges.py <new_city>
python src/08_score_graph.py   <new_city>
# 3. Test
python quick_test.py --city <new_city>
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Map data | OpenStreetMap via OSMnx |
| Routing | NetworkX Dijkstra |
| ML model | scikit-learn Random Forest |
| Geocoding | Nominatim (free, no key) |
| API | FastAPI + Uvicorn |
| Frontend | Leaflet.js + CartoDB tiles |
| Spatial ops | GeoPandas + Shapely |

All free and open-source. Zero API keys required.

---

## Datasets Used

| Dataset | Source | Format |
|---------|--------|--------|
| India Crime Statistics | NCRB / Kaggle | CSV |
| Road Accidents | MoRTH / data.gov.in | CSV |
| Accident Black Spots | data.gov.in | CSV + GPS |
| Flood Inventory | IIT-Delhi / GitHub | Shapefile |
| IMD Rainfall | imdlib Python package | Gridded |
| Road Network | OpenStreetMap / Geofabrik | .osm.pbf |
| District Boundaries | datameet / data.gov.in | GeoJSON |

---

## License

MIT License. Dataset licenses vary — check individual sources before commercial use.
