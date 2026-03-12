# 🛡️ SafeRoute India

AI-powered **risk-aware navigation system for Indian cities**.

SafeRoute recommends the **safest road route** between two locations by analysing:

- Crime statistics
- Road accident data
- Flood-prone zones
- Road infrastructure quality

Traditional navigation apps optimise for **distance or travel time**.

SafeRoute instead optimises for **personal safety**.

---

# 🚦 How SafeRoute Works

Each road segment receives a **risk score** derived from multiple datasets.

| Risk Factor | Weight | Data Source |
|--------------|--------|-------------|
| Crime severity | 40% | NCRB crime statistics |
| Road accidents | 30% | MoRTH accident datasets |
| Flood risk | 15% | IIT Delhi flood inventory |
| Road quality | 15% | OpenStreetMap road types |

Risk score formula:

```
ERS = 0.40×crime + 0.30×accident + 0.15×flood + 0.15×road_quality
```

Routing uses **NetworkX Dijkstra** with a modified cost function:

```
effective_weight = road_length × (1 + ERS × 3)
```

The system returns **three route options**:

| Route | Description |
|------|-------------|
🟢 Safest | lowest risk |
🟡 Balanced | mix of safety + distance |
🔴 Fastest | shortest path |

---

# 🌏 Supported Cities (MVP)

- Chennai  
- Mumbai  
- Delhi  
- Bengaluru  
- Hyderabad  

Adding new cities only requires downloading new OpenStreetMap graphs.

---

# 🧠 Tech Stack

| Component | Technology |
|-----------|------------|
Backend API | FastAPI |
Routing Engine | NetworkX |
Map Data | OSMnx |
Spatial Analysis | GeoPandas |
ML Model | scikit-learn Random Forest |
Frontend Map | Leaflet.js |
Visualisation | Folium + Matplotlib |

All tools are **free and open source**.

---

# 📂 Project Structure

```
safe-route-india
│
├── README.md
├── requirements.txt
├── config.py
├── run_pipeline.py
├── quick_test.py
│
├── api
│   └── main.py
│
├── frontend
│   ├── index.html
│   └── map.js
│
├── src
│   ├── clean_crime.py
│   ├── clean_accidents.py
│   ├── clean_flood.py
│   ├── download_data.py
│   ├── geocode.py
│   ├── snap_to_edges.py
│   ├── train_model.py
│   ├── score_graph.py
│   ├── routing.py
│   └── utils.py
│
├── notebooks
│
├── data
│   ├── raw
│   └── processed
│
├── models
│
├── graphs
│
├── validation
│
└── tests
```

---

# ⚙️ Installation Guide (Beginner Friendly)

## 1️⃣ Clone the Repository

Open **VS Code terminal** and run:

```
git clone https://github.com/amanhaalim/safe-Route-India.git
cd safe-Route-India
```

This downloads the project from GitHub to your system.

---

# 2️⃣ Create a Virtual Environment

Virtual environments isolate project dependencies.

### Windows

```
python -m venv venv
venv\Scripts\activate
```

### Mac / Linux

```
python3 -m venv venv
source venv/bin/activate
```

When activated your terminal will show:

```
(venv)
```

---

# 3️⃣ Install Dependencies

Install all required libraries:

```
pip install --upgrade pip
pip install -r requirements.txt
```

This installs libraries such as:

- geopandas
- osmnx
- fastapi
- networkx
- scikit-learn
- folium

---

# 4️⃣ Download Required Datasets

Create the following folders if they do not exist:

```
data/raw/crime
data/raw/accidents
data/raw/flood
```

Download datasets from:

Crime data  
https://www.kaggle.com/datasets/sudhanvahg/indian-crimes-dataset

Accident data  
https://www.kaggle.com/datasets/data125661/india-road-accident-dataset

Accident blackspots  
https://data.gov.in/catalog/accident-black-spots

Flood inventory  
https://github.com/hydrosenselab/India-Flood-Inventory

Place them inside:

```
data/raw/crime
data/raw/accidents
data/raw/flood
```

---

# 5️⃣ Run the Data Processing Pipeline

The pipeline performs:

1. Clean datasets  
2. Train AI risk model  
3. Build risk-weighted road graph  

Run:

```
python run_pipeline.py --city chennai
```

This will automatically:

- download OpenStreetMap road network
- clean crime data
- clean accident data
- process flood zones
- train risk model
- generate weighted road graph

Outputs will appear in:

```
graphs/
models/
data/processed/
```

---

# 6️⃣ Test the Routing Engine

Run a quick test route:

```
python quick_test.py --city chennai --origin "Chennai Central" --dest "T Nagar" --hour 21
```

Example output:

```
SAFEST    12.4 km  safety=78%
BALANCED  10.1 km  safety=65%
FASTEST    8.9 km  safety=51%
```

---

# 7️⃣ Start the API Server

Run:

```
uvicorn api.main:app --reload --port 8000
```

API will run at:

```
http://localhost:8000
```

API documentation:

```
http://localhost:8000/docs
```

---

# 8️⃣ Open the Map Interface

Open the frontend file in a browser:

```
frontend/index.html
```

The map allows you to:

- enter start and destination
- view safest routes
- visualise risk heatmaps

---

# 🧪 Running Tests

Run unit tests:

```
pytest tests/ -v
```

Run model validation:

```
python validation/validate_model.py --city chennai
```

---

# 🧩 Adding a New City

Example: adding **Kolkata**

Step 1 — edit `config.py`

Add city coordinates.

Step 2 — generate graph

```
python src/snap_to_edges.py kolkata
```

Step 3 — score the graph

```
python src/score_graph.py kolkata
```

Step 4 — test routing

```
python quick_test.py --city kolkata
```

---

# 📊 Datasets Used

| Dataset | Source |
|-------|------|
Crime data | NCRB |
Accident statistics | MoRTH |
Accident blackspots | data.gov.in |
Flood inventory | IIT Delhi |
Road network | OpenStreetMap |
Rainfall | IMD |

---

# 🧠 Future Improvements

- real-time crime reporting
- mobile app
- crowd-sourced safety alerts
- police station proximity routing
- women-safety mode
- night-time route optimisation

---

# 📜 License

MIT License.

Dataset licenses belong to their respective providers.
