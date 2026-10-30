# FleetGuard AI — Predictive Maintenance Platform

**Predict Before Failure. Maintain Before Breakdown.**

A complete, production-ready predictive vehicle maintenance platform built for the Teliport Season 4 Hackathon (Case Study 1: Self-Healing Machines).

---

## 🎯 Features

### Dual Operating Modes
- **Mode A: NASA C-MAPSS Benchmark** — Turbofan engine RUL prediction using Random Forest on NASA FD001 dataset
- **Mode B: Live Fleet Simulator** — 10+ simulated commercial vehicles with realistic sensor generation and fault injection

### Core Capabilities
- 📊 **Real-time Fleet Dashboard** — Health scores, alerts, work orders, sensor trends
- 🤖 **ML-Powered Predictions** — RUL regression, Isolation Forest anomaly detection, statistical outlier detection
- 🔧 **Rule-Based Maintenance Triage** — Deterministic alert generation (no LLM), automatic work order creation
- 📋 **Work Order Management** — Full lifecycle: Open → Assigned → In Progress → Completed
- 🎮 **Interactive Demo Controls** — Fault injection, simulation speed, reset, work order generation
- 🗄️ **Persistent PostgreSQL** — All data survives restarts on Render
- 📈 **MLflow Experiment Tracking** — Model versioning, metrics comparison, feature importance

---

## 🏗️ Architecture

```
fleetguard-ai/
├── app/
│   ├── main.py                 # Streamlit entry point
│   ├── models/
│   │   ├── database.py         # SQLAlchemy models
│   │   └── connection.py       # DB session management
│   ├── pages/                  # Dashboard pages
│   │   ├── overview.py         # Fleet overview KPIs & charts
│   │   ├── fleet.py            # Fleet management table
│   │   ├── vehicle_detail.py   # Per-vehicle deep dive
│   │   ├── predictions.py      # NASA benchmark + fleet health
│   │   ├── alerts.py           # Alert center with actions
│   │   ├── maintenance.py      # Work orders, technicians, calendar
│   │   └── data_management.py  # Dataset upload, model registry, sim settings
│   ├── ml/
│   │   ├── anomaly_detection.py  # Threshold, statistical, Isolation Forest
│   │   ├── rul_prediction.py     # NASA C-MAPSS RUL models
│   │   └── health_scoring.py     # Health score + rule-based triage
│   ├── simulation/
│   │   └── engine.py           # Deterministic fleet simulator
│   ├── services/
│   │   └── fleet_service.py    # Orchestration service
│   └── utils/
│       └── helpers.py          # Utility functions
├── scripts/
│   └── init_db.py              # Database initialization
├── tests/                      # Pytest test suite
├── Dockerfile                  # Render deployment
├── render.yaml                 # Render service definition
├── requirements.txt            # Python dependencies
└── .env.example                # Environment template
```

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.11+
- PostgreSQL 14+
- Docker (optional)

### 1. Clone & Install
```bash
cd fleetguard-ai
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your database URL
```

### 3. Initialize Database
```bash
# Option A: Using script
python scripts/init_db.py

# Option B: Using Docker Compose (includes PostgreSQL)
docker-compose up -d db
python scripts/init_db.py
```

### 4. Run Application
```bash
streamlit run app/main.py
```

Open http://localhost:8501

---

## 🐳 Docker Deployment (Local)

```bash
# Build image
docker build -t fleetguard-ai .

# Run with PostgreSQL
docker-compose up -d
```

---

## ☁️ Render Deployment

### 1. Create Render Account
Sign up at [render.com](https://render.com)

### 2. Connect Repository
- Push this repo to GitHub/GitLab
- In Render Dashboard: **New +** → **Blueprint**
- Select your repo → Render will detect `render.yaml`

### 3. Configure Resources
The `render.yaml` defines:
- **Web Service**: `fleetguard-ai` (Docker, Free tier)
- **PostgreSQL**: `fleetguard-db` (Free tier)

### 4. Environment Variables
Set in Render Dashboard → Environment:
```
DATABASE_URL          # Auto-populated from PostgreSQL
MLFLOW_TRACKING_URI   # file:///app/mlruns
STREAMLIT_SERVER_PORT # 8501
STREAMLIT_SERVER_ADDRESS # 0.0.0.0
SIMULATION_SPEED      # 1.0
DEMO_MODE             # true
```

### 5. Deploy
Click **Deploy** — Render builds Docker image, runs migrations, starts service.

### 6. Access
Your app will be live at: `https://fleetguard-ai.onrender.com`

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html

# Specific test modules
pytest tests/test_anomaly_detection.py -v
pytest tests/test_rul_prediction.py -v
pytest tests/test_health_scoring.py -v
pytest tests/test_simulation.py -v
pytest tests/test_database.py -v
```

---

## 🎮 Demo Walkthrough (For Judges)

### 1. Start Simulation
- Open **Demo Control Panel** (sidebar)
- Click **▶️ Start** — Live indicator turns 🔴 LIVE

### 2. Inject Faults
Click any fault button:
- 🌡️ **Overheating** — Engine temp spikes → Critical alert → Work order
- 📳 **Vibration** — Vibration spikes → High alert → Work order
- 🛢️ **Low Oil Pressure** — Oil pressure drops → Critical alert → Work order
- 🔋 **Battery Drain** — Voltage drops → High alert → Work order
- 📉 **Gradual Degradation** — All sensors drift → Trend alerts

### 3. Watch Dashboard React
- **Overview**: KPIs update, health distribution shifts, anomaly trend spikes
- **Fleet**: Vehicle status changes color (Green→Yellow→Red)
- **Vehicle Detail**: Sensor charts show fault, anomalies list populates
- **Alerts**: New alerts appear with severity badges
- **Maintenance**: Work orders auto-created, assign technicians

### 4. Complete Work Order
- Go to **Maintenance** → **Work Orders**
- Click **Assign** → Select technician → **Start Work** → Add notes → **Complete**
- Vehicle health updates, maintenance history records created

### 5. NASA Benchmark
- Go to **AI/ML Predictions** → **NASA Benchmark**
- Upload `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`
- Click **Train RUL Model** → View Actual vs Predicted RUL, residuals, metrics
- Compare Random Forest vs Linear Baseline

---

## 📊 Key Metrics (Demo Data)

| Metric | Value |
|--------|-------|
| Fleet Size | 10 vehicles (6 Trucks, 2 Buses, 2 more) |
| Sensors/Vehicle | 7 (temp, RPM, oil pressure, coolant, vibration, battery, fuel) |
| Simulation Speed | 0.1x – 5.0x configurable |
| Update Interval | 2 seconds (configurable) |
| Health Score Range | 0–100 (Green ≥80, Yellow 50–79, Red <50) |
| Alert Severities | Critical, High, Medium, Low |
| Work Order Priorities | Urgent, High, Medium, Low |

---

## ⚠️ Limitations & Disclaimers

### Prototype Limitations
1. **Simulated Data Only** — All sensor readings are generated by a deterministic simulator, not real vehicle telemetry
2. **NASA Model ≠ Truck Model** — The C-MAPSS RUL model is trained on turbofan engines; it is NOT applied to truck sensors
3. **Rule-Based Triage** — Maintenance decisions use explicit configurable thresholds (demo values), not validated OEM specifications
4. **No Safety Certification** — This is a hackathon prototype; never use for actual vehicle safety decisions
5. **Single-Instance** — Designed for single-container deployment; not horizontally scaled

### Model Assumptions
- NASA FD001: Single operating condition, 100 engines, run-to-failure
- Health scoring weights are heuristic (not statistically calibrated)
- Anomaly thresholds are demonstration defaults
- Degradation models are simplified linear/exponential curves

### Future Production Requirements
- Real vehicle telematics integration (J1939, OBD-II, MQTT)
- Validated models trained on fleet-specific historical data
- OEM-specified thresholds per vehicle model/year
- Redundant deployment, authentication, audit logging
- Regulatory compliance (ISO 26262, UN R155)

---

## 🔧 Configuration

### Simulation Parameters
```python
# In app/simulation/engine.py
HEALTHY_RANGES = {
    VehicleType.TRUCK: {
        "engine_temp": (85, 95),
        "rpm": (1200, 1800),
        "oil_pressure": (350, 450),
        # ...
    }
}

FAULT_CONFIGS = {
    SimulationEventType.OVERHEATING: {
        "engine_temp_multiplier": 1.35,
        "duration_cycles": 20,
    },
    # ...
}
```

### Alert Thresholds
```python
# In app/ml/anomaly_detection.py
DEFAULT_THRESHOLDS = {
    "engine_temperature_c": {"warning": 105, "critical": 115},
    "oil_pressure_kpa": {"warning": 250, "critical": 150},
    # ...
}
```

### Health Weights
```python
# In app/ml/health_scoring.py
HEALTH_WEIGHTS = {
    "engine_temperature": 0.20,
    "oil_pressure": 0.20,
    "vibration": 0.15,
    # ...
}
```

---

## 📁 Project Structure Details

### Database Tables
| Table | Purpose |
|-------|---------|
| `vehicles` | Fleet registry with specs, health, service schedule |
| `sensor_readings` | Time-series sensor data (simulated) |
| `vehicle_health_history` | Health score snapshots over time |
| `anomalies` | Detected anomalies with type, severity, evidence |
| `maintenance_alerts` | Triage-generated alerts with recommended actions |
| `work_orders` | Maintenance work order lifecycle |
| `maintenance_history` | Completed service records |
| `ml_models` | Trained model registry with metrics |
| `model_experiments` | MLflow experiment tracking |
| `dataset_uploads` | NASA dataset upload records |
| `simulation_events` | Fault injection audit trail |
| `technicians` | Maintenance personnel |

---

## 🤝 Contributing

This is a hackathon prototype. For production use:
1. Add authentication (OAuth2/OIDC)
2. Implement RBAC for technicians/managers/admins
3. Add API layer (FastAPI) for external integrations
4. Set up CI/CD with automated testing
5. Configure monitoring (Prometheus/Grafana)
6. Add data retention policies

---

## 📄 License

MIT License — Built for Teliport Season 4 Hackathon.

---

## 🙏 Acknowledgments

- **NASA** — C-MAPSS Turbofan Engine Degradation Dataset
- **Streamlit** — Rapid dashboard framework
- **Scikit-learn** — ML algorithms
- **Plotly** — Interactive visualizations
- **Render** — Free tier hosting
- **Teliport** — Hackathon platform