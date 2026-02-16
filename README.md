# IoT Anomaly Detection

Real-time anomaly detection system for IoT sensor data, combining multiple ML approaches with an ensemble scoring engine.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ IoT Sensors │────▶│  Streaming   │────▶│  Preprocessing  │
│ (Simulated) │     │  Pipeline    │     │  & Features     │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                    ┌──────────────────────────────┐│
                    │       Ensemble Detector       ││
                    │  ┌──────────┐ ┌───────────┐  │▼
                    │  │Isolation │ │Autoencoder│  │
                    │  │ Forest   │ │ (PyTorch) │  │
                    │  └──────────┘ └───────────┘  │
                    │  ┌──────────┐                │
                    │  │ DBSCAN   │                │
                    │  └──────────┘                │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
             ┌───────────┐ ┌────────────┐ ┌───────────┐
             │ Alerting  │ │ Root Cause │ │ REST API  │
             │ Engine    │ │ Analysis   │ │ (FastAPI) │
             └───────────┘ └────────────┘ └───────────┘
                                   │
                                   ▼
                          ┌────────────────┐
                          │   Streamlit    │
                          │   Dashboard    │
                          └────────────────┘
```

## Features

- **Multi-model ensemble** — Isolation Forest, PyTorch Autoencoder, and DBSCAN with weighted scoring
- **Real-time streaming** — Async pipeline with sliding window feature extraction
- **Alerting engine** — Configurable threshold, frequency, and combination rules with cooldown
- **Root cause analysis** — Per-sensor contribution ranking for detected anomalies
- **Interactive dashboard** — Streamlit app with real-time views, sensor health, and model comparison
- **REST API** — FastAPI endpoints for scoring, alerts, and monitoring
- **Synthetic data generator** — Configurable IoT sensor simulation with injected anomalies

## Quick Start

```bash
# Clone
git clone git@github.com:KarasiewiczStephane/iot-anomaly-detection.git
cd iot-anomaly-detection

# Install dependencies
pip install -r requirements.txt

# Generate sample data
make generate-data

# Run the API server
make run

# Launch the dashboard
make dashboard
```

## Docker

```bash
# Build and start all services
make docker-run

# Stop services
make docker-stop
```

The API runs on `http://localhost:8000` and the dashboard on `http://localhost:8501`.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check / readiness probe |
| `POST` | `/score` | Score a single sensor reading |
| `POST` | `/score/batch` | Score a batch of readings |
| `GET` | `/alerts` | Retrieve alert history (filterable) |
| `PUT` | `/alerts/{id}/resolve` | Resolve an alert |
| `GET` | `/models/status` | Model load status |
| `GET` | `/sensors/health` | Sensor health summary |

## Project Structure

```
iot-anomaly-detection/
├── src/
│   ├── api/            # FastAPI REST endpoints
│   ├── alerting/       # Rules engine and notification channels
│   ├── analysis/       # Root cause analysis
│   ├── dashboard/      # Streamlit interactive dashboard
│   ├── data/           # Data generation and preprocessing
│   ├── models/         # Isolation Forest, Autoencoder, DBSCAN, Ensemble
│   ├── streaming/      # Async streaming pipeline and real-time scorer
│   └── utils/          # Config, logging, database utilities
├── tests/              # Unit tests (160+ tests, >80% coverage)
├── configs/            # YAML configuration
├── .github/workflows/  # CI/CD pipeline
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── README.md
```

## Development

```bash
# Run tests
make test

# Lint
make lint

# Auto-fix lint issues
make lint-fix
```

## Tech Stack

- **ML**: scikit-learn, PyTorch, DBSCAN
- **API**: FastAPI, Uvicorn
- **Dashboard**: Streamlit, Plotly
- **Data**: Pandas, NumPy
- **Storage**: SQLite
- **CI/CD**: GitHub Actions, Docker

## License

MIT
