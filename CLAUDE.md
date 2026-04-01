# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.
Update this file whenever architecture decisions are made, new modules are added,
or conventions change. This is the single source of truth for the project.

---

## Project Overview

FlowWatch AI is a real-time network monitoring and anomaly detection system.
It ingests live network telemetry (latency, packet loss, DNS failures) via a
streaming pipeline, runs ML-based anomaly detection, exposes inference APIs,
and provides an LLM-powered assistant for root cause analysis and alerting.

---

## Tech Stack

### Backend
- Language: Python 3.11+
- API Framework: FastAPI (async)
- ML Frameworks: PyTorch (LSTM models), Scikit-learn (Isolation Forest)
- Data Pipeline: AWS Kinesis (local simulation via a Kinesis mock or LocalStack)
- LLM Assistant: Anthropic Claude API (claude-sonnet-4-20250514)
- Task Queue: (TBD — Celery or asyncio background tasks)
- Database: TimescaleDB or InfluxDB for time-series telemetry storage
- Caching: Redis (for recent telemetry state and alert deduplication)

### Frontend
- Framework: Next.js 14+ (App Router)
- Language: TypeScript
- Styling: Tailwind CSS
- Charts: Recharts or Chart.js for real-time network metrics
- Package Manager: npm

### Infrastructure
- Local: Docker Compose (all services containerized)
- Cloud: AWS EC2 (inference APIs), AWS Kinesis (telemetry ingestion),
         AWS CloudWatch (logging, alerting, dashboards)
- Containerization: Docker
- CI/CD: GitHub Actions (TBD)

---

## Project Structure

```
flowwatch-ai/
├── backend/
│   ├── api/                    # FastAPI route handlers
│   │   ├── routes/
│   │   │   ├── telemetry.py    # Ingest & query telemetry endpoints
│   │   │   ├── anomalies.py    # Anomaly detection endpoints
│   │   │   └── assistant.py    # LLM-based RCA assistant endpoints
│   │   └── main.py             # FastAPI app entrypoint
│   ├── models/
│   │   ├── lstm_model.py       # PyTorch LSTM anomaly detection model
│   │   ├── isolation_forest.py # Scikit-learn Isolation Forest model
│   │   └── feature_engineering.py  # Feature extraction from raw telemetry
│   ├── pipeline/
│   │   ├── kinesis_consumer.py # AWS Kinesis stream consumer
│   │   ├── kinesis_producer.py # Telemetry data producer (for simulation)
│   │   └── preprocessor.py     # Stream data cleaning and normalization
│   ├── assistant/
│   │   └── rca_agent.py        # Claude API integration for root cause analysis
│   ├── alerting/
│   │   └── alert_manager.py    # Real-time alerting logic and CloudWatch integration
│   ├── db/
│   │   └── timeseries.py       # DB connection and telemetry write/read helpers
│   ├── tests/                  # Unit and integration tests
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── app/                # Next.js App Router pages
│   │   ├── components/
│   │   │   ├── dashboard/      # Dashboard-specific components (charts, tables)
│   │   │   └── ui/             # Reusable UI components (cards, badges, alerts)
│   │   ├── lib/                # API clients, utilities, constants
│   │   └── styles/             # Global CSS
│   ├── public/                 # Static assets
│   ├── Dockerfile
│   └── package.json
│
├── infra/
│   ├── docker-compose.yml      # Local full-stack orchestration
│   ├── docker-compose.prod.yml # Production overrides
│   └── aws/                    # AWS CloudFormation or CDK scripts (TBD)
│
├── notebooks/                  # Jupyter notebooks for model training & EDA
├── data/                       # Sample/simulated telemetry data
├── scripts/                    # Utility scripts (data generation, model export)
├── .env.example                # Environment variable template
├── .gitignore
└── CLAUDE.md                   # This file
```

---

## Development Commands

### Backend
```bash
cd backend
pip install -r requirements.txt     # Install dependencies
uvicorn api.main:app --reload        # Start FastAPI dev server (localhost:8000)
pytest tests/                        # Run tests
```

### Frontend
```bash
cd frontend
npm install                          # Install dependencies
npm run dev                          # Start Next.js dev server (localhost:3000)
npm run build                        # Build for production
npm run lint                         # Run ESLint
```

### Docker (Full Stack Local)
```bash
docker-compose -f infra/docker-compose.yml up --build   # Start all services
docker-compose -f infra/docker-compose.yml down          # Stop all services
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in values. Never commit `.env`.

```
# Anthropic
ANTHROPIC_API_KEY=

# AWS
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
KINESIS_STREAM_NAME=flowwatch-telemetry

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/flowwatch

# Redis
REDIS_URL=redis://localhost:6379

# App
ENVIRONMENT=development
LOG_LEVEL=INFO
```

---

## Architecture Overview

```
[Network Agents / Simulators]
        |
        v
[AWS Kinesis Stream]  <-- telemetry: latency, packet_loss, dns_failures
        |
        v
[Kinesis Consumer (backend/pipeline)]
        |
        v
[Preprocessor + Feature Engineering]
        |
       / \
      v   v
[LSTM Model]  [Isolation Forest]
      \   /
       v v
[Anomaly Score Aggregator]
        |
        v
[FastAPI Inference API]  <-->  [LLM RCA Assistant (Claude API)]
        |
        v
[Alert Manager]  -->  [AWS CloudWatch / Notifications]
        |
        v
[TimescaleDB / InfluxDB]  <-->  [Next.js Frontend Dashboard]
```

---

## ML Model Details

### LSTM (PyTorch)
- Input: sliding window of time-series telemetry features (window size TBD, ~30 steps)
- Features: latency (ms), packet_loss (%), dns_failure_rate, jitter
- Output: reconstruction error → anomaly if error > threshold
- Training: offline on historical telemetry, exported as TorchScript for inference

### Isolation Forest (Scikit-learn)
- Input: feature vectors of current telemetry snapshot
- Output: anomaly score (-1 = anomaly, 1 = normal)
- Used as: fast, lightweight first-pass detector alongside LSTM

### Feature Engineering (backend/models/feature_engineering.py)
- Rolling mean / std of latency and packet loss
- Rate of change (delta) between consecutive readings
- DNS failure frequency over sliding window
- Time-of-day encoding (cyclic sin/cos features)

---

## API Endpoints (FastAPI)

| Method | Endpoint                  | Description                              |
|--------|---------------------------|------------------------------------------|
| POST   | /telemetry/ingest         | Ingest raw telemetry data point          |
| GET    | /telemetry/recent         | Fetch recent telemetry (last N minutes)  |
| GET    | /anomalies/latest         | Get latest detected anomalies            |
| POST   | /anomalies/detect         | Run on-demand anomaly detection          |
| POST   | /assistant/analyze        | Ask LLM assistant to explain anomaly     |
| GET    | /health                   | Health check                             |

---

## Code Conventions

### Python (Backend)
- Use Python 3.11+ type hints everywhere
- Async/await for all FastAPI route handlers
- Pydantic v2 models for all request/response schemas
- Keep route handlers thin — business logic lives in service/model layers
- Use `loguru` for structured logging
- All ML model files must include a `predict(features)` and `train(data)` method

### TypeScript (Frontend)
- Use TypeScript strictly — no `any` types
- Prefer functional components with hooks
- Use Tailwind utility classes — avoid inline styles
- Keep components focused and single-purpose
- API calls go in `src/lib/api.ts`

---

## What to Avoid

- Do NOT hardcode AWS credentials or API keys anywhere in code
- Do NOT block the async event loop in FastAPI routes (use `run_in_executor` for CPU-heavy ML inference if needed)
- Do NOT store raw telemetry in memory beyond the processing window — write to DB
- Do NOT skip Pydantic validation on incoming telemetry payloads
- Do NOT mix training logic with inference logic in the same file

---

## Current Status & Build Order

Track progress here as features are completed.

- [X] Phase 1: Project scaffolding (folder structure, Docker Compose, env setup)
- [X] Phase 2: Telemetry simulator (kinesis_producer.py) + Kinesis consumer
- [X] Phase 3: Feature engineering pipeline
- [X] Phase 4: Isolation Forest model (faster to build first)
- [~] Phase 5: LSTM model (PyTorch) + training notebook
- [ ] Phase 6: FastAPI inference endpoints
- [ ] Phase 7: LLM RCA assistant (Claude API integration)
- [ ] Phase 8: Alert manager + CloudWatch integration
- [ ] Phase 9: Next.js frontend dashboard
- [ ] Phase 10: Docker Compose full-stack wiring
- [ ] Phase 11: AWS deployment (EC2 + Kinesis + CloudWatch)

---

## Key Decisions Log

| Date       | Decision                                      | Reason                                      |
|------------|-----------------------------------------------|---------------------------------------------|
| 2026-04-01 | Use FastAPI over Flask                        | Native async support, better for streaming  |
| 2026-04-01 | Use both LSTM + Isolation Forest              | Complementary: LSTM=temporal, IF=snapshot   |
| 2026-04-01 | Claude API for RCA assistant                  | Best-in-class for log/telemetry analysis    |
| 2026-04-01 | LocalStack for local Kinesis simulation       | Avoid AWS costs during development          |
| 2026-04-01 | Next.js + Recharts for frontend               | SSR + real-time chart support               |

---

## Telemetry Schema

### TelemetryRecord (backend/pipeline/kinesis_consumer.py)

Pydantic v2 model. Produced by `kinesis_producer.py`, validated by the consumer.

| Field              | Type            | Constraints        | Description                                      |
|--------------------|-----------------|--------------------|--------------------------------------------------|
| `timestamp`        | `str`           | ISO 8601 UTC       | Record creation time                             |
| `host_id`          | `str`           | non-empty          | Simulated host identifier (e.g. `host-01`)       |
| `latency_ms`       | `float`         | ≥ 0                | Round-trip latency in milliseconds               |
| `packet_loss_pct`  | `float`         | [0, 100]           | Packet loss as a percentage                      |
| `dns_failure_rate` | `float`         | [0, 1]             | Fraction of DNS queries that failed              |
| `jitter_ms`        | `float`         | ≥ 0                | Network jitter in milliseconds                   |
| `is_anomaly`       | `bool`          | —                  | True when an anomaly was injected                |
| `anomaly_type`     | `str \| None`   | SPIKE/LOSS/DNS/CASCADE | Present only when `is_anomaly=True`          |

### ProcessedRecord (backend/pipeline/preprocessor.py)

All TelemetryRecord fields (post-clipping) plus:

| Field                   | Type    | Range   | Description                                                  |
|-------------------------|---------|---------|--------------------------------------------------------------|
| `latency_normalized`    | `float` | [0, 1]  | latency_ms min-max scaled over [0, 1000]                     |
| `loss_normalized`       | `float` | [0, 1]  | packet_loss_pct min-max scaled over [0, 100]                 |
| `dns_normalized`        | `float` | [0, 1]  | dns_failure_rate (already [0,1], min-max identity)           |
| `jitter_normalized`     | `float` | [0, 1]  | jitter_ms min-max scaled over [0, 200]                       |
| `composite_health_score`| `float` | [0, 1]  | Weighted avg: latency×0.4 + loss×0.3 + dns×0.2 + jitter×0.1 |
| `is_business_hours`     | `bool`  | —       | True when record UTC time is Mon–Fri 08:00–17:59             |

### FeatureVector (backend/models/feature_engineering.py)

Output of `FeatureExtractor.process()`.  Requires a full 30-record window per host.

**Statistical features (11)** — computed over the 30-record sliding window:

| Field                  | Type    | Description                                                  |
|------------------------|---------|--------------------------------------------------------------|
| `rolling_mean_latency` | `float` | Mean of `latency_normalized` over window                     |
| `rolling_std_latency`  | `float` | Std-dev of `latency_normalized` (ddof=1)                     |
| `rolling_mean_loss`    | `float` | Mean of `loss_normalized`                                    |
| `rolling_std_loss`     | `float` | Std-dev of `loss_normalized` (ddof=1)                        |
| `rolling_mean_dns`     | `float` | Mean of `dns_normalized`                                     |
| `rolling_mean_jitter`  | `float` | Mean of `jitter_normalized`                                  |
| `rolling_std_jitter`   | `float` | Std-dev of `jitter_normalized` (ddof=1)                      |
| `latency_trend`        | `float` | Linear regression slope of `latency_normalized` (+= worsening) |
| `health_score_trend`   | `float` | Linear regression slope of `composite_health_score`          |
| `spike_count`          | `float` | Records in window where `latency_normalized` > 0.7           |
| `loss_spike_count`     | `float` | Records in window where `loss_normalized` > 0.5              |

**Rate-of-change features (3)** — last two records:

| Field            | Type    | Description                                       |
|------------------|---------|---------------------------------------------------|
| `latency_delta`  | `float` | `latency_normalized[-1] − latency_normalized[-2]` |
| `loss_delta`     | `float` | `loss_normalized[-1] − loss_normalized[-2]`       |
| `dns_delta`      | `float` | `dns_normalized[-1] − dns_normalized[-2]`         |

**Temporal features (5)**:

| Field                | Type    | Description                                          |
|----------------------|---------|------------------------------------------------------|
| `hour_sin`           | `float` | `sin(2π × UTC_hour / 24)`                           |
| `hour_cos`           | `float` | `cos(2π × UTC_hour / 24)`                           |
| `day_sin`            | `float` | `sin(2π × weekday / 7)` (Mon=0)                     |
| `day_cos`            | `float` | `cos(2π × weekday / 7)`                             |
| `is_business_hours`  | `float` | `1.0` if Mon–Fri 08:00–17:59 UTC, else `0.0`        |

**Output shapes**:

| Method                          | Shape     | Use case               |
|---------------------------------|-----------|------------------------|
| `to_lstm_input()`               | `(30, 4)` | LSTM autoencoder input |
| `to_isolation_forest_input()`   | `(19,)`   | Isolation Forest input |

---

### AnomalyResult (backend/models/isolation_forest.py)

Output of `IsolationForestDetector.predict()` and `predict_batch()`.

| Field                       | Type          | Description                                                           |
|-----------------------------|---------------|-----------------------------------------------------------------------|
| `is_anomaly`                | `bool`        | True when raw IF prediction == -1                                     |
| `anomaly_score`             | `float`       | Normalised score [0, 1] — 0 = normal, 1 = certain anomaly            |
| `raw_score`                 | `float`       | Raw `decision_function` output (negative = anomalous)                 |
| `confidence`                | `float`       | `abs(anomaly_score − 0.5) × 2` — 0 = at boundary, 1 = far away       |
| `top_contributing_features` | `list[str]`   | Top-3 feature names most deviant from training-set means              |
| `host_id`                   | `str`         | Forwarded from input FeatureVector                                    |
| `timestamp`                 | `str`         | Forwarded from input FeatureVector                                    |
| `model_version`             | `str`         | e.g. `"1.0.0-20260401-120000"`                                        |
| `inference_time_ms`         | `float`       | Wall-clock milliseconds for this prediction                           |

### TrainingResult (backend/models/isolation_forest.py)

| Field                        | Type          | Description                                               |
|------------------------------|---------------|-----------------------------------------------------------|
| `n_samples`                  | `int`         | Training set size                                         |
| `contamination`              | `float`       | Expected anomaly fraction used for threshold              |
| `training_anomaly_rate`      | `float`       | Fraction of training samples flagged by fitted model      |
| `feature_importance`         | `list[float]` | Per-feature std-dev across training data (proxy importance) |
| `feature_names`              | `list[str]`   | Ordered feature names matching `feature_importance`       |
| `training_duration_seconds`  | `float`       | Wall-clock training time                                  |
| `model_version`              | `str`         | Version string                                            |
| `training_date`              | `str`         | UTC ISO 8601 timestamp                                    |

### Model Artifact (`backend/models/artifacts/isolation_forest.joblib`)

Saved by `IsolationForestDetector.save()`.  Contains: `model`, `metadata`,
`score_min`, `score_max`, `training_mean`.  Load with
`IsolationForestDetector.load()`.

---

### LSTMResult (backend/models/lstm_model.py)

Output of `LSTMDetector.predict()` and `predict_batch()`.

| Field                  | Type              | Description                                                           |
|------------------------|-------------------|-----------------------------------------------------------------------|
| `is_anomaly`           | `bool`            | True when reconstruction error exceeds the 95th-percentile threshold  |
| `anomaly_score`        | `float`           | Z-score normalised to [0, 1] via `clip((z+3)/6, 0, 1)`               |
| `reconstruction_error` | `float`           | Raw mean MSE across all 30 timesteps and 4 features                   |
| `threshold_used`       | `float`           | 95th-percentile threshold from training (fixed at train time)         |
| `per_feature_errors`   | `dict[str,float]` | MSE per feature: `latency`, `loss`, `dns`, `jitter`                   |
| `worst_feature`        | `str`             | Feature name with the highest individual reconstruction error          |
| `inference_time_ms`    | `float`           | Wall-clock milliseconds for the forward pass                          |
| `model_version`        | `str`             | Version string e.g. `"1.0.0-20260401-120000"`                         |

### LSTMTrainingResult (backend/models/lstm_model.py)

| Field                        | Type          | Description                                                   |
|------------------------------|---------------|---------------------------------------------------------------|
| `n_samples_train`            | `int`         | Normal windows in the training split                          |
| `n_samples_val`              | `int`         | Normal windows in the validation split                        |
| `best_val_loss`              | `float`       | Lowest validation MSE (best checkpoint)                       |
| `epochs_trained`             | `int`         | Actual epochs before early stopping                           |
| `threshold`                  | `float`       | 95th-percentile reconstruction error on full training set     |
| `error_mean`                 | `float`       | Mean error on training set (for z-score normalisation)        |
| `error_std`                  | `float`       | Std-dev of error on training set                              |
| `training_duration_seconds`  | `float`       | Wall-clock training time                                      |
| `model_version`              | `str`         | Version string                                                |
| `device_used`                | `str`         | `"cuda"` or `"cpu"`                                           |
| `train_losses`               | `list[float]` | Per-epoch training MSE                                        |
| `val_losses`                 | `list[float]` | Per-epoch validation MSE                                      |

### CombinedAnomalyResult (backend/models/lstm_model.py)

Output of `AnomalyDetector.detect()` — ensemble result fusing LSTM + Isolation Forest.

| Field                       | Type           | Description                                                         |
|-----------------------------|----------------|---------------------------------------------------------------------|
| `is_anomaly`                | `bool`         | `combined_score > 0.5` OR `(lstm.is_anomaly AND if.is_anomaly)`     |
| `combined_score`            | `float`        | Weighted average: LSTM×0.6 + IF×0.4                                 |
| `severity`                  | `str`          | `critical` (>0.8) / `high` (>0.6) / `medium` (>0.4) / `low`        |
| `lstm_result`               | `LSTMResult`   | Full LSTM detector output                                           |
| `if_result`                 | `AnomalyResult`| Full Isolation Forest output                                        |
| `detection_method`          | `str`          | `"lstm+if"` / `"lstm_only"` / `"if_only"` / `"none"`               |
| `worst_feature`             | `str`          | Highest-error feature from LSTM per-feature breakdown               |
| `top_contributing_features` | `list[str]`    | Top-3 features from IF deviation analysis                           |
| `timestamp`                 | `datetime`     | UTC datetime parsed from the input FeatureVector                    |

### LSTM Artifact (`backend/models/artifacts/lstm_model.pt`)

Saved by `LSTMTrainer._save_artifact()`.  Single `torch.save` dict containing:
`scripted_bytes` (TorchScript model), `threshold`, `error_mean`, `error_std`,
`model_version`, `training_date`.  Load with `LSTMDetector.load()`.

---

### Anomaly Baselines (producer simulation)

| Anomaly Type | Affected Metric(s)    | Injected Range          |
|--------------|-----------------------|-------------------------|
| SPIKE        | latency_ms            | 300 – 800 ms            |
| LOSS         | packet_loss_pct       | 15 – 40 %               |
| DNS          | dns_failure_rate      | 0.4 – 0.9               |
| CASCADE      | all four metrics      | All ranges above + jitter 50–150 ms |

---

## Notes for Claude Code

- Always check this file first before writing new code
- When adding a new module, update the Project Structure section above
- When a phase is completed, check it off in the Current Status section
- When a major architectural decision is made, log it in Key Decisions Log
- The `.env.example` file must be kept in sync with all new env vars added
