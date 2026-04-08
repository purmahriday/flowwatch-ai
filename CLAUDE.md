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

| Layer       | Technology                                              |
|-------------|---------------------------------------------------------|
| Backend     | Python 3.11+, FastAPI (async), asyncpg, loguru          |
| ML          | PyTorch (LSTM autoencoder), Scikit-learn (Isolation Forest) |
| Pipeline    | AWS Kinesis / LocalStack, Pydantic v2                   |
| LLM         | Anthropic Claude API (`claude-sonnet-4-20250514`)        |
| Database    | TimescaleDB (asyncpg) + in-memory deque fallback        |
| Cache       | Redis                                                   |
| Frontend    | Next.js 14+ (App Router), TypeScript, Tailwind, Recharts |
| Infra       | Docker Compose, AWS EC2/Kinesis/CloudWatch (prod)       |

---

## Project Structure

```
flowwatch-ai/
├── backend/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── telemetry.py    # Ingest & query telemetry endpoints
│   │   │   ├── anomalies.py    # Anomaly detection endpoints
│   │   │   └── assistant.py    # LLM-based RCA assistant endpoints
│   │   ├── main.py             # FastAPI app entrypoint + lifespan
│   │   ├── dependencies.py     # verify_api_key, get_anomaly_detector, get_feature_extractor
│   │   └── schemas.py          # All Pydantic request/response schemas
│   ├── models/
│   │   ├── lstm_model.py       # PyTorch LSTM anomaly detection model
│   │   ├── isolation_forest.py # Scikit-learn Isolation Forest model
│   │   └── feature_engineering.py
│   ├── pipeline/
│   │   ├── kinesis_consumer.py
│   │   ├── kinesis_producer.py
│   │   └── preprocessor.py
│   ├── assistant/
│   │   └── rca_agent.py        # Claude API integration for RCA
│   ├── alerting/
│   │   └── alert_manager.py    # Alerting logic + CloudWatch integration
│   ├── db/
│   │   └── timeseries.py       # asyncpg pool, init_db, insert_telemetry, etc.
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/                # Next.js App Router pages
│   │   ├── components/
│   │   │   ├── dashboard/      # Dashboard-specific components
│   │   │   └── ui/             # Reusable UI components
│   │   └── lib/                # API clients (api.ts), utilities
│   ├── Dockerfile
│   └── package.json
├── infra/
│   ├── docker-compose.yml          # Full-stack local orchestration
│   ├── docker-compose.override.yml # Dev hot-reload overrides
│   └── docker-compose.prod.yml
├── scripts/
│   └── init_kinesis.py         # Create Kinesis stream in LocalStack
├── notebooks/
├── data/
└── .env.example
```

---

## Development Commands

### Backend (local, no Docker)
```bash
cd backend
pip install -r requirements.txt
uvicorn backend.api.main:app --reload   # localhost:8000
pytest tests/
```

### Frontend (local, no Docker)
```bash
cd frontend
npm install
npm run dev    # localhost:3000
```

### Docker — full stack
```bash
# Start everything (uses docker-compose.override.yml for hot reload automatically)
docker-compose -f infra/docker-compose.yml up --build

# Init Kinesis stream after LocalStack is up
python scripts/init_kinesis.py

# Stop
docker-compose -f infra/docker-compose.yml down
```

---

## Docker Services

| Service      | Image / Build         | Port | Notes                              |
|--------------|-----------------------|------|------------------------------------|
| timescaledb  | timescale/timescaledb | 5432 | Hypertables for telemetry + anomaly|
| redis        | redis:7-alpine        | 6379 | Cache / deduplication              |
| localstack   | localstack/localstack | 4566 | Kinesis simulation                 |
| backend      | ./backend             | 8000 | FastAPI + ML models                |
| frontend     | ./frontend            | 3000 | Next.js dashboard                  |
| producer     | ./backend (commented) | —    | Optional telemetry simulator       |

---

## Environment Variables

Copy `.env.example` to `.env`. Never commit `.env`.

```
ANTHROPIC_API_KEY=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
AWS_ENDPOINT_URL=http://localstack:4566   # Docker; http://localhost:4566 locally
KINESIS_STREAM_NAME=flowwatch-telemetry
DATABASE_URL=postgresql://flowwatch:flowwatch@timescaledb:5432/flowwatch
REDIS_URL=redis://redis:6379
ENVIRONMENT=development
API_KEYS=flowwatch-dev-key-001
LOG_LEVEL=INFO
```

---

## Architecture

```
[Network Simulators / Agents]
        |
        v
[AWS Kinesis / LocalStack]
        |
        v
[Kinesis Consumer → Preprocessor → Feature Engineering]
        |
       / \
      v   v
[LSTM×0.6] [Isolation Forest×0.4]
      \   /
       v v
[Anomaly Score Aggregator]  ─→  [Alert Manager] ─→ [CloudWatch]
        |
        v
[FastAPI API]  ←→  [LLM RCA Assistant (Claude)]
        |
        v
[TimescaleDB]  ←→  [Next.js Dashboard]
```

---

## ML Models

### LSTM (PyTorch)
- Input: `(30, 4)` sliding window — latency, loss, dns, jitter (normalized)
- Output: reconstruction error → anomaly if error > 95th-percentile threshold
- Artifact: `backend/models/artifacts/lstm_model.pt` (TorchScript)

### Isolation Forest (Scikit-learn)
- Input: `(19,)` feature vector (statistical + rate-of-change + temporal)
- Output: anomaly score [0, 1]
- Artifact: `backend/models/artifacts/isolation_forest.joblib`

### Ensemble
- Combined score = LSTM×0.6 + IF×0.4
- Severity: critical >0.8 / high >0.6 / medium >0.4 / low otherwise

---

## API Endpoints

| Method | Endpoint             | Description                            |
|--------|----------------------|----------------------------------------|
| POST   | /telemetry/ingest    | Ingest raw telemetry data point        |
| GET    | /telemetry/recent    | Fetch recent telemetry (last N minutes)|
| GET    | /telemetry/hosts     | List active hosts with health snapshot |
| GET    | /anomalies/latest    | Get latest detected anomalies          |
| POST   | /anomalies/detect    | On-demand anomaly detection            |
| POST   | /assistant/analyze   | LLM root cause analysis                |
| POST   | /assistant/chat      | Conversational follow-up               |
| GET    | /alerts/recent       | Recent fired alerts                    |
| GET    | /alerts/stats        | Aggregate alert statistics             |
| GET    | /health              | Liveness probe (no auth)               |

Auth: `X-API-Key` header; keys from `API_KEYS` env var (comma-separated).

---

## Database Schema

### telemetry_records (hypertable on `timestamp`)
`host_id, timestamp, latency_ms, packet_loss_pct, dns_failure_rate, jitter_ms, health_score, is_anomaly, anomaly_score, severity`

### anomaly_events (hypertable on `timestamp`)
`host_id, timestamp, combined_score, severity, worst_feature, lstm_score, if_score, detection_method`

DB is optional — if `DATABASE_URL` is unset or unreachable, the backend falls back to in-memory deques automatically.

---

## In-Memory Stores (app.state)

| Store             | Type                        | Max per host |
|-------------------|-----------------------------|--------------|
| `telemetry_store` | `dict[host_id, deque]`      | 1 000        |
| `anomaly_store`   | `dict[host_id, deque]`      | 500          |
| `alert_manager`   | `AlertManager._alerts deque`| 1 000 total  |

---

## Code Conventions

### Python
- Python 3.11+ type hints everywhere
- `async/await` for all FastAPI handlers; `asyncio.to_thread` for CPU-bound ML
- Pydantic v2 for all request/response schemas
- Route handlers thin — logic in service/model layers
- `loguru` for structured logging

### TypeScript
- Strict TypeScript — no `any`
- Functional components with hooks
- Tailwind utility classes only
- API calls in `src/lib/api.ts`

---

## What to Avoid

- No hardcoded credentials or API keys
- No blocking the event loop in FastAPI routes
- No mixing training and inference logic in the same file
- No skipping Pydantic validation on incoming payloads

---

## Current Status

- [X] Phase 1: Project scaffolding
- [X] Phase 2: Telemetry simulator + Kinesis consumer
- [X] Phase 3: Feature engineering pipeline
- [X] Phase 4: Isolation Forest model
- [~] Phase 5: LSTM model (PyTorch) + training notebook
- [~] Phase 6: FastAPI inference endpoints
- [~] Phase 7: LLM RCA assistant (Claude API)
- [~] Phase 8: Alert manager + CloudWatch integration
- [~] Phase 9: Next.js frontend dashboard
- [~] Phase 10: Docker Compose full-stack wiring
- [ ] Phase 11: AWS deployment (EC2 + Kinesis + CloudWatch)

---

## Key Decisions

| Date       | Decision                                                | Reason                                          |
|------------|---------------------------------------------------------|-------------------------------------------------|
| 2026-04-01 | FastAPI over Flask                                      | Native async, better for streaming              |
| 2026-04-01 | LSTM + Isolation Forest ensemble                        | LSTM=temporal patterns, IF=snapshot outliers    |
| 2026-04-01 | Claude API for RCA                                      | Best-in-class log/telemetry analysis            |
| 2026-04-01 | LocalStack for local Kinesis                            | Avoid AWS costs during development              |
| 2026-04-05 | In-memory deques as primary store                       | No DB wired yet; DB is now additive (Phase 10)  |
| 2026-04-05 | X-API-Key auth                                          | Simple stateless; upgrade to JWT in Phase 11    |
| 2026-04-07 | AsyncAnthropic client in RCAAgent                       | Matches FastAPI async model                     |
| 2026-04-07 | AlertManager sync; callers use asyncio.to_thread        | boto3 is blocking                               |
| 2026-04-07 | CloudWatch dispatch disabled gracefully without creds   | Dev environment works without AWS               |
| 2026-04-07 | RCAPanel state lifted to page.tsx                       | AnomalyFeed and RCAPanel communicate via page   |
| 2026-04-07 | AlertFeed as primary view; anomalies as detail          | Alerts are the operator-facing surface          |
| 2026-04-07 | DB writes dual-path: memory (fast) + TimescaleDB (persist) | Survives restarts without slowing ingest    |
| 2026-04-07 | DB optional: graceful fallback to in-memory on failure  | Stack works without a running database          |

---

## Notes for Claude Code

- Always read this file before writing new code
- When adding a module, update the Project Structure section
- When a phase completes, check it off above
- When a major architectural decision is made, log it in Key Decisions
- Keep `.env.example` in sync with any new env vars
