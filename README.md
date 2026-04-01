# FlowWatch AI

Real-time network monitoring and anomaly detection system powered by ML and an LLM-based root cause analysis assistant.

## Overview

FlowWatch AI ingests live network telemetry (latency, packet loss, DNS failures, jitter) through a streaming pipeline, runs ML-based anomaly detection using both an LSTM autoencoder (temporal patterns) and an Isolation Forest (snapshot-based), exposes a FastAPI inference layer, and provides an Anthropic Claude-powered assistant for root cause analysis and alerting.

## Architecture

```
[Network Agents / Simulators]
        |
        v
[AWS Kinesis Stream]  ← latency, packet_loss, dns_failures, jitter
        |
        v
[Kinesis Consumer + Preprocessor]
        |
       / \
      v   v
[LSTM Model]  [Isolation Forest]
      \   /
       v v
[Anomaly Score Aggregator]
        |
        v
[FastAPI Inference API]  ↔  [Claude RCA Assistant]
        |
        v
[Alert Manager]  →  [AWS CloudWatch / Notifications]
        |
        v
[TimescaleDB]  ↔  [Next.js Dashboard]
```

## Tech Stack

| Layer        | Technology                                   |
|--------------|----------------------------------------------|
| Backend API  | Python 3.11+, FastAPI, Pydantic v2           |
| ML           | PyTorch (LSTM), Scikit-learn (Isolation Forest) |
| LLM          | Anthropic Claude API (`claude-sonnet-4-20250514`) |
| Streaming    | AWS Kinesis (LocalStack for local dev)       |
| Database     | TimescaleDB (PostgreSQL extension)           |
| Cache        | Redis                                        |
| Frontend     | Next.js 14+, TypeScript, Tailwind, Recharts  |
| Infra        | Docker Compose (local), AWS EC2 + CloudWatch |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Node.js 18+
- An Anthropic API key

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in ANTHROPIC_API_KEY and other values
```

### 2. Start full stack (Docker)

```bash
docker-compose -f infra/docker-compose.yml up --build
```

Services will be available at:
- **Backend API**: http://localhost:8000
- **Frontend Dashboard**: http://localhost:3000
- **API Docs (Swagger)**: http://localhost:8000/docs

### 3. Run backend only (local dev)

```bash
cd backend
pip install -r requirements.txt
uvicorn api.main:app --reload
```

### 4. Run frontend only (local dev)

```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

| Method | Endpoint               | Description                           |
|--------|------------------------|---------------------------------------|
| POST   | /telemetry/ingest      | Ingest a raw telemetry data point     |
| GET    | /telemetry/recent      | Fetch recent telemetry (last N mins)  |
| GET    | /anomalies/latest      | Get latest detected anomalies         |
| POST   | /anomalies/detect      | Run on-demand anomaly detection       |
| POST   | /assistant/analyze     | LLM root cause analysis of an anomaly |
| GET    | /health                | Health check                          |

## Project Structure

See [CLAUDE.md](CLAUDE.md) for the full project structure and architectural decisions.

## Build Phases

- [ ] Phase 1: Project scaffolding
- [ ] Phase 2: Telemetry simulator + Kinesis consumer
- [ ] Phase 3: Feature engineering pipeline
- [ ] Phase 4: Isolation Forest model
- [ ] Phase 5: LSTM model + training notebook
- [ ] Phase 6: FastAPI inference endpoints
- [ ] Phase 7: LLM RCA assistant
- [ ] Phase 8: Alert manager + CloudWatch
- [ ] Phase 9: Next.js frontend dashboard
- [ ] Phase 10: Docker Compose full-stack wiring
- [ ] Phase 11: AWS deployment

## License

MIT
