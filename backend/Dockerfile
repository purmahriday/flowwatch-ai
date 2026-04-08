FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make "from backend.xxx" imports work inside Docker.
# Build context is ../backend, so code lands at /app directly (no backend/ prefix).
# This symlink makes /app/backend point to /app itself, so both import styles resolve:
#   from backend.api.main import app  →  /app/backend/api/main.py  →  /app/api/main.py ✓
#   from api.main import app          →  /app/api/main.py                              ✓
RUN ln -s /app /app/backend

RUN mkdir -p models/artifacts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
