# Telemetry route handlers.
# POST /telemetry/ingest  — accepts a raw telemetry payload (latency, packet_loss,
#                           dns_failure_rate, jitter), validates with Pydantic, writes to DB.
# GET  /telemetry/recent  — returns the last N minutes of telemetry records from TimescaleDB.
