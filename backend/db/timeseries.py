# TimescaleDB (PostgreSQL) connection and time-series helpers.
# Manages async DB connection pool, exposes write_telemetry() for inserting
# new records and read_telemetry() for querying recent windows.
# Also stores detected anomaly events for historical review.
