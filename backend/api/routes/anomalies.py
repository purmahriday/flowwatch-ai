# Anomaly detection route handlers.
# GET  /anomalies/latest  — returns the most recently detected anomalies from the DB.
# POST /anomalies/detect  — runs on-demand anomaly detection against a telemetry snapshot
#                           using the Isolation Forest and/or LSTM model.
