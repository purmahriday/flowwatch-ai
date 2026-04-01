# Utility script to export trained ML models for production inference.
# Converts the PyTorch LSTM to TorchScript and serialises the Isolation Forest
# with joblib. Outputs model artifacts to backend/models/artifacts/.
# Run: python scripts/export_model.py --model lstm|isolation_forest
