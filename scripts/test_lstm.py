import sys
sys.path.insert(0, '.')

from backend.models.lstm_model import LSTMAutoencoder, LSTMTrainer, LSTMDetector, AnomalyDetector
from backend.models.isolation_forest import generate_training_data, IsolationForestDetector

print('Generating training data...')
vectors = generate_training_data(n_samples=500)

print('Training LSTM for 20 epochs...')
model = LSTMAutoencoder()
trainer = LSTMTrainer(model, epochs=20)
result = trainer.train(vectors)

print()
print('═══════════════════════════════════════')
print('        TRAINING LOSS CURVE')
print('═══════════════════════════════════════')
print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Note'}")
print('-' * 55)

best = float('inf')
for i, (tl, vl) in enumerate(zip(result.train_losses, result.val_losses), 1):
    note = ''
    if vl < best:
        best = vl
        note = 'best'
    print(f'{i:<8} {tl:<15.8f} {vl:<15.8f} {note}')

print()
print(f'Epochs trained:    {result.epochs_trained}')
print(f'Best val loss:     {result.best_val_loss:.8f}')
print(f'Threshold:         {result.threshold:.8f}')
print(f'Duration:          {result.training_duration_seconds:.2f}s')
print(f'Device:            {result.device_used}')

print()
print('Testing single prediction...')
detector = LSTMDetector.load()
pred = detector.predict(vectors[0])
print(f'Is anomaly:        {pred.is_anomaly}')
print(f'Anomaly score:     {pred.anomaly_score:.3f}')
print(f'Reconstruction:    {pred.reconstruction_error:.8f}')
print(f'Worst feature:     {pred.worst_feature}')
print(f'Per feature:       {pred.per_feature_errors}')
print(f'Inference time:    {pred.inference_time_ms:.2f}ms')

print()
print('Testing combined detector (LSTM + IF)...')
combined = AnomalyDetector()
res = combined.detect(vectors[0])
print(f'Is anomaly:        {res.is_anomaly}')
print(f'Combined score:    {res.combined_score:.3f}')
print(f'Severity:          {res.severity}')
print(f'Detection method:  {res.detection_method}')
print(f'Worst feature:     {res.worst_feature}')

print()
print('ALL TESTS PASSED')