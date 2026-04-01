# Real-time alerting logic and AWS CloudWatch integration.
# Evaluates anomaly scores against configurable thresholds, deduplicates alerts
# using Redis, publishes CloudWatch metric alarms, and (optionally) sends
# notifications via SNS or Slack webhooks.
