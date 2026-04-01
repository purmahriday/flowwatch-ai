# Claude API integration for root cause analysis (RCA).
# Accepts an anomaly context dict (anomaly scores, telemetry snapshot, recent history),
# constructs a prompt, calls the Anthropic Claude API (claude-sonnet-4-20250514),
# and returns a structured RCA explanation with probable causes and remediation steps.
