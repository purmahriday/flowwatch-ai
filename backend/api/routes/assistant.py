# LLM-based root cause analysis (RCA) assistant route handlers.
# POST /assistant/analyze — accepts an anomaly context payload and calls rca_agent.py
#                           which queries the Claude API; returns a structured RCA explanation.
