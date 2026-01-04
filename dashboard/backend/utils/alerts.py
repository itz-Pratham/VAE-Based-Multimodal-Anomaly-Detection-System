import requests
import time
from ..schemas import InputData

def trigger_webhook(webhook_url: str, data: InputData, score: float):
    """Sends a JSON payload to a configured URL when an anomaly is found"""
    if not webhook_url:
        print("⚠️ Anomaly detected, but no WEBHOOK_URL set.")
        return
    
    payload = {
        "alert": "Critical Anomaly Detected",
        "score": round(score, 4),
        "timestamp": data.timestamp or time.time(),
        "values": data.values,
        "message": "System detected a deviation exceeding calibrated threshold."
    }
    
    try:
        requests.post(webhook_url, json=payload, timeout=2)
        print("✅ Webhook alert sent.")
    except Exception as e:
        print(f"❌ Failed to send webhook: {e}")