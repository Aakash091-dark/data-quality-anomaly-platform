import logging
import requests
import os
from typing import Dict, Any

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alerting")

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

def send_alert(title: str, message: str, severity: str = "INFO", details: Dict[str, Any] = None):
    """
    Send alert to configured channels.
    Severity: INFO, WARNING, CRITICAL (P0-P3 mapping)
    """
    
    # 1. Log to console
    log_msg = f"[{severity}] {title}: {message}"
    print(f"\n>>> ALERT: {log_msg}") # Ensure visibility in demo
    
    if details:
        print(f"Details: {details}")

    if severity == "CRITICAL":
        logger.error(log_msg)
    else:
        logger.info(log_msg)
        
    # 2. Slack Alert (if configured)
    if SLACK_WEBHOOK_URL:
        payload = {
            "text": f"*{severity}* - {title}\n{message}\nDetails: {details}"
        }
        try:
            requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

def check_and_alert(anomaly_result: Dict[str, Any], context: str = "Batch"):
    """
    Helper to check anomaly result and trigger alert if needed.
    """
    if anomaly_result.get("prediction") == "anomaly":
        severity = "CRITICAL"
        title = f"Anomaly Detected in {context}"
        message = "Model detected anomalous data pattern."
        details = {
            "scores": anomaly_result.get("scores"),
            "threshold": anomaly_result.get("threshold")
        }
        send_alert(title, message, severity, details)
