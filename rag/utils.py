import hashlib
import json
import os
from datetime import datetime
from typing import Any


def uid(filepath: str) -> str:
    """Generate a unique ID for a document based on filepath and timestamp"""
    timestamp = datetime.now().isoformat()
    content = f"{filepath}_{timestamp}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def save_json(data: Any, filepath: str):
    """Save data as JSON"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Any:
    """Load JSON data"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def audit_log(action: str, doc_id: str, details: dict = None):
    """Log actions for audit purposes"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "doc_id": doc_id,
        "details": details or {}
    }

    log_file = "storage/logs/audit.jsonl"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')