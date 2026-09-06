from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parents[1]
request_path = project_root / "requests" / "smoke_request.json"

with request_path.open("r", encoding="utf-8") as file:
    request_body = json.load(file)

response = httpx.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    json=request_body,
    timeout=60,
)
response.raise_for_status()
print(json.dumps(response.json(), ensure_ascii=False, indent=2))


