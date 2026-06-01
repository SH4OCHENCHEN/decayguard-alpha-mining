from __future__ import annotations

import json
import requests


def call_ollama(prompt: str, model: str = "qwen3:8b", url: str = "http://localhost:11434/api/generate", temperature: float = 0.7, timeout: int = 120) -> str:
    resp = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def parse_json_list(text: str) -> list[dict]:
    data = json.loads(text)
    if isinstance(data, dict) and "alphas" in data:
        data = data["alphas"]
    if not isinstance(data, list):
        raise ValueError("LLM response must be a JSON list or {'alphas': [...]} object")
    return data
