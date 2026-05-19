from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def _parse_retry_seconds(error_body: str, fallback: float = 10.0) -> float:
    """Pull the suggested wait time out of Groq's 429 error message."""
    match = re.search(r"try again in\s+([\d.]+)s", error_body)
    if match:
        return float(match.group(1)) + 0.5  # small buffer on top
    return fallback


def _build_request(payload: dict, key: str) -> urllib.request.Request:
    return urllib.request.Request(
        GROQ_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": "python-groq-client/1.0",
        },
        method="POST",
    )


def call_groq(
    prompt: str,
    model: str,
    timeout: int,
    api_key: str | None = None,
    max_retries: int = 8,
) -> str:
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError(
            "Groq API key not found. Set the GROQ_API_KEY environment variable or pass --groq-api-key."
        )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(_build_request(payload, key), timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
                return str(body["choices"][0]["message"]["content"]).strip()

        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace").strip()

            if exc.code == 429:
                wait = _parse_retry_seconds(error_body)
                print(f"  [rate limit] waiting {wait:.1f}s before retry {attempt}/{max_retries}...", flush=True)
                time.sleep(wait)
                continue

            raise RuntimeError(
                f"Groq returned HTTP {exc.code} for model `{model}`. "
                f"Response body: {error_body or '<empty>'}"
            ) from exc

        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not connect to Groq API: {exc.reason}"
            ) from exc

    raise RuntimeError(f"Groq rate limit not resolved after {max_retries} retries.")
