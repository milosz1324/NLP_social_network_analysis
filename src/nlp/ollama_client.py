from __future__ import annotations

import json
import urllib.error
import urllib.request


def call_ollama(prompt: str, model: str, host: str, timeout: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_predict": 300,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Ollama returned HTTP {exc.code} for model `{model}`. "
            f"Response body: {error_body or '<empty>'}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not connect to Ollama. Start it with `ollama serve` and make sure "
            f"the model is available, e.g. `ollama pull {model}`."
        ) from exc

    return str(body.get("response", "")).strip()
