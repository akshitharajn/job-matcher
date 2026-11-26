"""Thin Groq client (OpenAI-compatible chat completions API) using plain httpx."""
import json
import time

import httpx

from . import config


class LLMError(Exception):
    pass


class GroqClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float = 30.0):
        self.api_key = api_key if api_key is not None else config.GROQ_API_KEY
        self.model = model or config.GROQ_MODEL
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat_json(self, system: str, user: str, retries: int = 2) -> dict:
        """Ask for a JSON object back. Retries on 429/5xx with backoff; raises LLMError otherwise."""
        if not self.available:
            raise LLMError("GROQ_API_KEY is not set")
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(retries + 1):
            try:
                r = httpx.post(config.GROQ_URL, json=payload, headers=headers, timeout=self.timeout)
            except httpx.HTTPError as e:
                if attempt == retries:
                    raise LLMError(f"network error: {e}") from e
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code in (429, 500, 502, 503) and attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code != 200:
                raise LLMError(f"Groq returned HTTP {r.status_code}")
            try:
                return json.loads(r.json()["choices"][0]["message"]["content"])
            except (KeyError, IndexError, ValueError) as e:
                raise LLMError("could not parse model output as JSON") from e
        raise LLMError("unreachable")
