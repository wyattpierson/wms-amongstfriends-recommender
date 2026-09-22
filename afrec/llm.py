"""Pluggable LLM client.

The pipeline (afrec.pipeline) only ever talks to an LLM through the small
`BaseLLM` interface in this module. Two backends are built in:

  - `LlamaCPPLM`  — a local llama.cpp `llama-server` (default), which exposes
                    an OpenAI-compatible `/v1/chat/completions` endpoint
  - `OllamaLM`    — a local Ollama server (`/api/generate`)

`make_llm()` picks the backend from the `LLM_BACKEND` env var and hands you an
instance configured from `LLM_URL` / `LLM_MODEL`. To try yet another backend
(vLLM, LM Studio, …), implement `generate()` with the same contract and hand
your instance to `pipeline.run()` / the evaluator. Nothing downstream cares
which backend runs.

Note on JSON parsing: `generate_json()` is deliberately *tolerant* — it strips
code fences, then falls back to extracting the outermost JSON array/object from
the raw text. Smaller local models tend to add a sentence of preamble or wrap
output in ```json fences; the tolerant parser scores model *quality*, not
formatting quirks.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import requests


# ── Errors ────────────────────────────────────────────────────────────────────

class LLMError(RuntimeError):
    """Base error for any LLM backend failure."""


class LLMConnectionError(LLMError):
    """Server unreachable."""


class LLMTimeoutError(LLMError):
    """Request exceeded its timeout."""


class LLMHTTPError(LLMError):
    """Server responded with a non-2xx status (e.g. model not pulled)."""

    def __init__(self, message: str, status_code: int, body: str):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class ResponseParseError(LLMError):
    """The model responded, but it wasn't usable JSON."""

    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


# ── Metadata ──────────────────────────────────────────────────────────────────

@dataclass
class GenerationMeta:
    name: str                       # e.g. "ollama:llama3.1"
    latency_s: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


# ── Parsing helpers ───────────────────────────────────────────────────────────

def strip_code_fences(raw: str) -> str:
    """Remove a leading/trailing ``` fence block if present."""
    raw = raw.strip()
    if not raw.startswith("```"):
        return raw
    lines = raw.split("\n")
    body = "\n".join(lines[1:])
    if body.strip().endswith("```"):
        body = body[: body.rfind("```")]
    return body.strip()


def _extract_json(raw: str):
    """Parse JSON, tolerating fences and prose around the actual JSON payload."""
    cleaned = strip_code_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fallback: grab the outermost JSON array or object in the text
    for opener, closer in (("[", "]"), ("{", "}")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("could not find a valid JSON value in the response")


# ── Interface ─────────────────────────────────────────────────────────────────

class BaseLLM:
    """Minimal LLM interface used by the pipeline and the evaluator.

    Subclasses must implement `generate()` returning the raw text response.
    `generate_json()` on top of it handles fence-stripping and validation.
    """

    name: str = "base"

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        num_predict: int | None = None,
        options: dict | None = None,
        timeout: float | None = None,
    ) -> tuple[str, GenerationMeta]:
        """Run the prompt, return (raw_text, meta). Raise LLMError on failure."""
        raise NotImplementedError

    def generate_json(
        self,
        prompt: str,
        *,
        expected: type = list,
        **kwargs,
    ) -> tuple[Any, str, GenerationMeta]:
        """Run the prompt and parse the response as JSON.

        Returns (parsed, raw_text, meta).
        Raises ResponseParseError if the response is not parseable or is not
        a JSON value of `expected` (use list, dict, or a type tuple via
        `expected_type` — the simple `type` check below uses isinstance).
        """
        raw, meta = self.generate(prompt, **kwargs)
        try:
            parsed = _extract_json(raw)
        except (json.JSONDecodeError, ValueError) as e:
            raise ResponseParseError(str(e), raw) from e
        if expected is not Any and not isinstance(parsed, expected):
            raise ResponseParseError(
                f"Expected a JSON {expected.__name__}, got {type(parsed).__name__}", raw
            )
        return parsed, raw, meta


# ── Ollama ────────────────────────────────────────────────────────────────────

def ollama_base_url(url: str) -> str:
    """Normalize a configured OLLAMA_URL (full generate URL or bare host) to a base URL."""
    u = url.strip()
    for suffix in ("/api/generate", "/api/chat", "/"):
        if u.endswith(suffix):
            u = u[: -len(suffix)]
    return u.rstrip("/")


class OllamaLM(BaseLLM):
    """Client for a local Ollama server (POST {base}/api/generate)."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 180,
    ):
        self.model = model
        self.base_url = ollama_base_url(base_url)
        self.default_timeout = timeout
        self.name = f"ollama:{model}"

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        num_predict: int | None = None,
        options: dict | None = None,
        timeout: float | None = None,
    ) -> tuple[str, GenerationMeta]:
        opts: dict[str, Any] = {"temperature": temperature}
        if num_predict is not None:
            opts["num_predict"] = num_predict
        opts.update(options or {})

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": opts,
        }

        t0 = time.perf_counter()
        url = f"{self.base_url}/api/generate"
        try:
            resp = requests.post(url, json=payload, timeout=timeout or self.default_timeout)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(
                f"Cannot reach Ollama at {self.base_url}. Is it running? (try: ollama serve)"
            ) from e
        except requests.exceptions.Timeout as e:
            raise LLMTimeoutError(f"Ollama request timed out after {timeout or self.default_timeout}s") from e
        except requests.exceptions.HTTPError as e:
            raise LLMHTTPError(
                f"Ollama returned HTTP {resp.status_code} for model '{self.model}'",
                resp.status_code,
                resp.text,
            ) from e

        data = resp.json()
        meta = GenerationMeta(
            name=self.name,
            latency_s=time.perf_counter() - t0,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )
        return (data.get("response") or "").strip(), meta


def list_ollama_models(base_url: str = "http://localhost:11434", timeout: float = 5) -> list[str]:
    """Convenience: list model names available on the server."""
    resp = requests.get(ollama_base_url(base_url) + "/api/tags", timeout=timeout)
    resp.raise_for_status()
    return [m.get("name", "") for m in resp.json().get("models", [])]


# ── llama.cpp llama-server ────────────────────────────────────────────────────

class LlamaCPPLM(BaseLLM):
    """Client for a local llama.cpp `llama-server` (POST {base}/v1/chat/completions).

    llama-server exposes an OpenAI-compatible API on its main port (8080 by
    default; we default to 8092). The `model` field is mostly cosmetic —
    llama-server just serves whatever GGUF you loaded, so any name works.
    """

    def __init__(
        self,
        model: str = "local-model",
        base_url: str = "http://localhost:8092",
        timeout: float = 180,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.default_timeout = timeout
        self.name = f"llama.cpp:{model}"

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        num_predict: int | None = None,
        options: dict | None = None,
        timeout: float | None = None,
    ) -> tuple[str, GenerationMeta]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": temperature,
        }
        if num_predict is not None:
            payload["max_tokens"] = num_predict
        # Extra options map onto OpenAI-style completion params (e.g. top_p).
        payload.update(options or {})

        t0 = time.perf_counter()
        url = f"{self.base_url}/v1/chat/completions"
        try:
            resp = requests.post(url, json=payload, timeout=timeout or self.default_timeout)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(
                f"Cannot reach llama-server at {self.base_url}. Is it running? "
                "(try: llama-server -m <your-model.gguf> --port 8092)"
            ) from e
        except requests.exceptions.Timeout as e:
            raise LLMTimeoutError(f"llama-server request timed out after {timeout or self.default_timeout}s") from e
        except requests.exceptions.HTTPError as e:
            raise LLMHTTPError(
                f"llama-server returned HTTP {resp.status_code}",
                resp.status_code,
                resp.text,
            ) from e

        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected llama-server response shape: {resp.text[:500]}") from e

        usage = data.get("usage") or {}
        meta = GenerationMeta(
            name=self.name,
            latency_s=time.perf_counter() - t0,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
        return (content or "").strip(), meta


# ── Factory ───────────────────────────────────────────────────────────────────

def make_llm(
    model: str | None = None,
    base_url: str | None = None,
    backend: str | None = None,
) -> BaseLLM:
    """Build an LLM client from env config.

    Env vars (all optional):
      LLM_BACKEND — "llama.cpp" (default) or "ollama"
      LLM_URL     — server base URL (falls back to OLLAMA_URL, then the
                    per-backend default: localhost:8092 / localhost:11434)
      LLM_MODEL   — model name (falls back to OLLAMA_MODEL, then "llama3")
    """
    backend = (backend or os.getenv("LLM_BACKEND", "llama.cpp")).strip().lower()
    model = model or os.getenv("LLM_MODEL") or os.getenv("OLLAMA_MODEL") or "llama3"
    base_url = base_url or os.getenv("LLM_URL") or os.getenv("OLLAMA_URL")

    if backend in ("llama.cpp", "llamacpp", "llama-server", "llama_server"):
        return LlamaCPPLM(model, base_url or "http://localhost:8092")
    if backend in ("ollama",):
        return OllamaLM(model, base_url or "http://localhost:11434")
    raise ValueError(f"Unknown LLM_BACKEND: '{backend}' (expected 'llama.cpp' or 'ollama')")
