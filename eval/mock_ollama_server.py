#!/usr/bin/env python3
"""
Tiny mock LLM server for offline harness testing.

Implements just enough of the llama.cpp llama-server API
(POST /v1/chat/completions) — and, for backwards compatibility, the Ollama API
(POST /api/generate) — to let evaluate.py run end-to-end against
deterministic canned responses.
Usage:  venv/bin/python eval/mock_ollama_server.py --port 11500
"""
import argparse, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARTISTS = json.dumps(["Sufjan Stevens", "James Blake", "Grouper", "Nicolas Jaar",
                      "Arca", "Shamir", "Carpenter Brut", "Lido"])
ALBUMS = json.dumps([
    {"artist": "Sufjan Stevens", "album": "Carrie & Lowell", "confidence": 0.9, "reason": "fits"},
    {"artist": "James Blake", "album": "overgrown", "confidence": 0.8, "reason": "fits"},
    {"artist": "Grouper", "album": "Beeswax", "confidence": 0.7, "reason": "fits"},
    {"artist": "Nicolas Jaar", "album": "Space Is Only Noise", "confidence": 0.6, "reason": "fits"},
    {"artist": "Arca", "album": "Mulatta", "confidence": 0.7, "reason": "fits"},
    {"artist": "Shamir", "album": "Nervous", "confidence": 0.6, "reason": "fits"},
    {"artist": "Carpenter Brut", "album": "Surgery", "confidence": 0.8, "reason": "fits"},
    {"artist": "Lido", "album": "The New World Order", "confidence": 0.7, "reason": "fits"},
])
RETRY = json.dumps([
    {"artist": "Nicolas Jaar", "album": "Cumbia", "confidence": 0.6, "reason": "fits"}
])


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _canned_response(self, prompt: str) -> str:
        if "In your previous response" in prompt:   # retry (both agents)
            return RETRY
        if "recommend exactly 8 real albums" in prompt:   # album-first one-shot
            return ALBUMS
        if "suggest 8" in prompt or "NOT already listened" in prompt:   # artist-then-album call 1
            return ARTISTS
        return ALBUMS   # artist-then-album call 2

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        path = self.path.rstrip("/")

        if path.endswith("/v1/chat/completions"):
            # llama.cpp llama-server (OpenAI-compatible)
            messages = req.get("messages") or []
            prompt = messages[-1].get("content", "") if messages else ""
            self._send({
                "id": "mock", "object": "chat.completion", "model": req.get("model", "mock"),
                "choices": [{"index": 0, "message": {"role": "assistant",
                              "content": self._canned_response(prompt)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50},
            })
        else:
            # Ollama (kept for backwards compatibility)
            self._send({"response": self._canned_response(req.get("prompt", "")),
                        "eval_count": 10, "prompt_eval_count": 40, "model": "mock"})

    def log_message(self, *a):  # keep output clean
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11500)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"mock ollama on http://127.0.0.1:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
