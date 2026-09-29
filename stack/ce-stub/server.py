"""Stub CE (manual RAG) backend for local router testing.

Not a real RAG: it streams a canned answer that echoes the query, in the
same NDJSON event format the gateway proxies (token events, then done).
That is enough for the gateway to save the turn to conversation history,
so follow-up queries can be routed with history in a live test.
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 7860


class Handler(BaseHTTPRequestHandler):
    def _json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "healthy", "stub": True})
        elif self.path == "/manuals":
            self._json(200, [])
        else:
            self._json(404, {"error": "not found (stub CE)"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            query = json.loads(raw).get("query", "")
        except json.JSONDecodeError:
            query = ""

        if self.path != "/chat":
            self._json(200, {"ok": True})
            return

        # No Content-Length: HTTP/1.0 semantics, body ends when we close.
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for event in (
            {"type": "token", "text": f"[CE stub] routed here. Query: {query}"},
            {"type": "done"},
        ):
            self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()

    def log_message(self, fmt, *args):
        print("ce-stub", fmt % args, flush=True)


if __name__ == "__main__":
    print(f"ce-stub listening on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
