"""Stdlib-only HTTP API and a single-file chat UI.

No framework, no build step, no node_modules. ``python3 tools/faqbot serve``
gives a working browser demo on a machine with nothing installed but Python,
which is the point: a buyer can see the thing work before installing anything.

Endpoints:

``GET  /``          the chat UI (one self-contained HTML page)
``GET  /health``    index stats and capability report
``POST /ask``       ``{"question": str, "session_id": str?, "k": int?}``
``POST /ingest``    ``{"paths": [str, ...]}``
``GET|POST /eval``  run a goldset and return the report

This is a demo server, not a production gateway. It has no authentication, no
rate limiting and no TLS; see the README's Limitations section.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .embedding import embedder_capabilities
from .eval import load_goldset, run_eval
from .guardrails import redact_pii
from .pipeline import FAQPipeline

__all__ = ["make_handler", "serve", "CHAT_HTML"]

MAX_BODY_BYTES = 1 << 20


CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>faqbot</title>
<style>
 :root { color-scheme: light dark; --bg:#101418; --fg:#e8edf2; --dim:#8b98a5;
         --card:#182028; --line:#26303a; --ok:#5ec27a; --warn:#e0a54a; }
 * { box-sizing: border-box; }
 body { margin:0; font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;
        background:var(--bg); color:var(--fg); }
 header { padding:14px 18px; border-bottom:1px solid var(--line); }
 header h1 { margin:0; font-size:15px; letter-spacing:.02em; }
 header p { margin:4px 0 0; color:var(--dim); font-size:12.5px; }
 main { max-width:820px; margin:0 auto; padding:18px; }
 .msg { margin:0 0 14px; padding:12px 14px; border-radius:10px;
        background:var(--card); border:1px solid var(--line); }
 .msg.you { background:transparent; border-style:dashed; color:var(--dim); }
 .meta { margin-top:9px; font-size:12px; color:var(--dim); }
 .badge { display:inline-block; padding:1px 7px; border-radius:20px;
          border:1px solid var(--line); margin-right:6px; }
 .refused .badge { color:var(--warn); border-color:var(--warn); }
 .src { font-size:12.5px; color:var(--dim); margin-top:7px; }
 .src code { color:var(--fg); }
 form { display:flex; gap:8px; margin-top:8px; }
 input { flex:1; padding:11px 13px; border-radius:9px; border:1px solid var(--line);
         background:var(--card); color:var(--fg); font-size:15px; }
 button { padding:11px 18px; border-radius:9px; border:1px solid var(--line);
          background:var(--card); color:var(--fg); cursor:pointer; font-size:15px; }
 button:hover { border-color:var(--ok); }
 .hint { color:var(--dim); font-size:12.5px; margin:10px 0 0; }
</style>
</head>
<body>
<header>
  <h1>faqbot &mdash; retrieval-grounded FAQ assistant</h1>
  <p id="stat">loading index&hellip;</p>
</header>
<main>
  <div id="log"></div>
  <form id="f">
    <input id="q" autocomplete="off" placeholder="Ask something about the indexed documents&hellip;">
    <button>Ask</button>
  </form>
  <p class="hint">Answers are extracted from the indexed documents and cited.
     When retrieval is weak, out of domain, ambiguous or self-contradictory,
     the bot refuses instead of guessing.</p>
</main>
<script>
const log = document.getElementById('log');
const session = 'web-' + Math.random().toString(36).slice(2, 10);

fetch('/health').then(r => r.json()).then(h => {
  document.getElementById('stat').textContent =
    h.chunks + ' chunks from ' + h.documents + ' documents | embedder: ' +
    h.embedder + ' | answerer: ' + h.answerer;
});

function esc(s) {
  return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}

function add(cls, html) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.innerHTML = html;
  log.appendChild(d);
  window.scrollTo(0, document.body.scrollHeight);
}

document.getElementById('f').addEventListener('submit', async ev => {
  ev.preventDefault();
  const box = document.getElementById('q');
  const question = box.value.trim();
  if (!question) return;
  box.value = '';
  add('you', esc(question));
  const res = await fetch('/ask', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question: question, session_id: session})
  });
  const a = await res.json();
  let html = esc(a.text);
  if (a.rewritten_question && a.rewritten_question !== a.question) {
    html += '<div class="src">resolved as: <code>' + esc(a.rewritten_question) + '</code></div>';
  }
  const seen = [];
  (a.citations || []).forEach(c => {
    if (seen.indexOf(c.marker) >= 0) return;
    seen.push(c.marker);
    html += '<div class="src">[' + c.marker + '] <code>' + esc(c.label) +
            '</code> &mdash; ' + esc(c.source) + '</div>';
  });
  html += '<div class="meta"><span class="badge">' +
          (a.refused ? 'refused: ' + esc(a.refusal_reason) : 'answered') + '</span>' +
          'confidence ' + a.confidence.toFixed(2) +
          ' | groundedness ' + a.groundedness.toFixed(2) +
          ' | ' + a.latency_ms.toFixed(0) + ' ms</div>';
  add(a.refused ? 'refused' : 'bot', html);
});
</script>
</body>
</html>
"""


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def make_handler(
    pipeline: FAQPipeline,
    *,
    goldset_path: Optional[str] = None,
    allow_ingest: bool = True,
    quiet: bool = False,
) -> type:
    """Build a request handler bound to one pipeline instance."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "faqbot/0.1"

        # -- plumbing ----------------------------------------------------
        def log_message(self, fmt: str, *args: Any) -> None:
            if quiet:
                return
            # Redact before writing: a question in an access log is user data.
            line = fmt % args
            print("[faqbot] %s %s" % (self.log_date_time_string(), redact_pii(line)[0]))

        def _send(self, code: int, payload: Any, content_type: str = "application/json") -> None:
            body = payload if isinstance(payload, bytes) else _json_bytes(payload)
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return None, "bad Content-Length"
            if length <= 0:
                return {}, None
            if length > MAX_BODY_BYTES:
                return None, "request body too large"
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                return None, "invalid JSON: %s" % exc
            if not isinstance(data, dict):
                return None, "expected a JSON object"
            return data, None

        # -- routes ------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, CHAT_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/health":
                self._send(200, self._health())
            elif path == "/eval":
                query = parse_qs(urlparse(self.path).query)
                self._run_eval((query.get("goldset") or [goldset_path or ""])[0])
            else:
                self._send(404, {"error": "not found", "path": path})

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            path = urlparse(self.path).path
            data, err = self._read_json()
            if err is not None:
                self._send(400, {"error": err})
                return
            assert data is not None
            if path == "/ask":
                self._ask(data)
            elif path == "/ingest":
                self._ingest(data)
            elif path == "/eval":
                self._run_eval(str(data.get("goldset") or goldset_path or ""))
            else:
                self._send(404, {"error": "not found", "path": path})

        # -- handlers ----------------------------------------------------
        def _health(self) -> Dict[str, Any]:
            stats = pipeline.stats()
            return {
                "status": "ok",
                "chunks": stats["chunks"],
                "documents": stats["documents"],
                "embedder": pipeline.embedder.name,
                "answerer": pipeline.answerer.name,
                "chunker": pipeline.chunker.name,
                "vocabulary_terms": stats["vocabulary_terms"],
                "embedder_capabilities": embedder_capabilities(),
                "goldset": goldset_path,
            }

        def _ask(self, data: Dict[str, Any]) -> None:
            question = str(data.get("question") or "").strip()
            if not question:
                self._send(400, {"error": "missing 'question'"})
                return
            session_id = data.get("session_id")
            k = data.get("k")
            answer = pipeline.ask(
                question,
                session_id=str(session_id) if session_id else None,
                k=int(k) if k else None,
            )
            self._send(200, answer.to_dict())

        def _ingest(self, data: Dict[str, Any]) -> None:
            if not allow_ingest:
                self._send(403, {"error": "ingest is disabled on this server"})
                return
            paths = data.get("paths")
            if not isinstance(paths, list) or not paths:
                self._send(400, {"error": "expected 'paths': [str, ...]"})
                return
            missing = [p for p in paths if not os.path.exists(str(p))]
            if missing:
                self._send(400, {"error": "no such path", "paths": missing})
                return
            try:
                stats = pipeline.ingest([str(p) for p in paths])
            except (OSError, ValueError) as exc:
                self._send(400, {"error": "%s: %s" % (type(exc).__name__, exc)})
                return
            self._send(200, stats)

        def _run_eval(self, path: str) -> None:
            if not path:
                self._send(400, {"error": "no goldset configured; pass ?goldset=PATH"})
                return
            if not os.path.exists(path):
                self._send(400, {"error": "no such goldset", "path": path})
                return
            report = run_eval(pipeline, load_goldset(path))
            self._send(200, report.to_dict(include_results=False))

    return Handler


def serve(
    pipeline: FAQPipeline,
    host: str = "127.0.0.1",
    port: int = 8080,
    *,
    goldset_path: Optional[str] = None,
    allow_ingest: bool = True,
    quiet: bool = False,
) -> None:
    """Run the demo server until interrupted.

    Binds to loopback by default. Binding to 0.0.0.0 exposes an unauthenticated
    ``/ingest`` endpoint to the network, so it has to be an explicit choice.
    """
    handler = make_handler(pipeline, goldset_path=goldset_path, allow_ingest=allow_ingest, quiet=quiet)
    httpd = ThreadingHTTPServer((host, port), handler)
    stats = pipeline.stats()
    print("faqbot serving on http://%s:%d" % (host, port))
    print("  index: %d chunks from %d documents (%s embedder)"
          % (stats["chunks"], stats["documents"], pipeline.embedder.name))
    print("  routes: /  /health  /ask  /ingest  /eval")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
