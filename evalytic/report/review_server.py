"""Local HTTP review server for human-in-the-loop scoring.

Serves a self-contained SPA for rating images with star scores and notes.
Uses only stdlib ``http.server`` -- no Flask/FastAPI dependency.

Usage::

    server = ReviewServer(report, port=3847)
    server.start()      # starts in background thread
    server.wait()       # blocks until user clicks "Done" or Ctrl+C
    updated = server.get_report()  # report with merged human scores
"""

from __future__ import annotations

import json
import threading
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..bench.types import BenchReport

# ---------------------------------------------------------------------------
# HTML SPA (inline)
# ---------------------------------------------------------------------------

_REVIEW_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evalytic Review</title>
<style>
  :root { --bg:#0f0f23; --fg:#e2e8f0; --card:#1e1e3a; --border:#2d2d5e;
          --accent:#6366f1; --green:#22c55e; --yellow:#eab308; --red:#ef4444; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         background:var(--bg); color:var(--fg); padding:1.5rem; max-width:1400px; margin:0 auto; }
  h1 { font-size:1.5rem; margin-bottom:0.3rem; }
  .meta { color:#888; font-size:0.85rem; margin-bottom:1.5rem; }
  .item { margin-bottom:2rem; border:1px solid var(--border); border-radius:8px; padding:1rem; background:var(--card); }
  .item-header { font-size:1.1rem; font-weight:600; margin-bottom:0.8rem; }
  .models { display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:1rem; }
  .model-card { border:1px solid var(--border); border-radius:6px; padding:0.8rem; background:var(--bg); }
  .model-card img { width:100%; border-radius:4px; margin-bottom:0.5rem; }
  .model-name { font-weight:600; margin-bottom:0.3rem; }
  .vlm-score { font-size:0.85rem; color:#888; margin-bottom:0.5rem; }
  .stars { display:flex; gap:4px; margin-bottom:0.5rem; cursor:pointer; }
  .star { font-size:1.6rem; color:#555; transition:color 0.15s; }
  .star.active { color:var(--yellow); }
  .star:hover, .star:hover ~ .star { color:var(--yellow); opacity:0.6; }
  .notes { width:100%; padding:0.4rem; border:1px solid var(--border); border-radius:4px;
           background:var(--bg); color:var(--fg); font-size:0.85rem; resize:vertical; min-height:2rem; }
  .toolbar { position:sticky; top:0; background:var(--bg); padding:0.8rem 0; border-bottom:1px solid var(--border);
             display:flex; justify-content:space-between; align-items:center; z-index:10; margin-bottom:1rem; }
  .btn { padding:0.5rem 1.5rem; border:none; border-radius:6px; font-size:0.95rem; cursor:pointer; font-weight:600; }
  .btn-primary { background:var(--accent); color:#fff; }
  .btn-primary:hover { opacity:0.9; }
  .progress { font-size:0.85rem; color:#888; }
  .saved-badge { color:var(--green); font-size:0.8rem; margin-left:0.5rem; opacity:0; transition:opacity 0.3s; }
  .saved-badge.show { opacity:1; }
</style>
</head>
<body>
<div class="toolbar">
  <div>
    <h1>Evalytic Review</h1>
    <span class="progress" id="progress">Loading...</span>
  </div>
  <div>
    <span class="saved-badge" id="saved-badge">Saved</span>
    <button class="btn btn-primary" id="done-btn">Done</button>
  </div>
</div>
<div id="app"></div>

<script>
let report = null;
let scores = {};  // key: "itemId/model/dimIndex" -> {score, notes}

async function loadReport() {
  const resp = await fetch('/api/report');
  report = await resp.json();
  render();
  updateProgress();
}

function render() {
  const app = document.getElementById('app');
  let html = '';
  for (const item of report.items) {
    const label = item.prompt || item.instruction || ('Input ' + item.item_id);
    html += '<div class="item"><div class="item-header">' + esc(label) + '</div><div class="models">';
    for (const model of report.models) {
      const r = item.results[model];
      if (!r) continue;
      html += '<div class="model-card">';
      html += '<div class="model-name">' + esc(model) + '</div>';
      if (r.status === 'success') {
        if (r.image_local) {
          html += '<img src="/images/' + encodeURIComponent(r.image_local) + '" alt="' + esc(model) + '">';
        }
        html += '<div class="vlm-score">VLM: ' + r.overall_score.toFixed(1) + '/5</div>';
        for (let di = 0; di < r.scores.length; di++) {
          const s = r.scores[di];
          const key = item.item_id + '/' + model + '/' + di;
          const cur = scores[key] || {};
          html += '<div style="margin-bottom:0.5rem">';
          html += '<div style="font-size:0.8rem;color:#aaa">' + esc(s.dimension.replace(/_/g,' ')) + ' (VLM: ' + s.score + ')</div>';
          html += '<div class="stars" data-key="' + esc(key) + '">';
          for (let i = 1; i <= 5; i++) {
            const active = cur.score && i <= cur.score ? ' active' : '';
            html += '<span class="star' + active + '" data-val="' + i + '">&#9733;</span>';
          }
          html += '</div>';
          html += '<textarea class="notes" data-key="' + esc(key) + '" placeholder="Notes...">' + esc(cur.notes || '') + '</textarea>';
          html += '</div>';
        }
      } else {
        html += '<div style="color:var(--red)">Failed</div>';
      }
      html += '</div>';
    }
    html += '</div></div>';
  }
  app.innerHTML = html;
  bindEvents();
}

function bindEvents() {
  document.querySelectorAll('.stars').forEach(el => {
    el.querySelectorAll('.star').forEach(star => {
      star.addEventListener('click', () => {
        const key = el.dataset.key;
        const val = parseInt(star.dataset.val);
        if (!scores[key]) scores[key] = {};
        scores[key].score = val;
        render();
        autoSave();
      });
    });
  });
  document.querySelectorAll('textarea.notes').forEach(el => {
    el.addEventListener('input', () => {
      const key = el.dataset.key;
      if (!scores[key]) scores[key] = {};
      scores[key].notes = el.value;
    });
    el.addEventListener('blur', () => autoSave());
  });
}

function updateProgress() {
  let total = 0, rated = 0;
  for (const item of report.items) {
    for (const model of report.models) {
      const r = item.results[model];
      if (!r || r.status !== 'success') continue;
      for (let di = 0; di < r.scores.length; di++) {
        total++;
        const key = item.item_id + '/' + model + '/' + di;
        if (scores[key] && scores[key].score) rated++;
      }
    }
  }
  document.getElementById('progress').textContent = rated + '/' + total + ' scored';
}

async function autoSave() {
  await fetch('/api/scores', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(scores)});
  const badge = document.getElementById('saved-badge');
  badge.classList.add('show');
  setTimeout(() => badge.classList.remove('show'), 1500);
  updateProgress();
}

document.getElementById('done-btn').addEventListener('click', async () => {
  await fetch('/api/scores', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(scores)});
  await fetch('/api/done', {method:'POST'});
  document.body.innerHTML = '<div style="text-align:center;margin-top:4rem"><h1>Review complete</h1><p>You can close this tab.</p></div>';
});

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

loadReport();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------


class _ReviewHandler(BaseHTTPRequestHandler):
    """Handles GET / POST requests for the review SPA."""

    report: BenchReport
    human_scores: dict[str, dict[str, Any]]
    shutdown_event: threading.Event

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # suppress default access log

    def do_GET(self) -> None:
        if self.path == "/":
            self._respond(200, "text/html", _REVIEW_HTML.encode())
        elif self.path == "/api/report":
            data = self._report_json()
            self._respond(200, "application/json", json.dumps(data).encode())
        elif self.path.startswith("/images/"):
            self._serve_image()
        else:
            self._respond(404, "text/plain", b"Not found")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""

        if self.path == "/api/scores":
            if body:
                self.human_scores.update(json.loads(body))
            self._respond(200, "application/json", b'{"ok":true}')
        elif self.path == "/api/done":
            self._respond(200, "application/json", b'{"ok":true}')
            self.shutdown_event.set()
        else:
            self._respond(404, "text/plain", b"Not found")

    def _respond(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_image(self) -> None:
        from pathlib import Path
        from urllib.parse import unquote

        raw = unquote(self.path[len("/images/"):])
        p = Path(raw)
        if not p.exists():
            self._respond(404, "text/plain", b"Image not found")
            return
        suffix = p.suffix.lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(
            suffix.lstrip("."), "application/octet-stream"
        )
        self._respond(200, mime, p.read_bytes())

    def _report_json(self) -> dict[str, Any]:
        """Build a JSON-friendly dict of the report for the SPA."""
        items = []
        for item in self.report.items:
            results: dict[str, Any] = {}
            for model, img in item.results.items():
                scores = [
                    {"dimension": s.dimension, "score": s.score, "explanation": s.explanation}
                    for s in img.scores
                ]
                results[model] = {
                    "image_local": img.image_local,
                    "overall_score": img.overall_score,
                    "status": img.status,
                    "scores": scores,
                }
            items.append({
                "item_id": item.item_id,
                "prompt": item.prompt,
                "instruction": item.instruction,
                "results": results,
            })
        return {"models": self.report.models, "items": items}


# ---------------------------------------------------------------------------
# ReviewServer
# ---------------------------------------------------------------------------


class ReviewServer:
    """Start a local HTTP server for human review of bench results.

    Parameters
    ----------
    report : BenchReport
        The completed bench report to review.
    port : int
        Port for the local server (default 3847).
    """

    def __init__(self, report: BenchReport, port: int = 3847) -> None:
        self._report = report
        self._port = port
        self._human_scores: dict[str, dict[str, Any]] = {}
        self._shutdown_event = threading.Event()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the review server and open the browser."""
        handler = partial(_ReviewHandler)
        # Attach shared state to the handler class (via class-level attrs)
        handler_cls = type(
            "_BoundHandler",
            (_ReviewHandler,),
            {
                "report": self._report,
                "human_scores": self._human_scores,
                "shutdown_event": self._shutdown_event,
            },
        )
        self._server = HTTPServer(("127.0.0.1", self._port), handler_cls)  # type: ignore[arg-type]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

        url = f"http://127.0.0.1:{self._port}"
        webbrowser.open(url)

    def _serve(self) -> None:
        assert self._server is not None
        while not self._shutdown_event.is_set():
            self._server.handle_request()

    def wait(self) -> None:
        """Block until the user clicks Done or Ctrl+C."""
        try:
            self._shutdown_event.wait()
        except KeyboardInterrupt:
            pass
        finally:
            if self._server:
                self._server.server_close()

    def get_report(self) -> BenchReport:
        """Return the report with human scores merged in."""
        self._merge_scores()
        return self._report

    def _merge_scores(self) -> None:
        """Merge human_scores dict into the BenchReport's DimensionResults."""
        for item in self._report.items:
            for model, img_result in item.results.items():
                for di, dim_result in enumerate(img_result.scores):
                    key = f"{item.item_id}/{model}/{di}"
                    entry = self._human_scores.get(key)
                    if entry:
                        if "score" in entry:
                            dim_result.human_score = float(entry["score"])
                        if "notes" in entry:
                            dim_result.human_notes = str(entry["notes"])
