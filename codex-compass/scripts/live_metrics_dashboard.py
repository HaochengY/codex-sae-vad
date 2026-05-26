#!/usr/bin/env python
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VAD Compass Training</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #1f2933; }
    header { padding: 18px 24px; background: #ffffff; border-bottom: 1px solid #d8dee8; display: flex; justify-content: space-between; align-items: center; }
    h1 { margin: 0; font-size: 20px; font-weight: 650; }
    main { padding: 20px 24px; max-width: 1280px; margin: 0 auto; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }
    .card { background: #ffffff; border: 1px solid #d8dee8; border-radius: 8px; padding: 14px; }
    .label { color: #65758b; font-size: 12px; }
    .value { font-size: 24px; font-weight: 700; margin-top: 4px; }
    .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    canvas { width: 100%; height: 260px; display: block; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #e6ebf2; }
    th { color: #65758b; font-weight: 600; }
    .status { color: #65758b; font-size: 13px; }
    @media (max-width: 900px) { .grid, .charts { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>VAD Compass Training</h1>
    <div class="status" id="status">waiting for metrics</div>
  </header>
  <main>
    <section class="grid" id="cards"></section>
    <section class="charts">
      <div class="card"><canvas id="loss"></canvas></div>
      <div class="card"><canvas id="reward"></canvas></div>
      <div class="card"><canvas id="prob"></canvas></div>
      <div class="card"><canvas id="slots"></canvas></div>
    </section>
    <section class="card" style="margin-top:16px">
      <table>
        <thead><tr><th>step</th><th>loss</th><th>bce</th><th>recon</th><th>reward</th><th>format</th><th>video_prob</th><th>label</th></tr></thead>
        <tbody id="rows"></tbody>
      </table>
    </section>
  </main>
<script>
const series = {};
function line(ctx, title, names, colors) {
  const c = document.getElementById(ctx);
  const g = c.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = c.getBoundingClientRect();
  c.width = Math.floor(rect.width * dpr);
  c.height = Math.floor(rect.height * dpr);
  g.scale(dpr, dpr);
  const w = rect.width, h = rect.height, pad = 34;
  g.clearRect(0, 0, w, h);
  g.fillStyle = '#1f2933'; g.font = '14px sans-serif'; g.fillText(title, 12, 20);
  const points = names.flatMap(n => series[n] || []);
  if (!points.length) return;
  const xs = points.map(p => p.step);
  const ys = points.map(p => p.value).filter(Number.isFinite);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  let minY = Math.min(...ys), maxY = Math.max(...ys);
  if (minY === maxY) { minY -= 1; maxY += 1; }
  g.strokeStyle = '#d8dee8'; g.lineWidth = 1;
  g.beginPath(); g.moveTo(pad, h - pad); g.lineTo(w - 10, h - pad); g.moveTo(pad, 30); g.lineTo(pad, h - pad); g.stroke();
  names.forEach((n, i) => {
    const arr = series[n] || [];
    g.strokeStyle = colors[i]; g.lineWidth = 2; g.beginPath();
    arr.forEach((p, j) => {
      const x = pad + ((p.step - minX) / Math.max(1, maxX - minX)) * (w - pad - 14);
      const y = h - pad - ((p.value - minY) / (maxY - minY)) * (h - pad - 36);
      if (j === 0) g.moveTo(x, y); else g.lineTo(x, y);
    });
    g.stroke();
    g.fillStyle = colors[i]; g.fillText(n, pad + i * 120, h - 8);
  });
}
function addSeries(name, rows, key) {
  series[name] = rows.map(r => ({step: r.step, value: Number(r[key])})).filter(p => Number.isFinite(p.value));
}
function render(rows) {
  const latest = rows[rows.length - 1] || {};
  document.getElementById('status').textContent = rows.length ? `last step ${latest.step}, updated ${new Date().toLocaleTimeString()}` : 'waiting for metrics';
  const cards = [
    ['step', latest.step], ['loss', latest.loss], ['reward', latest.reward_mean], ['format', latest.format_score_mean],
    ['video_prob', latest.video_prob], ['label', latest.label], ['bce', latest.bce_loss], ['recon', latest.recon_loss]
  ];
  document.getElementById('cards').innerHTML = cards.map(([k,v]) => `<div class="card"><div class="label">${k}</div><div class="value">${Number.isFinite(Number(v)) ? Number(v).toFixed(4).replace(/\\.0000$/, '') : '-'}</div></div>`).join('');
  addSeries('loss', rows, 'loss'); addSeries('bce', rows, 'bce_loss'); addSeries('recon', rows, 'recon_loss');
  addSeries('reward', rows, 'reward_mean'); addSeries('format', rows, 'format_score_mean'); addSeries('task', rows, 'task_score_mean');
  addSeries('video_prob', rows, 'video_prob');
  for (let i = 0; i < 4; i++) series[`slot_${i}`] = rows.map(r => ({step: r.step, value: (r.slot_probs || [])[i]})).filter(p => Number.isFinite(p.value));
  line('loss', 'Loss', ['loss','bce','recon'], ['#2563eb','#dc2626','#16a34a']);
  line('reward', 'Reward', ['reward','format','task'], ['#7c3aed','#0891b2','#ea580c']);
  line('prob', 'Video Probability', ['video_prob'], ['#111827']);
  line('slots', 'Slot Confidence', ['slot_0','slot_1','slot_2','slot_3'], ['#2563eb','#dc2626','#16a34a','#ea580c']);
  document.getElementById('rows').innerHTML = rows.slice(-20).reverse().map(r => `<tr><td>${r.step}</td><td>${fmt(r.loss)}</td><td>${fmt(r.bce_loss)}</td><td>${fmt(r.recon_loss)}</td><td>${fmt(r.reward_mean)}</td><td>${fmt(r.format_score_mean)}</td><td>${fmt(r.video_prob)}</td><td>${r.label}</td></tr>`).join('');
}
function fmt(v) { return Number.isFinite(Number(v)) ? Number(v).toFixed(4) : '-'; }
async function tick() {
  try {
    const res = await fetch('/metrics');
    render(await res.json());
  } catch (e) {
    document.getElementById('status').textContent = String(e);
  }
}
setInterval(tick, 2000);
window.addEventListener('resize', tick);
tick();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    metrics_path = Path("outputs/sht_window_smoke/metrics.jsonl")

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/metrics":
            rows = []
            if self.metrics_path.exists():
                for line in self.metrics_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rows.append(json.loads(line))
            body = json.dumps(rows, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="outputs/sht_window_smoke/metrics.jsonl")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    Handler.metrics_path = Path(args.metrics)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"dashboard: http://{args.host}:{args.port}  metrics={Handler.metrics_path}")
    server.serve_forever()


if __name__ == "__main__":
    main()
