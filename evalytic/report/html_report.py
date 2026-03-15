"""Self-contained HTML report generator for bench reports."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Template

if TYPE_CHECKING:
    from ..bench.types import BenchReport

# ---------------------------------------------------------------------------
# HTML template (Jinja2, inline)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evalytic Bench: {{ report.name }}</title>
<style>
  :root {
    --bg: #ffffff; --fg: #1a1a2e; --card: #f8f9fa; --border: #e0e0e0;
    --accent: #6366f1; --green: #22c55e; --yellow: #eab308; --red: #ef4444;
    --bar-bg: #e5e7eb;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f0f23; --fg: #e2e8f0; --card: #1e1e3a; --border: #2d2d5e;
      --bar-bg: #2d2d5e;
    }
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: var(--bg); color: var(--fg); line-height: 1.6; padding: 2rem; max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }
  h2 { font-size: 1.3rem; margin: 2rem 0 1rem; border-bottom: 2px solid var(--accent); padding-bottom: 0.3rem; }
  .meta { color: #888; font-size: 0.9rem; margin-bottom: 1.5rem; }
  table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
  th, td { padding: 0.6rem 1rem; text-align: left; border: 1px solid var(--border); }
  th { background: var(--card); font-weight: 600; }
  td { background: var(--bg); }
  .score { font-weight: 700; font-size: 1.1rem; }
  .score-green { color: var(--green); }
  .score-yellow { color: var(--yellow); }
  .score-red { color: var(--red); }
  .bar-container { display: flex; align-items: center; gap: 0.5rem; }
  .bar { height: 1.2rem; border-radius: 4px; min-width: 4px; }
  .bar-bg { background: var(--bar-bg); flex: 1; border-radius: 4px; overflow: hidden; }
  .bar-fill { height: 1.2rem; border-radius: 4px; }
  .winner-badge { background: var(--green); color: white; padding: 0.15rem 0.5rem;
                  border-radius: 4px; font-size: 0.75rem; font-weight: 600; margin-left: 0.5rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; margin: 1rem 0; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; }
  .card img { width: 100%; border-radius: 4px; margin-bottom: 0.5rem; }
  details { margin: 0.5rem 0; }
  summary { cursor: pointer; font-weight: 600; padding: 0.3rem 0; }
  .footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border);
            text-align: center; font-size: 0.8rem; color: #888; }
  .footer a { color: var(--accent); text-decoration: none; }

  /* --- Consensus Panel --- */
  .consensus-panel { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 1.2rem 1.5rem; margin: 1.5rem 0; }
  .consensus-panel h3 { font-size: 1rem; margin-bottom: 0.8rem; }
  .consensus-stats { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 1rem; }
  .consensus-stat { text-align: center; min-width: 80px; }
  .consensus-stat .stat-value { font-size: 1.6rem; font-weight: 700; }
  .consensus-stat .stat-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
  .judge-bias-table { font-size: 0.85rem; }
  .judge-bias-table td, .judge-bias-table th { padding: 0.35rem 0.7rem; }
  .dispute-list { font-size: 0.85rem; margin: 0; padding: 0; list-style: none; }
  .dispute-list li { padding: 0.25rem 0; border-bottom: 1px solid var(--border); }
  .dispute-list li:last-child { border-bottom: none; }

  /* --- Executive Summary Band --- */
  .summary-band { background: linear-gradient(135deg, var(--accent), #818cf8); color: white;
    padding: 1.2rem 1.5rem; border-radius: 10px; margin: 1rem 0 2rem; }
  .summary-band .summary-main { display: flex; align-items: center; gap: 0.7rem; flex-wrap: wrap; font-size: 1.15rem; }
  .summary-band .summary-rank { font-size: 1.6rem; font-weight: 800; opacity: 0.9; }
  .summary-band .summary-model { font-weight: 700; }
  .summary-band .summary-score { font-weight: 700; }
  .summary-band .summary-divider { opacity: 0.4; }
  .summary-band .summary-stats { margin-top: 0.4rem; font-size: 0.9rem; opacity: 0.85; }
  .summary-band .winner-pill { background: rgba(255,255,255,0.25); padding: 0.15rem 0.6rem;
    border-radius: 4px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.03em; }

  /* --- Radar Chart --- */
  .radar-section { display: flex; justify-content: center; margin: 1rem 0; }
  .radar-section svg { max-width: 380px; height: auto; }

  /* --- Lightbox --- */
  .lightbox-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.85); z-index: 1000; cursor: pointer;
    justify-content: center; align-items: center; flex-direction: column; }
  .lightbox-overlay.active { display: flex; }
  .lightbox-overlay img { max-width: 90vw; max-height: 80vh; object-fit: contain; border-radius: 8px; }
  .lightbox-caption { color: white; margin-top: 0.8rem; font-size: 0.95rem; text-align: center; }
  .lightbox-close { position: absolute; top: 1rem; right: 1.5rem; color: white; font-size: 2rem;
    cursor: pointer; opacity: 0.7; background: none; border: none; }
  .lightbox-close:hover { opacity: 1; }

  /* --- Sortable Tables --- */
  table[data-sortable] th { cursor: pointer; user-select: none; position: relative; }
  table[data-sortable] th:hover { background: var(--border); }
  table[data-sortable] th:not([data-no-sort])::after { content: ' \\25B4\\25BE'; font-size: 0.65em; opacity: 0.4; margin-left: 0.3rem; }
  table[data-sortable] th[data-no-sort] { cursor: default; }
  table[data-sortable] th.sort-asc::after { content: ' \\25B4'; opacity: 0.9; }
  table[data-sortable] th.sort-desc::after { content: ' \\25BE'; opacity: 0.9; }
</style>
</head>
<body>

<h1>{{ report.name }}</h1>
<div class="meta">
  {{ report.created_at[:19] | replace("T", " ") }} UTC &middot;
  {{ report.duration_seconds }}s &middot;
  {{ report.models | length }} models &middot;
  {{ report.items | length }} {{ "prompts" if report.pipeline == "text2img" else "inputs" }} &middot;
  Total cost: ${{ "%.2f" | format(report.cost.total_usd) }}
</div>

{% if report.winner and report.summary.get(report.winner) %}
{% set ws = report.summary[report.winner] %}
<div class="summary-band">
  <div class="summary-main">
    <span class="summary-rank">#1</span>
    <span class="summary-model">{{ report.winner }}</span>
    <span class="summary-score">{{ "%.1f" | format(ws.overall_score) }}/5</span>
    <span class="winner-pill">WINNER</span>
    {% if report.best_value and report.best_value != report.winner %}
    <span class="summary-divider">|</span>
    <span>Best Value: <strong>{{ report.best_value }}</strong></span>
    {% endif %}
  </div>
  <div class="summary-stats">
    {{ report.models | length }} models &middot;
    {{ report.items | length }} {{ "prompts" if report.pipeline == "text2img" else "inputs" }} (n={{ ws.sample_count }}) &middot;
    ${{ "%.2f" | format(report.cost.total_usd) }} &middot;
    {{ "%.0f" | format(report.duration_seconds) }}s
    {% if ws.sample_count < 10 %}&middot; <span style="opacity:0.7">⚠ Small sample — scores may vary with more prompts</span>{% endif %}
  </div>
</div>
{% endif %}

<h2>Model Rankings</h2>
<table data-sortable>
  <thead>
    <tr>
      <th>Rank</th>
      <th>Model</th>
      {% for dim in report.dimensions %}
      <th>{{ dim.replace("_", " ").title() }}{% if dim_weights.get(dim) %}<span style="font-size:0.7rem;color:#888;font-weight:400"> ({{ "%.0f" | format(dim_weights[dim] * 100) }}%)</span>{% endif %}{% if dim_variance.get(dim, "") == "differentiator" %}<span style="color:var(--accent);font-size:0.7rem;vertical-align:super" title="High cross-model variance — key differentiator"> ★</span>{% elif dim_variance.get(dim, "") == "ceiling" %}<span style="color:#888;font-size:0.7rem;vertical-align:super" title="All models score similarly — low differentiation"> ≈</span>{% endif %}</th>
      {% endfor %}
      {% for m_name in metric_names %}
      <th>{{ metric_labels.get(m_name, m_name) }}</th>
      {% endfor %}
      <th>{{ "Overall (weighted)" if has_weighted else "Overall" }}</th>
      <th>Conf</th>
      {% if is_consensus %}<th>Agree</th>{% endif %}
      {% if has_failures %}<th>Success</th>{% endif %}
      <th data-no-sort></th>
    </tr>
  </thead>
  <tbody>
    {% for model, score in report.ranking %}
    <tr>
      <td>{{ loop.index }}</td>
      <td><strong>{{ model }}</strong>{% if model == report.winner %}<span class="winner-badge">WINNER</span>{% endif %}</td>
      {% set ms = report.summary[model] %}
      {% for dim in report.dimensions %}
      {% set avg = ms.dimension_averages.get(dim, 0) %}
      {% set sd = ms.dimension_stddev.get(dim, 0) %}
      <td class="score {{ 'score-green' if avg >= 4 else ('score-yellow' if avg >= 3 else 'score-red') }}" {% if sd > 0 and ms.sample_count >= 2 %}title="stddev: ±{{ '%.1f' | format(sd) }}, n={{ ms.sample_count }}"{% endif %}>{{ "%.1f" | format(avg) }}</td>
      {% endfor %}
      {% for m_name in metric_names %}
      {% set flag_status = ms.metric_flags.get(m_name, "") %}
      <td style="font-size:0.9rem">
        {{ "%.3f" | format(ms.metric_averages.get(m_name, 0)) }}
        {% if flag_status == "flagged" %}<span style="color:var(--red)" title="Below threshold — excluded from overall"> ⚠</span>
        {% elif flag_status == "included" %}<span style="color:var(--green)" title="Above threshold — included in overall"> ✓</span>{% endif %}
      </td>
      {% endfor %}
      <td class="score {{ 'score-green' if ms.overall_score >= 4 else ('score-yellow' if ms.overall_score >= 3 else 'score-red') }}" {% if ms.overall_stddev > 0 and ms.sample_count >= 2 %}title="stddev: ±{{ '%.1f' | format(ms.overall_stddev) }}, n={{ ms.sample_count }}"{% endif %}>{{ "%.1f" | format(ms.overall_score) }}<span style="font-size:0.65rem;color:#888;font-weight:400"> n={{ ms.sample_count }}</span></td>
      <td class="{{ 'score-green' if ms.avg_confidence >= 0.8 else ('score-yellow' if ms.avg_confidence >= 0.5 else 'score-red') }}">{{ "%.0f" | format(ms.avg_confidence * 100) }}%</td>
      {% if is_consensus %}
      <td class="{{ 'score-green' if ms.agreement_rate >= 0.8 else ('score-yellow' if ms.agreement_rate >= 0.5 else 'score-red') }}">{{ "%.0f" | format(ms.agreement_rate * 100) }}%</td>
      {% endif %}
      {% if has_failures %}
      {% set sr = (ms.item_count / ms.total_items) if ms.total_items > 0 else 1.0 %}
      <td class="{{ 'score-green' if sr == 1.0 else ('score-yellow' if sr >= 0.5 else 'score-red') }}">{{ ms.item_count }}/{{ ms.total_items }}</td>
      {% endif %}
      <td>
        <div class="bar-container">
          <div class="bar-bg"><div class="bar-fill" style="width:{{ (ms.overall_score / 5 * 100) | int }}%;background:{{ 'var(--green)' if ms.overall_score >= 4 else ('var(--yellow)' if ms.overall_score >= 3 else 'var(--red)') }}"></div></div>
        </div>
      </td>
    </tr>
    {% endfor %}
  </tbody>
</table>

{% if is_consensus and consensus_stats %}
<div class="consensus-panel">
  <h3>Consensus Analysis &mdash; {{ report.judges | length }} Judges</h3>
  <div style="font-size:0.85rem;color:#888;margin-bottom:0.8rem">Two primary judges score each dimension independently. When they agree (score difference &le;0.5), the result is <strong style="color:var(--green)">high agreement</strong>. When they disagree, a third tiebreaker judge is called and the <strong style="color:var(--yellow)">median</strong> is used.</div>
  <div class="consensus-stats">
    <div class="consensus-stat">
      <div class="stat-value score-green">{{ consensus_stats.high_pct }}%</div>
      <div class="stat-label">High Agreement</div>
    </div>
    <div class="consensus-stat">
      <div class="stat-value score-yellow">{{ consensus_stats.disputed_pct }}%</div>
      <div class="stat-label">Disputed</div>
    </div>
    {% if consensus_stats.degraded_count > 0 %}
    <div class="consensus-stat">
      <div class="stat-value score-red">{{ consensus_stats.degraded_count }}</div>
      <div class="stat-label">Degraded</div>
    </div>
    {% endif %}
    <div class="consensus-stat">
      <div class="stat-value">{{ consensus_stats.total }}</div>
      <div class="stat-label">Total Scores</div>
    </div>
    <div class="consensus-stat">
      <div class="stat-value">{{ consensus_stats.tiebreaker_count }}</div>
      <div class="stat-label">Tiebreakers</div>
    </div>
  </div>

  {% if consensus_stats.judge_averages %}
  <details open>
    <summary>Judge Scoring Bias</summary>
    <table class="judge-bias-table" style="margin-top:0.5rem">
      <thead><tr><th>Judge</th><th>Role</th><th>Avg Score</th><th>vs Consensus</th><th>Scores Given</th></tr></thead>
      <tbody>
        {% for jname, javg in consensus_stats.judge_averages.items() %}
        {% set diff = javg - consensus_stats.consensus_avg %}
        {% set jcount = consensus_stats.judge_counts.get(jname, 0) %}
        {% set is_tiebreaker = jcount < consensus_stats.total %}
        <tr>
          <td>{{ jname }}</td>
          <td style="font-size:0.8rem;color:#888">{{ "tiebreaker" if is_tiebreaker else "primary" }}</td>
          <td class="score">{{ "%.2f" | format(javg) }}</td>
          <td>{% if is_tiebreaker %}<span style="color:#888" title="Tiebreaker only scores disputed dimensions — comparison is not apples-to-apples">n/a*</span>{% else %}<span class="{{ 'score-green' if diff|abs < 0.3 else ('score-yellow' if diff|abs < 0.6 else 'score-red') }}">{{ "%+.2f" | format(diff) }}</span>{% endif %}</td>
          <td>{{ jcount }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% if consensus_stats.tiebreaker_count > 0 %}
    <div style="font-size:0.75rem;color:#888;margin-top:0.3rem">* Tiebreaker only scores disputed dimensions, so its average is not directly comparable to primary judges.</div>
    {% endif %}
  </details>
  {% endif %}

  {% if consensus_stats.disputes %}
  <details>
    <summary>Disputed Dimensions ({{ consensus_stats.disputes | length }})</summary>
    <ul class="dispute-list" style="margin-top:0.5rem">
      {% for d in consensus_stats.disputes %}
      <li>
        <strong>{{ d.model }}/{{ d.dimension.replace("_", " ").title() }}</strong>:
        consensus <span class="score">{{ "%.1f" | format(d.final_score) }}</span>
        &larr; {% for jname, jscore in d.judge_scores.items() %}<span style="color:{{ 'var(--red)' if (jscore - d.final_score)|abs > 0.5 else '#888' }}">{{ jname }}={{ jscore }}</span>{{ ", " if not loop.last }}{% endfor %}
      </li>
      {% endfor %}
    </ul>
  </details>
  {% endif %}
</div>
{% endif %}

{% if report.dimensions | length >= 3 %}
<h2>Dimension Profile</h2>
<div class="radar-section">
  <svg id="radar-chart" viewBox="0 0 500 440" xmlns="http://www.w3.org/2000/svg"></svg>
</div>
<script>
(function() {
  var data = {{ radar_data_json }};
  var dims = {{ dimensions_json }};
  var svg = document.getElementById('radar-chart');
  if (!svg || dims.length < 3) return;
  var cx = 250, cy = 210, R = 150, n = dims.length;
  var colors = ['#6366f1','#22c55e','#f59e0b','#ef4444','#06b6d4','#ec4899','#8b5cf6','#14b8a6'];
  function poly(r) {
    var pts = [];
    for (var i = 0; i < n; i++) {
      var a = (2 * Math.PI * i / n) - Math.PI / 2;
      pts.push((cx + r * Math.cos(a)).toFixed(1) + ',' + (cy + r * Math.sin(a)).toFixed(1));
    }
    return pts.join(' ');
  }
  function el(tag, attrs) {
    var e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }
  // Grid levels
  for (var lv = 1; lv <= 5; lv++) {
    var r = (lv / 5) * R;
    svg.appendChild(el('polygon', {points: poly(r), fill: 'none', stroke: 'var(--border)', 'stroke-width': '1'}));
    svg.appendChild(Object.assign(el('text', {x: cx + 4, y: cy - r + 14, fill: '#888', 'font-size': '14'}),
      {textContent: lv}));
  }
  // Axes + labels
  for (var i = 0; i < n; i++) {
    var a = (2 * Math.PI * i / n) - Math.PI / 2;
    var ex = cx + R * Math.cos(a), ey = cy + R * Math.sin(a);
    svg.appendChild(el('line', {x1: cx, y1: cy, x2: ex, y2: ey, stroke: 'var(--border)', 'stroke-width': '1'}));
    var lx = cx + (R + 32) * Math.cos(a), ly = cy + (R + 32) * Math.sin(a);
    var txt = el('text', {x: lx, y: ly, 'text-anchor': 'middle', 'dominant-baseline': 'middle',
      fill: 'var(--fg)', 'font-size': '15', 'font-weight': '600'});
    txt.textContent = dims[i].replace(/_/g, ' ').replace(/\\b\\w/g, function(c) { return c.toUpperCase(); });
    svg.appendChild(txt);
  }
  // Data polygons
  var models = Object.keys(data);
  models.forEach(function(model, idx) {
    var c = colors[idx % colors.length];
    var pts = [];
    for (var i = 0; i < n; i++) {
      var s = data[model][dims[i]] || 0;
      var r = (s / 5) * R;
      var a = (2 * Math.PI * i / n) - Math.PI / 2;
      pts.push((cx + r * Math.cos(a)).toFixed(1) + ',' + (cy + r * Math.sin(a)).toFixed(1));
    }
    svg.appendChild(el('polygon', {points: pts.join(' '), fill: c, 'fill-opacity': '0.15',
      stroke: c, 'stroke-width': '2', 'stroke-opacity': '0.8'}));
    // Dots with tooltips
    for (var i = 0; i < n; i++) {
      var s = data[model][dims[i]] || 0;
      var r = (s / 5) * R;
      var a = (2 * Math.PI * i / n) - Math.PI / 2;
      var dot = el('circle', {cx: (cx + r * Math.cos(a)).toFixed(1),
        cy: (cy + r * Math.sin(a)).toFixed(1), r: '4', fill: c});
      var title = el('title', {});
      title.textContent = model + ': ' + dims[i].replace(/_/g, ' ') + ' = ' + s.toFixed(1);
      dot.appendChild(title);
      svg.appendChild(dot);
    }
  });
  // Legend
  var ly = 400;
  var totalW = 0;
  models.forEach(function(m) { totalW += m.length * 7 + 30; });
  var lx = Math.max(10, (500 - totalW) / 2);
  models.forEach(function(model, idx) {
    var c = colors[idx % colors.length];
    svg.appendChild(el('rect', {x: lx, y: ly, width: '12', height: '12', rx: '2', fill: c}));
    var t = el('text', {x: lx + 18, y: ly + 11, fill: 'var(--fg)', 'font-size': '14'});
    t.textContent = model;
    svg.appendChild(t);
    lx += model.length * 7 + 30;
  });
})();
</script>
{% endif %}

{% if report.efficiency_ranking and report.efficiency_ranking | length > 1 %}
{% set winner_ms = report.summary.get(report.winner) %}
<h2>Cost Efficiency</h2>
<table data-sortable>
  <thead>
    <tr><th>Rank</th><th>Model</th><th>Score</th><th>Cost/Image</th><th>Score/$</th><th>vs Winner</th><th data-no-sort></th></tr>
  </thead>
  <tbody>
    {% for model, spd in report.efficiency_ranking %}
    {% set ms = report.summary[model] %}
    <tr>
      <td>{{ loop.index }}</td>
      <td><strong>{{ model }}</strong>{% if model == report.best_value and model != report.winner %}<span style="background:var(--green);color:white;padding:0.1rem 0.4rem;border-radius:4px;font-size:0.7rem;font-weight:600;margin-left:0.4rem">BEST VALUE</span>{% elif model == report.winner %}<span style="background:var(--accent);color:white;padding:0.1rem 0.4rem;border-radius:4px;font-size:0.7rem;font-weight:600;margin-left:0.4rem">WINNER</span>{% endif %}</td>
      <td class="score {{ 'score-green' if ms.overall_score >= 4 else ('score-yellow' if ms.overall_score >= 3 else 'score-red') }}">{{ "%.1f" | format(ms.overall_score) }}</td>
      <td>${{ "%.4f" | format(ms.cost_per_image) }}</td>
      <td style="font-size:0.85rem;color:#888">{{ "%.0f" | format(spd) }}</td>
      <td style="font-size:0.85rem">
        {% if winner_ms and model != report.winner and winner_ms.cost_per_image > 0 %}
          {% set score_gap = ms.overall_score - winner_ms.overall_score %}
          {% set cost_ratio = ms.cost_per_image / winner_ms.cost_per_image %}
          <span class="{{ 'score-green' if score_gap >= 0 else ('score-yellow' if score_gap > -0.5 else 'score-red') }}">{{ "%+.1f" | format(score_gap) }} quality</span>,
          {% if cost_ratio > 1 %}<span class="score-red">{{ "%.0f" | format(cost_ratio) }}x cost</span>{% else %}<span class="score-green">{{ "%.0f" | format((1 - cost_ratio) * 100) }}% cheaper</span>{% endif %}
        {% elif model == report.winner %}
          <span style="color:#888">baseline</span>
        {% endif %}
      </td>
      <td>
        {% set best_spd = report.efficiency_ranking[0][1] %}
        {% set pct = (spd / best_spd * 100) | int if best_spd > 0 else 0 %}
        <div class="bar-bg"><div class="bar-fill" style="width:{{ pct }}%;background:var(--green)"></div></div>
      </td>
    </tr>
    {% endfor %}
  </tbody>
</table>
{% if report.best_value and report.best_value != report.winner %}
{% set bv = report.summary[report.best_value] %}
{% set w = report.summary[report.winner] %}
{% if w.cost_per_image > 0 %}
<p style="margin:0.5rem 0;font-size:0.9rem"><strong>{{ report.best_value }}</strong> is <span class="score-green">{{ "%.0f" | format((1 - bv.cost_per_image / w.cost_per_image) * 100) }}% cheaper</span> than {{ report.winner }} with only {{ "%.1f" | format(w.overall_score - bv.overall_score) }} points less quality</p>
{% endif %}
{% endif %}
{% endif %}

{% if metric_warnings %}
<div style="border:2px solid var(--yellow);border-radius:8px;padding:1rem;margin:1rem 0;background:rgba(234,179,8,0.08)">
  <strong style="color:var(--yellow)">⚠ Metric Warnings</strong>
  <ul style="margin:0.5rem 0 0 1.2rem;font-size:0.9rem">
    {% for warn in metric_warnings %}
    <li>{{ warn }}</li>
    {% endfor %}
  </ul>
</div>
{% endif %}

{% if report.correlations %}
<h2>Metric-VLM Correlation</h2>
<table>
  <thead><tr><th>Pair</th><th>Pearson r</th><th>p-value</th><th>Agreement</th></tr></thead>
  <tbody>
    {% for c in report.correlations %}
    <tr>
      <td>{{ c.metric_pair.replace("_vs_", " vs ").replace("_", " ") }}</td>
      <td class="score">{{ "%.2f" | format(c.pearson_r) }}</td>
      <td>{{ "%.4f" | format(c.p_value) }}</td>
      <td><span class="{{ 'score-green' if c.interpretation == 'high_agreement' else ('score-yellow' if c.interpretation == 'moderate' else 'score-red') }}">{{ c.interpretation.replace("_", " ") }}</span></td>
    </tr>
    {% endfor %}
  </tbody>
</table>
{% endif %}

<div id="items-container">
{% for item in report.items %}
<div class="bench-item" data-page="{{ ((loop.index0) // 20) + 1 }}"{% if loop.index0 >= 20 %} style="display:none"{% endif %}>
<h2>{{ item.prompt or item.instruction or ("Input " ~ loop.index) }}</h2>
{% if item.tags %}<div class="meta">Tags: {{ item.tags | join(", ") }}</div>{% endif %}
<div class="grid">
  {% set input_b64 = input_images.get(item.item_id, "") %}
  {% if input_b64 %}
  <div class="card" style="border:2px solid var(--accent)">
    <strong style="color:var(--accent)">Input</strong>
    <img src="data:image/jpeg;base64,{{ input_b64 }}" alt="Input image" loading="lazy">
    <div style="font-size:0.85rem;color:#888">Reference image</div>
  </div>
  {% endif %}
  {% for model in report.models %}
  {% set img = item.results.get(model) %}
  {% if img %}
  <div class="card">
    <strong>{{ model }}</strong>
    {% if img.status == "success" %}
      {% set b64 = images.get(model ~ "/" ~ item.item_id, "") %}
      {% if b64 %}<img src="data:image/jpeg;base64,{{ b64 }}" alt="{{ model }}" loading="lazy">{% endif %}
      <div class="bar-container" style="margin:0.3rem 0">
        <span class="score {{ 'score-green' if img.overall_score >= 4 else ('score-yellow' if img.overall_score >= 3 else 'score-red') }}">{{ "%.1f" | format(img.overall_score) }}/5</span>
        <div class="bar-bg"><div class="bar-fill" style="width:{{ (img.overall_score / 5 * 100) | int }}%;background:{{ 'var(--green)' if img.overall_score >= 4 else ('var(--yellow)' if img.overall_score >= 3 else 'var(--red)') }}"></div></div>
      </div>
      <div style="font-size:0.85rem;color:#888">{{ img.generation_time_ms }}ms &middot; ${{ "%.4f" | format(img.generation_cost_usd) }}</div>
      <details>
        <summary>Score Details</summary>
        {% for s in img.scores %}
        <div style="margin:0.3rem 0;font-size:0.85rem">
          <strong>{{ s.dimension.replace("_", " ").title() }}:</strong>
          <span class="{{ 'score-green' if s.score >= 4 else ('score-yellow' if s.score >= 3 else 'score-red') }}">{{ s.score }}/5</span>
          <span style="font-size:0.8rem;color:#888">({{ "%.0f" | format(s.confidence * 100) }}% conf)</span>
          {% if s.agreement %}<span style="font-size:0.75rem;padding:0.1rem 0.3rem;border-radius:3px;margin-left:0.3rem;background:{{ 'rgba(34,197,94,0.15)' if s.agreement == 'high' else ('rgba(234,179,8,0.15)' if s.agreement == 'disputed' else 'rgba(239,68,68,0.15)') }}">{{ s.agreement }}</span>{% endif %}
          {% if s.judge_scores %}<div style="font-size:0.75rem;color:#888;margin-top:0.1rem">{% for jname, jscore in s.judge_scores.items() %}{{ jname }}: {{ jscore }}{{ ", " if not loop.last }}{% endfor %}</div>{% endif %}
          <br><em>{{ s.explanation }}</em>
        </div>
        {% endfor %}
        {% if img.metrics %}
        <div style="margin-top:0.5rem;padding-top:0.3rem;border-top:1px solid var(--border)">
          {% for m in img.metrics %}
          <div style="font-size:0.85rem"><strong>{{ metric_labels.get(m.metric, m.metric) }}:</strong> {{ "%.4f" | format(m.value) }}</div>
          {% endfor %}
        </div>
        {% endif %}
        {% if img.flags %}
        <div style="margin-top:0.5rem;padding:0.4rem;background:rgba(234,179,8,0.12);border-radius:4px;font-size:0.8rem">
          {% for f in img.flags %}
          <div style="color:var(--yellow)">⚠ {{ f.message }}</div>
          {% endfor %}
        </div>
        {% endif %}
      </details>
    {% else %}
      <div style="color:var(--red)">Failed: {{ img.error }}</div>
    {% endif %}
  </div>
  {% endif %}
  {% endfor %}
</div>
</div>
{% endfor %}
</div>

{% if report.items | length > 20 %}
<div id="pagination" style="display:flex;align-items:center;justify-content:center;gap:1rem;margin:1.5rem 0">
  <button id="prev-btn" style="padding:0.4rem 1rem;border:1px solid var(--border);border-radius:4px;background:var(--card);color:var(--fg);cursor:pointer" disabled>Prev</button>
  <span id="page-info" style="font-size:0.9rem"></span>
  <button id="next-btn" style="padding:0.4rem 1rem;border:1px solid var(--border);border-radius:4px;background:var(--card);color:var(--fg);cursor:pointer">Next</button>
</div>
<script>
(function() {
  var items = document.querySelectorAll('.bench-item');
  var totalPages = Math.ceil(items.length / 20);
  var currentPage = 1;
  var prevBtn = document.getElementById('prev-btn');
  var nextBtn = document.getElementById('next-btn');
  var pageInfo = document.getElementById('page-info');

  function showPage(page) {
    currentPage = page;
    items.forEach(function(el) {
      el.style.display = parseInt(el.dataset.page) === page ? '' : 'none';
    });
    prevBtn.disabled = page <= 1;
    nextBtn.disabled = page >= totalPages;
    pageInfo.textContent = 'Page ' + page + ' of ' + totalPages + ' (' + items.length + ' items)';
  }

  prevBtn.addEventListener('click', function() { if (currentPage > 1) showPage(currentPage - 1); });
  nextBtn.addEventListener('click', function() { if (currentPage < totalPages) showPage(currentPage + 1); });
  showPage(1);
})();
</script>
{% endif %}

<h2>Cost Summary</h2>
<table>
  <thead><tr><th>Category</th><th>Requests</th><th>Cost</th></tr></thead>
  <tbody>
    <tr><td>fal.ai generation</td><td>{{ report.cost.generation_request_count }}</td><td>${{ "%.4f" | format(report.cost.generation_total_usd) }}</td></tr>
    {% if report.cost.judge_provider %}
    <tr><td>{{ report.cost.judge_provider }} judge</td><td>{{ report.cost.judge_request_count }}</td><td>${{ "%.4f" | format(report.cost.judge_total_usd) }}</td></tr>
    {% else %}
    <tr><td>VLM judge</td><td>0</td><td>$0.0000</td></tr>
    {% endif %}
    {% if metric_names %}<tr><td>Local metrics</td><td></td><td>${{ "%.4f" | format(report.cost.metrics_total_usd) }}</td></tr>{% endif %}
    <tr><td><strong>Total</strong></td><td></td><td><strong>${{ "%.4f" | format(report.cost.total_usd) }}</strong></td></tr>
  </tbody>
</table>

<details>
  <summary><h2 style="display:inline">Configuration</h2></summary>
  <table>
    <tr><td>Models</td><td>{{ report.models | join(", ") }}</td></tr>
    <tr><td>Judge</td><td>{{ report.judge or "None (metrics only)" }}</td></tr>
    {% if is_consensus %}
    <tr><td>Judges</td><td>{{ report.judges | join(", ") }}</td></tr>
    <tr><td>Consensus Mode</td><td>Adaptive 2+1</td></tr>
    {% endif %}
    <tr><td>Dimensions</td><td>{{ report.dimensions | join(", ") }}</td></tr>
    <tr><td>Pipeline</td><td>{{ report.pipeline }}</td></tr>
    <tr><td>Evalytic Version</td><td>{{ report.metadata.get("evalytic_version", "unknown") }}</td></tr>
    <tr><td>Platform</td><td>{{ report.metadata.get("platform", "unknown") }}</td></tr>
    {% if dim_weights %}
    <tr><td>Dimension Weights</td><td>
      {% for dim, w in dim_weights.items() %}
      {{ dim.replace("_", " ").title() }}: {{ "%.0f" | format(w * 100) }}%{{ ", " if not loop.last }}
      {% endfor %}
    </td></tr>
    {% endif %}
    {% if scoring_config %}
    <tr><td>Metric Scoring</td><td>
      {% for name, cfg in scoring_config.items() %}
      {% if name != "dimension_weights" %}{{ name }}: threshold={{ cfg.flag_threshold }}, weight={{ cfg.weight }}{{ ", " if not loop.last }}{% endif %}
      {% endfor %}
    </td></tr>
    {% endif %}
  </table>
</details>

<div class="footer">
  Generated by <a href="https://evalytic.ai">Evalytic</a> &mdash; evalytic.ai
</div>

<!-- Lightbox overlay -->
<div class="lightbox-overlay" id="lightbox">
  <button class="lightbox-close" title="Close">&times;</button>
  <img src="" alt="Zoomed image">
  <div class="lightbox-caption"></div>
</div>

<script>
/* --- Lightbox --- */
(function() {
  var overlay = document.getElementById('lightbox');
  if (!overlay) return;
  var lbImg = overlay.querySelector('img');
  var lbCap = overlay.querySelector('.lightbox-caption');

  document.querySelectorAll('.card img').forEach(function(img) {
    img.style.cursor = 'zoom-in';
    img.addEventListener('click', function() {
      var card = img.closest('.card');
      var model = card ? card.querySelector('strong') : null;
      var score = card ? card.querySelector('.score') : null;
      lbImg.src = img.src;
      lbCap.textContent = (model ? model.textContent : '') + (score ? '  ' + score.textContent : '');
      overlay.classList.add('active');
    });
  });

  overlay.addEventListener('click', function(e) {
    if (e.target === overlay || e.target.classList.contains('lightbox-close')) {
      overlay.classList.remove('active');
    }
  });
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') overlay.classList.remove('active');
  });
})();

/* --- Sortable Tables --- */
(function() {
  document.querySelectorAll('table[data-sortable]').forEach(function(table) {
    var headers = table.querySelectorAll('thead th');
    headers.forEach(function(th, colIdx) {
      if (th.hasAttribute('data-no-sort')) return;
      th.addEventListener('click', function() {
        var tbody = table.querySelector('tbody');
        var rows = Array.from(tbody.querySelectorAll('tr'));
        var asc = th.classList.contains('sort-asc');
        // Clear all sort classes on siblings
        headers.forEach(function(h) { h.classList.remove('sort-asc', 'sort-desc'); });
        th.classList.add(asc ? 'sort-desc' : 'sort-asc');
        var dir = asc ? -1 : 1;

        rows.sort(function(a, b) {
          var at = a.children[colIdx] ? a.children[colIdx].textContent.trim() : '';
          var bt = b.children[colIdx] ? b.children[colIdx].textContent.trim() : '';
          // Extract numeric value (strip $, %, letters)
          var an = parseFloat(at.replace(/[^0-9.\\-]/g, ''));
          var bn = parseFloat(bt.replace(/[^0-9.\\-]/g, ''));
          if (!isNaN(an) && !isNaN(bn)) return (an - bn) * dir;
          return at.localeCompare(bt) * dir;
        });
        rows.forEach(function(r) { tbody.appendChild(r); });
      });
    });
  });
})();
</script>

</body>
</html>
""")


def _load_image_base64(path: str) -> str:
    """Read a local image file and return JPEG base64 data.

    Compresses to JPEG quality 80 via Pillow to keep HTML file size down.
    Also handles URLs by downloading to a temp file first.
    """
    if path.startswith(("http://", "https://")):
        try:
            from ..bench.metrics import _resolve_image_path
            path = _resolve_image_path(path)
        except Exception:
            return ""

    p = Path(path)
    if not p.exists():
        return ""
    try:
        from PIL import Image
        import io

        img = Image.open(p)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        # Fallback: raw base64 without compression
        return base64.b64encode(p.read_bytes()).decode("utf-8")


def write_html(report: BenchReport, path: str) -> None:
    """Render a self-contained HTML report file."""
    # Build image map: "model/item_id" -> base64
    images: dict[str, str] = {}
    for item in report.items:
        for model, img_result in item.results.items():
            if img_result.image_local and img_result.status == "success":
                key = f"{model}/{item.item_id}"
                b64 = _load_image_base64(img_result.image_local)
                if b64:
                    images[key] = b64

    # Build input image map for img2img: "item_id" -> base64
    input_images: dict[str, str] = {}
    if report.pipeline == "img2img":
        for item in report.items:
            if item.image_url:
                b64 = _load_image_base64(item.image_url)
                if b64:
                    input_images[item.item_id] = b64

    # Collect metric names across all model summaries
    metric_names: list[str] = []
    for ms in report.summary.values():
        for m in ms.metric_averages:
            if m not in metric_names:
                metric_names.append(m)

    metric_labels = {"clip_score": "CLIP", "lpips": "LPIPS"}

    # Weighted scoring detection
    has_weighted = any(ms.weighted_overall_score is not None for ms in report.summary.values())

    # Metric warnings
    metric_warnings: list[str] = []
    for model in report.models:
        ms = report.summary.get(model)
        if ms is None:
            continue
        for m_name, status in ms.metric_flags.items():
            if status == "flagged":
                val = ms.metric_averages.get(m_name, 0.0)
                label = metric_labels.get(m_name, m_name)
                metric_warnings.append(
                    f"{model}: {label} ({val:.4f}) below threshold — excluded from overall"
                )

    # Detect if any model has failures
    has_failures = any(
        ms.total_items > 0 and ms.item_count < ms.total_items
        for ms in report.summary.values()
    )

    # Scoring config from report.config
    scoring_config = report.config.get("scoring_config")

    # Dimension weights from report.config
    dim_weights: dict[str, float] = report.config.get("dimension_weights", {})

    # Radar chart data: model -> {dimension -> score}
    radar_data = {
        model: ms.dimension_averages
        for model, ms in report.summary.items()
    }

    # Consensus statistics (computed from item-level score data)
    consensus_stats: dict[str, Any] | None = None
    if report.consensus_mode:
        consensus_stats = _compute_consensus_stats(report)

    # Dimension variance analysis: which dimensions differentiate models?
    dim_variance = _compute_dimension_variance(report)

    html = HTML_TEMPLATE.render(
        report=report,
        images=images,
        input_images=input_images,
        has_failures=has_failures,
        is_consensus=report.consensus_mode,
        consensus_stats=consensus_stats,
        dim_variance=dim_variance,
        dim_weights=dim_weights,
        metric_names=metric_names,
        metric_labels=metric_labels,
        has_weighted=has_weighted,
        metric_warnings=metric_warnings,
        scoring_config=scoring_config,
        radar_data_json=json.dumps(radar_data),
        dimensions_json=json.dumps(report.dimensions),
    )
    Path(path).write_text(html, encoding="utf-8")


def _compute_dimension_variance(report: BenchReport) -> dict[str, str]:
    """Classify dimensions as 'differentiator', 'ceiling', or '' based on cross-model variance.

    - differentiator: cross-model stddev >= 0.5 (models score very differently)
    - ceiling: all models within 0.3 of each other AND average >= 4.5 (everyone maxes out)
    """
    result: dict[str, str] = {}
    for dim in report.dimensions:
        model_avgs = [
            ms.dimension_averages.get(dim, 0.0)
            for ms in report.summary.values()
        ]
        if len(model_avgs) < 2:
            result[dim] = ""
            continue
        mean = sum(model_avgs) / len(model_avgs)
        variance = sum((s - mean) ** 2 for s in model_avgs) / len(model_avgs)
        stddev = variance ** 0.5
        spread = max(model_avgs) - min(model_avgs)
        if stddev >= 0.5:
            result[dim] = "differentiator"
        elif spread <= 0.3 and mean >= 4.5:
            result[dim] = "ceiling"
        else:
            result[dim] = ""
    return result


def _compute_consensus_stats(report: BenchReport) -> dict[str, Any]:
    """Aggregate consensus statistics across all items for the HTML panel."""
    high = disputed = degraded = total = 0
    # Per-judge score accumulators
    judge_totals: dict[str, float] = {}
    judge_counts: dict[str, int] = {}
    consensus_score_sum = 0.0
    consensus_score_count = 0
    disputes: list[dict[str, Any]] = []

    for item in report.items:
        for model, result in item.results.items():
            for score_data in result.scores:
                agree = score_data.agreement
                if not agree:
                    continue
                total += 1
                if agree == "high":
                    high += 1
                elif agree == "disputed":
                    disputed += 1
                    disputes.append({
                        "model": model,
                        "dimension": score_data.dimension,
                        "final_score": score_data.score,
                        "judge_scores": dict(score_data.judge_scores),
                    })
                elif agree == "degraded":
                    degraded += 1

                # Accumulate per-judge scores
                for jname, jscore in score_data.judge_scores.items():
                    judge_totals[jname] = judge_totals.get(jname, 0.0) + jscore
                    judge_counts[jname] = judge_counts.get(jname, 0) + 1

                consensus_score_sum += score_data.score
                consensus_score_count += 1

    # Compute averages
    judge_averages = {
        jname: judge_totals[jname] / judge_counts[jname]
        for jname in judge_totals
    }
    consensus_avg = consensus_score_sum / consensus_score_count if consensus_score_count else 0.0

    # Count tiebreakers (disputed dimensions that have 3 judge scores)
    tiebreaker_count = sum(1 for d in disputes if len(d["judge_scores"]) >= 3)

    return {
        "high_pct": round(high / total * 100) if total else 0,
        "disputed_pct": round(disputed / total * 100) if total else 0,
        "degraded_count": degraded,
        "total": total,
        "tiebreaker_count": tiebreaker_count,
        "judge_averages": judge_averages,
        "judge_counts": judge_counts,
        "consensus_avg": round(consensus_avg, 2),
        "disputes": disputes,
    }
