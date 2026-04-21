"""Leaderboard HTML report generator.

Produces a self-contained, standalone HTML page with:
- Sortable table (vanilla JS)
- Adjustable weight panel (VLM / deterministic split)
- Lock pins on sub-sliders to prevent redistribution
- Family + seed filtering
- Dark/light mode toggle
- URL-based shareable custom views
- Archive dropdown for dated snapshots
- Prompt gallery with model output thumbnails
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Template

from .leaderboard_data import LEADERBOARD_MODELS

if TYPE_CHECKING:
    from ..bench.types import BenchReport


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class LeaderboardEntry:
    """One row in the leaderboard table."""

    model_key: str
    display_name: str
    family: str
    fal_endpoint: str
    cost_per_image: float
    seed_label: str  # "42" or "—" (no seed support)
    imgsys_elo: int | None
    license_type: str  # "open" or "proprietary"
    avg_time_s: float
    # VLM scores (1-5)
    visual_quality: float
    prompt_adherence: float
    text_rendering: float | None  # None if no text prompts
    # Deterministic metrics (0-1)
    clip_score: float
    sharpness: float
    nima_score: float | None  # None if not available
    arniqa_score: float | None = None
    topiq_score: float | None = None
    musiq_score: float | None = None
    ocr_accuracy: float | None = None  # None if not available or no text prompts


@dataclass
class LeaderboardData:
    """Complete data passed to the Jinja2 template."""

    title: str
    date: str
    model_count: int
    prompt_count: int
    seed_count: int
    noseed_count: int
    judge: str
    entries: list[LeaderboardEntry]
    archive_versions: list[dict[str, Any]]
    show_fal_links: bool
    prompts: list[dict[str, Any]] = field(default_factory=list)
    prompt_categories: list[str] = field(default_factory=list)  # sorted unique categories
    open_count: int = 0
    proprietary_count: int = 0
    config: dict[str, Any] = field(default_factory=dict)  # bench config (seed, image_size, etc.)
    display_names: dict[str, str] = field(default_factory=dict)  # model_key -> display name
    best_value: str = ""  # model_key with highest score/$
    best_value_display: str = ""  # display name
    best_value_cost: float = 0.0  # $/img
    judge_agree_pct: float = 0.0  # % of dimensions where judges agree within 0.5
    judge_avg_gap: float = 0.0  # avg gap between highest and lowest judge
    judge_extreme_pct: float = 0.0  # % with gap > 1.0
    cdn_base: str = ""  # CDN base URL for images (e.g. https://evalytic.ai/leaderboard)


# ---------------------------------------------------------------------------
# Prompt category detection
# ---------------------------------------------------------------------------

_ID_PREFIX_CATEGORIES = {
    "text": "Text Rendering",
    "comp": "Composition",
    "port": "Portrait",
    "anim": "Animal / Nature",
    "land": "Landscape",
    "art": "Artistic Style",
    "prod": "Product",
    "arch": "Architecture",
    "food": "Food",
    "fash": "Fashion",
    "sci": "Sci-Fi / Fantasy",
    "abs": "Abstract",
}

# (category, keywords) — first match wins, order matters
_KEYWORD_RULES: list[tuple[str, list[str]]] = [
    ("Text Rendering", ["sign reading", "written", "text", "lettering", "chalkboard", "book cover", "title", "label", "neon sign"]),
    ("Portrait", ["portrait", "woman", "man", "person", "bride", "blacksmith", "samurai", "face", "elderly", "child"]),
    ("Animal / Nature", ["cat", "dog", "bird", "hummingbird", "butterfly", "fox", "horse", "animal", "owl", "fish"]),
    ("Landscape", ["landscape", "field", "mountain", "ocean", "forest", "sunset", "lightning", "valley", "desert", "lake"]),
    ("Composition", ["arranged", "in a row", "in a circle", "three ", "five ", "counted", "group of"]),
    ("Artistic Style", ["ukiyo-e", "woodblock", "impressionist", "watercolor", "oil painting", "art nouveau", "cubist", "surreal"]),
    ("Product", ["product", "sneaker", "watch", "bottle", "packaging", "marble countertop"]),
    ("Architecture", ["building", "cathedral", "mosque", "temple", "riad", "architecture", "skyscraper", "bridge"]),
    ("Food", ["food", "cake", "coffee", "sushi", "bread", "fruit", "meal", "dish"]),
    ("Sci-Fi / Fantasy", ["sci-fi", "fantasy", "spaceship", "dragon", "robot", "alien", "cyberpunk", "futuristic"]),
]


def _categorize_prompt(item_id: str, prompt: str, tags: list[str] | None = None) -> str:
    """Derive a display category from item_id prefix, tags, or prompt keywords."""
    # 1. Try item_id prefix (e.g. "text-001" → "text")
    if "-" in item_id:
        prefix = item_id.rsplit("-", 1)[0]
        if prefix in _ID_PREFIX_CATEGORIES:
            return _ID_PREFIX_CATEGORIES[prefix]

    # 2. Try tags (e.g. ["portrait", "hard"])
    if tags:
        for tag in tags:
            t = tag.lower()
            if t in _ID_PREFIX_CATEGORIES:
                return _ID_PREFIX_CATEGORIES[t]

    # 3. Keyword match on prompt text
    prompt_lower = prompt.lower()
    for category, keywords in _KEYWORD_RULES:
        for kw in keywords:
            if kw in prompt_lower:
                return category

    return "Other"


# ---------------------------------------------------------------------------
# Enrichment: BenchReport -> LeaderboardData
# ---------------------------------------------------------------------------


def _calc_judge_agreement(report: BenchReport) -> dict[str, float]:
    """Calculate inter-judge agreement statistics from a BenchReport."""
    gaps: list[float] = []
    agree = extreme = 0
    for item in report.items:
        for _model, result in item.results.items():
            if result.status != "success":
                continue
            for dr in result.scores:
                if not dr.judge_scores or len(dr.judge_scores) < 2:
                    continue
                vals = list(dr.judge_scores.values())
                gap = max(vals) - min(vals)
                gaps.append(gap)
                if gap <= 0.5:
                    agree += 1
                if gap > 1.0:
                    extreme += 1
    total = len(gaps)
    if total == 0:
        return {"judge_agree_pct": 0, "judge_avg_gap": 0, "judge_extreme_pct": 0}
    return {
        "judge_agree_pct": round(agree / total * 100, 1),
        "judge_avg_gap": round(sum(gaps) / total, 2),
        "judge_extreme_pct": round(extreme / total * 100, 1),
    }


def enrich_report(
    report: BenchReport,
    *,
    archive_versions: list[dict[str, Any]] | None = None,
    show_fal_links: bool = True,
) -> LeaderboardData:
    """Enrich a BenchReport with leaderboard metadata.

    Merges BenchReport scores with LEADERBOARD_MODELS metadata (family,
    display name, ELO, seed support) and registry info (endpoint, cost).
    """
    from ..bench.registry import MODEL_REGISTRY

    entries: list[LeaderboardEntry] = []

    for model_key, ms in report.summary.items():
        # Metadata from lookup
        meta = LEADERBOARD_MODELS.get(model_key)
        reg = MODEL_REGISTRY.get(model_key)

        display_name = meta.display_name if meta else model_key
        family = meta.family if meta else "Other"
        imgsys_elo = meta.imgsys_elo if meta else None
        seed_supported = meta.seed_supported if meta else True
        license_type = meta.license_type if meta else "proprietary"
        fal_endpoint = reg.endpoint if reg else model_key
        cost = ms.cost_per_image if ms.cost_per_image > 0 else (reg.cost_per_image if reg else 0.0)

        # Avg generation time
        avg_time_s = 0.0
        if ms.item_count > 0 and ms.total_generation_time_ms > 0:
            avg_time_s = ms.total_generation_time_ms / ms.item_count / 1000

        # text_rendering: None if dimension not present or score is 0
        tr = ms.dimension_averages.get("text_rendering")
        if tr is not None and tr == 0.0 and "text_rendering" not in report.dimensions:
            tr = None

        entries.append(LeaderboardEntry(
            model_key=model_key,
            display_name=display_name,
            family=family,
            fal_endpoint=fal_endpoint,
            cost_per_image=cost,
            seed_label=str(report.config.get("seed", 42)) if seed_supported else "\u2014",
            imgsys_elo=imgsys_elo,
            license_type=license_type,
            avg_time_s=round(avg_time_s, 1),
            visual_quality=ms.dimension_averages.get("visual_quality", 0.0),
            prompt_adherence=ms.dimension_averages.get("prompt_adherence", 0.0),
            text_rendering=tr,
            clip_score=ms.metric_averages.get("clip_score", 0.0),
            sharpness=ms.metric_averages.get("sharpness", 0.0),
            nima_score=ms.metric_averages.get("nima_score"),
            arniqa_score=ms.metric_averages.get("arniqa_score"),
            topiq_score=ms.metric_averages.get("topiq_score"),
            musiq_score=ms.metric_averages.get("musiq_score"),
            ocr_accuracy=ms.metric_averages.get("ocr_accuracy"),
        ))

    seed_count = sum(1 for e in entries if e.seed_label not in ("\u2014", "median of 3"))
    noseed_count = len(entries) - seed_count
    open_count = sum(1 for e in entries if e.license_type == "open")
    proprietary_count = len(entries) - open_count

    # Judge string
    if report.consensus_mode and report.judges:
        # Pretty-print judge names for display
        _JUDGE_DISPLAY = {
            "fal/claude-sonnet-4-6": "Claude Sonnet 4.6",
            "bedrock/eu.anthropic.claude-sonnet-4-6": "Claude Sonnet 4.6",
            "fal/gpt-5.2": "GPT-5.2",
            "gemini-2.5-flash": "Gemini 2.5 Flash",
        }
        pretty = [_JUDGE_DISPLAY.get(j, j) for j in report.judges]
        judge_str = f"3-Judge Median ({', '.join(pretty)})"
    elif report.judge:
        judge_str = report.judge
    else:
        judge_str = ""

    # Date from report
    date = report.created_at[:10] if report.created_at else ""

    # Build model_key -> display_name map for prompt browser
    display_names: dict[str, str] = {}
    for e in entries:
        display_names[e.model_key] = e.display_name

    # Extract prompts + images for gallery
    prompts_gallery: list[dict[str, Any]] = []
    categories_seen: set[str] = set()
    for item in report.items:
        images: dict[str, str] = {}
        for model_key, result in item.results.items():
            if result.status == "success" and result.image_url:
                images[model_key] = result.image_url
        prompt_text = item.prompt or item.instruction or f"Item {item.item_id}"
        category = _categorize_prompt(item.item_id, prompt_text, item.tags)
        categories_seen.add(category)
        prompts_gallery.append({
            "item_id": item.item_id,
            "prompt": prompt_text,
            "category": category,
            "images": images,
        })

    return LeaderboardData(
        title="Evalytic Image Models Leaderboard",
        date=date,
        model_count=len(entries),
        prompt_count=len(report.items) or max((ms.sample_count for ms in report.summary.values()), default=0),
        seed_count=seed_count,
        noseed_count=noseed_count,
        judge=judge_str,
        entries=entries,
        archive_versions=archive_versions or [],
        show_fal_links=show_fal_links,
        prompts=prompts_gallery,
        prompt_categories=sorted(categories_seen),
        open_count=open_count,
        proprietary_count=proprietary_count,
        config=report.config if report.config else {},
        display_names=display_names,
        best_value=report.best_value,
        best_value_display=display_names.get(report.best_value, report.best_value),
        best_value_cost=report.summary[report.best_value].cost_per_image if report.best_value and report.best_value in report.summary else 0.0,
        **_calc_judge_agreement(report),
    )


# ---------------------------------------------------------------------------
# Default overall calculation (server-side, 80/20 weights)
# ---------------------------------------------------------------------------

_DEFAULT_VLM_WEIGHT = 0.60
_DEFAULT_DET_WEIGHT = 0.40
_DEFAULT_VLM_SUB = {"visual_quality": 0.40, "prompt_adherence": 0.40, "text_rendering": 0.20}
_DEFAULT_DET_SUB = {"clip_score": 0.25, "sharpness": 0.15, "nima_score": 0.25, "arniqa_score": 0.00, "topiq_score": 0.35, "musiq_score": 0.00}

# CLIP normalize range (raw 0.15-0.40 -> 0-1)
_CLIP_RANGE = (0.15, 0.40)


def _calc_overall(e: LeaderboardEntry) -> float:
    """Calculate default overall for one entry (server-side).

    Formula: overall = 1 + 4 * (vlm_w * vlm_unit + det_w * det_unit)
    where vlm_unit = (vlm_avg - 1) / 4   (min-max 1-5 -> 0-1)
    and   det_unit = weighted avg of calibrated metrics
    """
    # VLM weighted average
    vlm_scores = []
    vlm_weights = []
    for dim, w in _DEFAULT_VLM_SUB.items():
        val = getattr(e, dim, None) if dim != "text_rendering" else e.text_rendering
        if dim == "visual_quality":
            val = e.visual_quality
        elif dim == "prompt_adherence":
            val = e.prompt_adherence
        if val is not None and val > 0:
            vlm_scores.append(val)
            vlm_weights.append(w)

    if not vlm_scores:
        return 1.0

    # Renormalize VLM sub-weights
    w_sum = sum(vlm_weights)
    vlm_avg = sum(s * w / w_sum for s, w in zip(vlm_scores, vlm_weights))
    vlm_unit = (vlm_avg - 1.0) / 4.0

    # Deterministic calibrated average
    det_vals = []
    det_ws = []

    # CLIP -- calibrate
    if e.clip_score > 0:
        lo, hi = _CLIP_RANGE
        clip_norm = max(0.0, min(1.0, (e.clip_score - lo) / (hi - lo)))
        det_vals.append(clip_norm)
        det_ws.append(_DEFAULT_DET_SUB["clip_score"])

    # Sharpness -- raw
    if e.sharpness > 0:
        det_vals.append(min(1.0, e.sharpness))
        det_ws.append(_DEFAULT_DET_SUB["sharpness"])

    # NIMA -- raw
    if e.nima_score is not None and e.nima_score > 0:
        det_vals.append(min(1.0, e.nima_score))
        det_ws.append(_DEFAULT_DET_SUB["nima_score"])

    # ARNIQA -- raw
    if e.arniqa_score is not None and e.arniqa_score > 0:
        det_vals.append(min(1.0, e.arniqa_score))
        det_ws.append(_DEFAULT_DET_SUB["arniqa_score"])

    # TOPIQ -- raw
    if e.topiq_score is not None and e.topiq_score > 0:
        det_vals.append(min(1.0, e.topiq_score))
        det_ws.append(_DEFAULT_DET_SUB["topiq_score"])

    # MUSIQ -- raw
    if e.musiq_score is not None and e.musiq_score > 0:
        det_vals.append(min(1.0, e.musiq_score))
        det_ws.append(_DEFAULT_DET_SUB["musiq_score"])

    if det_vals:
        dw_sum = sum(det_ws)
        det_unit = sum(v * w / dw_sum for v, w in zip(det_vals, det_ws))
    else:
        det_unit = 0.0

    # Adjust weights if no deterministic metrics
    vlm_w = _DEFAULT_VLM_WEIGHT
    det_w = _DEFAULT_DET_WEIGHT
    if not det_vals:
        vlm_w = 1.0
        det_w = 0.0

    combined = vlm_w * vlm_unit + det_w * det_unit
    return 1.0 + 4.0 * combined  # no rounding — let display handle precision


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

LEADERBOARD_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ data.title }} — Best AI Image Generators Ranked ({{ data.date[:7] }})</title>
<meta name="description" content="Independent benchmark of {{ data.model_count }} AI image generation models including Flux, Imagen, Seedream, GPT Image, and more. Ranked by {{ data.judge }} + CLIP Score + NIMA + TOPIQ. Updated {{ data.date }}.">
<meta name="keywords" content="AI image generation, image model benchmark, text to image comparison, Flux vs Imagen vs Seedream vs GPT Image, AI image quality, CLIP score, NIMA score, TOPIQ, image generation leaderboard, best AI image generator 2026, Evalytic, fal.ai benchmark">
<meta name="robots" content="index, follow">
<link rel="canonical" href="https://evalytic.ai/leaderboard">
<meta property="og:title" content="{{ data.title }} — Best AI Image Generators Ranked">
<meta property="og:description" content="Independent benchmark of {{ data.model_count }} AI image models. Flux, Imagen, Seedream, GPT Image and more ranked by 3-judge median + 5 deterministic metrics.">
<meta property="og:type" content="website">
<meta property="og:url" content="https://evalytic.ai/leaderboard">
<meta property="og:site_name" content="Evalytic">
<meta property="og:image" content="https://evalytic.ai/og-leaderboard.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{{ data.title }}">
<meta name="twitter:description" content="{{ data.model_count }} AI image models benchmarked with 3-judge median + CLIP + NIMA + TOPIQ.">
<meta name="twitter:image" content="https://evalytic.ai/og-leaderboard.png">
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "WebPage",
  "name": "{{ data.title }}",
  "description": "Independent benchmark of {{ data.model_count }} AI image generation models ranked by visual quality, prompt adherence, and text rendering using 3-judge median and 5 deterministic metrics.",
  "url": "https://evalytic.ai/leaderboard",
  "dateModified": "{{ data.date }}",
  "publisher": {
    "@type": "Organization",
    "name": "Evalytic",
    "url": "https://evalytic.ai",
    "logo": {
      "@type": "ImageObject",
      "url": "https://evalytic.ai/og-image.png"
    }
  },
  "image": "https://evalytic.ai/og-leaderboard.png",
  "breadcrumb": {
    "@type": "BreadcrumbList",
    "itemListElement": [
      {"@type": "ListItem", "position": 1, "name": "Evalytic", "item": "https://evalytic.ai"},
      {"@type": "ListItem", "position": 2, "name": "Leaderboard", "item": "https://evalytic.ai/leaderboard"}
    ]
  },
  "mainEntity": {
    "@type": "Dataset",
    "name": "Evalytic Image Models Benchmark",
    "description": "{{ data.model_count }} text-to-image AI models evaluated across {{ data.prompt_count }} prompts using consensus VLM judges and deterministic metrics.",
    "license": "https://opensource.org/licenses/MIT",
    "dateModified": "{{ data.date }}",
    "variableMeasured": ["Visual Quality", "Prompt Adherence", "Text Rendering", "CLIP Score", "Sharpness", "NIMA Score", "TOPIQ"]
  }
}
</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0a0a0a; --surface: #111827; --border: rgba(31, 41, 55, 0.5);
  --text: #ffffff; --text-muted: #9ca3af; --accent: #4ade80; --accent-dark: #22c55e;
  --blue: #3b82f6; --gold: #fbbf24; --silver: #94a3b8;
  --bronze: #d97706; --red: #ef4444; --yellow: #eab308; --orange: #f97316;
  color-scheme: dark;
}
.light {
  --bg: #ffffff; --surface: #f9fafb; --border: #e5e7eb; --text: #111827;
  --text-muted: #6b7280; --accent: #16a34a; --accent-dark: #16a34a; --blue: #2563eb;
  color-scheme: light;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
::selection { background: var(--blue); color: white; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
.light ::-webkit-scrollbar-thumb { background: #d1d5db; }
body { font-family: 'Inter', system-ui, -apple-system, sans-serif; background: var(--bg);
       color: var(--text); line-height: 1.5; padding: 1.5rem; max-width: 1400px; margin: 0 auto;
       -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
a { color: var(--blue); text-decoration: none; }
a:hover { text-decoration: underline; }

/* Header */
.header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.5rem; }
.header h1 { font-size: 1.5rem; font-weight: 700; color: var(--text); }
.header-right { display: flex; align-items: center; gap: 0.75rem; }
.powered-by { font-size: 0.7rem; color: var(--text-muted); letter-spacing: 0.02em; }
.powered-by span { color: var(--accent); font-weight: 600; }
.subtitle { color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1rem; }
.badge { background: var(--surface); border: 1px solid var(--border); border-radius: 4px;
         padding: 0.15rem 0.5rem; font-size: 0.75rem; color: var(--text-muted); }

/* Controls */
.controls { display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 1rem; }
select, button { font-family: 'Inter', system-ui, sans-serif; font-size: 0.82rem; padding: 0.4rem 0.75rem;
  background: var(--surface); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; cursor: pointer; min-height: 2.2rem; }
button:hover { border-color: var(--accent); }
.btn-active { background: var(--accent-dark); color: #fff; border-color: var(--accent-dark); }

/* Toggle */
.theme-toggle { background: none; border: none; font-size: 1.2rem; cursor: pointer; padding: 0.2rem; color: var(--text); }

/* Weight Panel */
.weight-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 1rem 1.2rem; margin-bottom: 1rem; display: none; }
.weight-panel.open { display: block; }
.weight-panel h3 { font-size: 0.85rem; margin-bottom: 0.6rem; }
.weight-row { display: flex; align-items: center; gap: 0.5rem; margin: 0.3rem 0; font-size: 0.8rem; }
.weight-row label { width: 140px; flex-shrink: 0; }
.weight-row input[type=range] { flex: 1; accent-color: var(--accent); }
.weight-row output { width: 36px; text-align: right; font-variant-numeric: tabular-nums;
  font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; }
.weight-group { margin: 0.5rem 0; padding: 0.5rem 0; border-top: 1px solid var(--border); }
.weight-group:first-child { border-top: none; padding-top: 0; }
.presets { display: flex; gap: 0.4rem; margin-bottom: 0.8rem; }
.weight-actions { display: flex; gap: 0.5rem; margin-top: 0.8rem; align-items: center; }
.weight-actions .apply-btn { background: var(--accent); color: #fff; font-weight: 600;
  padding: 0.4rem 1.2rem; border-radius: 6px; border: none; cursor: pointer; font-size: 0.82rem;
  transition: opacity 0.15s; }
.weight-actions .apply-btn:hover { opacity: 0.85; }
.toast { position: fixed; bottom: 1.5rem; left: 50%; transform: translateX(-50%);
  background: var(--accent-dark); color: #fff; padding: 0.4rem 1rem; border-radius: 4px;
  font-size: 0.85rem; opacity: 0; transition: opacity 0.3s; pointer-events: none; z-index: 999; }
.toast.show { opacity: 1; }
.info-icon { color: var(--text-muted); cursor: help; font-size: 0.7rem; }

/* Group sum indicator */
.group-sum { font-size: 0.72rem; font-family: 'JetBrains Mono', monospace;
  font-variant-numeric: tabular-nums; margin-left: auto; padding: 0.15rem 0.5rem;
  border-radius: 3px; transition: color 0.15s, background 0.15s; }
.group-sum.ok { color: var(--accent); background: rgba(34,197,94,0.08); }
.group-sum.err { color: #ef4444; background: rgba(239,68,68,0.08); }

/* Validation modal */
.wt-modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.5); z-index: 1100;
  display: none; align-items: center; justify-content: center; }
.wt-modal-overlay.open { display: flex; }
.wt-modal { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 1.5rem 2rem; max-width: 400px; width: 90%; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
.wt-modal h3 { margin: 0 0 0.8rem; font-size: 1rem; color: #ef4444; }
.wt-modal p { font-size: 0.85rem; color: var(--text-muted); margin: 0.4rem 0; line-height: 1.5; }
.wt-modal .err-detail { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
  background: rgba(239,68,68,0.06); padding: 0.5rem 0.8rem; border-radius: 4px;
  margin: 0.6rem 0; border-left: 3px solid #ef4444; }
.wt-modal button { margin-top: 1rem; background: var(--accent); color: #fff; border: none;
  padding: 0.4rem 1.2rem; border-radius: 6px; cursor: pointer; font-weight: 600; }

/* Column tooltip (JS-positioned) */
.col-tooltip { position: fixed; z-index: 1200; background: var(--surface); color: var(--text);
  border: 1px solid var(--border); padding: 0.5rem 0.7rem; border-radius: 6px;
  font-size: 0.75rem; font-weight: 400; text-transform: none; letter-spacing: 0;
  white-space: normal; width: 240px; line-height: 1.45;
  box-shadow: 0 4px 16px rgba(0,0,0,0.2); pointer-events: none;
  opacity: 0; transition: opacity 0.15s; }
.col-tooltip.visible { opacity: 1; }

/* Table */
.try-link { display: inline-block; color: var(--blue); font-size: 0.72rem; font-weight: 500;
  text-decoration: none; opacity: 0.8; padding: 0.15rem 0.4rem; border-radius: 4px;
  border: 1px solid rgba(59,130,246,0.2); background: rgba(59,130,246,0.05); }
.try-link:hover { opacity: 1; background: rgba(59,130,246,0.12); border-color: rgba(59,130,246,0.35); text-decoration: none; }
td[data-col="fal"] { text-align: center; }
.table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table { width: 100%; border-collapse: collapse; font-size: 0.8rem; white-space: nowrap; }
caption { text-align: left; font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.3rem; }
th, td { padding: 0.5rem 0.7rem; text-align: right; border-bottom: 1px solid var(--border); }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
th { background: var(--surface); font-weight: 500; font-size: 0.7rem; color: var(--text-muted);
     text-transform: uppercase; letter-spacing: 0.05em; position: sticky; top: 0; z-index: 10;
     cursor: pointer; user-select: none; }
th[aria-sort]::after { content: " \\25B4\\25BE"; font-size: 0.6em; opacity: 0.3; }
th[aria-sort="ascending"]::after { content: " \\25B4"; opacity: 0.8; }
th[aria-sort="descending"]::after { content: " \\25BE"; opacity: 0.8; }
tbody tr { transition: background 0.1s; }
tbody tr:hover { background: rgba(74,222,128,0.04); }
.light tbody tr:hover { background: rgba(22,163,106,0.04); }
td { font-variant-numeric: tabular-nums; font-family: 'JetBrains Mono', 'Inter', monospace; font-size: 0.78rem; }
td[data-col="name"] { font-family: 'Inter', system-ui, sans-serif; }
td[data-col="family"] { font-family: 'Inter', system-ui, sans-serif; }
td[data-col="seed"] { font-family: 'Inter', system-ui, sans-serif; font-size: 0.75rem; min-width: 2.5rem; text-align: center; }
td[data-col="spd"], th[data-sort="spd"] { padding-left: 0.2rem; padding-right: 0.2rem; font-size: 0.7rem; }

/* Rank medals — top 3 rows */
tr[data-rank="1"] { background: rgba(234,179,8,0.06); }
tr[data-rank="2"] { background: rgba(192,192,192,0.06); }
tr[data-rank="3"] { background: rgba(205,127,50,0.06); }
tr[data-rank="1"] td:first-child,
tr[data-rank="2"] td:first-child,
tr[data-rank="3"] td:first-child { border-left: 3px solid; font-weight: 700; }
tr[data-rank="1"] td:first-child { border-left-color: var(--gold); }
tr[data-rank="2"] td:first-child { border-left-color: var(--silver); }
tr[data-rank="3"] td:first-child { border-left-color: var(--bronze); }
/* All rows same text size — no special sizing for top 3 */
.rank-medal { font-size: 1rem; }

/* License badge */
.license-badge { display: inline-block; font-size: 0.5rem; font-weight: 600; padding: 0.08rem 0.3rem;
  border-radius: 3px; vertical-align: middle; margin-left: 0.25rem; letter-spacing: 0.02em;
  font-family: 'Inter', system-ui, sans-serif; }
.license-badge.open { background: rgba(34,197,94,0.15); color: #16a34a; border: 1px solid rgba(34,197,94,0.3); }
.light .license-badge.open { background: rgba(22,163,74,0.1); color: #15803d; border-color: rgba(22,163,74,0.25); }


/* Score colors (low-opacity backgrounds) */
.sc-5 { background: rgba(34,197,94,0.12); }
.sc-4 { background: rgba(234,179,8,0.10); }
.sc-3 { background: rgba(249,115,22,0.10); }
.sc-2 { background: rgba(239,68,68,0.10); }
.sc-null { color: var(--text-muted); }

/* Prompt Browser */
.prompt-browser { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 1rem 1.2rem; margin-top: 1.5rem; }
.prompt-browser summary { cursor: pointer; font-weight: 600; font-size: 0.9rem; }
.pb-nav { display: flex; align-items: center; gap: 0.5rem; margin: 0.75rem 0; flex-wrap: wrap; }
.pb-nav button { padding: 0.25rem 0.5rem; font-size: 0.78rem; min-width: 2rem; }
.pb-nav button:disabled { opacity: 0.3; cursor: default; }
.pb-select { padding: 0.3rem 0.5rem; font-size: 0.78rem; border: 1px solid var(--border);
  border-radius: 4px; background: var(--bg); color: var(--text); }
.pb-prompt-dropdown { flex: 1; min-width: 200px; max-width: 500px;
  overflow: hidden; text-overflow: ellipsis; }
.pb-counter { font-size: 0.78rem; color: var(--text-muted); font-variant-numeric: tabular-nums;
  font-family: 'JetBrains Mono', monospace; min-width: 3.5rem; text-align: center; }
.pb-cat-tag { display: inline-block; font-size: 0.65rem; font-weight: 600; padding: 0.1rem 0.4rem;
  border-radius: 3px; background: var(--accent); color: #fff; margin-right: 0.4rem;
  vertical-align: middle; font-style: normal; letter-spacing: 0.02em; }
.pb-prompt-text { font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.75rem;
  font-style: italic; line-height: 1.4; padding: 0.5rem 0; border-bottom: 1px solid var(--border); }
.pb-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 0.5rem; margin-top: 0.75rem; }
.pb-card { text-align: center; }
.pb-card img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 4px;
  border: 1px solid var(--border); transition: transform 0.15s, border-color 0.15s; cursor: pointer;
  image-rendering: auto; }
.pb-card img:hover { transform: scale(1.05); border-color: var(--accent-dark); }
.pb-label { display: block; font-size: 0.7rem; color: var(--text-muted); margin-top: 0.25rem;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pb-slide { display: none; }
.pb-slide.active { display: block; }
/* Lightbox */
.pb-lightbox { position: fixed; inset: 0; background: rgba(0,0,0,0.9); z-index: 1000;
  display: none; align-items: center; justify-content: center; cursor: pointer; }
.pb-lightbox.open { display: flex; }
.pb-lightbox img { max-width: 85vw; max-height: 80vh; border-radius: 8px; cursor: default;
  transition: opacity 0.15s; }
.pb-lightbox img.loading { opacity: 0.15; }
.pb-lightbox .lb-spinner { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
  width: 2.5rem; height: 2.5rem; border: 3px solid rgba(255,255,255,0.2);
  border-top-color: #fff; border-radius: 50%; animation: lb-spin 0.6s linear infinite;
  display: none; }
.pb-lightbox .lb-spinner.active { display: block; }
@keyframes lb-spin { to { transform: translate(-50%,-50%) rotate(360deg); } }
.pb-lightbox-info { position: fixed; bottom: 1.5rem; left: 50%; transform: translateX(-50%);
  text-align: center; pointer-events: none; }
.pb-lb-model { color: #fff; font-size: 1.25rem; font-weight: 700;
  background: rgba(0,0,0,0.75); padding: 0.4rem 1rem; border-radius: 6px; display: inline-block; }
.pb-lb-prompt { color: rgba(255,255,255,0.75); font-size: 0.95rem; margin-top: 0.4rem;
  max-width: 80vw; line-height: 1.4; }
.pb-lb-nav { position: fixed; top: 50%; transform: translateY(-50%); z-index: 1001;
  background: rgba(255,255,255,0.12); border: none; color: #fff; font-size: 1.5rem;
  width: 3rem; height: 3rem; border-radius: 50%; cursor: pointer;
  display: flex; align-items: center; justify-content: center; transition: background 0.15s; }
.pb-lb-nav:hover { background: rgba(255,255,255,0.25); }
.pb-lb-prev { left: 1rem; }
.pb-lb-next { right: 1rem; }

/* Methodology */
.methodology { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 1rem 1.2rem; margin-top: 1.5rem; }
.methodology summary { cursor: pointer; font-weight: 600; font-size: 0.9rem; }
.methodology ul { margin: 0.5rem 0 0 1.2rem; font-size: 0.85rem; color: var(--text-muted); }
.methodology li { margin: 0.2rem 0; }

/* Site header */
.site-header { width: 100%; padding: 0.75rem 1.5rem; border-bottom: 1px solid var(--border);
  font-family: 'Manrope', 'Inter', system-ui, sans-serif; }
.site-header-inner { display: flex; align-items: center; justify-content: space-between;
  max-width: 1280px; margin: 0 auto; }
.site-logo { display: flex; align-items: center; gap: 0.5rem; text-decoration: none; color: var(--text); }
.site-logo span { font-size: 1.05rem; font-weight: 700; letter-spacing: -0.02em; }
.site-nav { display: flex; align-items: center; gap: 1.25rem; }
.site-nav a { color: var(--text-muted); font-size: 0.81rem; font-weight: 500; text-decoration: none;
  transition: color 0.15s; }
.site-nav a:hover { color: var(--text); }
.site-nav .pip-btn { height: 2rem; padding: 0 0.75rem; display: inline-flex; align-items: center;
  justify-content: center; border-radius: 6px; background: var(--text); color: var(--bg);
  font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 500;
  border: none; cursor: pointer; transition: opacity 0.15s; }
.site-nav .pip-btn:hover { opacity: 0.85; }
/* Hamburger menu button — hidden on desktop */
.hamburger-btn { display: none; background: none; border: none; color: var(--text);
  font-size: 1.4rem; cursor: pointer; padding: 0.2rem; line-height: 1; }

/* Jump links */
.jump-links { display: inline-flex; gap: 0.15rem; margin-left: 0.75rem; }
.jump-links a { font-size: 0.78rem; color: var(--text-muted); text-decoration: none;
  padding: 0.15rem 0.45rem; border-radius: 4px; transition: all 0.15s; }
.jump-links a:hover { color: var(--text); background: var(--surface); }

/* Disclaimer banner */
.disclaimer-banner { background: rgba(34,197,94,0.06); border: 1px solid rgba(34,197,94,0.25);
  border-radius: 8px; padding: 0.6rem 1rem; margin-bottom: 1rem; text-align: center;
  font-size: 0.78rem; color: var(--text-muted); line-height: 1.5; }
.disclaimer-banner strong { color: var(--accent); }
.light .disclaimer-banner { background: rgba(22,163,74,0.06); border-color: rgba(22,163,74,0.2); }
.light .disclaimer-banner strong { color: #15803d; }

/* Footer */
.footer { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border);
  text-align: center; font-size: 0.75rem; color: var(--text-muted); }
.footer a { color: var(--accent); }

/* Responsive — Tablet */
@media (max-width: 1024px) {
  .weight-panel { flex-direction: column; gap: 0.75rem; }
  .weight-group { min-width: 100%; }
}

/* Responsive — Mobile */
@media (max-width: 768px) {
  body { padding: 0.75rem 1rem; }
  h1 { font-size: 1.1rem; }
  .header { flex-direction: column; gap: 0.5rem; text-align: center; }
  .header-right { justify-content: center; }

  /* Site header mobile — hamburger menu */
  .site-header { padding: 0.5rem 0.75rem; }
  .hamburger-btn { display: block; }
  .site-nav { display: none; position: absolute; top: 100%; left: 0; right: 0;
    flex-direction: column; align-items: flex-start; gap: 0; padding: 0.5rem 0;
    background: var(--bg); border-bottom: 1px solid var(--border);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15); z-index: 100; }
  .site-nav.open { display: flex; }
  .site-nav a { font-size: 0.85rem; padding: 0.6rem 1rem; width: 100%; }
  .site-nav a:hover { background: var(--surface); }
  .site-nav .pip-btn { display: none; }
  .site-header-inner { position: relative; }
  .disclaimer-banner { font-size: 0.72rem; padding: 0.5rem 0.75rem; margin: 0.5rem; }

  /* Table: hide # col, sticky model name col */
  .table-wrap { margin: 0; border-radius: 0; position: relative; }
  .table-wrap::after { content: "\2190 scroll \2192"; display: block; text-align: center;
    font-size: 0.65rem; color: var(--text-muted); opacity: 0.6; padding: 0.3rem 0; }
  table { font-size: 0.7rem; }
  th, td { padding: 0.3rem 0.4rem; }
  /* Hide rank (#) and family columns on mobile */
  th:first-child, td:first-child { display: none; }
  th:nth-child(3), td:nth-child(3) { display: none; }
  /* Sticky model name (2nd col, now visually first) */
  th:nth-child(2), td:nth-child(2) { position: sticky; left: 0; z-index: 5;
    background: var(--bg); min-width: 6rem; max-width: 9rem;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    border-right: 1px solid var(--border); font-size: 0.68rem; }
  td { font-size: 0.65rem; }
  .license-badge { font-size: 0.4rem; padding: 0.05rem 0.2rem; }

  /* Filters */
  .filters { flex-direction: column; gap: 0.4rem; }
  .filter-group { flex-wrap: wrap; gap: 0.3rem; }
  .filter-group button { font-size: 0.68rem; padding: 0.2rem 0.45rem; }
  .filter-group select { font-size: 0.72rem; }
  /* Filter panel responsive */
  .controls { position: relative; }
  #filter-panel { left: auto !important; right: 0; min-width: 0 !important; width: calc(100vw - 3rem); max-width: 280px; }

  /* Weight panel */
  .weight-panel { padding: 0.6rem; }
  .weight-group { min-width: 100%; }
  .sub-sliders { gap: 0.3rem; }

  /* Prompt browser */
  .prompt-browser { padding: 0.6rem; }
  .pb-nav { gap: 0.35rem; }
  .pb-select { font-size: 0.72rem; padding: 0.25rem 0.4rem; }
  .pb-prompt-dropdown { min-width: 120px; }
  .pb-grid { grid-template-columns: repeat(3, 1fr); gap: 0.4rem; }
  .pb-prompt-text { font-size: 0.78rem; }
  .pb-label { font-size: 0.55rem; }
  .pb-cat-tag { font-size: 0.55rem; padding: 0.05rem 0.3rem; }

  /* Lightbox mobile */
  .pb-lightbox img { max-width: 95vw; max-height: 70vh; }
  .pb-lb-model { font-size: 1rem; }
  .pb-lb-prompt { font-size: 0.8rem; max-width: 90vw; }
  .pb-lb-nav { width: 2.5rem; height: 2.5rem; font-size: 1.2rem; }
  .pb-lb-prev { left: 0.4rem; }
  .pb-lb-next { right: 0.4rem; }

  /* Methodology */
  .methodology { padding: 0.6rem; }
  .methodology ul { font-size: 0.78rem; }
}

/* Responsive — Small phones */
@media (max-width: 480px) {
  body { padding: 0.5rem 0.75rem; }
  h1 { font-size: 0.95rem; }
  .site-logo span { font-size: 0.9rem; }
  .pb-grid { grid-template-columns: repeat(3, 1fr); gap: 0.3rem; }
  .pb-lightbox img { max-width: 98vw; max-height: 65vh; border-radius: 4px; }
  .pb-lightbox-info { bottom: 0.8rem; }
  .pb-lb-model { font-size: 0.9rem; padding: 0.3rem 0.7rem; }
  .pb-lb-prompt { font-size: 0.72rem; }
  table { font-size: 0.6rem; }
  th, td { padding: 0.2rem 0.3rem; }
  th:nth-child(2), td:nth-child(2) { min-width: 5rem; max-width: 7rem; font-size: 0.6rem; }
  .license-badge { display: none; }
  /* Filter panel: full-width on small screens */
  #filter-panel { left: auto !important; right: -1rem; min-width: 0 !important; width: calc(100vw - 2rem); max-width: 280px; }
  .controls { position: relative; }
}
</style>
</head>
<body>

<!-- Site Header -->
<header class="site-header">
  <div class="site-header-inner">
    <a href="https://evalytic.ai" class="site-logo">
      <svg width="28" height="28" viewBox="0 0 100 100"><rect width="100" height="100" rx="22" fill="#1c1917"/><path d="M22 50 Q50 26 78 50 Q50 74 22 50Z" stroke="white" stroke-width="5" stroke-linejoin="round" fill="none"/></svg>
      <span>evalytic</span>
    </a>
    <div style="display:flex;align-items:center;gap:0.75rem">
      <nav class="site-nav" id="siteNav">
        <a href="https://evalytic.ai/showcase">Showcases</a>
        <a href="https://evalytic.ai/leaderboard" style="color:var(--text);font-weight:600">Leaderboard</a>
        <a href="https://docs.evalytic.ai/">Docs</a>
        <a href="https://github.com/evalytic/evalytic">GitHub</a>
        <button class="pip-btn" onclick="navigator.clipboard.writeText('pip install evalytic');this.textContent='Copied!';setTimeout(()=>this.textContent='pip install evalytic',1500)">pip install evalytic</button>
      </nav>
      <button class="hamburger-btn" onclick="document.getElementById('siteNav').classList.toggle('open');this.textContent=this.textContent==='&#x2715;'?'&#x2630;':'&#x2715;'" aria-label="Menu">&#x2630;</button>
    </div>
  </div>
</header>

<div class="header">
  <h1>{{ data.title }}</h1>
  <div class="header-right">
    {% if data.archive_versions %}
    <select id="archive-select" aria-label="Select leaderboard version">
      {% for v in data.archive_versions %}
      <option value="{{ v.file }}"{{ ' selected' if v.date == data.date }}>{{ v.label }}</option>
      {% endfor %}
    </select>
    {% endif %}
    <button class="theme-toggle" id="theme-toggle" aria-label="Toggle dark mode" role="switch" aria-checked="true" title="Toggle theme">&#9790;</button>
  </div>
</div>
<div class="subtitle">
  Last run: {{ data.date }} &middot; {{ data.model_count }} models &middot; {{ data.prompt_count }} prompts &middot; {{ data.judge or "&mdash;" }}
  &middot; Built with <a href="https://github.com/evalytic/evalytic" style="color:var(--accent)">Evalytic</a>
  <span class="jump-links">
    <a href="#rankings-table">#Rankings</a>
    <a href="#prompt-browser">#Images</a>
    <a href="#methodology">#Methodology</a>
  </span>
</div>

<!-- Weight panel toggle -->
<div class="controls">
  <button id="weight-toggle">Customize Weights &#9662;</button>
  <span style="color:var(--text-muted);font-size:0.75rem">|</span>
  <button id="share-btn" title="Copy shareable link">&#128279; Copy Link</button>
  <button id="share-x-btn" title="Share on X/Twitter" style="font-size:0.75rem;padding:0.4rem 0.6rem">&#120143; Post</button>
  <button id="share-li-btn" title="Share on LinkedIn" style="font-size:0.75rem;padding:0.4rem 0.6rem">in Share</button>
  <span style="color:var(--text-muted);font-size:0.75rem">|</span>
  <div style="position:relative;display:inline-block">
    <button id="filter-toggle" style="cursor:pointer">Filters <span id="filter-badge" style="display:none;background:var(--accent);color:#000;font-size:0.6rem;padding:0.05rem 0.35rem;border-radius:99px;margin-left:0.2rem;font-weight:700"></span> &#9662;</button>
    <div id="filter-panel" style="display:none;position:absolute;top:calc(100% + 0.4rem);left:0;z-index:60;background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:0.75rem 1rem;min-width:220px;box-shadow:0 8px 24px rgba(0,0,0,0.25)">
      <div style="margin-bottom:0.6rem">
        <label style="font-size:0.75rem;color:var(--text-muted);display:block;margin-bottom:0.25rem">Family</label>
        <select id="family-filter" aria-label="Filter by family" style="width:100%;font-size:0.8rem;padding:0.3rem 0.4rem;border-radius:4px;border:1px solid var(--border);background:var(--surface);color:var(--text)">
          <option value="all">All Families</option>
          {% for fam in families %}
          <option value="{{ fam }}">{{ fam }}</option>
          {% endfor %}
        </select>
      </div>
      <div style="margin-bottom:0.6rem">
        <label style="font-size:0.75rem;color:var(--text-muted);display:block;margin-bottom:0.25rem">Seed Support</label>
        <div class="filter-group" style="display:flex;gap:0.3rem">
          <button class="seed-btn btn-active" data-seed="all">All</button>
          <button class="seed-btn" data-seed="seed">Seed ({{ data.seed_count }})</button>
          <button class="seed-btn" data-seed="noseed">No-Seed ({{ data.noseed_count }})</button>
        </div>
      </div>
      <div>
        <label style="font-size:0.75rem;color:var(--text-muted);display:block;margin-bottom:0.25rem">License</label>
        <div class="filter-group" style="display:flex;gap:0.3rem">
          <button class="license-btn btn-active" data-license="all">All</button>
          <button class="license-btn" data-license="open">Open ({{ data.open_count }})</button>
          <button class="license-btn" data-license="proprietary">Proprietary ({{ data.proprietary_count }})</button>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Weight Panel -->
<div class="weight-panel" id="weight-panel">
  <div class="presets">
    <button class="preset-btn btn-active" data-preset="default">Default</button>
    <button class="preset-btn" data-preset="quality">Quality</button>
    <button class="preset-btn" data-preset="text">Text-Heavy</button>
    <button class="preset-btn" data-preset="ecommerce">E-Commerce</button>
    <button class="preset-btn" data-preset="speed">Speed</button>
    <button class="preset-btn" data-preset="budget">Budget</button>
    <button class="preset-btn" data-preset="photorealistic">Photorealistic</button>
  </div>
  <div class="weight-group">
    <div style="display:flex;align-items:center;gap:0.5rem">
      <h3 style="margin:0">VLM Dimensions</h3>
      <span class="group-sum ok" id="vlm-sum">= 100%</span>
    </div>
    <div class="weight-row">
      <label for="w-vlm">VLM Total</label>
      <input type="range" id="w-vlm" min="0" max="100" value="60">
      <output for="w-vlm">60%</output>
    </div>
    <div class="weight-row" style="padding-left:1rem">
      <label for="w-vq">Visual Quality</label>
      <input type="range" id="w-vq" min="0" max="100" value="40">
      <output for="w-vq">40%</output>
    </div>
    <div class="weight-row" style="padding-left:1rem">
      <label for="w-pa">Prompt Adherence</label>
      <input type="range" id="w-pa" min="0" max="100" value="40">
      <output for="w-pa">40%</output>
    </div>
    <div class="weight-row" style="padding-left:1rem">
      <label for="w-tr">Text Rendering <span class="info-icon" title="Only scored on prompts that contain text (e.g. signs, labels). ~12% of prompts.">&#9432;</span></label>
      <input type="range" id="w-tr" min="0" max="100" value="20">
      <output for="w-tr">20%</output>
    </div>
  </div>
  <div class="weight-group">
    <div style="display:flex;align-items:center;gap:0.5rem">
      <h3 style="margin:0">Deterministic Metrics</h3>
      <span class="group-sum ok" id="det-sum">= 100%</span>
    </div>
    <div class="weight-row">
      <label for="w-det">Deterministic Total</label>
      <input type="range" id="w-det" min="0" max="100" value="40">
      <output for="w-det">40%</output>
    </div>
    <div class="weight-row" style="padding-left:1rem">
      <label for="w-clip">CLIP Score</label>
      <input type="range" id="w-clip" min="0" max="100" value="25">
      <output for="w-clip">25%</output>
    </div>
    <div class="weight-row" style="padding-left:1rem">
      <label for="w-sharp">Sharpness</label>
      <input type="range" id="w-sharp" min="0" max="100" value="15">
      <output for="w-sharp">15%</output>
    </div>
    <div class="weight-row" style="padding-left:1rem">
      <label for="w-nima">NIMA</label>
      <input type="range" id="w-nima" min="0" max="100" value="25">
      <output for="w-nima">25%</output>
    </div>
    <div class="weight-row" style="padding-left:1rem">
      <label for="w-topiq">TOPIQ</label>
      <input type="range" id="w-topiq" min="0" max="100" value="35">
      <output for="w-topiq">35%</output>
    </div>
    <!-- ARNIQA removed (r=0.93 with TOPIQ), MUSIQ removed (CPU-slow, r=0.90 with TOPIQ) -->
    <input type="hidden" id="w-arniqa" value="0">
    <input type="hidden" id="w-musiq" value="0">
    <div class="weight-row" style="padding-left:1rem">
      <label for="w-time">Speed <span class="info-icon" title="Faster generation = higher score. Normalized: fastest model = 1.0, slowest = 0.0">&#9432;</span></label>
      <input type="range" id="w-time" min="0" max="100" value="0">
      <output for="w-time">0%</output>
    </div>
  </div>
  <div class="weight-actions">
    <button class="apply-btn" id="apply-btn">&#9654; Apply Weights</button>
    <button id="reset-btn">&#8634; Reset</button>
  </div>
</div>

<!-- Validation Modal -->
<div class="wt-modal-overlay" id="wt-modal">
  <div class="wt-modal">
    <h3>&#9888; Weight Validation Error</h3>
    <div id="wt-modal-body"></div>
    <button id="wt-modal-close">Got it</button>
  </div>
</div>

<!-- Leaderboard Table -->
<main>
<section aria-label="Rankings" class="table-wrap" id="rankings-table">
<table id="lb-table">
  <caption>Image model leaderboard ranked by overall score</caption>
  <thead>
    <tr>
      <th data-sort="rank" aria-sort="none" data-tip="Ranking based on weighted overall score">#</th>
      <th data-sort="name" aria-sort="none" data-tip="Model name and provider">Model</th>
      <th data-sort="family" aria-sort="none" data-tip="Model provider / family (e.g. BFL, Google, Ideogram)">Family</th>
      <th data-sort="cost" aria-sort="none" data-tip="Cost per image generation on fal.ai (USD)">$/img</th>
      <th data-sort="overall" aria-sort="descending" data-tip="Weighted combination of VLM judge scores and deterministic metrics. Adjustable via weight panel above.">Overall</th>
      <th data-sort="vq" aria-sort="none" data-tip="Visual Quality (1-5) — VLM judge score for overall image quality, coherence, and realism">VQ</th>
      <th data-sort="pa" aria-sort="none" data-tip="Prompt Adherence (1-5) — VLM judge score for how well the image matches the text prompt">PA</th>
      <th data-sort="tr" aria-sort="none" data-tip="Text Rendering (1-5) — VLM judge score for accuracy of text in the image. Only scored on prompts containing text (signs, labels). Shows '—' otherwise.">TR</th>
      <th data-sort="clip" aria-sort="none" data-tip="CLIP Score (0-1) — Deterministic text-image similarity via OpenAI CLIP ViT-L/14. Higher = better prompt match.">CLIP</th>
      <th data-sort="sharp" aria-sort="none" data-tip="Sharpness (0-1) — Laplacian variance measure. Higher = sharper, more detailed image.">Sharp</th>
      <th data-sort="nima" aria-sort="none" data-tip="NIMA Score (0-1) — Neural Image Assessment trained on human aesthetic ratings (AVA dataset). Higher = more aesthetically pleasing.">NIMA</th>
      <th data-sort="arniqa" aria-sort="none" data-tip="ARNIQA (0-1) — Learned quality regression trained on KonIQ-10k." style="display:none">ARNIQA</th>
      <th data-sort="topiq" aria-sort="none" data-tip="TOPIQ (0-1) — Top-down quality via CFANet (KonIQ-10k trained). State-of-the-art no-reference quality metric.">TOPIQ</th>
      <th data-sort="musiq" aria-sort="none" data-tip="MUSIQ (0-1) — Multi-Scale Image Quality transformer." style="display:none">MUSIQ</th>
      <th data-sort="spd" aria-sort="none" data-tip="Score per Dollar — Overall score / cost per image. Higher = better value for money.">S/$</th>
      <th data-sort="elo" aria-sort="none" data-tip="imgsys.org ELO rating — Community-voted ranking for independent reference.">imgsys</th>
      <th data-sort="time" aria-sort="none" data-tip="Average generation time per image (seconds)">Time</th>
      <th data-sort="seed" aria-sort="none" data-tip="Fixed random seed for reproducibility. '&mdash;' = model doesn't support seed.">Seed</th>
      {% if data.show_fal_links %}<th data-tip="Try this model on fal.ai">Try</th>{% endif %}
    </tr>
  </thead>
  <tbody>
    {% for e in entries_with_overall %}
    <tr data-rank="{{ loop.index }}"
        data-family="{{ e.entry.family }}"
        data-seed="{{ 'noseed' if e.entry.seed_label in ('\u2014', 'median of 3') else 'seed' }}"
        data-license="{{ e.entry.license_type }}"
        data-key="{{ e.entry.model_key }}">
      <td data-col="rank" data-value="{{ loop.index }}">{% if loop.index == 1 %}<span class="rank-medal">&#x1F947;</span>{% elif loop.index == 2 %}<span class="rank-medal">&#x1F948;</span>{% elif loop.index == 3 %}<span class="rank-medal">&#x1F949;</span>{% else %}{{ loop.index }}{% endif %}</td>
      <td data-col="name" data-value="{{ e.entry.display_name }}" title="{{ e.entry.model_key }}"><strong>{{ e.entry.display_name }}</strong>{% if e.entry.license_type == "open" %} <span class="license-badge open">Open</span>{% endif %}{% if e.entry.model_key == data.best_value %} <span class="license-badge" style="background:rgba(251,191,36,0.15);color:var(--gold);border:1px solid rgba(251,191,36,0.3);">Best Value</span>{% endif %}</td>
      <td data-col="family" data-value="{{ e.entry.family }}">{{ e.entry.family }}</td>
      <td data-col="cost" data-value="{{ e.entry.cost_per_image }}">${{ "%.3f" | format(e.entry.cost_per_image) }}</td>
      <td data-col="overall" data-value="{{ e.overall }}"><strong>{{ "%.3f" | format(e.overall) }}</strong></td>
      <td data-col="vq" data-value="{{ e.entry.visual_quality }}" class="{{ _score_class_vlm(e.entry.visual_quality) }}">{{ "%.1f" | format(e.entry.visual_quality) }}</td>
      <td data-col="pa" data-value="{{ e.entry.prompt_adherence }}" class="{{ _score_class_vlm(e.entry.prompt_adherence) }}">{{ "%.1f" | format(e.entry.prompt_adherence) }}</td>
      {% if e.entry.text_rendering is not none %}
      <td data-col="tr" data-value="{{ e.entry.text_rendering }}" class="{{ _score_class_vlm(e.entry.text_rendering) }}">{{ "%.1f" | format(e.entry.text_rendering) }}</td>
      {% else %}
      <td data-col="tr" data-value="" class="sc-null" aria-label="Not available">&mdash;</td>
      {% endif %}
      <td data-col="clip" data-value="{{ e.entry.clip_score }}" class="{{ _score_class_det(e.entry.clip_score) }}">{{ "%.3f" | format(e.entry.clip_score) }}</td>
      <td data-col="sharp" data-value="{{ e.entry.sharpness }}" class="{{ _score_class_det(e.entry.sharpness) }}">{{ "%.3f" | format(e.entry.sharpness) }}</td>
      {% if e.entry.nima_score is not none %}
      <td data-col="nima" data-value="{{ e.entry.nima_score }}" class="{{ _score_class_det(e.entry.nima_score) }}">{{ "%.3f" | format(e.entry.nima_score) }}</td>
      {% else %}
      <td data-col="nima" data-value="" class="sc-null" aria-label="Not available">&mdash;</td>
      {% endif %}
      {% if e.entry.arniqa_score is not none %}
      <td data-col="arniqa" data-value="{{ e.entry.arniqa_score }}" style="display:none">{{ "%.3f" | format(e.entry.arniqa_score) }}</td>
      {% else %}
      <td data-col="arniqa" data-value="" style="display:none">&mdash;</td>
      {% endif %}
      {% if e.entry.topiq_score is not none %}
      <td data-col="topiq" data-value="{{ e.entry.topiq_score }}" class="{{ _score_class_det(e.entry.topiq_score) }}">{{ "%.3f" | format(e.entry.topiq_score) }}</td>
      {% else %}
      <td data-col="topiq" data-value="" class="sc-null" aria-label="Not available">&mdash;</td>
      {% endif %}
      {% if e.entry.musiq_score is not none %}
      <td data-col="musiq" data-value="{{ e.entry.musiq_score }}" style="display:none">{{ "%.3f" | format(e.entry.musiq_score) }}</td>
      {% else %}
      <td data-col="musiq" data-value="" style="display:none">&mdash;</td>
      {% endif %}
      <td data-col="spd" data-value="{{ e.spd }}">{{ "%.0f" | format(e.spd) }}</td>
      {% if e.entry.imgsys_elo is not none %}
      <td data-col="elo" data-value="{{ e.entry.imgsys_elo }}">{{ e.entry.imgsys_elo }}</td>
      {% else %}
      <td data-col="elo" data-value="" class="sc-null" aria-label="Not available">&mdash;</td>
      {% endif %}
      <td data-col="time" data-value="{{ e.entry.avg_time_s }}">{{ "%.1f" | format(e.entry.avg_time_s) }}s</td>
      <td data-col="seed" data-value="{{ e.entry.seed_label }}">{{ e.entry.seed_label }}</td>
      {% if data.show_fal_links %}<td data-col="fal"><a href="https://fal.ai/models/{{ e.entry.fal_endpoint }}" target="_blank" rel="noopener" class="try-link" title="Try on fal.ai">Try</a></td>{% endif %}
    </tr>
    {% endfor %}
  </tbody>
</table>
</div>

</section>

<!-- Prompt Browser -->
{% if data.prompts %}
<section aria-label="Prompt Browser">
<details class="prompt-browser" id="prompt-browser">
  <summary>Prompt Browser ({{ data.prompt_count }} prompts &times; {{ data.model_count }} models)</summary>
  <div class="pb-nav">
    <select id="pb-category" class="pb-select" aria-label="Filter by category">
      <option value="all">All Categories</option>
      {% for cat in data.prompt_categories %}
      <option value="{{ cat }}">{{ cat }}</option>
      {% endfor %}
    </select>
    <select id="pb-prompt-select" class="pb-select pb-prompt-dropdown" aria-label="Jump to prompt">
      {% for p in data.prompts %}
      <option value="{{ loop.index0 }}" data-category="{{ p.category }}">{{ p.item_id }} — {{ p.prompt[:60] }}{{ '…' if p.prompt|length > 60 }}</option>
      {% endfor %}
    </select>
    <span class="pb-counter" id="pb-counter">1 / {{ data.prompts | length }}</span>
    <button id="pb-prev" disabled>&larr;</button>
    <button id="pb-next"{{ ' disabled' if data.prompts | length <= 1 }}>&rarr;</button>
  </div>
  {% for p in data.prompts %}
  {% set is_first_slide = loop.first %}
  <div class="pb-slide{{ ' active' if is_first_slide }}" data-slide="{{ loop.index0 }}" data-prompt="{{ p.prompt }}" data-category="{{ p.category }}">
    <div class="pb-prompt-text"><span class="pb-cat-tag">{{ p.category }}</span> &ldquo;{{ p.prompt }}&rdquo;</div>
    <div class="pb-grid">
      {% for model_key, img_url in p.images.items() %}
      {% set thumb_url = data.cdn_base ~ '/thumbs/' ~ model_key ~ '/' ~ p.item_id ~ '.webp' if data.cdn_base else img_url %}
      {% set full_url = data.cdn_base ~ '/images/' ~ model_key ~ '/' ~ p.item_id ~ '.webp' if data.cdn_base else img_url %}
      <div class="pb-card">
        <img {{ 'src' if is_first_slide else 'data-src' }}="{{ thumb_url }}"
             alt="{{ data.display_names.get(model_key, model_key) }} — {{ p.prompt[:50] }}"
             loading="lazy" width="120" height="120"
             onerror="this.parentElement.style.display='none'"
             data-full="{{ full_url }}" data-model="{{ data.display_names.get(model_key, model_key) }}">
        <span class="pb-label">{{ data.display_names.get(model_key, model_key) }}</span>
      </div>
      {% endfor %}
    </div>
  </div>
  {% endfor %}
</details>
<div class="pb-lightbox" id="pb-lightbox">
  <div class="lb-spinner" id="lb-spinner"></div>
  <img id="pb-lightbox-img" src="" alt="">
  <div class="pb-lightbox-info" id="pb-lightbox-info">
    <div class="pb-lb-model" id="pb-lb-model"></div>
    <div class="pb-lb-prompt" id="pb-lb-prompt"></div>
  </div>
  <button class="pb-lb-nav pb-lb-prev" id="pb-lb-prev" aria-label="Previous image">&larr;</button>
  <button class="pb-lb-nav pb-lb-next" id="pb-lb-next" aria-label="Next image">&rarr;</button>
</div>
</section>
{% endif %}

<!-- Methodology -->
<section aria-label="Methodology">
<details class="methodology" id="methodology">
  <summary>Methodology</summary>
  <h2 style="font-size:0.82rem;margin:0.8rem 0 0.3rem;color:var(--text)">Benchmark Setup</h2>
  <ul>
    <li><strong>{{ data.model_count }} models</strong> evaluated on fal.ai infrastructure</li>
    <li><strong>{{ data.prompt_count }} curated prompts</strong> from PartiPrompts, DrawBench, OneIG-Bench + custom. Each prompt contributes equally to final scores (simple average).</li>
    <li><strong>Judge:</strong> {{ data.judge or "None" }}. Each image is scored by 3 independent VLM judges. Final score = median of 3, mitigating single-judge bias.</li>
    <li><strong>Seed:</strong> {{ data.seed_count }} models with seed={{ data.config.get('seed', 42) }}{% if data.noseed_count %}, {{ data.noseed_count }} models without seed support (single run &mdash; {{ data.prompt_count }} prompts provides statistical robustness){% endif %}. A fixed seed ensures the same prompt always produces the same image, making results reproducible and comparisons fair.</li>
    {% if data.config.get('image_size') %}<li><strong>Image size:</strong> {{ data.config['image_size'] }}</li>{% endif %}
    <li><strong>Concurrency:</strong> {{ data.config.get('concurrency', 8) }} parallel requests</li>
    <li><strong>Pipeline:</strong> {{ data.config.get('pipeline', 'text2img') }}</li>
    <li><strong>Date:</strong> {{ data.date }}</li>
  </ul>
  <h2 style="font-size:0.82rem;margin:0.8rem 0 0.3rem;color:var(--text)">VLM Judge Evaluation Criteria</h2>
  <ul>
    <li><strong>Visual Quality:</strong> Overall image quality &mdash; sharpness, color fidelity, coherence, realism, absence of artifacts. Scale: 1.0&ndash;5.0 with 0.1 increments.</li>
    <li><strong>Prompt Adherence:</strong> How faithfully the image matches the prompt &mdash; objects, attributes, spatial relationships, style. Scale: 1.0&ndash;5.0.</li>
    <li><strong>Text Rendering:</strong> Accuracy of rendered text (signs, labels). Only evaluated on {{ data.prompt_count // 8 }} text-containing prompts. Scale: 1.0&ndash;5.0.</li>
  </ul>
  <p style="font-size:0.78rem;color:var(--text-muted);margin:0.3rem 0;">
    <a href="https://github.com/evalytic/evalytic" target="_blank" rel="noopener">View judge prompt source code &nearr;</a>
  </p>
  <h2 style="font-size:0.82rem;margin:0.8rem 0 0.3rem;color:var(--text)">Inter-Judge Agreement</h2>
  <ul>
    <li><strong>{{ data.judge_agree_pct }}%</strong> of dimensions: judges agree within 0.5 points</li>
    <li>Average judge gap: <strong>{{ data.judge_avg_gap }}</strong> points</li>
    <li>Extreme disagreement (gap &gt; 1.0): <strong>{{ data.judge_extreme_pct }}%</strong> &mdash; most common in text rendering</li>
    <li>Median-of-3 approach reduces outlier impact from any single judge</li>
  </ul>
  <h2 style="font-size:0.82rem;margin:0.8rem 0 0.3rem;color:var(--text)">Data &amp; Reproducibility</h2>
  <ul>
    <li><a href="https://github.com/evalytic/evalytic" target="_blank" rel="noopener">Prompt dataset (100 prompts, JSON)</a> &mdash; curated from PartiPrompts, DrawBench, OneIG-Bench + custom</li>
    <li><a href="https://github.com/evalytic/evalytic" target="_blank" rel="noopener">Evalytic SDK (open source)</a> &mdash; the same tool that generated this leaderboard</li>
    <li>Raw scores embedded in this page (<code style="font-size:0.7rem">View Source &rarr; #leaderboard-data</code>)</li>
  </ul>
  <details style="margin:0.5rem 0">
    <summary style="font-size:0.8rem;cursor:pointer;color:var(--accent)">Reproduce this benchmark</summary>
    <pre style="background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:0.6rem 0.8rem;font-size:0.72rem;margin:0.4rem 0;overflow-x:auto;color:var(--text-muted)"><code>pip install evalytic
evalytic bench \\
  --models flux-schnell flux-pro imagen-4 \\
  --judges "gemini-2.5-flash,gpt-5.2,claude-sonnet-4-6" \\
  --prompts prompts-text2img-v1.json \\
  --image-size square_hd --seed 42 --yes</code></pre>
  </details>
  <h2 style="font-size:0.82rem;margin:0.8rem 0 0.3rem;color:var(--text)">Column Definitions</h2>
  <ul>
    <li><strong>Overall</strong> &mdash; Weighted combination of VLM judge scores and deterministic metrics. Formula: <code style="font-size:0.75rem">1 + 4 &times; (vlm_w &times; (vlm_avg&minus;1)/4 + det_w &times; det_avg)</code>. Default: 60% VLM + 40% deterministic. Adjustable via the weight panel above.</li>
    <li><strong>VQ (Visual Quality)</strong> &mdash; VLM judge score (1&ndash;5). Evaluates overall image quality: sharpness, color fidelity, coherence, realism, and absence of visual artifacts.</li>
    <li><strong>PA (Prompt Adherence)</strong> &mdash; VLM judge score (1&ndash;5). How faithfully the generated image matches the text prompt &mdash; objects, attributes, spatial relationships, and style.</li>
    <li><strong>TR (Text Rendering)</strong> &mdash; VLM judge score (1&ndash;5). Accuracy of text rendered within the image (signs, labels, logos). Only scored on prompts that contain text elements (~12% of prompts). Shows &ldquo;&mdash;&rdquo; for models/prompts without text.</li>
    <li><strong>CLIP</strong> &mdash; Deterministic metric (0&ndash;1). CLIP ViT-L/14 cosine similarity between prompt text and generated image. Higher = better semantic alignment with prompt.</li>
    <li><strong>Sharp (Sharpness)</strong> &mdash; Deterministic metric (0&ndash;1). Variance of Laplacian applied to the image. Higher = sharper, more detailed image. Low values may indicate blur or softness.</li>
    <li><strong>NIMA</strong> &mdash; Deterministic metric (0&ndash;1). Neural Image Assessment (NIMA) trained on human aesthetic ratings (AVA dataset). Higher = more aesthetically pleasing.</li>
    <li><strong>TOPIQ</strong> &mdash; Deterministic metric (0&ndash;1). Top-down Image Quality via CFANet architecture (KonIQ-10k trained). State-of-the-art no-reference quality metric.</li>
    <li><strong>Score/$</strong> &mdash; Value efficiency: Overall score divided by cost per image. Higher = better quality per dollar spent.</li>
    <li><strong>imgsys</strong> &mdash; ELO rating from <a href="https://imgsys.org" target="_blank">imgsys.org</a> community voting. Independent reference point for cross-validation.</li>
    <li><strong>Time</strong> &mdash; Average image generation time in seconds.</li>
    <li><strong>Seed</strong> &mdash; Fixed random seed for reproducibility. Models that support seed get deterministic output. Models without seed support are run 3&times; per prompt with median score taken.</li>
    <li><strong>$/img</strong> &mdash; Cost per image generation in USD on fal.ai.</li>
  </ul>
</details>
</section>
</main>

<!-- CTA: Run Your Own -->
<section style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1.2rem 1.5rem;margin-top:1.5rem;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
  <h2 style="font-size:1rem;margin:0 0 0.4rem;color:var(--text)">Run this benchmark on your own models</h2>
  <p style="font-size:0.82rem;color:var(--text-muted);margin:0 0 0.8rem;max-width:600px;margin-left:auto;margin-right:auto;">
    This leaderboard was built with <strong>Evalytic</strong> &mdash; an open-source SDK for evaluating AI-generated images.
    Same tool works for model selection, regression detection, and CI/CD quality gating.
  </p>
  <div style="display:flex;gap:0.6rem;justify-content:center;flex-wrap:wrap;align-items:center;">
    <code style="background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:0.4rem 0.8rem;font-size:0.8rem;font-family:'JetBrains Mono',monospace;color:var(--accent)">pip install evalytic</code>
    <a href="https://docs.evalytic.ai/quickstart" target="_blank" rel="noopener" style="background:var(--accent);color:#fff;padding:0.4rem 1rem;border-radius:6px;font-size:0.82rem;font-weight:600;text-decoration:none;">Get Started &rarr;</a>
    <a href="https://github.com/evalytic/evalytic" target="_blank" rel="noopener" style="border:1px solid var(--border);padding:0.4rem 1rem;border-radius:6px;font-size:0.82rem;text-decoration:none;color:var(--text);">GitHub &nearr;</a>
  </div>
</section>

<footer class="footer">
  <a href="https://evalytic.ai">evalytic.ai</a> &mdash; Evals for AI outputs &mdash;
  <a href="https://docs.evalytic.ai/">Docs</a> &middot;
  <a href="https://github.com/evalytic/evalytic">GitHub</a> &middot;
  <a href="https://evalytic.ai/showcase">Showcases</a>
</footer>

<div class="toast" id="toast">Link copied!</div>
<div class="col-tooltip" id="col-tooltip"></div>

<!-- Embedded data for JS recompute -->
<script id="leaderboard-data" type="application/json">{{ entries_json }}</script>

<script>
(function() {
  "use strict";
  var DATA = JSON.parse(document.getElementById("leaderboard-data").textContent);
  var table = document.getElementById("lb-table");
  var tbody = table.querySelector("tbody");
  var CLIP_LO = 0.15, CLIP_HI = 0.40;

  // Time normalization: fastest = 1.0, slowest = 0.0
  var times = DATA.map(function(d) { return d.time || 0; }).filter(function(t) { return t > 0; });
  var TIME_MIN = times.length > 0 ? Math.min.apply(null, times) : 1;
  var TIME_MAX = times.length > 0 ? Math.max.apply(null, times) : 60;

  // --- Presets ---
  var PRESETS = {
    "default":       {vlm:60,vq:40,pa:40,tr:20,det:40,clip:25,sharp:15,nima:25,arniqa:0,topiq:35,musiq:0,time:0},
    "quality":       {vlm:90,vq:50,pa:20,tr:30,det:10,clip:15,sharp:10,nima:30,arniqa:0,topiq:45,musiq:0,time:0},
    "text":          {vlm:80,vq:30,pa:20,tr:50,det:20,clip:20,sharp:10,nima:30,arniqa:0,topiq:40,musiq:0,time:0},
    "ecommerce":     {vlm:70,vq:50,pa:40,tr:10,det:30,clip:20,sharp:15,nima:25,arniqa:0,topiq:40,musiq:0,time:0},
    "speed":         {vlm:60,vq:40,pa:40,tr:20,det:40,clip:15,sharp:10,nima:20,arniqa:0,topiq:25,musiq:0,time:30},
    "budget":        {vlm:50,vq:35,pa:40,tr:25,det:30,clip:20,sharp:15,nima:25,arniqa:0,topiq:40,musiq:0,time:20},
    "photorealistic":{vlm:80,vq:60,pa:30,tr:10,det:20,clip:15,sharp:20,nima:25,arniqa:0,topiq:40,musiq:0,time:0}
  };

  // --- Slider refs ---
  var sliders = {
    vlm: document.getElementById("w-vlm"),
    det: document.getElementById("w-det"),
    vq: document.getElementById("w-vq"),
    pa: document.getElementById("w-pa"),
    tr: document.getElementById("w-tr"),
    clip: document.getElementById("w-clip"),
    sharp: document.getElementById("w-sharp"),
    nima: document.getElementById("w-nima"),
    arniqa: document.getElementById("w-arniqa"),
    topiq: document.getElementById("w-topiq"),
    musiq: document.getElementById("w-musiq"),
    time: document.getElementById("w-time")
  };

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  // --- Group sum indicators ---
  var vlmSumEl = document.getElementById("vlm-sum");
  var detSumEl = document.getElementById("det-sum");

  function updateGroupSums() {
    var vlmTotal = (+sliders.vq.value) + (+sliders.pa.value) + (+sliders.tr.value);
    var detTotal = (+sliders.clip.value) + (+sliders.sharp.value) + (+sliders.nima.value) + (+sliders.arniqa.value) + (+sliders.topiq.value) + (+sliders.musiq.value) + (+sliders.time.value);
    vlmSumEl.textContent = "= " + vlmTotal + "%";
    vlmSumEl.className = "group-sum " + (Math.abs(vlmTotal - 100) <= 1 ? "ok" : "err");
    detSumEl.textContent = "= " + detTotal + "%";
    detSumEl.className = "group-sum " + (Math.abs(detTotal - 100) <= 1 ? "ok" : "err");
  }

  // --- Validation modal ---
  var modal = document.getElementById("wt-modal");
  var modalBody = document.getElementById("wt-modal-body");
  document.getElementById("wt-modal-close").addEventListener("click", function() {
    modal.classList.remove("open");
  });
  modal.addEventListener("click", function(e) {
    if (e.target === modal) modal.classList.remove("open");
  });

  function showModal(errors) {
    modalBody.innerHTML = '<p>Sub-weights within each group must add up to exactly 100%.</p>' +
      errors.map(function(e) { return '<div class="err-detail">' + e + '</div>'; }).join('');
    modal.classList.add("open");
  }

  function validateWeights() {
    var errors = [];
    var vlmTotal = (+sliders.vq.value) + (+sliders.pa.value) + (+sliders.tr.value);
    var detTotal = (+sliders.clip.value) + (+sliders.sharp.value) + (+sliders.nima.value) + (+sliders.arniqa.value) + (+sliders.topiq.value) + (+sliders.musiq.value) + (+sliders.time.value);
    if (Math.abs(vlmTotal - 100) > 1) errors.push("VLM Dimensions: " + sliders.vq.value + " + " + sliders.pa.value + " + " + sliders.tr.value + " = <strong>" + vlmTotal + "%</strong> (need 100%)");
    if (Math.abs(detTotal - 100) > 1) errors.push("Deterministic Metrics total = <strong>" + detTotal + "%</strong> (need 100%)");
    return errors;
  }

  function getOutput(slider) {
    // Find the <output> sibling — it's the next element sibling that is an output
    var sib = slider.nextElementSibling;
    while (sib && sib.tagName !== "OUTPUT") sib = sib.nextElementSibling;
    // Fallback: check previous sibling direction or use parentNode query
    if (!sib) {
      var row = slider.closest(".weight-row");
      if (row) sib = row.querySelector("output");
    }
    return sib;
  }

  function readWeights() {
    return {
      vlm: +sliders.vlm.value, det: +sliders.det.value,
      vq: +sliders.vq.value, pa: +sliders.pa.value, tr: +sliders.tr.value,
      clip: +sliders.clip.value, sharp: +sliders.sharp.value, nima: +sliders.nima.value,
      arniqa: +sliders.arniqa.value, topiq: +sliders.topiq.value, musiq: +sliders.musiq.value, time: +sliders.time.value
    };
  }

  function setWeights(w) {
    for (var k in w) {
      if (sliders[k]) {
        sliders[k].value = w[k];
        var out = getOutput(sliders[k]);
        if (out) out.textContent = w[k] + "%";
      }
    }
  }

  // --- Calculate overall for one entry ---
  function calcOverall(d, w) {
    var vlmW = w.vlm / 100, detW = w.det / 100;
    // Renormalize top-level
    var topSum = vlmW + detW;
    if (topSum > 0) { vlmW /= topSum; detW /= topSum; }
    else { vlmW = 1; detW = 0; }

    // VLM sub-weights
    var vlmSubs = [], vlmSubW = [];
    var subTotal = 0;
    if (d.vq > 0) { vlmSubs.push(d.vq); vlmSubW.push(w.vq); subTotal += w.vq; }
    if (d.pa > 0) { vlmSubs.push(d.pa); vlmSubW.push(w.pa); subTotal += w.pa; }
    if (d.tr !== null && d.tr > 0) { vlmSubs.push(d.tr); vlmSubW.push(w.tr); subTotal += w.tr; }
    if (vlmSubs.length === 0) return 1;
    var vlmAvg = 0;
    for (var i = 0; i < vlmSubs.length; i++) vlmAvg += vlmSubs[i] * (vlmSubW[i] / subTotal);
    var vlmUnit = (vlmAvg - 1) / 4;

    // Det sub-weights
    var detVals = [], detSubW = [], detTotal = 0;
    if (d.clip > 0) {
      var cn = clamp((d.clip - CLIP_LO) / (CLIP_HI - CLIP_LO), 0, 1);
      detVals.push(cn); detSubW.push(w.clip); detTotal += w.clip;
    }
    if (d.sharp > 0) { detVals.push(Math.min(1, d.sharp)); detSubW.push(w.sharp); detTotal += w.sharp; }
    if (d.nima !== null && d.nima > 0) { detVals.push(Math.min(1, d.nima)); detSubW.push(w.nima); detTotal += w.nima; }
    if (d.arniqa !== null && d.arniqa > 0) { detVals.push(Math.min(1, d.arniqa)); detSubW.push(w.arniqa); detTotal += w.arniqa; }
    if (d.topiq !== null && d.topiq > 0) { detVals.push(Math.min(1, d.topiq)); detSubW.push(w.topiq); detTotal += w.topiq; }
    if (d.musiq !== null && d.musiq > 0) { detVals.push(Math.min(1, d.musiq)); detSubW.push(w.musiq); detTotal += w.musiq; }
    if (d.time > 0 && w.time > 0) {
      // Invert: faster = higher score. Normalize to 0-1 range.
      var timeNorm = TIME_MAX > TIME_MIN ? clamp(1 - (d.time - TIME_MIN) / (TIME_MAX - TIME_MIN), 0, 1) : 0.5;
      detVals.push(timeNorm); detSubW.push(w.time); detTotal += w.time;
    }

    var detUnit = 0;
    if (detVals.length > 0) {
      for (var j = 0; j < detVals.length; j++) detUnit += detVals[j] * (detSubW[j] / detTotal);
    } else {
      vlmW = 1; detW = 0;
    }

    return 1 + 4 * (vlmW * vlmUnit + detW * detUnit);
  }

  // --- Score CSS class ---
  function scVlm(v) {
    if (v >= 4.5) return "sc-5";
    if (v >= 3.5) return "sc-4";
    if (v >= 2.5) return "sc-3";
    return "sc-2";
  }


  var MEDALS = ["&#x1F947;", "&#x1F948;", "&#x1F949;"];

  // --- Recompute + re-sort ---
  function recompute() {
    var w = readWeights();
    var rows = Array.from(tbody.querySelectorAll("tr"));
    rows.forEach(function(row) {
      var key = row.dataset.key;
      var d = DATA.find(function(e) { return e.key === key; });
      if (!d) return;
      var ov = calcOverall(d, w);
      var ovCell = row.querySelector('[data-col="overall"]');
      ovCell.dataset.value = ov.toFixed(4);
      ovCell.innerHTML = "<strong>" + ov.toFixed(3) + "</strong>";
      // Score/$
      var spdCell = row.querySelector('[data-col="spd"]');
      var spd = d.cost > 0 ? ov / d.cost : 0;
      spdCell.dataset.value = spd.toFixed(1);
      spdCell.textContent = Math.round(spd);
    });
    sortByCol("overall", false);  // desc
    updateURL(w);
  }

  // --- Sort ---
  var currentSort = "overall", currentAsc = false;

  function sortByCol(col, asc) {
    currentSort = col; currentAsc = asc;
    var ths = table.querySelectorAll("th[data-sort]");
    ths.forEach(function(h) { h.setAttribute("aria-sort", "none"); });
    var th = table.querySelector('th[data-sort="' + col + '"]');
    if (th) th.setAttribute("aria-sort", asc ? "ascending" : "descending");

    var rows = Array.from(tbody.querySelectorAll("tr"));
    var dir = asc ? 1 : -1;
    rows.sort(function(a, b) {
      var ac = a.querySelector('[data-col="' + col + '"]');
      var bc = b.querySelector('[data-col="' + col + '"]');
      var av = ac ? ac.dataset.value : "";
      var bv = bc ? bc.dataset.value : "";
      // Null always last
      if (av === "" && bv === "") return 0;
      if (av === "") return 1;
      if (bv === "") return -1;
      var an = parseFloat(av), bn = parseFloat(bv);
      if (!isNaN(an) && !isNaN(bn)) {
        if (an !== bn) return (an - bn) * dir;
        // Tiebreaker 1: higher score/$ wins
        var aSpd = parseFloat((a.querySelector('[data-col="spd"]') || {}).dataset.value || "0");
        var bSpd = parseFloat((b.querySelector('[data-col="spd"]') || {}).dataset.value || "0");
        if (aSpd !== bSpd) return bSpd - aSpd;
        // Tiebreaker 2: open model wins
        var aName = (a.querySelector('[data-col="name"]') || {}).innerHTML || "";
        var bName = (b.querySelector('[data-col="name"]') || {}).innerHTML || "";
        var aOpen = aName.indexOf("Open") !== -1 ? 0 : 1;
        var bOpen = bName.indexOf("Open") !== -1 ? 0 : 1;
        if (aOpen !== bOpen) return aOpen - bOpen;
        // Tiebreaker 3: lower cost wins
        var aCost = parseFloat((a.querySelector('[data-col="cost"]') || {}).dataset.value || "0");
        var bCost = parseFloat((b.querySelector('[data-col="cost"]') || {}).dataset.value || "0");
        return aCost - bCost;
      }
      return av.localeCompare(bv) * dir;
    });
    rows.forEach(function(r) { tbody.appendChild(r); });
    updateRanks();
  }

  function updateRanks() {
    var visible = Array.from(tbody.querySelectorAll("tr:not([hidden])"));
    visible.forEach(function(row, i) {
      row.dataset.rank = i + 1;
      var rc = row.querySelector('[data-col="rank"]');
      if (rc) {
        rc.dataset.value = i + 1;
        if (i < 3) {
          rc.innerHTML = '<span class="rank-medal">' + MEDALS[i] + '</span>';
        } else {
          rc.textContent = i + 1;
        }
      }
    });
  }

  // --- Column click sort ---
  table.querySelectorAll("th[data-sort]").forEach(function(th) {
    th.addEventListener("click", function() {
      var col = th.dataset.sort;
      var wasAsc = th.getAttribute("aria-sort") === "ascending";
      sortByCol(col, !wasAsc);
    });
  });

  // --- Filter ---
  var familyFilter = document.getElementById("family-filter");
  var seedBtns = document.querySelectorAll(".seed-btn");
  var licenseBtns = document.querySelectorAll(".license-btn");
  var currentSeed = "all";
  var currentLicense = "all";

  function applyFilters() {
    var fam = familyFilter.value;
    var rows = Array.from(tbody.querySelectorAll("tr"));
    rows.forEach(function(row) {
      var matchFam = fam === "all" || row.dataset.family === fam;
      var matchSeed = currentSeed === "all" || row.dataset.seed === currentSeed;
      var matchLic = currentLicense === "all" || row.dataset.license === currentLicense;
      row.hidden = !(matchFam && matchSeed && matchLic);
    });
    updateRanks();
  }

  familyFilter.addEventListener("change", applyFilters);
  seedBtns.forEach(function(btn) {
    btn.addEventListener("click", function() {
      seedBtns.forEach(function(b) { b.classList.remove("btn-active"); });
      btn.classList.add("btn-active");
      currentSeed = btn.dataset.seed;
      applyFilters();
    });
  });
  licenseBtns.forEach(function(btn) {
    btn.addEventListener("click", function() {
      licenseBtns.forEach(function(b) { b.classList.remove("btn-active"); });
      btn.classList.add("btn-active");
      currentLicense = btn.dataset.license;
      applyFilters();
    });
  });

  // --- Filter panel toggle + badge ---
  var filterPanel = document.getElementById("filter-panel");
  var filterBadge = document.getElementById("filter-badge");
  document.getElementById("filter-toggle").addEventListener("click", function(e) {
    e.stopPropagation();
    filterPanel.style.display = filterPanel.style.display === "none" ? "block" : "none";
  });
  filterPanel.addEventListener("click", function(e) { e.stopPropagation(); });
  document.addEventListener("click", function() { filterPanel.style.display = "none"; });

  function updateFilterBadge() {
    var count = 0;
    if (familyFilter.value !== "all") count++;
    if (currentSeed !== "all") count++;
    if (currentLicense !== "all") count++;
    if (count > 0) {
      filterBadge.textContent = count;
      filterBadge.style.display = "inline";
    } else {
      filterBadge.style.display = "none";
    }
  }
  // Hook badge update into filter changes
  var origApplyFilters = applyFilters;
  applyFilters = function() { origApplyFilters(); updateFilterBadge(); };
  familyFilter.removeEventListener("change", origApplyFilters);
  familyFilter.addEventListener("change", applyFilters);

  // --- Weight panel toggle ---
  var weightPanel = document.getElementById("weight-panel");
  document.getElementById("weight-toggle").addEventListener("click", function() {
    weightPanel.classList.toggle("open");
    this.textContent = weightPanel.classList.contains("open") ? "Customize Weights \\u25B4" : "Customize Weights \\u25BE";
  });

  // --- Slider input: update output label + group sums, NO auto-balance ---
  function onSliderInput(slider) {
    var outEl = getOutput(slider);
    if (outEl) outEl.textContent = slider.value + "%";
    // Deactivate preset buttons
    document.querySelectorAll(".preset-btn").forEach(function(b) { b.classList.remove("btn-active"); });

    // VLM <-> Det coupling (top-level must sum to 100)
    if (slider === sliders.vlm) {
      sliders.det.value = 100 - +sliders.vlm.value;
      var detOut = getOutput(sliders.det);
      if (detOut) detOut.textContent = sliders.det.value + "%";
    } else if (slider === sliders.det) {
      sliders.vlm.value = 100 - +sliders.det.value;
      var vlmOut = getOutput(sliders.vlm);
      if (vlmOut) vlmOut.textContent = sliders.vlm.value + "%";
    }
    updateGroupSums();
  }

  Object.values(sliders).forEach(function(s) {
    s.addEventListener("input", function() { onSliderInput(s); });
  });

  // --- Apply button: recompute (sub-weights are auto-normalized in calcOverall) ---
  document.getElementById("apply-btn").addEventListener("click", function() {
    recompute();
    // Close panel + update toggle text
    weightPanel.classList.remove("open");
    document.getElementById("weight-toggle").textContent = "Customize Weights \u25BE";
    // Show success toast
    var toast = document.getElementById("toast");
    toast.textContent = "Weights applied!";
    toast.classList.add("show");
    setTimeout(function() { toast.classList.remove("show"); }, 1500);
  });

  // --- Presets ---
  document.querySelectorAll(".preset-btn").forEach(function(btn) {
    btn.addEventListener("click", function() {
      document.querySelectorAll(".preset-btn").forEach(function(b) { b.classList.remove("btn-active"); });
      btn.classList.add("btn-active");
      setWeights(PRESETS[btn.dataset.preset]);
      updateGroupSums();
      recompute();
    });
  });

  // --- Reset ---
  document.getElementById("reset-btn").addEventListener("click", function() {
    document.querySelectorAll(".preset-btn").forEach(function(b) { b.classList.remove("btn-active"); });
    document.querySelector('[data-preset="default"]').classList.add("btn-active");
    setWeights(PRESETS["default"]);
    updateGroupSums();
    recompute();
  });

  // --- URL state ---
  function updateURL(w) {
    var params = new URLSearchParams();
    params.set("wv", "1");
    var urlKeys = ["vlm","det","vq","pa","tr","clip","sharp","nima","topiq","time"];
    for (var i = 0; i < urlKeys.length; i++) {
      var k = urlKeys[i];
      if (w[k] !== undefined) params.set(k, w[k]);
    }
    history.replaceState(null, "", "?" + params.toString());
  }

  function loadURL() {
    var params = new URLSearchParams(location.search);
    if (!params.has("wv")) return false;
    var w = {};
    ["vlm","det","vq","pa","tr","clip","sharp","nima","time"].forEach(function(k) {
      if (params.has(k)) w[k] = clamp(parseInt(params.get(k), 10) || 0, 0, 100);
    });
    if (Object.keys(w).length > 0) {
      setWeights(w);
      return true;
    }
    return false;
  }

  // --- Share ---
  function copyUrl() {
    var url = location.href;
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(url);
    } else {
      var ta = document.createElement("textarea");
      ta.value = url; ta.style.position = "fixed"; ta.style.opacity = "0";
      document.body.appendChild(ta); ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
    var toast = document.getElementById("toast");
    toast.classList.add("show");
    setTimeout(function() { toast.classList.remove("show"); }, 2000);
  }
  document.getElementById("share-btn").addEventListener("click", copyUrl);
  document.getElementById("share-x-btn").addEventListener("click", function() {
    var text = "{{ data.model_count }} AI image models benchmarked by 3 independent VLM judges — Evalytic Image Models Leaderboard";
    window.open("https://x.com/intent/post?text=" + encodeURIComponent(text) + "&url=" + encodeURIComponent(location.href), "_blank");
  });
  document.getElementById("share-li-btn").addEventListener("click", function() {
    window.open("https://www.linkedin.com/sharing/share-offsite/?url=" + encodeURIComponent(location.href), "_blank");
  });

  // --- Dark/light toggle ---
  var toggle = document.getElementById("theme-toggle");
  function setTheme(light) {
    document.body.classList.toggle("light", light);
    toggle.textContent = light ? "\\u2600" : "\\u263E";
    toggle.setAttribute("aria-checked", !light);
    try { localStorage.setItem("evalytic-theme", light ? "light" : "dark"); } catch(e) {}
  }
  toggle.addEventListener("click", function() {
    setTheme(!document.body.classList.contains("light"));
  });
  // Init theme
  try {
    var saved = localStorage.getItem("evalytic-theme");
    if (saved === "light") setTheme(true);
    else if (!saved && window.matchMedia("(prefers-color-scheme: light)").matches) setTheme(true);
  } catch(e) {}

  // --- Archive dropdown ---
  var archiveSelect = document.getElementById("archive-select");
  if (archiveSelect) {
    archiveSelect.addEventListener("change", function() {
      location.href = this.value;
    });
  }

  // --- Prompt Browser ---
  var pbSlides = document.querySelectorAll(".pb-slide");
  var pbPrev = document.getElementById("pb-prev");
  var pbNext = document.getElementById("pb-next");
  var pbCounter = document.getElementById("pb-counter");
  var pbCatSelect = document.getElementById("pb-category");
  var pbPromptSelect = document.getElementById("pb-prompt-select");
  var pbIdx = 0;
  var pbVisible = [];  // indices of visible slides after category filter

  function pbBuildVisible() {
    var cat = pbCatSelect ? pbCatSelect.value : "all";
    pbVisible = [];
    var opts = pbPromptSelect ? pbPromptSelect.options : [];
    for (var i = 0; i < opts.length; i++) {
      var show = cat === "all" || opts[i].dataset.category === cat;
      opts[i].hidden = !show;
      if (show) pbVisible.push(parseInt(opts[i].value));
    }
  }

  function pbLoadImages(slide) {
    slide.querySelectorAll("img[data-src]").forEach(function(img) {
      img.src = img.dataset.src;
      img.removeAttribute("data-src");
    });
  }

  function pbShow(slideIdx) {
    if (!pbSlides.length) return;
    slideIdx = Math.max(0, Math.min(slideIdx, pbSlides.length - 1));
    pbIdx = slideIdx;
    pbSlides.forEach(function(s) { s.classList.remove("active"); });
    pbSlides[pbIdx].classList.add("active");
    pbLoadImages(pbSlides[pbIdx]);
    // Update prompt select
    if (pbPromptSelect) pbPromptSelect.value = pbIdx;
    // Counter shows position within visible set
    var posInVisible = pbVisible.indexOf(pbIdx);
    if (pbCounter) pbCounter.textContent = (posInVisible + 1) + " / " + pbVisible.length;
    if (pbPrev) pbPrev.disabled = posInVisible <= 0;
    if (pbNext) pbNext.disabled = posInVisible >= pbVisible.length - 1;
  }

  function pbShowVisibleOffset(delta) {
    var pos = pbVisible.indexOf(pbIdx);
    var next = pos + delta;
    if (next >= 0 && next < pbVisible.length) pbShow(pbVisible[next]);
  }

  // Init
  pbBuildVisible();

  if (pbPrev) pbPrev.addEventListener("click", function() { pbShowVisibleOffset(-1); });
  if (pbNext) pbNext.addEventListener("click", function() { pbShowVisibleOffset(1); });

  if (pbCatSelect) pbCatSelect.addEventListener("change", function() {
    pbBuildVisible();
    // Jump to first visible prompt in this category
    if (pbVisible.length > 0) pbShow(pbVisible[0]);
  });

  if (pbPromptSelect) pbPromptSelect.addEventListener("change", function() {
    pbShow(parseInt(pbPromptSelect.value));
  });

  // Keyboard navigation for prompt browser
  document.addEventListener("keydown", function(e) {
    var browser = document.getElementById("prompt-browser");
    if (!browser || !browser.open) return;
    if (lightbox && lightbox.classList.contains("open")) return;
    if (e.key === "ArrowLeft") { pbShowVisibleOffset(-1); e.preventDefault(); }
    if (e.key === "ArrowRight") { pbShowVisibleOffset(1); e.preventDefault(); }
  });

  // --- Lightbox with prev/next ---
  var lightbox = document.getElementById("pb-lightbox");
  var lbImg = document.getElementById("pb-lightbox-img");
  var lbModel = document.getElementById("pb-lb-model");
  var lbPrompt = document.getElementById("pb-lb-prompt");
  var lbPrev = document.getElementById("pb-lb-prev");
  var lbNext = document.getElementById("pb-lb-next");
  var lbImages = [];  // current slide's images [{src, model}]
  var lbIdx = 0;
  var lbCurrentPrompt = "";

  function lbOpen(imgEl) {
    // Build image list from current active slide
    var activeSlide = document.querySelector(".pb-slide.active");
    lbCurrentPrompt = activeSlide ? (activeSlide.dataset.prompt || "") : "";
    lbImages = [];
    var cards = activeSlide ? activeSlide.querySelectorAll(".pb-card img") : [];
    cards.forEach(function(img, i) {
      lbImages.push({ src: img.dataset.full || img.src, model: img.dataset.model || "" });
      if (img === imgEl) lbIdx = i;
    });
    lbShowCurrent();
    lightbox.classList.add("open");
  }

  var lbSpinner = document.getElementById("lb-spinner");
  function lbShowCurrent() {
    if (!lbImages.length) return;
    lbIdx = Math.max(0, Math.min(lbIdx, lbImages.length - 1));
    var item = lbImages[lbIdx];
    // Show spinner + fade out while loading
    lbImg.classList.add("loading");
    if (lbSpinner) lbSpinner.classList.add("active");
    lbImg.onload = function() {
      lbImg.classList.remove("loading");
      if (lbSpinner) lbSpinner.classList.remove("active");
    };
    lbImg.src = item.src;
    if (lbModel) lbModel.textContent = item.model;
    if (lbPrompt) lbPrompt.textContent = lbCurrentPrompt;
    if (lbPrev) lbPrev.style.display = lbIdx > 0 ? "" : "none";
    if (lbNext) lbNext.style.display = lbIdx < lbImages.length - 1 ? "" : "none";
  }

  function lbClose() {
    lightbox.classList.remove("open");
    lbImg.src = "";
    lbImages = [];
  }

  if (lightbox) {
    // Attach click to all gallery images (use event delegation for lazy-loaded)
    document.addEventListener("click", function(e) {
      var img = e.target.closest(".pb-card img");
      if (img) { lbOpen(img); e.stopPropagation(); }
    });
    if (lbPrev) lbPrev.addEventListener("click", function(e) { lbIdx--; lbShowCurrent(); e.stopPropagation(); });
    if (lbNext) lbNext.addEventListener("click", function(e) { lbIdx++; lbShowCurrent(); e.stopPropagation(); });
    lightbox.addEventListener("click", function(e) {
      if (e.target === lightbox) lbClose();
    });
    document.addEventListener("keydown", function(e) {
      if (!lightbox.classList.contains("open")) return;
      if (e.key === "Escape") { lbClose(); e.preventDefault(); }
      if (e.key === "ArrowLeft" && lbIdx > 0) { lbIdx--; lbShowCurrent(); e.preventDefault(); }
      if (e.key === "ArrowRight" && lbIdx < lbImages.length - 1) { lbIdx++; lbShowCurrent(); e.preventDefault(); }
    });
  }

  // --- Column tooltip (JS-positioned, no overflow clipping) ---
  var tipEl = document.getElementById("col-tooltip");
  var tipTimer = null;
  document.querySelectorAll("th[data-tip]").forEach(function(th) {
    th.addEventListener("mouseenter", function(e) {
      clearTimeout(tipTimer);
      tipEl.textContent = th.dataset.tip;
      // Position: above the header, centered on the th
      var rect = th.getBoundingClientRect();
      var tipW = 240;
      var left = rect.left + rect.width / 2 - tipW / 2;
      // Clamp to viewport
      if (left < 8) left = 8;
      if (left + tipW > window.innerWidth - 8) left = window.innerWidth - tipW - 8;
      var top = rect.top - 8;
      tipEl.style.left = left + "px";
      tipEl.style.width = tipW + "px";
      // Measure height to position above
      tipEl.style.top = "0px";
      tipEl.classList.add("visible");
      var tipH = tipEl.offsetHeight;
      tipEl.style.top = (top - tipH) + "px";
      // If goes above viewport, show below instead
      if (top - tipH < 4) {
        tipEl.style.top = (rect.bottom + 8) + "px";
      }
    });
    th.addEventListener("mouseleave", function() {
      tipTimer = setTimeout(function() { tipEl.classList.remove("visible"); }, 100);
    });
  });

  // --- Init ---
  updateGroupSums();
  if (loadURL()) {
    updateGroupSums();
    recompute();
    weightPanel.classList.add("open");
    document.getElementById("weight-toggle").textContent = "Customize Weights \\u25B4";
  }
})();
</script>

</body>
</html>
""")


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------


def _score_class_vlm(score: float) -> str:
    if score >= 4.5:
        return "sc-5"
    if score >= 3.5:
        return "sc-4"
    if score >= 2.5:
        return "sc-3"
    return "sc-2"


def _score_class_det(score: float) -> str:
    if score >= 0.75:
        return "sc-5"
    if score >= 0.50:
        return "sc-4"
    if score >= 0.25:
        return "sc-3"
    return "sc-2"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_leaderboard(
    report: BenchReport,
    output_path: str,
    *,
    show_fal_links: bool = True,
    archive_versions: list[dict[str, Any]] | None = None,
    cdn_base: str = "",
) -> str:
    """Generate a self-contained leaderboard HTML page from a BenchReport.

    1. Enriches with LEADERBOARD_MODELS metadata (family, display name, ELO)
    2. Calculates default overall (80/20 VLM/det)
    3. Server-renders the table via Jinja2
    4. Embeds entry data as JSON for client-side weight recompute

    Returns the output file path.
    """
    data = enrich_report(
        report,
        archive_versions=archive_versions,
        show_fal_links=show_fal_links,
    )
    data.cdn_base = cdn_base

    # Calculate default overall + Score/$, sort by overall desc
    entries_with_overall = []
    for e in data.entries:
        overall = _calc_overall(e)
        spd = overall / e.cost_per_image if e.cost_per_image > 0 else 0.0
        entries_with_overall.append({
            "entry": e,
            "overall": round(overall, 4),
            "spd": round(spd, 2),
        })
    # Sort: overall desc → score/$ desc → open before proprietary → cost asc
    entries_with_overall.sort(key=lambda x: (
        -x["overall"],
        -x["spd"],
        0 if x["entry"].license_type == "open" else 1,
        x["entry"].cost_per_image,
    ))

    # JSON data for JS recompute
    entries_json_list = []
    for e in data.entries:
        entries_json_list.append({
            "key": e.model_key,
            "vq": e.visual_quality,
            "pa": e.prompt_adherence,
            "tr": e.text_rendering,
            "clip": e.clip_score,
            "sharp": e.sharpness,
            "nima": e.nima_score,
            "arniqa": e.arniqa_score,
            "topiq": e.topiq_score,
            "musiq": e.musiq_score,
            "cost": e.cost_per_image,
            "time": e.avg_time_s,
        })

    # Unique families for filter dropdown
    families = sorted({e.family for e in data.entries})

    # Wrap entries for Jinja (need to make SimpleNamespace-like objects)
    class _EO:
        def __init__(self, entry: LeaderboardEntry, overall: float, spd: float):
            self.entry = entry
            self.overall = overall
            self.spd = spd

    eo_list = [_EO(x["entry"], x["overall"], x["spd"]) for x in entries_with_overall]

    html = LEADERBOARD_TEMPLATE.render(
        data=data,
        entries_with_overall=eo_list,
        entries_json=json.dumps(entries_json_list),
        families=families,
        _score_class_vlm=_score_class_vlm,
        _score_class_det=_score_class_det,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")
    return output_path
