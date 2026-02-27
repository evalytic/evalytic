# ShopLens -- AI Product Photography Platform

ShopLens is a sample e-commerce AI photography platform that demonstrates [Evalytic](https://github.com/anthropics/evalytic) end-to-end. It runs real image generation on **fal.ai** and real quality scoring via **Gemini 2.0 Flash** -- the same VLM prompts used by the Evalytic judge Lambda.

## The Story

ShopLens runs 3 image pipelines for e-commerce product photography:

| Pipeline | What it does | Models | Evalytic Use Case |
|----------|-------------|--------|-------------------|
| **Background Studio** | Generate product backgrounds from prompts | Flux Schnell, Dev, Pro v1.1 | Multi-model comparison |
| **Product Enhancer** | Enhance product photos with studio lighting | Flux Dev i2i vs Flux Kontext | Regression detection |
| **Clean Cut** | Remove product backgrounds | BiRefNet v2 | Single-image quality gate |

ShopLens wants to:
1. **Pick the best Flux model** for background generation
2. **Check if upgrading** the enhancer model causes regression
3. **Run quality gates** on background removal edge cases

## Quick Start

```bash
cd samples/shoplens
pip install -e .

# Copy and fill in your API keys
cp .env.example .env
# Edit .env: add FAL_KEY and GEMINI_API_KEY

# Run a single pipeline (~15s, ~$0.01)
python scripts/03_eval_bg_removal.py

# Run all 3 pipelines (~3min, ~$0.69)
python scripts/run_all.py
```

## What You'll See

Colored terminal output with:
- Per-model comparison tables with scored dimensions
- Regression alerts when the candidate model drops > 0.3 points
- Quality gate pass/fail for edge-case background removal
- Consolidated platform health summary

## How It Works

### LocalJudge (no backend needed)

`shoplens/judge.py` implements a `LocalJudge` that uses the **exact same prompts** as the Evalytic Lambda judge and the **exact same Gemini REST API pattern**. This means you get real VLM quality scores with zero infrastructure.

### Dual-mode: local or Evalytic SDK

When the Evalytic API is deployed, set `EVALYTIC_API_KEY` in your `.env` and the scripts route through the real SDK instead of LocalJudge. Same scores, same dimensions, now with experiment tracking and dashboards.

## Cost Estimate

| Item | Count | Cost |
|------|-------|------|
| Flux Schnell (5 imgs) | 5 | $0.02 |
| Flux Dev (5 bg + 4 enhance) | 9 | $0.23 |
| Flux Pro v1.1 (5 imgs) | 5 | $0.28 |
| Flux Kontext (4 imgs) | 4 | $0.16 |
| BiRefNet (3 imgs) | 3 | ~free |
| Gemini 2.0 Flash (26 evals) | 26 | free tier |
| **Total** | | **~$0.69** |

## Project Structure

```
samples/shoplens/
├── shoplens/
│   ├── config.py       # Pipeline configs, fal.ai model registry
│   ├── generate.py     # fal.ai wrappers (text2img, img2img, utility)
│   ├── judge.py        # LocalJudge: direct Gemini scoring
│   └── report.py       # Rich terminal report renderer
│
├── golden_sets/
│   ├── backgrounds.py      # 5 text2img prompts
│   ├── product_enhance.py  # 4 product photo URLs + instructions
│   └── bg_removal.py       # 3 product photo URLs for BG removal
│
└── scripts/
    ├── 01_compare_bg_models.py   # Multi-model comparison
    ├── 02_check_enhancer.py      # Regression detection
    ├── 03_eval_bg_removal.py     # Single-image quality gate
    └── run_all.py                # Full suite + consolidated report
```

## API Keys

- **FAL_KEY**: Get one at [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys)
- **GEMINI_API_KEY**: Get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (free tier: 1000 req/day)
