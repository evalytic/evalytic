# Evalytic

**Evals for visual AI.** Automated quality evaluation for AI-generated visuals.

[![PyPI](https://img.shields.io/pypi/v/evalytic)](https://pypi.org/project/evalytic/)
[![Python](https://img.shields.io/pypi/pyversions/evalytic)](https://pypi.org/project/evalytic/)
[![License](https://img.shields.io/pypi/l/evalytic)](https://github.com/evalytic/evalytic/blob/main/LICENSE)

Know if your AI-generated visuals are good — before your users tell you they're not.

```bash
pip install evalytic

evaly bench \
  --models flux-schnell flux-dev flux-pro \
  --prompts "A photorealistic cat on a windowsill" \
  --output report.html
```

## What It Does

Evalytic benchmarks AI image generation models by generating images, scoring them with VLM judges (Gemini, GPT-5, Claude, Ollama), and producing rich reports — all in one command.

- **Model Selection** — Compare Flux Schnell vs Dev vs Pro with real prompts
- **Prompt Optimization** — Measure how well models follow your prompts
- **Regression Detection** — Catch quality drops when models update
- **CI/CD Quality Gate** — Block deploys when image quality falls below threshold
- **6 Semantic Dimensions** — visual_quality, prompt_adherence, text_rendering, input_fidelity, transformation_quality, artifact_detection

## Quickstart

### 1. Install

```bash
pip install evalytic
```

### 2. Set API Keys

```bash
export FAL_KEY=your_fal_key          # fal.ai for image generation
export GEMINI_API_KEY=your_gemini_key  # Free tier: 1,000 req/day
```

### 3. Run

```bash
# Single model benchmark
evaly bench -m flux-schnell -p "A cat sitting on a windowsill" --yes

# Compare models with HTML report
evaly bench -m flux-schnell -m flux-dev -m flux-pro \
  -p prompts.json -o report.html --review

# Score an existing image
evaly eval --image output.png --prompt "A sunset over mountains"

# CI/CD quality gate
evaly gate --report report.json --threshold 3.5
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `evaly bench` | Generate, score, and report in one command |
| `evaly eval` | Score a single image without generation |
| `evaly gate` | CI/CD quality gate with pass/fail exit codes |

## Judges

Any VLM that can analyze images works as a judge:

```bash
evaly bench -m flux-schnell -p "A cat" -j gemini-2.5-flash     # Default (free)
evaly bench -m flux-schnell -p "A cat" -j openai/gpt-5.2       # OpenAI
evaly bench -m flux-schnell -p "A cat" -j anthropic/claude-sonnet-4  # Anthropic
evaly bench -m flux-schnell -p "A cat" -j ollama/qwen2.5-vl:7b # Local
```

## Optional Extras

```bash
pip install "evalytic[metrics]"  # CLIP Score + LPIPS (~2GB)
pip install "evalytic[all]"      # Everything
```

## Configuration

Create `evalytic.toml` in your project root:

```toml
[keys]
fal = "your_fal_key"
gemini = "your_gemini_key"

[bench]
judge = "gemini-2.5-flash"
dimensions = ["visual_quality", "prompt_adherence"]
concurrency = 4
```

## Documentation

Full docs at [docs.evalytic.dev](https://docs.evalytic.dev)

## License

MIT
