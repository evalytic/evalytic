# Evalytic

**Evals for visual AI.** Automated quality evaluation for AI-generated images and video.

[![PyPI](https://img.shields.io/pypi/v/evalytic)](https://pypi.org/project/evalytic/)
[![Python](https://img.shields.io/pypi/pyversions/evalytic)](https://pypi.org/project/evalytic/)
[![License](https://img.shields.io/pypi/l/evalytic)](https://github.com/evalytic/evalytic/blob/main/LICENSE)

Know if your AI-generated visuals are good — before your users tell you they're not.

```bash
pip install evalytic

evaly bench \
  -m flux-schnell -m flux-dev -m flux-pro \
  -p "A photorealistic cat on a windowsill" \
  -o report.html --yes
```

## What It Does

Evalytic benchmarks AI image generation models by generating images, scoring them with VLM judges (Gemini, GPT, Claude, Ollama), and producing rich reports — all in one command.

- **Model Selection** — Compare Flux Schnell vs Dev vs Pro with real prompts
- **Prompt Optimization** — Measure how well models follow your prompts
- **Regression Detection** — Catch quality drops when models update
- **CI/CD Quality Gate** — Block deploys when image quality falls below threshold
- **7 Semantic Dimensions** — visual_quality, prompt_adherence, text_rendering, input_fidelity, transformation_quality, artifact_detection, identity_preservation
- **Consensus Judging** — Multi-judge scoring with automatic agreement analysis

## Quickstart

### 1. Install

```bash
pip install evalytic
```

### 2. See Real Examples (no API key needed)

```bash
evaly demo              # Opens showcase with 4 real benchmark case studies
evaly demo face         # Face identity preservation comparison
evaly demo flagship     # Flux Schnell vs Dev vs Pro cost/quality
```

### 3. Set API Keys

```bash
export FAL_KEY=your_fal_key          # fal.ai for image generation
export GEMINI_API_KEY=your_gemini_key  # Default judge
```

### 4. Run

```bash
# Single model benchmark
evaly bench -m flux-schnell -p "A cat sitting on a windowsill" --yes

# Compare models with HTML report
evaly bench -m flux-schnell -m flux-dev -m flux-pro \
  -p prompts.json -o report.html --review

# img2img benchmark
evaly bench -m flux-kontext -m seedream-edit -m reve-edit \
  -p prompts.json --input product.jpg --yes

# Score an existing image
evaly eval --image output.png --prompt "A sunset over mountains"

# CI/CD quality gate
evaly gate --report report.json --threshold 3.5
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `evaly demo` | Browse real benchmark showcases (no API key needed) |
| `evaly bench` | Generate, score, and report in one command |
| `evaly eval` | Score a single image without generation |
| `evaly gate` | CI/CD quality gate with pass/fail exit codes |

## Judges

Any VLM that can analyze images works as a judge:

```bash
evaly bench -m flux-schnell -p "A cat" -j gemini-2.5-flash        # Default
evaly bench -m flux-schnell -p "A cat" -j gemini-2.5-pro           # Gemini Pro
evaly bench -m flux-schnell -p "A cat" -j openai/gpt-5.2           # OpenAI
evaly bench -m flux-schnell -p "A cat" -j anthropic/claude-sonnet-4-6  # Anthropic
evaly bench -m flux-schnell -p "A cat" -j ollama/qwen2.5-vl:7b    # Local
```

### Consensus Mode

Use multiple judges for more reliable scores:

```bash
evaly bench -m flux-schnell -p "A cat" \
  --judges "gemini-2.5-flash,openai/gpt-5.2"
```

Two judges score in parallel. If they disagree, a third breaks the tie.

## Optional Extras

```bash
pip install "evalytic[metrics]"  # CLIP Score + LPIPS + ArcFace (~2GB)
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

Full docs at [docs.evalytic.ai](https://docs.evalytic.ai)

## License

MIT
