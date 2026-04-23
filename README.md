# Evalytic

**Evals for AI outputs.** Automated quality evaluation for images, video, text, RAG, and agent runs.

[![PyPI](https://img.shields.io/pypi/v/evalytic)](https://pypi.org/project/evalytic/)
[![Python](https://img.shields.io/pypi/pyversions/evalytic)](https://pypi.org/project/evalytic/)
[![License](https://img.shields.io/pypi/l/evalytic)](https://github.com/evalytic/evalytic/blob/main/LICENSE)

Know if your AI outputs are good before your users tell you they're not. One SDK for visual generation, LLM text, RAG pipelines, and tool-using agents.

```bash
pip install evalytic

# Visual: compare image generation models
evaly bench -m flux-schnell -m flux-dev -m flux-pro \
  -p "A product photo on marble countertop" --yes

# RAG: evaluate retrieval-augmented answers
evaly rag eval \
  --query "What is Evalytic?" \
  --response "Evalytic evaluates AI outputs." \
  --context "Evalytic is an evaluation platform." \
  --metrics faithfulness,answer_relevancy,contextual_relevancy,hallucination

# Text: evaluate LLM outputs
evaly text eval \
  --input "Translate: Hello" --output-text "Merhaba" --expected "Merhaba" \
  --metrics exact_match,semantic_similarity

# Agent: evaluate tool-using agent runs
evaly agent eval \
  --input "Find the score" --final-output "0.9" \
  --tool-call search --expected-tool search
```

## What It Does

Evalytic scores AI outputs across four eval domains with the same consensus-judge architecture:

- **Visual** (images, video) — 7 semantic dimensions scored by VLM judges + deterministic metrics (sharpness, CLIP, LPIPS, ArcFace, NIMA, TOPIQ)
- **RAG** — reference-free `faithfulness`, `answer_relevancy`, `contextual_relevancy`, `hallucination`; reference-based `context_precision` + `context_recall`
- **Text** — `factual_correctness`, `semantic_similarity`, `g_eval` (custom rubric), BLEU, ROUGE, exact_match, levenshtein, string_presence
- **Agent** — `tool_call_accuracy`, `goal_accuracy`, `step_efficiency`

VLM / LLM judges (Gemini, GPT, Claude, Ollama) + local metrics work together or independently. Every domain supports consensus mode (2+1 adaptive multi-judge).

### Use Cases

- **Model Selection** — Compare any fal.ai / OpenAI / Anthropic models head-to-head
- **RAG Hallucination Detection** — Claim-level faithfulness against retrieved context
- **Prompt Optimization** — Measure output quality across semantic dimensions
- **Regression Detection** — Catch quality drops when models, prompts, or retrievers update
- **CI/CD Quality Gate** — Block deploys when any metric falls below threshold (visual OR text OR agent)
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

### 3. Score an Existing Image

```bash
# Local metrics only (free, no API key)
evaly eval --image output.png --prompt "A sunset over mountains" --no-judge

# With VLM judge
export GEMINI_API_KEY=your_gemini_key
evaly eval --image output.png --prompt "A sunset over mountains" --yes
```

### 4. Benchmark Models

```bash
export FAL_KEY=your_fal_key

# Text-to-image
evaly bench -m flux-schnell -m flux-dev -m flux-pro \
  -p "A cat sitting on a windowsill" --yes

# Image-to-image
evaly bench -m flux-kontext -m seedream-edit -m reve-edit \
  --inputs product.jpg -p "Place on a marble countertop" --yes

# Metrics only, no VLM judge
evaly bench -m flux-schnell -m flux-dev -p "A cat" --no-judge
```

### 5. Interactive Setup

```bash
evaly init   # Guided setup: use case, API keys, config file
```

## CLI Commands

| Command | Domain | Description |
|---------|--------|-------------|
| `evaly init` | Any | Interactive setup wizard |
| `evaly demo` | Visual | Browse real benchmark showcases (no API key needed) |
| `evaly bench` | Visual | Generate, score, and report in one command |
| `evaly eval` | Visual | Score a single image without generation |
| `evaly rag eval` | RAG | Evaluate RAG answers (reference-free + reference-based) |
| `evaly text eval` | Text | Evaluate LLM outputs (factual, semantic, BLEU, ROUGE, G-Eval) |
| `evaly agent eval` | Agent | Evaluate tool-using agent runs |
| `evaly compare` | All | Delta between two report files (same-type only) |
| `evaly gate` | All | CI/CD quality gate (`--threshold` for visual, `--metric-threshold` for text/RAG/agent) |
| `evaly dataset` | All | Manage evaluation datasets (rag, text, agent, visual) |

## Judges

Any VLM that can analyze images works as a judge:

```bash
evaly bench -m flux-schnell -p "A cat" -j gemini-2.5-flash            # Default
evaly bench -m flux-schnell -p "A cat" -j openai/gpt-5.2              # OpenAI
evaly bench -m flux-schnell -p "A cat" -j anthropic/claude-sonnet-4-6 # Anthropic
evaly bench -m flux-schnell -p "A cat" -j fal/gemini-2.5-flash        # Via fal.ai (single key)
evaly bench -m flux-schnell -p "A cat" -j ollama/qwen2.5-vl:7b        # Local
```

### Consensus Mode

Use multiple judges for more reliable scores:

```bash
evaly bench -m flux-schnell -p "A cat" \
  --judges "gemini-2.5-flash,openai/gpt-5.2"
```

Two judges score in parallel. If they disagree, a third breaks the tie.

## Pytest Integration

Turn any metric into a pytest assertion. Fails the test when any score falls below its threshold, and reports every failing metric at once.

```python
from evalytic.testing import assert_test
from evalytic.text.types import RAGTestCase, RetrievedChunk

def test_rag_quality():
    case = RAGTestCase(
        query="What is Evalytic?",
        response="Evalytic evaluates AI outputs.",
        contexts=[RetrievedChunk(text="Evalytic is an evaluation SDK for AI outputs.")],
    )
    assert_test(case, metrics={
        "faithfulness": 0.8,
        "hallucination": 0.9,
        "contextual_relevancy": 0.75,
    })
```

Same helper works for `TextTestCase` and `AgentTestCase`. No plugin install required. For single-metric assertions use `assert_metric(case, "hallucination", threshold=0.95)`.

## Optional Extras

```bash
pip install "evalytic[metrics]"      # CLIP + LPIPS + ArcFace + NIMA + TOPIQ (~2GB)
pip install "evalytic[ocr]"          # OCR text accuracy (pytesseract)
pip install "evalytic[embeddings]"   # Local sentence-transformers for RAG/text embeddings (~500MB)
pip install "evalytic[all]"          # Everything
```

For RAG `answer_relevancy` and text `semantic_similarity`, either install `evalytic[embeddings]` or set `OPENAI_API_KEY` / `FAL_KEY` (embeddings auto-resolve).

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

[bench.dimension_weights]
input_fidelity = 0.5
visual_quality = 0.1

[rag]
judge = "gemini-2.5-flash"
judges = ["gemini-2.5-flash", "openai/gpt-5.2"]  # Consensus mode

[rag.thresholds]
faithfulness = 0.8
answer_relevancy = 0.7

[text.thresholds]
factual_correctness = 0.8
semantic_similarity = 0.7

[embeddings]
provider = "sentence-transformers"  # or "openai" / "fal"
model = "all-MiniLM-L6-v2"
```

## Documentation

Full docs at [docs.evalytic.ai](https://docs.evalytic.ai)

## License

MIT
