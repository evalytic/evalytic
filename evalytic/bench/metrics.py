"""Optional local metrics: CLIP Score (text2img) and LPIPS (img2img).

Requires ``pip install evalytic[metrics]`` for torch, transformers, and lpips.
Gracefully degrades when dependencies are missing -- sets METRICS_AVAILABLE = False.

Lightweight metrics (sharpness) run with only numpy+PIL — no torch required.
"""

from __future__ import annotations

import math
import re
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import BenchItem, CorrelationStat, MetricFlag, MetricResult, MetricScoringConfig

# ---------------------------------------------------------------------------
# Graceful import
# ---------------------------------------------------------------------------

METRICS_AVAILABLE = False
_torch: Any = None
_transformers: Any = None
_lpips_lib: Any = None

try:
    import torch as _torch  # type: ignore[no-redef]
    import transformers as _transformers  # type: ignore[no-redef]
    import lpips as _lpips_lib  # type: ignore[no-redef]

    METRICS_AVAILABLE = True
except ImportError:
    pass

NIMA_AVAILABLE = False
_pyiqa: Any = None
try:
    import pyiqa as _pyiqa  # type: ignore[no-redef]
    NIMA_AVAILABLE = True
except ImportError:
    pass

OCR_AVAILABLE = False
_pytesseract: Any = None
try:
    import pytesseract as _pytesseract  # type: ignore[no-redef]
    OCR_AVAILABLE = True
except ImportError:
    pass

_DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"
_DEFAULT_CACHE_DIR = "~/.evalytic/cache/models"


# ---------------------------------------------------------------------------
# CLIP scorer
# ---------------------------------------------------------------------------


class CLIPScorer:
    """Compute CLIP similarity between an image and a text prompt.

    Returns cosine similarity in [0.0, 1.0].
    """

    def __init__(self, model_name: str = _DEFAULT_CLIP_MODEL, cache_dir: str | None = None) -> None:
        if not METRICS_AVAILABLE:
            raise RuntimeError(
                "CLIP scoring requires evalytic[metrics]. "
                "Install with: pip install evalytic[metrics]"
            )
        self._cache_dir = str(Path(cache_dir or _DEFAULT_CACHE_DIR).expanduser())
        self._model_name = model_name
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        self._model = _transformers.CLIPModel.from_pretrained(
            self._model_name, cache_dir=self._cache_dir
        )
        self._processor = _transformers.CLIPProcessor.from_pretrained(
            self._model_name, cache_dir=self._cache_dir
        )
        self._model.eval()

    def score(self, image_path: str, text: str) -> float:
        """Return CLIP cosine similarity between *image_path* and *text*."""
        self._load()
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(text=[text], images=image, return_tensors="pt", padding=True)
        with _torch.no_grad():
            outputs = self._model(**inputs)
            # Normalised cosine similarity, scaled to [0, 1]
            logits = outputs.logits_per_image  # shape (1, 1)
            # CLIP logits are cosine_sim * 100; scale to [0, 1]
            similarity = logits.item() / 100.0
            return max(0.0, min(1.0, similarity))


# ---------------------------------------------------------------------------
# LPIPS scorer
# ---------------------------------------------------------------------------


class LPIPSScorer:
    """Compute LPIPS perceptual distance between two images.

    Returns *similarity* (1 - raw_lpips) so higher = better, consistent
    with all other Evalytic scores.
    """

    def __init__(self, cache_dir: str | None = None) -> None:
        if not METRICS_AVAILABLE:
            raise RuntimeError(
                "LPIPS scoring requires evalytic[metrics]. "
                "Install with: pip install evalytic[metrics]"
            )
        self._cache_dir = str(Path(cache_dir or _DEFAULT_CACHE_DIR).expanduser())
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        self._model = _lpips_lib.LPIPS(net="alex")
        self._model.eval()

    def score(self, image1_path: str, image2_path: str) -> float:
        """Return perceptual similarity (1 - LPIPS distance)."""
        self._load()
        from PIL import Image
        import torchvision.transforms as T  # type: ignore[import-untyped]

        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        img1 = transform(Image.open(image1_path).convert("RGB")).unsqueeze(0)
        img2 = transform(Image.open(image2_path).convert("RGB")).unsqueeze(0)

        with _torch.no_grad():
            dist = self._model(img1, img2).item()
        return max(0.0, min(1.0, 1.0 - dist))


# ---------------------------------------------------------------------------
# Face (ArcFace) scorer
# ---------------------------------------------------------------------------


class FaceScorer:
    """ArcFace face identity similarity between input and output images.

    Returns cosine similarity [0, 1] of face embeddings.
    Returns None if no face detected in either image.

    Requires ``pip install evalytic[metrics]`` (insightface + onnxruntime).
    """

    def __init__(self, cache_dir: str | None = None) -> None:
        if not METRICS_AVAILABLE:
            raise RuntimeError(
                "Face scoring requires evalytic[metrics]. "
                "Install with: pip install evalytic[metrics]"
            )
        self._cache_dir = str(Path(cache_dir or _DEFAULT_CACHE_DIR).expanduser())
        self._app: Any = None

    def _load(self) -> None:
        if self._app is not None:
            return
        from insightface.app import FaceAnalysis  # type: ignore[import-untyped]

        self._app = FaceAnalysis(
            name="buffalo_l",
            root=self._cache_dir,
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=(640, 640))

    def score(self, image1_path: str, image2_path: str) -> float | None:
        """Return face cosine similarity or None if no face in either image."""
        self._load()
        import numpy as np
        from PIL import Image

        img1 = np.array(Image.open(image1_path).convert("RGB"))
        img2 = np.array(Image.open(image2_path).convert("RGB"))

        faces1 = self._app.get(img1)
        faces2 = self._app.get(img2)

        if not faces1 or not faces2:
            return None

        # Use largest face (by bounding box area)
        def _largest(faces: list[Any]) -> Any:
            return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        emb1 = _largest(faces1).embedding
        emb2 = _largest(faces2).embedding

        # Cosine similarity
        sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        return max(0.0, min(1.0, sim))


# ---------------------------------------------------------------------------
# Aesthetic scorer (LAION CLIP-MLP)
# ---------------------------------------------------------------------------


class AestheticScorer:
    """Aesthetic quality predictor — MLP on CLIP ViT-L/14 embeddings.

    Uses the improved-aesthetic-predictor (SAC+logos+AVA) trained on
    ``openai/clip-vit-large-patch14`` embeddings.  Normalises the raw
    1-10 AVA score to 0-1.

    Optionally accepts a :class:`CLIPScorer` to reuse its loaded model
    instead of loading a second copy.

    Requires ``pip install evalytic[metrics]`` (torch + transformers).
    """

    _HF_REPO = "camenduru/improved-aesthetic-predictor"
    _HF_FILE = "sac+logos+ava1-l14-linearMSE.pth"

    def __init__(
        self,
        clip_scorer: CLIPScorer | None = None,
        cache_dir: str | None = None,
    ) -> None:
        if not METRICS_AVAILABLE:
            raise RuntimeError(
                "Aesthetic scoring requires evalytic[metrics]. "
                "Install with: pip install evalytic[metrics]"
            )
        self._clip_scorer = clip_scorer
        self._cache_dir = str(Path(cache_dir or _DEFAULT_CACHE_DIR).expanduser())
        self._model: Any = None
        self._processor: Any = None
        self._mlp: Any = None

    def _load(self) -> None:
        if self._mlp is not None:
            return

        # Reuse CLIP model from CLIPScorer if available
        if self._clip_scorer is not None:
            self._clip_scorer._load()
            self._model = self._clip_scorer._model
            self._processor = self._clip_scorer._processor
        else:
            self._model = _transformers.CLIPModel.from_pretrained(
                _DEFAULT_CLIP_MODEL, cache_dir=self._cache_dir,
            )
            self._processor = _transformers.CLIPProcessor.from_pretrained(
                _DEFAULT_CLIP_MODEL, cache_dir=self._cache_dir,
            )
            self._model.eval()

        # MLP: 768 → 1024 → 128 → 64 → 16 → 1 (no activations)
        self._mlp = _torch.nn.Sequential(
            _torch.nn.Linear(768, 1024),
            _torch.nn.Dropout(0.2),
            _torch.nn.Linear(1024, 128),
            _torch.nn.Dropout(0.2),
            _torch.nn.Linear(128, 64),
            _torch.nn.Dropout(0.1),
            _torch.nn.Linear(64, 16),
            _torch.nn.Linear(16, 1),
        )

        # Download weights from HuggingFace
        from huggingface_hub import hf_hub_download  # available via transformers

        weights_path = hf_hub_download(
            self._HF_REPO, self._HF_FILE, cache_dir=self._cache_dir,
        )
        state_dict = _torch.load(weights_path, map_location="cpu", weights_only=True)
        # Remap keys: "layers.X.weight" → "X.weight"
        remapped = {k.replace("layers.", ""): v for k, v in state_dict.items()}
        self._mlp.load_state_dict(remapped)
        self._mlp.eval()

    def score(self, image_path: str) -> float:
        """Return aesthetic score in [0.0, 1.0].  Higher = more aesthetic."""
        self._load()
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt")
        with _torch.no_grad():
            emb = self._model.get_image_features(**inputs)
            if not isinstance(emb, _torch.Tensor):
                emb = emb.pooler_output if hasattr(emb, "pooler_output") else emb[1]
            emb = emb / emb.norm(dim=-1, keepdim=True)
            raw = self._mlp(emb).item()

        # AVA scores 1-10 → normalise to 0-1
        return round(max(0.0, min(1.0, (raw - 1.0) / 9.0)), 4)


# ---------------------------------------------------------------------------
# NIMA scorer (pyiqa) — replaces AestheticScorer
# ---------------------------------------------------------------------------


class NimaScorer:
    """Neural Image Assessment (NIMA) — aesthetic quality predictor.

    Uses the NIMA model from pyiqa, trained on the AVA dataset of human
    aesthetic ratings.  Returns a normalised 0-1 score (raw NIMA is 1-10).

    Requires ``pip install pyiqa``.
    """

    def __init__(self) -> None:
        if not NIMA_AVAILABLE:
            raise RuntimeError(
                "NIMA scoring requires pyiqa. "
                "Install with: pip install pyiqa"
            )
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        self._model = _pyiqa.create_metric("nima", device="cpu")

    def score(self, image_path: str) -> float:
        """Return NIMA aesthetic score in [0.0, 1.0].  Higher = more aesthetic."""
        self._load()
        raw = self._model(image_path).item()
        # NIMA outputs 1-10, normalise to 0-1
        return round(max(0.0, min(1.0, (raw - 1.0) / 9.0)), 4)


# ---------------------------------------------------------------------------
# OCR scorer (pytesseract) — text accuracy in generated images
# ---------------------------------------------------------------------------


class OCRScorer:
    """OCR accuracy scorer — compares expected text with OCR-extracted text.

    Uses pytesseract to extract text from the generated image and compares
    it against expected text using SequenceMatcher ratio.

    Returns accuracy 0.0-1.0, or None if no expected text is provided.

    Requires ``pip install pytesseract`` (and Tesseract binary installed).
    """

    def __init__(self) -> None:
        if not OCR_AVAILABLE:
            raise RuntimeError(
                "OCR scoring requires pytesseract. "
                "Install with: pip install pytesseract"
            )

    def score(self, image_path: str, expected_text: str) -> float | None:
        """Return OCR accuracy in [0.0, 1.0] or None if no expected text."""
        if not expected_text:
            return None
        try:
            from difflib import SequenceMatcher

            from PIL import Image

            img = Image.open(image_path)
            extracted = _pytesseract.image_to_string(img).strip()
            if not extracted:
                return 0.0
            ratio = SequenceMatcher(None, expected_text.lower(), extracted.lower()).ratio()
            return round(max(0.0, min(1.0, ratio)), 4)
        except Exception:
            return None


def _extract_expected_text(prompt: str) -> str:
    """Extract quoted strings from a prompt as expected text for OCR.

    Looks for text inside single or double quotes, e.g.:
    'A sign that says "Hello World"' -> 'Hello World'
    """
    matches = re.findall(r"""['"]([^'"]+)['"]""", prompt)
    return " ".join(matches) if matches else ""


# ---------------------------------------------------------------------------
# Sharpness scorer (Variance of Laplacian) — no torch required
# ---------------------------------------------------------------------------


class SharpnessScorer:
    """Measure image sharpness via Variance of Laplacian.

    Higher values = sharper image.  No heavy dependencies — uses only
    numpy and PIL which are already available in the base SDK.

    The raw variance is resolution-dependent, so we normalise to a 0-1
    scale using an empirical sigmoid: ``1 / (1 + exp(-k * (log(vol) - midpoint)))``.
    """

    def score(self, image_path: str) -> float:
        """Return sharpness score in [0.0, 1.0].  Higher = sharper."""
        import numpy as np
        from PIL import Image, ImageFilter

        img = Image.open(image_path).convert("L")
        # Resize to consistent resolution so scores are comparable
        img = img.resize((512, 512), Image.LANCZOS)
        # Laplacian via PIL kernel (3x3 approximation)
        laplacian = img.filter(ImageFilter.Kernel(
            size=(3, 3),
            kernel=[0, 1, 0, 1, -4, 1, 0, 1, 0],
            scale=1,
            offset=128,
        ))
        arr = np.asarray(laplacian, dtype=np.float64) - 128.0
        vol = float(np.var(arr))
        # Sigmoid normalisation: midpoint ~500, k tuned for AI-generated images
        log_vol = np.log1p(vol)
        score = 1.0 / (1.0 + np.exp(-1.5 * (log_vol - 5.0)))
        return round(max(0.0, min(1.0, float(score))), 4)


# ---------------------------------------------------------------------------
# Batch compute
# ---------------------------------------------------------------------------


def _resolve_image_path(url_or_path: str, cache_dir: str | None = None) -> str:
    """Download a URL to a local temp file, or return local path as-is."""
    if not url_or_path.startswith(("http://", "https://")):
        return url_or_path

    import hashlib
    import tempfile

    import httpx

    # Cache downloaded files by URL hash
    url_hash = hashlib.md5(url_or_path.encode()).hexdigest()[:12]
    ext = ".jpg"
    lower = url_or_path.lower().split("?")[0]
    if lower.endswith(".png"):
        ext = ".png"
    elif lower.endswith(".webp"):
        ext = ".webp"

    cache_path = Path(cache_dir or tempfile.gettempdir()) / f"evalytic_input_{url_hash}{ext}"
    if cache_path.exists():
        return str(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    resp = httpx.get(url_or_path, follow_redirects=True, timeout=30)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return str(cache_path)


def compute_metrics(
    items: list[BenchItem],
    metric_types: list[str],
    pipeline: str,
    prompts: list[dict[str, Any]],
    cache_dir: str | None = None,
) -> None:
    """Compute requested metrics and attach to ImageResult.metrics in-place.

    * ``clip`` — only for text2img (needs a text prompt)
    * ``lpips`` — only for img2img (needs an input image)
    * ``face`` — only for img2img (needs an input image with faces)
    * ``sharpness`` — always available (no torch, uses numpy+PIL)
    * ``aesthetic`` — (deprecated, alias for ``nima``) LAION aesthetic MLP on CLIP embeddings
    * ``nima`` — NIMA aesthetic quality predictor (requires pyiqa)
    * ``ocr`` — OCR text accuracy for prompts with quoted text (requires pytesseract)
    """
    from .types import MetricResult

    clip_scorer: CLIPScorer | None = None
    lpips_scorer: LPIPSScorer | None = None
    face_scorer: FaceScorer | None = None
    sharpness_scorer: SharpnessScorer | None = None
    aesthetic_scorer: AestheticScorer | None = None
    nima_scorer: NimaScorer | None = None
    ocr_scorer: OCRScorer | None = None

    if "clip" in metric_types and pipeline == "text2img":
        clip_scorer = CLIPScorer(cache_dir=cache_dir)
    if "lpips" in metric_types and pipeline == "img2img":
        lpips_scorer = LPIPSScorer(cache_dir=cache_dir)
    if "face" in metric_types and pipeline == "img2img":
        face_scorer = FaceScorer(cache_dir=cache_dir)
    if "sharpness" in metric_types:
        sharpness_scorer = SharpnessScorer()
    if "aesthetic" in metric_types and "nima" not in metric_types:
        # "aesthetic" is deprecated alias — use NimaScorer if available, else fall back
        if NIMA_AVAILABLE:
            nima_scorer = NimaScorer()
        else:
            aesthetic_scorer = AestheticScorer(clip_scorer=clip_scorer, cache_dir=cache_dir)
    if "nima" in metric_types:
        nima_scorer = NimaScorer()
    if "ocr" in metric_types:
        ocr_scorer = OCRScorer()

    if clip_scorer is None and lpips_scorer is None and face_scorer is None and sharpness_scorer is None and aesthetic_scorer is None and nima_scorer is None and ocr_scorer is None:
        return

    # Build a prompt lookup: item_id -> prompt text
    prompt_map: dict[str, str] = {}
    input_map: dict[str, str] = {}
    expected_text_map: dict[str, str] = {}
    for p in prompts:
        item_id = p.get("item_id", "")
        prompt_map[item_id] = p.get("prompt", "")
        input_map[item_id] = p.get("image_url", "")
        # expected_text can be a string or list of strings
        et = p.get("expected_text", "")
        if isinstance(et, list):
            et = " ".join(et)
        expected_text_map[item_id] = et

    # Resolve input images (download URLs to local files for LPIPS/face)
    resolved_inputs: dict[str, str] = {}

    for item in items:
        prompt_text = prompt_map.get(item.item_id, item.prompt)
        input_image = input_map.get(item.item_id, item.image_url)

        # Resolve URL to local path once per item (shared by LPIPS + face)
        if input_image and (lpips_scorer or face_scorer):
            if input_image not in resolved_inputs:
                try:
                    resolved_inputs[input_image] = _resolve_image_path(input_image, cache_dir)
                except Exception:
                    resolved_inputs[input_image] = ""
            input_image_local = resolved_inputs[input_image]
        else:
            input_image_local = input_image

        for _model, img_result in item.results.items():
            if img_result.status != "success" or not img_result.image_local:
                continue

            if clip_scorer and prompt_text:
                try:
                    val = clip_scorer.score(img_result.image_local, prompt_text)
                    img_result.metrics.append(
                        MetricResult(
                            metric="clip_score",
                            value=round(val, 4),
                            description="CLIP cosine similarity (text-image)",
                        )
                    )
                except Exception:
                    pass  # skip on failure

            if lpips_scorer and input_image_local:
                try:
                    val = lpips_scorer.score(input_image_local, img_result.image_local)
                    img_result.metrics.append(
                        MetricResult(
                            metric="lpips",
                            value=round(val, 4),
                            description="LPIPS perceptual similarity (1 - distance)",
                        )
                    )
                except Exception:
                    pass  # skip on failure

            if face_scorer and input_image_local:
                try:
                    val = face_scorer.score(input_image_local, img_result.image_local)
                    if val is not None:  # Face detected in both images
                        img_result.metrics.append(
                            MetricResult(
                                metric="face_similarity",
                                value=round(val, 4),
                                description="ArcFace identity similarity (face embedding cosine)",
                            )
                        )
                except Exception:
                    pass  # skip on failure

            if sharpness_scorer:
                try:
                    val = sharpness_scorer.score(img_result.image_local)
                    img_result.metrics.append(
                        MetricResult(
                            metric="sharpness",
                            value=val,
                            description="Variance of Laplacian sharpness (0-1, higher=sharper)",
                        )
                    )
                except Exception:
                    pass  # skip on failure

            if aesthetic_scorer:
                try:
                    val = aesthetic_scorer.score(img_result.image_local)
                    img_result.metrics.append(
                        MetricResult(
                            metric="aesthetic_score",
                            value=val,
                            description="Aesthetic quality (LAION MLP on CLIP, 0-1)",
                        )
                    )
                except Exception:
                    pass  # skip on failure

            if nima_scorer:
                try:
                    val = nima_scorer.score(img_result.image_local)
                    img_result.metrics.append(
                        MetricResult(
                            metric="nima_score",
                            value=val,
                            description="NIMA aesthetic quality (AVA-trained, 0-1)",
                        )
                    )
                except Exception:
                    pass  # skip on failure

            if ocr_scorer:
                try:
                    # Prefer explicit expected_text from prompt metadata, fall back to regex extraction
                    expected = expected_text_map.get(item.item_id, "") or _extract_expected_text(prompt_text)
                    if expected:
                        val = ocr_scorer.score(img_result.image_local, expected)
                        if val is not None:
                            img_result.metrics.append(
                                MetricResult(
                                    metric="ocr_accuracy",
                                    value=val,
                                    description="OCR text accuracy (SequenceMatcher ratio, 0-1)",
                                )
                            )
                except Exception:
                    pass  # skip on failure


# ---------------------------------------------------------------------------
# Metric scoring: normalization + threshold
# ---------------------------------------------------------------------------


def normalize_metric(value: float, range_min: float, range_max: float) -> float:
    """Linearly map *value* from [range_min, range_max] to [0, 5], clamped."""
    if range_max <= range_min:
        return 0.0
    ratio = (value - range_min) / (range_max - range_min)
    return max(0.0, min(5.0, ratio * 5.0))


def check_metric_threshold(
    metric: str,
    value: float,
    config: MetricScoringConfig,
) -> MetricFlag | None:
    """Return a :class:`MetricFlag` if *value* is below the threshold, else ``None``."""
    from .types import MetricFlag

    if value <= config.flag_threshold:
        label = {"clip_score": "CLIP score", "lpips": "LPIPS similarity", "face_similarity": "Face similarity", "sharpness": "Sharpness", "aesthetic_score": "Aesthetic score", "nima_score": "NIMA score", "ocr_accuracy": "OCR accuracy"}.get(metric, metric)
        return MetricFlag(
            metric=metric,
            value=value,
            threshold=config.flag_threshold,
            message=f"{label} ({value:.4f}) below threshold ({config.flag_threshold}) — excluded from overall",
        )
    return None


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def _interpret_r(r: float) -> str:
    """Classify Pearson r magnitude into human-readable label."""
    ar = abs(r)
    if ar >= 0.7:
        return "high_agreement"
    if ar >= 0.4:
        return "moderate"
    return "low_agreement"


def compute_correlation(
    items: list[BenchItem],
    metric: str,
    dimension: str,
) -> CorrelationStat | None:
    """Pearson r between a local metric and a VLM dimension across all images.

    Returns ``None`` if fewer than 3 paired data points.
    """
    from .types import CorrelationStat

    metric_vals: list[float] = []
    dim_vals: list[float] = []

    for item in items:
        for _model, img_result in item.results.items():
            if img_result.status != "success":
                continue
            # Find metric value
            m_val: float | None = None
            for mr in img_result.metrics:
                if mr.metric == metric:
                    m_val = mr.value
                    break
            # Find dimension score
            d_val: float | None = None
            for dr in img_result.scores:
                if dr.dimension == dimension:
                    d_val = dr.score
                    break
            if m_val is not None and d_val is not None:
                metric_vals.append(m_val)
                dim_vals.append(d_val)

    if len(metric_vals) < 3:
        return None

    try:
        r = statistics.correlation(metric_vals, dim_vals)
    except statistics.StatisticsError:
        # One or both inputs are constant (zero variance) — no correlation
        return None

    # Approximate p-value via t-distribution
    n = len(metric_vals)
    if abs(r) >= 1.0:
        p_value = 0.0
    else:
        t_stat = r * math.sqrt((n - 2) / (1 - r * r))
        # Two-tailed p-value approximation using t-distribution CDF
        # For large n this is reasonable; exact requires scipy
        df = n - 2
        x = df / (df + t_stat * t_stat)
        # Regularised incomplete beta approximation (simple)
        p_value = _approx_t_pvalue(t_stat, df)

    return CorrelationStat(
        metric_pair=f"{metric}_vs_{dimension}",
        pearson_r=round(r, 4),
        p_value=round(p_value, 6),
        interpretation=_interpret_r(r),
    )


def _approx_t_pvalue(t: float, df: int) -> float:
    """Rough two-tailed p-value for the t-distribution.

    Uses the approximation from Abramowitz & Stegun for |t| > 0.
    Accurate enough for reporting; not for publication.
    """
    if df <= 0:
        return 1.0
    x = abs(t)
    # Normal approximation for large df
    if df > 100:
        # Use normal CDF approximation
        z = x
        p_one_tail = 0.5 * math.erfc(z / math.sqrt(2))
        return 2 * p_one_tail
    # Simple approximation for smaller df
    a = 1.0 + x * x / df
    p_one_tail = 0.5 * a ** (-(df + 1) / 2.0)
    # Normalise roughly — this is a rough estimate
    return min(1.0, 2 * p_one_tail)
