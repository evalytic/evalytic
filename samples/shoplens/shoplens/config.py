"""Pipeline configs and fal.ai model registry."""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# fal.ai model registry
# ---------------------------------------------------------------------------

FAL_MODELS: dict[str, str] = {
    # text2img
    "flux-schnell": "fal-ai/flux/schnell",
    "flux-dev": "fal-ai/flux/dev",
    "flux-pro": "fal-ai/flux-pro/v1.1",
    # img2img
    "flux-dev-i2i": "fal-ai/flux/dev/image-to-image",
    "flux-kontext": "fal-ai/flux-pro/kontext",
    # utility
    "birefnet": "fal-ai/birefnet/v2",
}


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    pipeline_id: str
    name: str
    pipeline_type: str  # "text2img" | "img2img"
    default_model: str
    challenger_models: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)
    fal_params: dict = field(default_factory=dict)


PIPELINES: dict[str, PipelineConfig] = {
    "bg-generator": PipelineConfig(
        pipeline_id="bg-generator",
        name="Background Studio",
        pipeline_type="text2img",
        default_model="flux-schnell",
        challenger_models=["flux-dev", "flux-pro"],
        dimensions=["visual_quality", "prompt_adherence"],
        fal_params={"image_size": "landscape_16_9"},
    ),
    "product-enhancer": PipelineConfig(
        pipeline_id="product-enhancer",
        name="Product Enhancer",
        pipeline_type="img2img",
        default_model="flux-dev-i2i",
        challenger_models=["flux-kontext"],
        dimensions=[
            "visual_quality",
            "input_fidelity",
            "transformation_quality",
            "artifact_detection",
        ],
    ),
    "clean-cut": PipelineConfig(
        pipeline_id="clean-cut",
        name="Clean Cut",
        pipeline_type="img2img",
        default_model="birefnet",
        dimensions=[
            "visual_quality",
            "input_fidelity",
            "artifact_detection",
        ],
    ),
}
