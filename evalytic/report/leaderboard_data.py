"""Leaderboard model metadata — display names, families, imgsys ELO, seed support.

This lookup table enriches BenchReport data with leaderboard-specific metadata
that is not carried by the report itself (family, official name, ELO, seed info).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeaderboardModelMeta:
    """Static metadata for one leaderboard model."""

    display_name: str
    family: str
    imgsys_elo: int | None = None
    seed_supported: bool = True
    license_type: str = "proprietary"  # "open" or "proprietary"


# fmt: off
LEADERBOARD_MODELS: dict[str, LeaderboardModelMeta] = {
    # Google / Gemini (Nano Banana)
    "nano-banana-2":    LeaderboardModelMeta("Gemini 3.1 Flash Image",  "Google",   1268),
    "nano-banana-pro":  LeaderboardModelMeta("Gemini 3 Pro Image",      "Google",   1233),
    "nano-banana":      LeaderboardModelMeta("Gemini 2.5 Flash Image",  "Google",   1156),
    # Google / Imagen
    "imagen-4-ultra":   LeaderboardModelMeta("Imagen 4 Ultra",          "Google",   1148),
    "imagen-4":         LeaderboardModelMeta("Imagen 4",                "Google",   1133),
    "imagen-4-fast":    LeaderboardModelMeta("Imagen 4 Fast",           "Google",   None),
    "imagen-3":         LeaderboardModelMeta("Imagen 3",                "Google",   1059),
    # BFL / FLUX
    "flux-2-max":       LeaderboardModelMeta("FLUX.2 [max]",            "Black Forest Labs", 1167),
    "flux-2-flex":      LeaderboardModelMeta("FLUX.2 [flex]",           "Black Forest Labs", 1160),
    "flux-2-pro":       LeaderboardModelMeta("FLUX.2 [pro]",            "Black Forest Labs", 1155),
    "flux-2-dev":       LeaderboardModelMeta("FLUX.2 [dev]",            "Black Forest Labs", 1149, license_type="open"),
    "flux-pro":         LeaderboardModelMeta("FLUX Pro 1.1",            "Black Forest Labs", 1016),
    "flux-schnell":     LeaderboardModelMeta("FLUX.1 [schnell]",        "Black Forest Labs", 950, license_type="open"),
    # Reve
    "reve":             LeaderboardModelMeta("Reve v1.0",               "Reve",     1177, seed_supported=False),
    # xAI
    "grok-imagine":     LeaderboardModelMeta("Grok Imagine",            "xAI",      1175, seed_supported=False),
    # Tencent / Hunyuan
    "hunyuan-3":        LeaderboardModelMeta("Hunyuan Image 3.0",       "Tencent",  1152, license_type="open"),
    # ByteDance / Seedream (API-only, no open weights)
    "seedream-v4.5":    LeaderboardModelMeta("Seedream v4.5",           "ByteDance", 1144),
    "seedream-v4":      LeaderboardModelMeta("Seedream v4",             "ByteDance", 1118),
    "seedream":         LeaderboardModelMeta("Seedream v5 Lite",        "ByteDance", 1111),
    # Alibaba
    "qwen-image-2":     LeaderboardModelMeta("Qwen Image 2.0",         "Alibaba",  1139, license_type="open"),
    "wan-2.6":          LeaderboardModelMeta("Wan 2.6 T2I",            "Alibaba",  1135, license_type="open"),
    "qwen-image":       LeaderboardModelMeta("Qwen Image",             "Alibaba",  1058, license_type="open"),
    # Recraft
    "recraft-v4":       LeaderboardModelMeta("Recraft V4",              "Recraft",  1100, seed_supported=False),
    "recraft-v3":       LeaderboardModelMeta("Recraft V3",              "Recraft",  1021, seed_supported=False),
    # Ideogram
    "ideogram-v3":      LeaderboardModelMeta("Ideogram V3 Turbo",      "Ideogram", 1050),
    # OpenAI
    "gpt-image-1":      LeaderboardModelMeta("GPT Image 1",            "OpenAI",   1115, seed_supported=False),
    "gpt-image-1-mini": LeaderboardModelMeta("GPT Image 1 Mini",       "OpenAI",   1104, seed_supported=False),
    # Other
    "z-image-turbo":    LeaderboardModelMeta("Z-Image Turbo",          "Other",    1080),
    "glm-image":        LeaderboardModelMeta("GLM Image",              "Other",    1012, license_type="open"),
    "hidream":          LeaderboardModelMeta("HiDream i1 Full",        "Other",    None, license_type="open"),
    "sd35-large":       LeaderboardModelMeta("SD 3.5 Large",           "Other",    939, license_type="open"),
    "bria-fibo":        LeaderboardModelMeta("Bria FIBO",              "Other",    None),
    "imagineart":       LeaderboardModelMeta("ImagineArt 1.5",         "Other",    None),
}
# fmt: on
