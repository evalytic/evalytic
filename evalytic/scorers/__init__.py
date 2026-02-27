"""Built-in evaluation dimension constants.

Import these to avoid hard-coding dimension name strings throughout your
code:

    from evalytic.scorers import VISUAL_QUALITY, PROMPT_ADHERENCE

    evalytic.Eval(
        project="my-project",
        data=ds,
        scores=[VISUAL_QUALITY, PROMPT_ADHERENCE],
    )
"""

# -- Shared (all pipeline types) ------------------------------------------
VISUAL_QUALITY: str = "visual_quality"

# -- text2img dimensions --------------------------------------------------
PROMPT_ADHERENCE: str = "prompt_adherence"
TEXT_RENDERING: str = "text_rendering"

# -- img2img dimensions ---------------------------------------------------
INPUT_FIDELITY: str = "input_fidelity"
TRANSFORMATION_QUALITY: str = "transformation_quality"
ARTIFACT_DETECTION: str = "artifact_detection"
IDENTITY_PRESERVATION: str = "identity_preservation"

# Convenience list of ALL implemented dimensions.
ALL_DIMENSIONS: list[str] = [
    VISUAL_QUALITY,
    PROMPT_ADHERENCE,
    TEXT_RENDERING,
    INPUT_FIDELITY,
    TRANSFORMATION_QUALITY,
    ARTIFACT_DETECTION,
    IDENTITY_PRESERVATION,
]

# Future: anatomical_correctness, physics_lighting, style_consistency,
#         detail_preservation, temporal_coherence
