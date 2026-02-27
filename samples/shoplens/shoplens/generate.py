"""fal.ai image generation wrappers."""

from __future__ import annotations

import fal_client

from .config import FAL_MODELS


def generate_text2img(prompt: str, model_key: str, **params: object) -> str:
    """Generate an image from a text prompt. Returns the image URL."""
    model_id = FAL_MODELS[model_key]
    arguments = {"prompt": prompt, **params}
    result = fal_client.subscribe(model_id, arguments=arguments)
    return result["images"][0]["url"]


def generate_img2img(
    image_url: str,
    instruction: str,
    model_key: str,
    **params: object,
) -> str:
    """Transform an image with a text instruction. Returns the image URL."""
    model_id = FAL_MODELS[model_key]
    arguments = {"image_url": image_url, "prompt": instruction, **params}
    result = fal_client.subscribe(model_id, arguments=arguments)
    # Some models return images[0], others return image
    if "images" in result:
        return result["images"][0]["url"]
    return result["image"]["url"]


def remove_background(image_url: str) -> str:
    """Remove background from an image using BiRefNet. Returns the image URL."""
    model_id = FAL_MODELS["birefnet"]
    result = fal_client.subscribe(model_id, arguments={"image_url": image_url})
    return result["image"]["url"]
