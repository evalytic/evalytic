"""Golden set: 5 text2img prompts for product background generation."""

BACKGROUNDS = [
    {
        "prompt": "Clean white studio background with soft diffused shadows, product photography, minimalist, high-key lighting",
        "expected": "white studio",
        "tags": ["studio", "minimal"],
    },
    {
        "prompt": "Rustic wooden table surface with warm morning sunlight streaming from the left, shallow depth of field, lifestyle product photography",
        "expected": "wooden table",
        "tags": ["lifestyle", "warm"],
    },
    {
        "prompt": "Polished marble surface with fresh eucalyptus leaves and soft shadows, luxury beauty product photography, elegant and clean",
        "expected": "marble with eucalyptus",
        "tags": ["luxury", "beauty"],
    },
    {
        "prompt": "Flat lay on light gray linen fabric texture, overhead view, soft natural lighting, artisan product photography",
        "expected": "gray linen flat lay",
        "tags": ["flatlay", "textile"],
    },
    {
        "prompt": "Dark carbon fiber surface with subtle blue LED accent lighting, tech product photography, dramatic and modern",
        "expected": "carbon fiber tech",
        "tags": ["tech", "dark"],
    },
]
