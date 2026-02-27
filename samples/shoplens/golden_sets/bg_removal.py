"""Golden set: 3 product photos for background removal (edge cases)."""

CUTOUTS = [
    {
        "image_url": "https://images.pexels.com/photos/1115128/pexels-photo-1115128.jpeg?auto=compress&w=800",
        "product_type": "makeup brush",
        "challenge": "Fine bristles at edges require precise masking",
    },
    {
        "image_url": "https://images.pexels.com/photos/8361540/pexels-photo-8361540.jpeg?auto=compress&w=800",
        "product_type": "glass bottle",
        "challenge": "Transparency and reflections make edge detection difficult",
    },
    {
        "image_url": "https://images.pexels.com/photos/2529147/pexels-photo-2529147.jpeg?auto=compress&w=800",
        "product_type": "sneaker",
        "challenge": "Complex lace silhouette with thin overlapping elements",
    },
]
