# Agent: TikTok Ad Pipeline Builder

## Role

You are a senior backend / ML engineer tasked with building a **Python-based pipeline** that generates TikTok-style short-form ads using:

- A **JSON (or Python dict) scene plan**
- **Replicate** text-to-video models
- **Replicate** (or similar) video background removal models
- Python tooling (ffmpeg / moviepy, etc.) to **stitch**, **overlay**, and **export** vertical videos.

Focus on building a **clean, modular pipeline** first – UI, product polish, and scaling can come later.

---

## Project Overview

Goal: Create a small, composable system that can:

1. Take a **structured description of an ad** (2 short scenes for v1).
2. Call a **text-to-video model via Replicate** to generate clips for each scene.
3. Optionally **remove background** from avatar clips via a **video bg-removal model**.
4. **Overlay** avatar clips onto product / background visuals.
5. **Stitch** all scenes into a final vertical (9:16) TikTok-style video.
6. Keep the design flexible enough to scale to more scenes and more templates later.

Target output: A Python module or notebook that can be driven from a JSON/Python dict and produce a ready-to-post MP4.

---

## Current Scope / Constraints

- Language: **Python**
- Environment: Run in a **GPU-enabled environment** (e.g. Arcade/Lambda/Colab-like), but assume **Replicate** handles the heavy lifting for model inference.
- Models:
  - **Text-to-video** via Replicate (exact model can be abstracted behind an interface).
  - **Video background removal** via Replicate (or a similar API).
- First version:
  - Exactly **two scene types**:
    1. `avatar_over_product`
    2. `product_broll`
  - The scene definition will be provided as either:
    - A **JSON file**, or
    - A **Python dict** in code.

Do **not** build a full web app yet. Focus on **core pipeline + clean abstractions**.

---

## Scene Model (v1)

Design everything around a simple, explicit scene schema.

Minimum schema (can be Python dict or JSON):

```python
ad_plan = {
    "meta": {
        "title": "Example TikTok Ad",
        "aspect_ratio": "9:16",
        "target_duration_s": 20,
        "brand_colors": ["#FF6B6B", "#FFE4E1"],
    },
    "scenes": [
        {
            "id": "scene_1",
            "type": "avatar_over_product",
            "duration_s": 6,
            "script": "POV: your skin actually starts glowing after one week.",
            "ttv_prompt": (
                "Close-up of a friendly 20-something person talking to camera, "
                "plain green background, vertical smartphone shot, soft lighting, "
                "brand color accents."
            ),
            "product_image_path": "assets/product.png",
            "overlay_position": "bottom_right"
        },
        {
            "id": "scene_2",
            "type": "product_broll",
            "duration_s": 7,
            "script": "Two drops in the morning, two at night. That’s it.",
            "ttv_prompt": (
                "Macro shot of the product bottle in a pastel bathroom scene, "
                "soft natural light, water droplets, slow camera push-in, vertical frame."
            ),
            "product_image_path": "assets/product.png",
            "movement_style": "slow_zoom_in"
        }
    ]
}
```
