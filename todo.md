# Short Video Agent - Pipeline TODO

## Architecture Notes

- Develop in stages and verify incrementally to save costs
- Reference images of speaker need to be passed as well as product image if its an ad
- Current approach: green screen + alpha mask compositing

## Pipeline Steps

1. **Generate person with input image of speaker**

   - Have a picture as input frame of persona, multiple inputs / different personas
   - ✅ Implemented in `step1.py`

2. **Get alpha mask / green screen video**

   - Apply it and overlay onto something
   - ✅ Implemented in `step2.py`

3. **Composite the generated person with the alpha mask onto the background**

   - ✅ Implemented in `step3.py`

4. **Generate start images for scenes and use text to video with start frame image**

   - ✅ Implemented in `step4.py`
   - Now supports start/end frame constraints

5. **Stitch everything together into final video**

   - ✅ Implemented in `step5.py`

6. **Add audio (voiceover, background music, sound effects)**

   - [ ] TODO

7. **Add text overlays for captions/transcript**
   - [ ] TODO

## Infrastructure

- [x] Replicate client / API wrapper
- [x] Unified interface to call text-to-video or text-to-image models (abstract replicate-specific API)
- [x] Script-like JSON or YAML definition of scene (`scene_schema.py`, `example_ad.yaml`)
- [x] Text-to-speech support with duration estimation

## Available Models (via `replicate_client.py`)

### Video Models

| Model                 | ID                                   | Cost Tier | Features                           |
| --------------------- | ------------------------------------ | --------- | ---------------------------------- |
| `veo-3.1-fast`        | google/veo-3.1-fast                  | standard  | start/end frame, audio (Replicate) |
| `veo-3.1`             | google/veo-3.1                       | premium   | start/end frame, audio (Replicate) |
| `ltx-video`           | Lightricks/LTX-Video                 | self-host | text-to-video, image-to-video (HF) |
| `ltx-video-distilled` | Lightricks/LTX-Video-0.9.7-distilled | self-host | faster, fewer steps (HF)           |

### Image Models

| Model          | ID                             | Cost Tier |
| -------------- | ------------------------------ | --------- |
| `flux-2-pro`   | black-forest-labs/flux-2-pro   | standard  |
| `flux-1.1-pro` | black-forest-labs/flux-1.1-pro | cheap     |

### Video Matting

| Model                  | ID                                  | Cost Tier |
| ---------------------- | ----------------------------------- | --------- |
| `robust-video-matting` | arielreplicate/robust_video_matting | cheap     |

### Text-to-Speech

| Model             | ID                      | Cost Tier | Notes                      |
| ----------------- | ----------------------- | --------- | -------------------------- |
| `speech-02-hd`    | minimax/speech-02-hd    | standard  | High quality ($50/M chars) |
| `speech-02-turbo` | minimax/speech-02-turbo | cheap     | Fast ($30/M chars)         |
| `voice-cloning`   | minimax/voice-cloning   | standard  | Clone voice ($3/voice)     |

## Usage Examples

```python
from replicate_client import (
    generate_video,
    generate_video_with_image,
    generate_image,
    remove_background,
    save_output,
    list_models,
    ModelType,
)

# Generate video with start frame
output = generate_video_with_image(
    prompt="Camera zooms into her eye...",
    image="start_frame.png",
    duration=8,
    last_frame="end_frame.png",  # Optional end frame constraint
    model="veo-3.1-fast",
)
save_output(output, "output.mp4")

# Switch to cheaper model for testing
output = generate_image(
    prompt="Beautiful scene...",
    model="flux-1.1-pro",  # Cheaper than flux-2-pro
)

# List available video models
video_models = list_models(ModelType.IMAGE_TO_VIDEO)

# Text-to-speech with duration fitting
from replicate_client import generate_speech, fit_text_to_duration, estimate_speech_duration

text = "This is my script that needs to fit in 5 seconds."
fitted_text = fit_text_to_duration(text, target_duration=5.0)
audio = generate_speech(fitted_text, emotion="happy")
save_output(audio, "voiceover.mp3")

# Load scene from YAML
from scene_schema import load_config
config = load_config("example_ad.yaml")
print(f"Total duration: {config.total_duration}s")
for scene in config.scenes:
    valid, est = scene.validate_script_duration()
    print(f"  {scene.id}: {est:.1f}s script in {scene.duration_s}s scene {'✓' if valid else '✗'}")
```

[x] add LTX video open source video generation model to run on own GPU for cost advantages on runpod GPU

---

## LTX-Video Model Reference

**Model Card:** [Lightricks/LTX-Video](https://hf.co/Lightricks/LTX-Video)

- **Task:** text-to-video, image-to-video
- **Library:** diffusers
- **Downloads:** 3.9M+ | **Likes:** 2000+
- **License:** other (Lightricks)

### Key Features

- **High compression Video-VAE** (1:192 pixel-to-latent ratio) for fast generation
- **~10GB VRAM** with memory optimizations (fp8 + group offloading)
- **Supports conditioning** on images/videos for image-to-video
- **LoRA support** for stylistic modifications
- **Spatial upscaler** available (LTX-Video 0.9.7+)

### Available Variants

| Variant          | Model ID                                 | Notes                         |
| ---------------- | ---------------------------------------- | ----------------------------- |
| Base             | `Lightricks/LTX-Video`                   | Original model                |
| 0.9.7-dev        | `Lightricks/LTX-Video-0.9.7-dev`         | Supports upscaling pipeline   |
| 0.9.7-distilled  | `Lightricks/LTX-Video-0.9.7-distilled`   | Faster, fewer inference steps |
| Spatial Upscaler | `Lightricks/ltxv-spatial-upscaler-0.9.7` | 2x latent upscaling           |

### Basic Usage (Text-to-Video)

```python
import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

pipeline = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video", torch_dtype=torch.bfloat16
)
pipeline.to("cuda")

prompt = "A woman with long brown hair smiles at camera, soft lighting, vertical shot"
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=512,
    num_frames=161,
    decode_timestep=0.03,
    decode_noise_scale=0.025,
    num_inference_steps=50,
).frames[0]

export_to_video(video, "output.mp4", fps=24)
```

### Memory-Optimized Usage (~10GB VRAM)

```python
import torch
from diffusers import LTXPipeline, AutoModel
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video

# fp8 layerwise weight-casting
transformer = AutoModel.from_pretrained(
    "Lightricks/LTX-Video",
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)
transformer.enable_layerwise_casting(
    storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
)

pipeline = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video", transformer=transformer, torch_dtype=torch.bfloat16
)

# group-offloading
onload_device = torch.device("cuda")
offload_device = torch.device("cpu")
pipeline.transformer.enable_group_offload(
    onload_device=onload_device, offload_device=offload_device,
    offload_type="leaf_level", use_stream=True
)
apply_group_offloading(pipeline.text_encoder, onload_device=onload_device,
                       offload_type="block_level", num_blocks_per_group=2)
apply_group_offloading(pipeline.vae, onload_device=onload_device,
                       offload_type="leaf_level")

video = pipeline(
    prompt="Your prompt here",
    negative_prompt="worst quality, blurry, jittery",
    width=768, height=512, num_frames=161,
    decode_timestep=0.03, decode_noise_scale=0.025,
    num_inference_steps=50,
).frames[0]

export_to_video(video, "output.mp4", fps=24)
```

### Image-to-Video with Conditioning

```python
import torch
from diffusers import LTXConditionPipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_image

pipeline = LTXConditionPipeline.from_pretrained(
    "Lightricks/LTX-Video-0.9.7-dev", torch_dtype=torch.bfloat16
)
pipeline.to("cuda")
pipeline.vae.enable_tiling()

# Load conditioning image
image = load_image("path/to/start_frame.png")
condition = LTXVideoCondition(video=[image], frame_index=0)

video = pipeline(
    conditions=[condition],
    prompt="Camera slowly zooms in on the subject",
    negative_prompt="worst quality, blurry",
    width=768, height=512, num_frames=161,
    num_inference_steps=30,
    decode_timestep=0.05,
    decode_noise_scale=0.025,
    guidance_scale=5.0,
).frames[0]

export_to_video(video, "output.mp4", fps=24)
```

### Distilled Model (Faster, Fewer Steps)

```python
import torch
from diffusers import LTXConditionPipeline
from diffusers.utils import export_to_video

pipeline = LTXConditionPipeline.from_pretrained(
    "Lightricks/LTX-Video-0.9.7-distilled", torch_dtype=torch.bfloat16
)
pipeline.to("cuda")
pipeline.vae.enable_tiling()

# Use custom timesteps for distilled model
video = pipeline(
    prompt="Your prompt here",
    negative_prompt="worst quality, blurry",
    width=512, height=768,  # 9:16 vertical
    num_frames=161,
    timesteps=[1000, 993, 987, 981, 975, 909, 725, 0.03],  # Custom timesteps
    guidance_scale=1.0,  # Use 1.0 for distilled
).frames[0]

export_to_video(video, "output.mp4", fps=24)
```

### Recommended Settings

| Parameter             | Value                              | Notes                                       |
| --------------------- | ---------------------------------- | ------------------------------------------- |
| `torch_dtype`         | `torch.bfloat16`                   | Required for transformer, VAE, text encoder |
| `decode_timestep`     | `0.03-0.05`                        | For timestep-aware VAE (0.9.1+)             |
| `decode_noise_scale`  | `0.025`                            | Noise scale for VAE decoding                |
| `guidance_scale`      | `1.0` (distilled) / `5.0` (normal) | Distilled models use 1.0                    |
| `num_inference_steps` | `50` (normal) / `8` (distilled)    | Fewer steps for distilled                   |
| `num_frames`          | `161`                              | ~6.7s at 24fps                              |

### Integration with Pipeline

To integrate LTX-Video into the existing pipeline, create a new video model interface:

```python
# In replicate_client.py or new ltx_client.py
from abc import ABC, abstractmethod

class VideoModelInterface(ABC):
    """Abstract interface for video generation models."""

    @abstractmethod
    def generate_video(
        self,
        prompt: str,
        duration: int = 8,
        width: int = 768,
        height: int = 512,
        start_image: str | None = None,
        end_image: str | None = None,
    ) -> str:
        """Generate video and return path to output file."""
        pass

class ReplicateVideoModel(VideoModelInterface):
    """Replicate-based video models (Veo, etc.)."""
    def __init__(self, model_name: str = "veo-3.1-fast"):
        self.model_name = model_name

    def generate_video(self, prompt, duration=8, width=768, height=512,
                       start_image=None, end_image=None) -> str:
        # Use existing replicate_client functions
        ...

class LTXVideoModel(VideoModelInterface):
    """Self-hosted LTX-Video model."""
    def __init__(self, model_id: str = "Lightricks/LTX-Video",
                 use_distilled: bool = False):
        self.model_id = model_id
        self.pipeline = None  # Lazy load

    def _load_pipeline(self):
        if self.pipeline is None:
            from diffusers import LTXPipeline
            self.pipeline = LTXPipeline.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16
            ).to("cuda")

    def generate_video(self, prompt, duration=8, width=768, height=512,
                       start_image=None, end_image=None) -> str:
        self._load_pipeline()
        num_frames = int(duration * 24)  # 24fps
        video = self.pipeline(
            prompt=prompt,
            width=width, height=height,
            num_frames=num_frames,
            num_inference_steps=50,
        ).frames[0]
        output_path = f"output_{hash(prompt)}.mp4"
        export_to_video(video, output_path, fps=24)
        return output_path
```

### Demo Spaces

- [ltx-video-distilled](https://hf.co/spaces/Lightricks/ltx-video-distilled)
- [LTXV-lora-the-explorer](https://hf.co/spaces/linoyts/LTXV-lora-the-explorer)
- [VideoModelStudio](https://hf.co/spaces/jbilcke-hf/VideoModelStudio)
