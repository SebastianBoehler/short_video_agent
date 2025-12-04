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

| Model          | ID                  | Cost Tier | Features               |
| -------------- | ------------------- | --------- | ---------------------- |
| `veo-3.1-fast` | google/veo-3.1-fast | standard  | start/end frame, audio |
| `veo-3.1`      | google/veo-3.1      | premium   | start/end frame, audio |

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
