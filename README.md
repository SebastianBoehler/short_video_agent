# Short Video Agent

A modular Python pipeline for generating TikTok-style short-form videos using AI models.

## Features

- **Multi-image speaker support** - Use folders with multiple reference images per speaker
- **Backend flexibility** - Switch between Replicate API and local LTX-2 models
- **Docker deployment** - GPU-enabled images for cloud deployment (RunPod, GCloud, etc.)
- **Modular architecture** - Clean separation of generators, processors, and pipeline logic
- **YAML configuration** - Define speakers, products, and scenes declaratively

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add your REPLICATE_API_TOKEN to .env

# Run with a config file
python run.py --config schemes/example_multi_speaker.yaml

# Check available models
python run.py --list-models

# Check GPU availability for local models
python run.py --check-gpu
```

## Architecture

```
src/
├── generators/          # Video, image, audio generation
│   ├── base.py         # Abstract interfaces
│   ├── replicate.py    # Replicate API backend
│   └── ltx.py          # Local LTX-2 model
├── processors/          # Video processing
│   ├── matting.py      # Background removal
│   ├── compositor.py   # Overlay/compositing
│   ├── captions.py     # TikTok-style captions
│   └── stitcher.py     # Scene concatenation
├── config/              # Configuration
│   ├── models.py       # Model registry
│   └── schema.py       # Scene/speaker schema
├── pipeline/            # Orchestration
│   ├── scene.py        # Scene processing
│   └── runner.py       # Main pipeline
└── utils/               # Utilities
```

## Multi-Image Speaker Support

Define speakers with folders containing multiple reference images:

```yaml
speakers:
  - id: "host"
    name: "Sarah"
    image_dir: "speakers/sarah/" # Folder with multiple images
    description: "Young woman, mid-20s, long brown hair"

scenes:
  - id: "intro"
    type: "speaker_in_scene"
    speaker_id: "host" # Uses all images from folder
    scene_prompt: "Woman in modern bathroom..."
```

## Backends

| Backend     | Description                      | Use Case                     |
| ----------- | -------------------------------- | ---------------------------- |
| `replicate` | Replicate API for all generation | Default, no GPU needed       |
| `local`     | Local LTX-2 for video            | GPU deployment, cost savings |
| `hybrid`    | Local video + API images         | Balance of speed and cost    |

```bash
# Use Replicate API (default)
python run.py --config my_ad.yaml --backend replicate

# Use local LTX-2 model
python run.py --config my_ad.yaml --backend local
```

## Docker Deployment

```bash
# CPU-only (uses Replicate API)
docker build -f Dockerfile.cpu -t video-agent-cpu .
docker run -v $(pwd)/outputs:/app/outputs video-agent-cpu --config schemes/my_ad.yaml

# GPU-enabled (uses local LTX-2)
docker build -t video-agent-gpu .
docker run --gpus all -v $(pwd)/outputs:/app/outputs video-agent-gpu --config schemes/my_ad.yaml --backend local
```

## Scene Types

| Type                   | Description                            |
| ---------------------- | -------------------------------------- |
| `speaker`              | Person talking with background overlay |
| `speaker_in_scene`     | Transform speaker into environment     |
| `speaker_angle_change` | Continue with new camera angle         |
| `broll`                | Pure video without speaker             |
| `product`              | Product-focused scene                  |

## Models

**Video Generation:**

- `veo-3.1-fast` - Google Veo with audio (speaker scenes)
- `wan-2.5-i2v` - Cheap image-to-video
- `ltx-2` - Local model with audio support

**Image Generation:**

- `seedream-4.5` - Multi-image reference support
- `nano-banana-pro` - Scene transformation
- `flux-2-pro` - High quality

## Requirements

- Python 3.11+
- ffmpeg
- Replicate API token (for API backend)
- NVIDIA GPU with 10GB+ VRAM (for local backend)
