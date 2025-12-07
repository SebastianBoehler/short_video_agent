"""
Unified Replicate API client with model registry for easy switching between models.

Supports:
- Text-to-video generation
- Image-to-video generation (with optional start/end frames)
- Text-to-image generation
- Video background removal
"""

import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import replicate
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Model Registry
# =============================================================================

class ModelType(Enum):
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    TEXT_TO_IMAGE = "text_to_image"
    TEXT_TO_SPEECH = "text_to_speech"
    VOICE_CLONING = "voice_cloning"
    VIDEO_MATTING = "video_matting"
    VIDEO_CAPTIONING = "video_captioning"


@dataclass
class ModelConfig:
    """Configuration for a Replicate model."""
    name: str
    model_id: str
    model_type: ModelType
    default_params: dict = field(default_factory=dict)
    supports_start_frame: bool = False
    supports_end_frame: bool = False
    supports_audio: bool = False
    cost_tier: str = "standard"  # "cheap", "standard", "premium"
    description: str = ""


# Model registry - add new models here
MODEL_REGISTRY: dict[str, ModelConfig] = {
    # Video models
    "veo-3.1-fast": ModelConfig(
        name="veo-3.1-fast",
        model_id="google/veo-3.1-fast",
        model_type=ModelType.IMAGE_TO_VIDEO,
        default_params={
            "resolution": "720p",
            "aspect_ratio": "9:16",
            "duration": 8,
        },
        supports_start_frame=True,
        supports_end_frame=True,
        supports_audio=True,
        cost_tier="standard",
        description="Google Veo 3.1 Fast - high quality video generation with start/end frame support"
    ),
    "veo-3.1": ModelConfig(
        name="veo-3.1",
        model_id="google/veo-3.1",
        model_type=ModelType.IMAGE_TO_VIDEO,
        default_params={
            "resolution": "720p",
            "aspect_ratio": "9:16",
            "duration": 8,
        },
        supports_start_frame=True,
        supports_end_frame=True,
        supports_audio=True,
        cost_tier="premium",
        description="Google Veo 3.1 - highest quality, slower generation"
    ),
    
    # Image models
    "flux-2-pro": ModelConfig(
        name="flux-2-pro",
        model_id="black-forest-labs/flux-2-pro",
        model_type=ModelType.TEXT_TO_IMAGE,
        default_params={
            "resolution": "1 MP",
            "aspect_ratio": "9:16",
            "output_format": "png",
        },
        cost_tier="standard",
        description="FLUX 2 Pro - high quality image generation"
    ),
    "seedream-4.5": ModelConfig(
        name="seedream-4.5",
        model_id="bytedance/seedream-4.5",
        model_type=ModelType.TEXT_TO_IMAGE,
        default_params={
            "size": "4K",
            #"width": 2048, overriden by aspect ratio
            #"height": 2048,
            "max_images": 1,
            "aspect_ratio": "9:16",
            "sequential_image_generation": "disabled",
        },
        cost_tier="cheap",
        description="Seedream 4.5 - cheap image generation with reference image support"
    ),
    "flux-1.1-pro": ModelConfig(
        name="flux-1.1-pro",
        model_id="black-forest-labs/flux-1.1-pro",
        model_type=ModelType.TEXT_TO_IMAGE,
        default_params={
            "aspect_ratio": "9:16",
            "output_format": "png",
        },
        cost_tier="cheap",
        description="FLUX 1.1 Pro - fast, cheaper image generation"
    ),
    
    # Z-Image Turbo - ultra fast image generation
    "z-image-turbo": ModelConfig(
        name="z-image-turbo",
        model_id="prunaai/z-image-turbo",
        model_type=ModelType.TEXT_TO_IMAGE,
        default_params={
            "width": 768,
            "height": 1344,  # 9:16 vertical
            "output_format": "png",
            "guidance_scale": 0,
            "output_quality": 94,
            "num_inference_steps": 22,
        },
        cost_tier="cheap",
        description="Z-Image Turbo - ultra fast image generation (~$0.001/run)"
    ),
    
    # Video matting models
    "robust-video-matting": ModelConfig(
        name="robust-video-matting",
        model_id="arielreplicate/robust_video_matting:73d2128a371922d5d1abf0712a1d974be0e4e2358cc1218e4e34714767232bac",
        model_type=ModelType.VIDEO_MATTING,
        default_params={
            "output_type": "alpha-mask",
        },
        cost_tier="cheap",
        description="Robust Video Matting - background removal for videos"
    ),
    
    # Video captioning models
    "tiktok-captions": ModelConfig(
        name="tiktok-captions",
        model_id="shreejalmaharjan-27/tiktok-short-captions:46bf1c12c77ad1782d6f87828d4d8ba4d48646b8e1271b490cb9e95ccdbc4504",
        model_type=ModelType.VIDEO_CAPTIONING,
        default_params={
            "model": "large-v3",
            "language": "auto",
            "temperature": 0,
            "caption_size": 100,
            "highlight_color": "#FFFFFF",
            "suppress_tokens": "-1",
            "logprob_threshold": -1,
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": True,
            "compression_ratio_threshold": 2.4,
            "temperature_increment_on_fallback": 0.2,
        },
        cost_tier="cheap",
        description="TikTok-style captions - adds animated captions to video"
    ),
    
    # Text-to-speech models
    "speech-02-hd": ModelConfig(
        name="speech-02-hd",
        model_id="minimax/speech-02-hd",
        model_type=ModelType.TEXT_TO_SPEECH,
        default_params={},
        cost_tier="standard",
        description="MiniMax Speech-02-HD - high quality TTS for voiceovers ($50/M chars)"
    ),
    "speech-02-turbo": ModelConfig(
        name="speech-02-turbo",
        model_id="minimax/speech-02-turbo",
        model_type=ModelType.TEXT_TO_SPEECH,
        default_params={},
        cost_tier="cheap",
        description="MiniMax Speech-02-Turbo - fast TTS for real-time ($30/M chars)"
    ),
    
    # Voice cloning
    "voice-cloning": ModelConfig(
        name="voice-cloning",
        model_id="minimax/voice-cloning",
        model_type=ModelType.VOICE_CLONING,
        default_params={
            "model": "speech-02-hd",
        },
        cost_tier="standard",
        description="MiniMax Voice Cloning - clone voice from 10s-5min audio ($3/voice)"
    ),
    
    # LTX-Video - cheap text-to-video model
    "ltx-video-replicate": ModelConfig(
        name="ltx-video-replicate",
        model_id="lightricks/ltx-video:8c47da666861d081eeb4d1261853087de23923a268a69b63febdf5dc1dee08e4",
        model_type=ModelType.TEXT_TO_VIDEO,
        default_params={
            "cfg": 3,
            "model": "0.9.1",
            "steps": 30,
            "length": 97,  # ~4 seconds at 24fps
            "target_size": 640,
            "aspect_ratio": "9:16",
            "negative_prompt": "low quality, worst quality, deformed, distorted, watermark",
            "image_noise_scale": 0.15,
        },
        supports_start_frame=False,
        supports_end_frame=False,
        supports_audio=False,
        cost_tier="cheap",
        description="LTX-Video via Replicate - fast, cheap text-to-video (~$0.02/run)"
    ),
    
    # Wan 2.5 Image-to-Video - cheap with audio support
    # Valid durations: 5 or 10 seconds only
    "wan-2.5-i2v": ModelConfig(
        name="wan-2.5-i2v",
        model_id="wan-video/wan-2.5-i2v",
        model_type=ModelType.IMAGE_TO_VIDEO,
        default_params={
            "duration": 5,
            "resolution": "720p",
            "negative_prompt": "",
            "enable_prompt_expansion": True,
        },
        supports_start_frame=True,
        supports_end_frame=False,
        supports_audio=True,
        cost_tier="cheap",
        description="Wan 2.5 I2V - cheap image-to-video with audio, durations: 5 or 10s (~$0.05/run)"
    ),
    
    # Wan 2.5 Image-to-Video Fast - faster, cheaper version
    # Valid durations: 5 or 10 seconds only
    "wan-2.5-i2v-fast": ModelConfig(
        name="wan-2.5-i2v-fast",
        model_id="wan-video/wan-2.5-i2v-fast",
        model_type=ModelType.IMAGE_TO_VIDEO,
        default_params={
            "duration": 5,
            "resolution": "720p",
            "negative_prompt": "",
            "enable_prompt_expansion": True,
        },
        supports_start_frame=True,
        supports_end_frame=False,
        supports_audio=True,
        cost_tier="cheap",
        description="Wan 2.5 I2V Fast - faster image-to-video, durations: 5 or 10s"
    ),
    
    # Wan 2.5 Text-to-Video - text-to-video without image input
    # Valid durations: 5 or 10 seconds only
    "wan-2.5-t2v": ModelConfig(
        name="wan-2.5-t2v",
        model_id="wan-video/wan-2.5-t2v",
        model_type=ModelType.TEXT_TO_VIDEO,
        default_params={
            "size": "720*1280",  # 9:16 vertical (width*height)
            "duration": 5,
            "negative_prompt": "",
            "enable_prompt_expansion": True,
        },
        supports_start_frame=False,
        supports_end_frame=False,
        supports_audio=True,
        cost_tier="cheap",
        description="Wan 2.5 T2V - text-to-video with audio, durations: 5 or 10s (~$0.05/run)"
    ),
    
    # Wan 2.5 Text-to-Video Fast - faster, cheaper version
    # Valid durations: 5 or 10 seconds only
    "wan-2.5-t2v-fast": ModelConfig(
        name="wan-2.5-t2v-fast",
        model_id="wan-video/wan-2.5-t2v-fast",
        model_type=ModelType.TEXT_TO_VIDEO,
        default_params={
            "size": "720*1280",  # 9:16 vertical (width*height)
            "duration": 5,
            "negative_prompt": "",
            "enable_prompt_expansion": True,
        },
        supports_start_frame=False,
        supports_end_frame=False,
        supports_audio=True,
        cost_tier="cheap",
        description="Wan 2.5 T2V Fast - faster text-to-video, durations: 5 or 10s"
    ),
    
    # Seedance 1 Pro Fast - ByteDance high quality I2V
    # $0.025/sec, 720p, good for speaker videos
    "seedance-1-pro-fast": ModelConfig(
        name="seedance-1-pro-fast",
        model_id="bytedance/seedance-1-pro-fast",
        model_type=ModelType.IMAGE_TO_VIDEO,
        default_params={
            "fps": 24,
            "duration": 6,
            "resolution": "720p",
            "aspect_ratio": "9:16",
            "camera_fixed": True,
        },
        supports_start_frame=True,
        supports_end_frame=False,
        supports_audio=False,
        cost_tier="standard",
        description="Seedance 1 Pro Fast - high quality I2V, $0.025/sec, 720p"
    ),
    
    # Nano Banana Pro - scene transformation with reference images
    # Great for putting people into new scenes/environments
    "nano-banana-pro": ModelConfig(
        name="nano-banana-pro",
        model_id="google/nano-banana-pro",
        model_type=ModelType.TEXT_TO_IMAGE,
        default_params={
            "resolution": "1K",
            "aspect_ratio": "9:16",
            "output_format": "png",
            "safety_filter_level": "block_only_high",
        },
        supports_start_frame=False,
        supports_end_frame=False,
        supports_audio=False,
        cost_tier="standard",
        description="Nano Banana Pro - scene transformation, puts reference person into new environments"
    ),
}

# Default models for each task
DEFAULT_VIDEO_MODEL = "wan-2.5-i2v"
DEFAULT_IMAGE_MODEL = "seedream-4.5"
DEFAULT_MATTING_MODEL = "robust-video-matting"
DEFAULT_TTS_MODEL = "speech-02-hd"


# =============================================================================
# Client Setup
# =============================================================================

client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))


def get_model(model_name: str) -> ModelConfig:
    """Get model config by name."""
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    return MODEL_REGISTRY[model_name]


def list_models(model_type: Optional[ModelType] = None) -> list[ModelConfig]:
    """List available models, optionally filtered by type."""
    models = list(MODEL_REGISTRY.values())
    if model_type:
        models = [m for m in models if m.model_type == model_type]
    return models


# =============================================================================
# Retry Logic
# =============================================================================

def retry_replicate_call(func, max_retries=3, delay=1):
    """Retry a Replicate API call with exponential backoff and rate limit handling."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_str = str(e)
            # Handle rate limiting
            if "status: 429" in error_str and "rate limit resets in" in error_str:
                reset_match = re.search(r'resets in ~(\d+)s', error_str)
                if reset_match:
                    wait_time = int(reset_match.group(1)) + 2
                    print(f"⏳ Rate limited, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
            if attempt == max_retries - 1:
                raise
            wait = delay * (2 ** attempt)
            print(f"⚠️ Attempt {attempt + 1} failed, retrying in {wait}s...")
            time.sleep(wait)


def _prepare_file_input(path: Union[str, Path]) -> str:
    """Prepare a file for upload to Replicate as a data URI."""
    import base64
    import mimetypes
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Determine MIME type
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        # Default based on extension
        ext = path.suffix.lower()
        mime_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp',
            '.mp4': 'video/mp4',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
        }
        mime_type = mime_map.get(ext, 'application/octet-stream')
    
    # Read and encode as base64 data URI
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    
    return f"data:{mime_type};base64,{data}"


# =============================================================================
# Video Generation
# =============================================================================

def generate_video(
    prompt: str,
    duration: int = 8,
    generate_audio: bool = True,
    aspect_ratio: str = "9:16",
    resolution: str = "720p",
    model: str = "wan-2.5-t2v",  # Default to T2V model for text-to-video
) -> object:
    """
    Generate a video from a text prompt (no image input).
    
    Args:
        prompt: Text description of the video
        duration: Video duration in seconds
        generate_audio: Whether to generate audio
        aspect_ratio: Video aspect ratio (e.g., "9:16", "16:9", "1:1")
        resolution: Video resolution (e.g., "720p", "1080p")
        model: Model name from registry
    
    Returns:
        Replicate output object (use .read() for bytes, .url() for URL)
    """
    model_config = get_model(model)
    
    # Handle LTX-Video model
    if model == "ltx-video-replicate":
        return generate_video_ltx(
            prompt=prompt,
            duration=duration,
            aspect_ratio=aspect_ratio,
            model=model,
        )
    
    # Handle Wan T2V models (regular and fast)
    if model in ("wan-2.5-t2v", "wan-2.5-t2v-fast"):
        return generate_video_wan_t2v(
            prompt=prompt,
            duration=duration,
            aspect_ratio=aspect_ratio,
            model=model,
        )
    
    def call_replicate():
        input_data = {
            **model_config.default_params,
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
        }
        if model_config.supports_audio:
            input_data["generate_audio"] = generate_audio
            
        print(f"🎬 Generating video with {model_config.name}")
        print(f"   Prompt: {prompt[:100]}...")
        return client.run(model_config.model_id, input=input_data)
    
    return retry_replicate_call(call_replicate)


def generate_video_wan_t2v(
    prompt: str,
    duration: int = 5,
    aspect_ratio: str = "9:16",
    model: str = "wan-2.5-t2v",
    generate_audio: bool = True,  # Wan T2V always generates audio, param for API compat
) -> object:
    """
    Generate a video using Wan 2.5 T2V model (text-to-video with audio).
    
    Args:
        prompt: Text description of the video
        duration: Video duration in seconds (5 or 10)
        aspect_ratio: Video aspect ratio (e.g., "9:16", "16:9", "1:1")
        model: Model name ("wan-2.5-t2v" or "wan-2.5-t2v-fast")
        generate_audio: Ignored - Wan T2V always generates audio
    
    Returns:
        Replicate output object
    """
    model_config = get_model(model)
    
    # Clamp duration to valid values (5 or 10)
    duration = 5 if duration <= 7 else 10
    
    # Convert aspect ratio to size format (width*height)
    size_map = {
        "9:16": "720*1280",
        "16:9": "1280*720",
        "1:1": "720*720",
    }
    size = size_map.get(aspect_ratio, "720*1280")
    
    def call_replicate():
        input_data = {
            **model_config.default_params,
            "prompt": prompt,
            "duration": duration,
            "size": size,
        }
            
        print(f"🎬 Generating video with {model_config.name} (T2V)")
        print(f"   Prompt: {prompt[:100]}...")
        print(f"   Duration: {duration}s, Size: {size}")
        
        return client.run(model_config.model_id, input=input_data)
    
    return retry_replicate_call(call_replicate)


def generate_video_ltx(
    prompt: str,
    duration: int = 4,
    aspect_ratio: str = "9:16",
    model: str = "ltx-video-replicate",
    cfg: float = 3.0,
    steps: int = 30,
    target_size: int = 640,
) -> object:
    """
    Generate a video using LTX-Video model (cheap, fast text-to-video).
    
    Args:
        prompt: Text description of the video
        duration: Video duration in seconds (max ~4s per run)
        aspect_ratio: Video aspect ratio (e.g., "9:16", "16:9", "1:1")
        model: Model name from registry
        cfg: Classifier-free guidance scale
        steps: Number of inference steps
        target_size: Target size for the video
    
    Returns:
        Replicate output object (use .read() for bytes, .url() for URL)
    """
    model_config = get_model(model)
    
    # Convert duration to frame length (24fps)
    # LTX-Video uses 'length' parameter for frame count
    frame_length = min(duration * 24, 97)  # Max 97 frames (~4s)
    
    def call_replicate():
        input_data = {
            **model_config.default_params,
            "prompt": prompt,
            "length": frame_length,
            "aspect_ratio": aspect_ratio,
            "cfg": cfg,
            "steps": steps,
            "target_size": target_size,
        }
            
        print(f"🎬 Generating video with {model_config.name} (cheap mode)")
        print(f"   Prompt: {prompt[:100]}...")
        print(f"   Frames: {frame_length} (~{frame_length/24:.1f}s)")
        
        output = client.run(model_config.model_id, input=input_data)
        
        # LTX-Video returns a list, get first element
        if isinstance(output, list) and len(output) > 0:
            return output[0]
        return output
    
    return retry_replicate_call(call_replicate)


def generate_video_with_image(
    prompt: str,
    image: Union[str, Path],
    duration: int = 8,
    generate_audio: bool = True,
    last_frame: Optional[Union[str, Path]] = None,
    aspect_ratio: str = "9:16",
    resolution: str = "720p",
    model: str = DEFAULT_VIDEO_MODEL,
) -> object:
    """
    Generate a video from a prompt with a start image (and optional end frame).
    
    Args:
        prompt: Text description of the video
        image: Path to start frame image OR URL
        duration: Video duration in seconds
        generate_audio: Whether to generate audio
        last_frame: Optional path/URL to end frame image (for models that support it)
        aspect_ratio: Video aspect ratio
        resolution: Video resolution
        model: Model name from registry
    
    Returns:
        Replicate output object
    """
    model_config = get_model(model)
    
    # Clamp duration for Wan models (only supports 5s or 10s)
    if model_config.name.startswith("wan"):
        duration = 5 if duration <= 7 else 10
    
    # Clamp duration for Veo models (only supports 4, 6, or 8s)
    if model_config.name.startswith("veo"):
        if duration <= 5:
            duration = 4
        elif duration <= 7:
            duration = 6
        else:
            duration = 8
    
    if not model_config.supports_start_frame:
        raise ValueError(f"Model {model} does not support start frame input")
    
    if last_frame and not model_config.supports_end_frame:
        raise ValueError(f"Model {model} does not support end frame input")
    
    def call_replicate():
        # Handle image input - URL or file path (convert to data URI)
        if isinstance(image, str) and image.startswith(('http://', 'https://', 'data:')):
            image_input = image
        else:
            image_input = _prepare_file_input(image)
        
        # Handle last_frame input
        last_frame_input = None
        if last_frame:
            if isinstance(last_frame, str) and last_frame.startswith(('http://', 'https://', 'data:')):
                last_frame_input = last_frame
            else:
                last_frame_input = _prepare_file_input(last_frame)
        
        input_data = {
            **model_config.default_params,
            "prompt": prompt,
            "image": image_input,
            "duration": duration,
        }
        
        # Only add aspect_ratio and resolution for models that support them
        # Wan 2.5 I2V doesn't use aspect_ratio (derives from image)
        if model_config.name.startswith("veo"):
            input_data["aspect_ratio"] = aspect_ratio
            input_data["resolution"] = resolution
            if model_config.supports_audio:
                input_data["generate_audio"] = generate_audio
        
        if last_frame_input:
            input_data["last_frame"] = last_frame_input
        
        print(f"🎬 Generating video with {model_config.name} (start frame)")
        print(f"   Prompt: {prompt[:100]}...")
        print(f"   Image: {image[:50] if isinstance(image, str) else 'file'}")
        if last_frame:
            print(f"   With end frame constraint")
        
        return client.run(model_config.model_id, input=input_data)
    
    return retry_replicate_call(call_replicate)


# =============================================================================
# Image Generation
# =============================================================================

def generate_image(
    prompt: str,
    aspect_ratio: str = "9:16",
    resolution: str = "1 MP",
    output_format: str = "webp",
    model: str = DEFAULT_IMAGE_MODEL,
    reference_image: Optional[str] = None,
    reference_images: Optional[list[str]] = None,
) -> object:
    """
    Generate an image from a text prompt.
    
    Args:
        prompt: Text description of the image
        aspect_ratio: Image aspect ratio
        resolution: Image resolution (for models that support it)
        output_format: Output format (webp, png, jpg)
        model: Model name from registry
        reference_image: Optional single reference image URL (legacy)
        reference_images: Optional list of reference image paths/URLs (max 4)
    
    Returns:
        Replicate output object
    """
    model_config = get_model(model)
    
    def call_replicate():
        # Nano Banana Pro - scene transformation with reference person
        if model == "nano-banana-pro":
            input_data = {
                **model_config.default_params,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
            }
            # Add reference images (person to put into scene)
            refs = []
            if reference_images:
                for ref in reference_images[:4]:
                    if isinstance(ref, str) and ref.startswith(('http://', 'https://', 'data:')):
                        refs.append(ref)
                    else:
                        refs.append(_prepare_file_input(ref))
            elif reference_image:
                if isinstance(reference_image, str) and reference_image.startswith(('http://', 'https://', 'data:')):
                    refs.append(reference_image)
                else:
                    refs.append(_prepare_file_input(reference_image))
            if refs:
                input_data["image_input"] = refs
                print(f"   Using {len(refs)} reference image(s) for scene transformation")
        # Seedream 4.5 uses different parameters and supports reference images
        elif model == "seedream-4.5":
            input_data = {
                **model_config.default_params,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
            }
            # Add reference images if provided (max 4)
            refs = []
            if reference_images:
                for ref in reference_images[:4]:
                    if isinstance(ref, str) and ref.startswith(('http://', 'https://', 'data:')):
                        refs.append(ref)
                    else:
                        refs.append(_prepare_file_input(ref))
            elif reference_image:
                if isinstance(reference_image, str) and reference_image.startswith(('http://', 'https://', 'data:')):
                    refs.append(reference_image)
                else:
                    refs.append(_prepare_file_input(reference_image))
            if refs:
                input_data["image_input"] = refs
                print(f"   Using {len(refs)} reference image(s)")
        # Z-Image Turbo uses width/height instead of aspect_ratio
        elif model == "z-image-turbo":
            # Convert aspect ratio to width/height
            size_map = {
                "9:16": (768, 1344),
                "16:9": (1344, 768),
                "1:1": (1024, 1024),
            }
            width, height = size_map.get(aspect_ratio, (768, 1344))
            input_data = {
                **model_config.default_params,
                "prompt": prompt,
                "width": width,
                "height": height,
            }
        # Flux 2 Pro uses input_images for reference images
        elif model == "flux-2-pro":
            input_data = {
                **model_config.default_params,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
            }
            # Add reference images if provided
            refs = []
            if reference_images:
                for ref in reference_images[:4]:
                    if isinstance(ref, str) and ref.startswith(('http://', 'https://', 'data:')):
                        refs.append(ref)
                    else:
                        refs.append(_prepare_file_input(ref))
            elif reference_image:
                if isinstance(reference_image, str) and reference_image.startswith(('http://', 'https://', 'data:')):
                    refs.append(reference_image)
                else:
                    refs.append(_prepare_file_input(reference_image))
            if refs:
                input_data["input_images"] = refs
                print(f"   Using {len(refs)} reference image(s)")
        else:
            input_data = {
                **model_config.default_params,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
            }
            # Only add resolution if model supports it
            if "resolution" in model_config.default_params:
                input_data["resolution"] = resolution
            
        print(f"🖼️ Generating image with {model_config.name}")
        print(f"   Prompt: {prompt[:100]}...")
        if reference_image:
            print(f"   Reference: {reference_image}")
        return client.run(model_config.model_id, input=input_data)
    
    return retry_replicate_call(call_replicate)


# =============================================================================
# Video Background Removal
# =============================================================================

def remove_background(
    input_video: Union[str, Path],
    output_type: str = "alpha-mask",
    model: str = DEFAULT_MATTING_MODEL,
) -> object:
    """
    Remove background from a video.
    
    Args:
        input_video: Path to input video OR URL
        output_type: Output type ("alpha-mask", "green-screen", "foreground")
        model: Model name from registry
    
    Returns:
        Replicate output object
    """
    model_config = get_model(model)
    
    def call_replicate():
        # Handle video input - URL or file path
        if isinstance(input_video, str) and input_video.startswith(('http://', 'https://')):
            video_input = input_video
        else:
            video_input = _prepare_file_input(input_video)
        
        try:
            input_data = {
                **model_config.default_params,
                "input_video": video_input,
                "output_type": output_type,
            }
            
            print(f"🎭 Removing background with {model_config.name}")
            print(f"   Output type: {output_type}")
            return client.run(model_config.model_id, input=input_data)
        finally:
            if hasattr(video_input, 'close'):
                video_input.close()
    
    return retry_replicate_call(call_replicate)


# =============================================================================
# Video Captioning
# =============================================================================

DEFAULT_CAPTION_MODEL = "tiktok-captions"


def add_captions(
    input_video: Union[str, Path],
    language: str = "auto",
    highlight_color: str = "#FFFFFF",
    caption_size: int = 100,
    model: str = DEFAULT_CAPTION_MODEL,
) -> object:
    """
    Add TikTok-style animated captions to a video.
    
    Args:
        input_video: Path to input video OR URL
        language: Language code or "auto" for auto-detection
        highlight_color: Hex color for caption highlight (e.g., "#39E508" for green)
        caption_size: Size of captions (default 100)
        model: Model name from registry
    
    Returns:
        Replicate output object with captioned video
    """
    model_config = get_model(model)
    
    def call_replicate():
        # Handle video input - URL or file path
        # For large video files, upload to Replicate's file hosting first to avoid timeouts
        if isinstance(input_video, str) and input_video.startswith(('http://', 'https://')):
            video_url = input_video
        else:
            # Upload file to Replicate's file hosting for large videos
            video_path = Path(input_video)
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            print(f"   Uploading video ({file_size_mb:.1f} MB) to Replicate...")
            
            with open(video_path, "rb") as f:
                file_output = replicate.files.create(f)
            video_url = file_output.urls["get"]
            print(f"   Upload complete: {video_url[:80]}...")
        
        input_data = {
            **model_config.default_params,
            "video": video_url,
            "language": language,
            "highlight_color": highlight_color,
            "caption_size": caption_size,
        }
        
        print(f"📝 Adding captions with {model_config.name}")
        print(f"   Language: {language}, Color: {highlight_color}")
        return client.run(model_config.model_id, input=input_data)
    
    return retry_replicate_call(call_replicate)


# =============================================================================
# Utility Functions
# =============================================================================

def save_output(output, output_path: Union[str, Path]) -> Path:
    """
    Save Replicate output to a file.
    
    Args:
        output: Replicate output object (single or array)
        output_path: Path to save the file
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle array outputs (like seedream) by taking first item
    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            raise ValueError("Output array is empty")
        output = output[0]
    
    with open(output_path, "wb") as f:
        f.write(output.read())
    
    print(f"✅ Saved to: {output_path}")
    return output_path


def get_output_url(output) -> str:
    """Get the URL of a Replicate output."""
    return output.url()


# =============================================================================
# Text-to-Speech
# =============================================================================

# Speaking rate constants
# Average speaking rate: ~150 words per minute = 2.5 words per second
# Average word length: ~5 characters
# So roughly 12-15 characters per second of speech
CHARS_PER_SECOND = 14
WORDS_PER_SECOND = 2.5


def get_word_range_for_duration(duration_s: float) -> tuple[int, int]:
    """
    Get recommended word count range for a given duration.
    
    Args:
        duration_s: Duration in seconds
    
    Returns:
        Tuple of (min_words, max_words)
    """
    target_words = int(duration_s * WORDS_PER_SECOND)
    min_words = max(1, target_words - 2)
    max_words = target_words + 2
    return min_words, max_words


def estimate_speech_duration(text: str) -> float:
    """
    Estimate speech duration in seconds for given text.
    
    Args:
        text: Text to estimate duration for
    
    Returns:
        Estimated duration in seconds
    """
    # Remove pause markers like <#0.5#> and count their duration
    import re
    pause_pattern = r'<#(\d+\.?\d*)#>'
    pauses = re.findall(pause_pattern, text)
    total_pause_time = sum(float(p) for p in pauses)
    
    # Remove pause markers from text for character count
    clean_text = re.sub(pause_pattern, '', text)
    
    # Estimate speech duration
    speech_duration = len(clean_text) / CHARS_PER_SECOND
    
    return speech_duration + total_pause_time


def fit_text_to_duration(text: str, target_duration: float) -> str:
    """
    Truncate or validate text to fit within target duration.
    
    Args:
        text: Text to fit
        target_duration: Target duration in seconds
    
    Returns:
        Text that fits within duration (may be truncated)
    """
    estimated = estimate_speech_duration(text)
    
    if estimated <= target_duration:
        return text
    
    # Need to truncate - estimate max characters
    max_chars = int(target_duration * CHARS_PER_SECOND * 0.9)  # 10% buffer
    
    # Truncate at word boundary
    if len(text) > max_chars:
        truncated = text[:max_chars].rsplit(' ', 1)[0]
        print(f"⚠️ Text truncated from {len(text)} to {len(truncated)} chars to fit {target_duration}s")
        return truncated
    
    return text


def generate_speech(
    text: str,
    voice_id: Optional[str] = None,
    emotion: Optional[str] = None,
    speed: float = 1.0,
    model: str = DEFAULT_TTS_MODEL,
) -> object:
    """
    Generate speech from text.
    
    Args:
        text: Text to convert to speech. Use <#x#> for pauses (x = seconds, 0.01-99.99)
        voice_id: Voice ID (from voice cloning) or preset voice name
        emotion: Emotion for speech (happy, sad, angry, fearful, disgusted, surprised, neutral)
        speed: Speech speed multiplier (0.5-2.0)
        model: TTS model to use
    
    Returns:
        Replicate output object (audio file)
    """
    model_config = get_model(model)
    
    def call_replicate():
        input_data = {
            **model_config.default_params,
            "text": text,
            "speed": speed,
        }
        
        if voice_id:
            input_data["voice_id"] = voice_id
        
        if emotion:
            input_data["emotion"] = emotion
        
        estimated_duration = estimate_speech_duration(text)
        print(f"🎙️ Generating speech with {model_config.name}")
        print(f"   Text: {text[:80]}...")
        print(f"   Estimated duration: {estimated_duration:.1f}s")
        
        return client.run(model_config.model_id, input=input_data)
    
    return retry_replicate_call(call_replicate)


def clone_voice(
    voice_file: Union[str, Path],
    target_model: str = "speech-02-hd",
) -> str:
    """
    Clone a voice from an audio file.
    
    Args:
        voice_file: Path to audio file (MP3, M4A, WAV). 10s-5min, <20MB
        target_model: Target TTS model for the cloned voice
    
    Returns:
        Voice ID for use with generate_speech()
    """
    model_config = get_model("voice-cloning")
    
    def call_replicate():
        if isinstance(voice_file, str) and voice_file.startswith(('http://', 'https://')):
            file_input = voice_file
        else:
            file_input = _prepare_file_input(voice_file)
        
        try:
            input_data = {
                "voice_file": file_input,
                "model": target_model,
            }
            
            print(f"🎤 Cloning voice with {model_config.name}")
            result = client.run(model_config.model_id, input=input_data)
            
            voice_id = result.get("voice_id") if isinstance(result, dict) else result
            print(f"✅ Voice cloned: {voice_id}")
            return voice_id
        finally:
            if hasattr(file_input, 'close'):
                file_input.close()
    
    return retry_replicate_call(call_replicate)
