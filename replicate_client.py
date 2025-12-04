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
            "output_format": "webp",
        },
        cost_tier="standard",
        description="FLUX 2 Pro - high quality image generation"
    ),
    "flux-1.1-pro": ModelConfig(
        name="flux-1.1-pro",
        model_id="black-forest-labs/flux-1.1-pro",
        model_type=ModelType.TEXT_TO_IMAGE,
        default_params={
            "aspect_ratio": "9:16",
            "output_format": "webp",
        },
        cost_tier="cheap",
        description="FLUX 1.1 Pro - fast, cheaper image generation"
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
}

# Default models for each task
DEFAULT_VIDEO_MODEL = "veo-3.1-fast"
DEFAULT_IMAGE_MODEL = "flux-2-pro"
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


def _prepare_file_input(path: Union[str, Path]) -> object:
    """Prepare a file for upload to Replicate."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return open(path, 'rb')


# =============================================================================
# Video Generation
# =============================================================================

def generate_video(
    prompt: str,
    duration: int = 8,
    generate_audio: bool = True,
    aspect_ratio: str = "9:16",
    resolution: str = "720p",
    model: str = DEFAULT_VIDEO_MODEL,
) -> object:
    """
    Generate a video from a text prompt.
    
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
    
    if not model_config.supports_start_frame:
        raise ValueError(f"Model {model} does not support start frame input")
    
    if last_frame and not model_config.supports_end_frame:
        raise ValueError(f"Model {model} does not support end frame input")
    
    def call_replicate():
        # Handle image input - URL or file path
        if isinstance(image, str) and image.startswith(('http://', 'https://')):
            image_input = image
        else:
            image_input = _prepare_file_input(image)
        
        # Handle last_frame input
        last_frame_input = None
        if last_frame:
            if isinstance(last_frame, str) and last_frame.startswith(('http://', 'https://')):
                last_frame_input = last_frame
            else:
                last_frame_input = _prepare_file_input(last_frame)
        
        try:
            input_data = {
                **model_config.default_params,
                "prompt": prompt,
                "image": image_input,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
            }
            
            if model_config.supports_audio:
                input_data["generate_audio"] = generate_audio
            
            if last_frame_input:
                input_data["last_frame"] = last_frame_input
            
            print(f"🎬 Generating video with {model_config.name} (start frame)")
            print(f"   Prompt: {prompt[:100]}...")
            if last_frame:
                print(f"   With end frame constraint")
            
            return client.run(model_config.model_id, input=input_data)
        finally:
            # Close file handles if we opened them
            if hasattr(image_input, 'close'):
                image_input.close()
            if last_frame_input and hasattr(last_frame_input, 'close'):
                last_frame_input.close()
    
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
) -> object:
    """
    Generate an image from a text prompt.
    
    Args:
        prompt: Text description of the image
        aspect_ratio: Image aspect ratio
        resolution: Image resolution (for models that support it)
        output_format: Output format (webp, png, jpg)
        model: Model name from registry
    
    Returns:
        Replicate output object
    """
    model_config = get_model(model)
    
    def call_replicate():
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
# Utility Functions
# =============================================================================

def save_output(output, output_path: Union[str, Path]) -> Path:
    """
    Save Replicate output to a file.
    
    Args:
        output: Replicate output object
        output_path: Path to save the file
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
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
