"""
Model registry and configuration for video generation backends.

Supports:
- Replicate API models (Veo, Wan, etc.)
- Local models (LTX-2, etc.)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ModelType(Enum):
    """Types of models supported."""
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    AUDIO_TO_VIDEO = "audio_to_video"
    TEXT_TO_IMAGE = "text_to_image"
    TEXT_TO_SPEECH = "text_to_speech"
    TEXT_TO_MUSIC = "text_to_music"
    VOICE_CLONING = "voice_cloning"
    VIDEO_MATTING = "video_matting"
    VIDEO_CAPTIONING = "video_captioning"


class ModelBackend(Enum):
    """Backend for model execution."""
    REPLICATE = "replicate"
    LOCAL = "local"
    FAL = "fal"
    RUNPOD = "runpod"


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    model_id: str
    model_type: ModelType
    backend: ModelBackend = ModelBackend.REPLICATE
    default_params: dict = field(default_factory=dict)
    supports_start_frame: bool = False
    supports_end_frame: bool = False
    supports_audio: bool = False
    supports_multi_image: bool = False
    max_reference_images: int = 1
    cost_tier: str = "standard"
    description: str = ""


class ModelRegistry:
    """Registry of available models."""
    
    _models: dict[str, ModelConfig] = {}
    
    # Default models for each task
    DEFAULT_VIDEO_MODEL = "wan-2.5-i2v"
    DEFAULT_SPEAKER_MODEL = "veo-3.1-fast"
    DEFAULT_IMAGE_MODEL = "seedream-4.5"
    DEFAULT_MATTING_MODEL = "robust-video-matting"
    DEFAULT_TTS_MODEL = "speech-02-hd"
    DEFAULT_CAPTION_MODEL = "tiktok-captions"
    
    @classmethod
    def register(cls, config: ModelConfig) -> None:
        """Register a model configuration."""
        cls._models[config.name] = config
    
    @classmethod
    def get(cls, name: str) -> ModelConfig:
        """Get model config by name."""
        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise ValueError(f"Unknown model: {name}. Available: {available}")
        return cls._models[name]
    
    @classmethod
    def list_models(cls, model_type: Optional[ModelType] = None) -> list[ModelConfig]:
        """List available models, optionally filtered by type."""
        models = list(cls._models.values())
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        return models
    
    @classmethod
    def get_default(cls, model_type: ModelType) -> str:
        """Get default model name for a type."""
        defaults = {
            ModelType.IMAGE_TO_VIDEO: cls.DEFAULT_VIDEO_MODEL,
            ModelType.TEXT_TO_VIDEO: cls.DEFAULT_VIDEO_MODEL,
            ModelType.TEXT_TO_IMAGE: cls.DEFAULT_IMAGE_MODEL,
            ModelType.VIDEO_MATTING: cls.DEFAULT_MATTING_MODEL,
            ModelType.TEXT_TO_SPEECH: cls.DEFAULT_TTS_MODEL,
            ModelType.VIDEO_CAPTIONING: cls.DEFAULT_CAPTION_MODEL,
        }
        return defaults.get(model_type, cls.DEFAULT_VIDEO_MODEL)


def _register_default_models():
    """Register all default models."""
    
    # === Video Models (Replicate) ===
    
    ModelRegistry.register(ModelConfig(
        name="veo-3.1-fast",
        model_id="google/veo-3.1-fast",
        model_type=ModelType.IMAGE_TO_VIDEO,
        backend=ModelBackend.REPLICATE,
        default_params={
            "resolution": "720p",
            "aspect_ratio": "9:16",
            "duration": 8,
        },
        supports_start_frame=True,
        supports_end_frame=True,
        supports_audio=True,
        cost_tier="standard",
        description="Google Veo 3.1 Fast - high quality video with audio"
    ))
    
    ModelRegistry.register(ModelConfig(
        name="veo-3.1",
        model_id="google/veo-3.1",
        model_type=ModelType.IMAGE_TO_VIDEO,
        backend=ModelBackend.REPLICATE,
        default_params={
            "resolution": "720p",
            "aspect_ratio": "9:16",
            "duration": 8,
        },
        supports_start_frame=True,
        supports_end_frame=True,
        supports_audio=True,
        cost_tier="premium",
        description="Google Veo 3.1 - highest quality, slower"
    ))
    
    ModelRegistry.register(ModelConfig(
        name="wan-2.5-i2v",
        model_id="wan-video/wan-2.5-i2v",
        model_type=ModelType.IMAGE_TO_VIDEO,
        backend=ModelBackend.REPLICATE,
        default_params={
            "duration": 5,
            "resolution": "720p",
            "negative_prompt": "",
            "enable_prompt_expansion": True,
        },
        supports_start_frame=True,
        supports_audio=True,
        cost_tier="cheap",
        description="Wan 2.5 I2V - cheap image-to-video, 5 or 10s"
    ))
    
    ModelRegistry.register(ModelConfig(
        name="wan-2.5-t2v",
        model_id="wan-video/wan-2.5-t2v",
        model_type=ModelType.TEXT_TO_VIDEO,
        backend=ModelBackend.REPLICATE,
        default_params={
            "size": "720*1280",
            "duration": 5,
            "negative_prompt": "",
            "enable_prompt_expansion": True,
        },
        supports_audio=True,
        cost_tier="cheap",
        description="Wan 2.5 T2V - text-to-video with audio, 5 or 10s"
    ))
    
    # === Image Models (Replicate) ===
    
    ModelRegistry.register(ModelConfig(
        name="seedream-4.5",
        model_id="bytedance/seedream-4.5",
        model_type=ModelType.TEXT_TO_IMAGE,
        backend=ModelBackend.REPLICATE,
        default_params={
            "size": "4K",
            "max_images": 1,
            "aspect_ratio": "9:16",
            "sequential_image_generation": "disabled",
        },
        supports_multi_image=True,
        max_reference_images=4,
        cost_tier="cheap",
        description="Seedream 4.5 - cheap with multi-image reference support"
    ))
    
    ModelRegistry.register(ModelConfig(
        name="flux-2-pro",
        model_id="black-forest-labs/flux-2-pro",
        model_type=ModelType.TEXT_TO_IMAGE,
        backend=ModelBackend.REPLICATE,
        default_params={
            "resolution": "1 MP",
            "aspect_ratio": "9:16",
            "output_format": "png",
        },
        supports_multi_image=True,
        max_reference_images=4,
        cost_tier="standard",
        description="FLUX 2 Pro - high quality with multi-image support"
    ))
    
    ModelRegistry.register(ModelConfig(
        name="nano-banana-pro",
        model_id="google/nano-banana-pro",
        model_type=ModelType.TEXT_TO_IMAGE,
        backend=ModelBackend.REPLICATE,
        default_params={
            "resolution": "1K",
            "aspect_ratio": "9:16",
            "output_format": "png",
            "safety_filter_level": "block_only_high",
        },
        supports_multi_image=True,
        max_reference_images=4,
        cost_tier="standard",
        description="Nano Banana Pro - scene transformation with reference images"
    ))
    
    # === Video Matting ===
    
    ModelRegistry.register(ModelConfig(
        name="robust-video-matting",
        model_id="arielreplicate/robust_video_matting:73d2128a371922d5d1abf0712a1d974be0e4e2358cc1218e4e34714767232bac",
        model_type=ModelType.VIDEO_MATTING,
        backend=ModelBackend.REPLICATE,
        default_params={
            "output_type": "alpha-mask",
        },
        cost_tier="cheap",
        description="Robust Video Matting - background removal"
    ))
    
    # === Captioning ===
    
    ModelRegistry.register(ModelConfig(
        name="tiktok-captions",
        model_id="shreejalmaharjan-27/tiktok-short-captions:46bf1c12c77ad1782d6f87828d4d8ba4d48646b8e1271b490cb9e95ccdbc4504",
        model_type=ModelType.VIDEO_CAPTIONING,
        backend=ModelBackend.REPLICATE,
        default_params={
            "model": "large-v3",
            "language": "auto",
            "temperature": 0,
            "caption_size": 100,
            "highlight_color": "#FFFFFF",
        },
        cost_tier="cheap",
        description="TikTok-style animated captions"
    ))
    
    # === TTS ===
    
    ModelRegistry.register(ModelConfig(
        name="speech-02-hd",
        model_id="minimax/speech-02-hd",
        model_type=ModelType.TEXT_TO_SPEECH,
        backend=ModelBackend.REPLICATE,
        default_params={},
        cost_tier="standard",
        description="MiniMax Speech-02-HD - high quality TTS"
    ))
    
    # === Music Generation ===
    
    # MusicGen by Meta - text to music
    ModelRegistry.register(ModelConfig(
        name="musicgen",
        model_id="meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
        model_type=ModelType.TEXT_TO_MUSIC,
        backend=ModelBackend.REPLICATE,
        default_params={
            "model_version": "stereo-melody-large",
            "duration": 8,
            "normalization_strategy": "peak",
            "top_k": 250,
            "top_p": 0,
            "temperature": 1,
            "classifier_free_guidance": 3,
            "output_format": "mp3",
        },
        cost_tier="cheap",
        description="MusicGen by Meta - text to music generation"
    ))
    
    # MusicGen Stereo with chord control
    ModelRegistry.register(ModelConfig(
        name="musicgen-stereo-chord",
        model_id="sakemin/musicgen-stereo-chord",
        model_type=ModelType.TEXT_TO_MUSIC,
        backend=ModelBackend.REPLICATE,
        default_params={
            "duration": 8,
            "output_format": "mp3",
        },
        cost_tier="cheap",
        description="MusicGen Stereo - music with chord/tempo control"
    ))
    
    # Lyria 2 by Google - high quality music generation
    ModelRegistry.register(ModelConfig(
        name="lyria-2",
        model_id="google/lyria-2",
        model_type=ModelType.TEXT_TO_MUSIC,
        backend=ModelBackend.REPLICATE,
        default_params={
            "prompt": "",
        },
        cost_tier="standard",
        description="Lyria 2 by Google - 48kHz stereo, 30s, professional quality"
    ))
    
    # === Local Models ===
    
    ModelRegistry.register(ModelConfig(
        name="ltx-2",
        model_id="Lightricks/LTX-2",
        model_type=ModelType.IMAGE_TO_VIDEO,
        backend=ModelBackend.LOCAL,
        default_params={
            "num_frames": 121,
            "width": 768,
            "height": 512,
            "num_inference_steps": 50,
            "guidance_scale": 5.0,
        },
        supports_start_frame=True,
        supports_audio=True,  # Audio-to-video variant
        cost_tier="self-hosted",
        description="LTX-2 - local video generation with audio support"
    ))
    
    ModelRegistry.register(ModelConfig(
        name="ltx-2-distilled",
        model_id="Lightricks/LTX-Video-0.9.7-distilled",
        model_type=ModelType.IMAGE_TO_VIDEO,
        backend=ModelBackend.LOCAL,
        default_params={
            "num_frames": 121,
            "width": 768,
            "height": 512,
            "guidance_scale": 1.0,
            "timesteps": [1000, 993, 987, 981, 975, 909, 725, 0.03],
        },
        supports_start_frame=True,
        cost_tier="self-hosted",
        description="LTX-2 Distilled - faster local generation"
    ))
    
    # Chroma-4B - multimodal TTS with voice cloning (local)
    # Excellent for German and other languages
    ModelRegistry.register(ModelConfig(
        name="chroma-4b",
        model_id="FlashLabs/Chroma-4B",
        model_type=ModelType.TEXT_TO_SPEECH,
        backend=ModelBackend.LOCAL,
        default_params={
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        supports_audio=True,  # Supports voice cloning via reference audio
        cost_tier="self-hosted",
        description="Chroma-4B - multimodal TTS with voice cloning, excellent for German"
    ))


# Register models on import
_register_default_models()
