"""
Abstract base classes for generators.

Provides unified interfaces for:
- Video generation (text-to-video, image-to-video, audio-to-video)
- Image generation (with multi-image reference support)
- Audio generation (TTS, voice cloning)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union


@dataclass
class GeneratorOutput:
    """Output from a generator."""
    path: str
    duration_s: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    
    def exists(self) -> bool:
        """Check if output file exists."""
        return Path(self.path).exists()


class VideoGenerator(ABC):
    """Abstract interface for video generation."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model/generator name for logging."""
        pass
    
    @property
    @abstractmethod
    def supports_start_frame(self) -> bool:
        """Whether model supports start frame conditioning."""
        pass
    
    @property
    @abstractmethod
    def supports_end_frame(self) -> bool:
        """Whether model supports end frame conditioning."""
        pass
    
    @property
    @abstractmethod
    def supports_audio(self) -> bool:
        """Whether model can generate audio."""
        pass
    
    @property
    def supports_audio_input(self) -> bool:
        """Whether model supports audio-to-video (audio conditioning)."""
        return False
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        duration: int = 8,
        width: int = 768,
        height: int = 1280,
        start_image: Optional[str] = None,
        end_image: Optional[str] = None,
        audio_input: Optional[str] = None,
        generate_audio: bool = False,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> GeneratorOutput:
        """
        Generate video from prompt.
        
        Args:
            prompt: Text description of the video
            duration: Duration in seconds
            width: Video width
            height: Video height
            start_image: Optional path to start frame image
            end_image: Optional path to end frame image
            audio_input: Optional path to audio for audio-to-video
            generate_audio: Whether to generate audio (if supported)
            output_path: Optional output path (auto-generated if None)
            **kwargs: Additional model-specific parameters
        
        Returns:
            GeneratorOutput with path and metadata
        """
        pass


class ImageGenerator(ABC):
    """Abstract interface for image generation with multi-image support."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model/generator name for logging."""
        pass
    
    @property
    @abstractmethod
    def supports_multi_image(self) -> bool:
        """Whether model supports multiple reference images."""
        pass
    
    @property
    @abstractmethod
    def max_reference_images(self) -> int:
        """Maximum number of reference images supported."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        width: int = 768,
        height: int = 1280,
        reference_images: Optional[list[str]] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> GeneratorOutput:
        """
        Generate image from prompt with optional reference images.
        
        Args:
            prompt: Text description of the image
            width: Image width
            height: Image height
            reference_images: List of reference image paths (speaker, product, etc.)
            output_path: Optional output path
            **kwargs: Additional model-specific parameters
        
        Returns:
            GeneratorOutput with path and metadata
        """
        pass
    
    def generate_with_speaker(
        self,
        prompt: str,
        speaker_images: list[str],
        product_images: Optional[list[str]] = None,
        width: int = 768,
        height: int = 1280,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> GeneratorOutput:
        """
        Generate image with speaker and optional product references.
        
        Combines speaker and product images up to max_reference_images.
        Speaker images take priority.
        
        Args:
            prompt: Text description
            speaker_images: List of speaker reference images
            product_images: Optional list of product reference images
            width: Image width
            height: Image height
            output_path: Optional output path
            **kwargs: Additional parameters
        
        Returns:
            GeneratorOutput with path and metadata
        """
        # Combine references, prioritizing speaker images
        max_refs = self.max_reference_images
        references = []
        
        # Add speaker images first
        for img in speaker_images[:max_refs]:
            references.append(img)
        
        # Add product images if space remains
        remaining = max_refs - len(references)
        if product_images and remaining > 0:
            for img in product_images[:remaining]:
                references.append(img)
        
        return self.generate(
            prompt=prompt,
            width=width,
            height=height,
            reference_images=references if references else None,
            output_path=output_path,
            **kwargs,
        )


class AudioGenerator(ABC):
    """Abstract interface for audio generation (TTS, voice cloning)."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model/generator name for logging."""
        pass
    
    @property
    @abstractmethod
    def supports_voice_cloning(self) -> bool:
        """Whether model supports voice cloning."""
        pass
    
    @abstractmethod
    def generate_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "en",
        emotion: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> GeneratorOutput:
        """
        Generate speech from text.
        
        Args:
            text: Text to speak
            voice_id: Voice ID or cloned voice ID
            language: Language code
            emotion: Optional emotion/style
            output_path: Optional output path
            **kwargs: Additional parameters
        
        Returns:
            GeneratorOutput with path and metadata
        """
        pass
    
    def clone_voice(
        self,
        audio_sample: str,
        voice_name: str,
        **kwargs,
    ) -> str:
        """
        Clone a voice from an audio sample.
        
        Args:
            audio_sample: Path to audio sample (10s-5min)
            voice_name: Name for the cloned voice
            **kwargs: Additional parameters
        
        Returns:
            Voice ID for the cloned voice
        """
        raise NotImplementedError("Voice cloning not supported by this generator")


class MattingGenerator(ABC):
    """Abstract interface for video matting/background removal."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model/generator name for logging."""
        pass
    
    @abstractmethod
    def remove_background(
        self,
        video_path: str,
        output_type: str = "alpha-mask",
        output_path: Optional[str] = None,
        **kwargs,
    ) -> GeneratorOutput:
        """
        Remove background from video.
        
        Args:
            video_path: Path to input video
            output_type: Output type (alpha-mask, green-screen, foreground)
            output_path: Optional output path
            **kwargs: Additional parameters
        
        Returns:
            GeneratorOutput with path to matted video
        """
        pass


class CaptionGenerator(ABC):
    """Abstract interface for video captioning."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model/generator name for logging."""
        pass
    
    @abstractmethod
    def add_captions(
        self,
        video_path: str,
        language: str = "auto",
        highlight_color: str = "#FFFFFF",
        output_path: Optional[str] = None,
        **kwargs,
    ) -> GeneratorOutput:
        """
        Add captions to video.
        
        Args:
            video_path: Path to input video
            language: Language code or "auto"
            highlight_color: Hex color for caption highlight
            output_path: Optional output path
            **kwargs: Additional parameters
        
        Returns:
            GeneratorOutput with path to captioned video
        """
        pass
