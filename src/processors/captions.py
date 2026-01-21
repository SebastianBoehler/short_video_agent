"""Video caption processor."""

from pathlib import Path
from typing import Optional

from ..generators.base import GeneratorOutput
from ..generators.replicate import ReplicateCaptionGenerator


class CaptionGenerator:
    """
    Video caption processor.
    
    Adds TikTok-style animated captions to videos.
    """
    
    def __init__(
        self,
        backend: str = "replicate",
        model_name: str = "tiktok-captions",
    ):
        self._backend = backend
        
        if backend == "replicate":
            self._generator = ReplicateCaptionGenerator(model_name)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def add_captions(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        language: str = "auto",
        highlight_color: str = "#FFFFFF",
    ) -> GeneratorOutput:
        """
        Add captions to video.
        
        Args:
            video_path: Path to input video
            output_path: Optional output path
            language: Language code or "auto"
            highlight_color: Hex color for caption highlight
        
        Returns:
            GeneratorOutput with path to captioned video
        """
        return self._generator.add_captions(
            video_path=video_path,
            language=language,
            highlight_color=highlight_color,
            output_path=output_path,
        )
