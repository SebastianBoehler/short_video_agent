"""Video background removal/matting processor."""

from pathlib import Path
from typing import Optional

from ..generators.base import GeneratorOutput
from ..generators.replicate import ReplicateMattingGenerator


class BackgroundRemover:
    """
    Video background removal processor.
    
    Supports multiple backends:
    - Replicate API (robust-video-matting)
    - Local models (future)
    """
    
    def __init__(
        self,
        backend: str = "replicate",
        model_name: str = "robust-video-matting",
    ):
        self._backend = backend
        
        if backend == "replicate":
            self._generator = ReplicateMattingGenerator(model_name)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def remove_background(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        output_type: str = "alpha-mask",
    ) -> GeneratorOutput:
        """
        Remove background from video.
        
        Args:
            video_path: Path to input video
            output_path: Optional output path
            output_type: Output type (alpha-mask, green-screen, foreground)
        
        Returns:
            GeneratorOutput with path to matted video
        """
        return self._generator.remove_background(
            video_path=video_path,
            output_type=output_type,
            output_path=output_path,
        )
