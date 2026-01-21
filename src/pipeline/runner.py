"""
Main pipeline runner.

Orchestrates the complete video generation workflow.
"""

from pathlib import Path
from typing import Optional

from ..config.schema import AdConfig, load_config
from ..config.models import ModelRegistry, ModelBackend
from ..generators.base import VideoGenerator, ImageGenerator
from ..generators.replicate import (
    ReplicateVideoGenerator,
    ReplicateImageGenerator,
)
from ..generators.ltx import LTX2VideoGenerator
from ..processors.matting import BackgroundRemover
from ..processors.compositor import VideoCompositor
from ..processors.captions import CaptionGenerator
from ..processors.stitcher import VideoStitcher
from ..utils.files import ensure_dir
from .scene import SceneProcessor, SceneOutput


class VideoPipeline:
    """
    Main video generation pipeline.
    
    Supports multiple backends:
    - replicate: Use Replicate API for all generation
    - local: Use local models (LTX-2, etc.)
    - hybrid: Use local for video, API for other tasks
    """
    
    def __init__(
        self,
        backend: str = "replicate",
        video_model: Optional[str] = None,
        speaker_model: Optional[str] = None,
        image_model: Optional[str] = None,
    ):
        """
        Initialize pipeline with specified backend.
        
        Args:
            backend: Backend type (replicate, local, hybrid)
            video_model: Override default video model
            speaker_model: Override default speaker model
            image_model: Override default image model
        """
        self._backend = backend
        
        # Initialize generators based on backend
        if backend == "replicate":
            self._video_gen = ReplicateVideoGenerator(
                video_model or ModelRegistry.DEFAULT_VIDEO_MODEL
            )
            self._speaker_gen = ReplicateVideoGenerator(
                speaker_model or ModelRegistry.DEFAULT_SPEAKER_MODEL
            )
            self._image_gen = ReplicateImageGenerator(
                image_model or ModelRegistry.DEFAULT_IMAGE_MODEL
            )
        elif backend == "local":
            self._video_gen = LTX2VideoGenerator(variant="ltx-2")
            self._speaker_gen = self._video_gen  # Same model for now
            self._image_gen = ReplicateImageGenerator(
                image_model or ModelRegistry.DEFAULT_IMAGE_MODEL
            )
        elif backend == "hybrid":
            # Local for video, API for images
            self._video_gen = LTX2VideoGenerator(variant="ltx-2")
            self._speaker_gen = self._video_gen
            self._image_gen = ReplicateImageGenerator(
                image_model or ModelRegistry.DEFAULT_IMAGE_MODEL
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        # Initialize processors
        self._bg_remover = BackgroundRemover()
        self._compositor = VideoCompositor()
        self._caption_gen = CaptionGenerator()
        self._stitcher = VideoStitcher()
        
        # Scene processor
        self._scene_processor = SceneProcessor(
            video_generator=self._speaker_gen,
            image_generator=self._image_gen,
            background_remover=self._bg_remover,
            compositor=self._compositor,
        )
    
    def run(
        self,
        config_path: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Run the complete video generation pipeline.
        
        Args:
            config_path: Path to YAML/JSON config file
            output_dir: Output directory (default: outputs/{scheme_name}/)
        
        Returns:
            Path to final video
        """
        # Generate output directory from scheme name if not specified
        if output_dir is None:
            scheme_name = Path(config_path).stem
            output_dir = f"outputs/{scheme_name}"
        
        print(f"\n{'#'*60}")
        print(f"# Short Video Pipeline v2.0")
        print(f"# Backend: {self._backend}")
        print(f"# Config: {config_path}")
        print(f"# Output: {output_dir}")
        print(f"{'#'*60}")
        
        # Load configuration
        config = load_config(config_path)
        print(f"\n📋 Loaded: {config.title}")
        print(f"   Scenes: {len(config.scenes)}")
        print(f"   Total duration: {config.total_duration}s")
        print(f"   Speakers: {len(config.speakers)}")
        print(f"   Products: {len(config.products)}")
        
        # Validate
        errors = config.validate()
        if errors:
            print("\n⚠️ Validation warnings:")
            for error in errors:
                print(f"   - {error}")
        
        # Setup output directory
        output_path = ensure_dir(output_dir)
        
        # Process each scene
        scene_outputs: list[SceneOutput] = []
        previous_frame: Optional[str] = None
        
        for i, scene in enumerate(config.scenes):
            print(f"\n\n{'*'*60}")
            print(f"* Scene {i+1}/{len(config.scenes)}: {scene.id}")
            print(f"{'*'*60}")
            
            output = self._scene_processor.process(
                scene=scene,
                output_dir=output_path,
                previous_frame=previous_frame,
                config=config,
            )
            scene_outputs.append(output)
            previous_frame = output.final_frame_path
        
        # Stitch everything together
        video_paths = [s.video_path for s in scene_outputs]
        final_video_path = output_path / f"{config.title.replace(' ', '_').lower()}_final.mp4"
        self._stitcher.stitch(video_paths, str(final_video_path))
        
        # Add captions if enabled
        if config.caption_final_video or config.add_captions:
            print(f"\n{'='*60}")
            print(f"📝 Adding captions to final video...")
            print(f"{'='*60}")
            
            captioned_output = self._caption_gen.add_captions(
                video_path=str(final_video_path),
                language=config.caption_language,
                highlight_color=config.caption_color,
            )
            
            captioned_path = output_path / f"{config.title.replace(' ', '_').lower()}_captioned.mp4"
            Path(captioned_output.path).rename(captioned_path)
            final_video_path = captioned_path
            print(f"✅ Captioned video saved: {final_video_path}")
        
        print(f"\n\n{'#'*60}")
        print(f"# Pipeline Complete!")
        print(f"# Output: {final_video_path}")
        print(f"{'#'*60}\n")
        
        return str(final_video_path)
    
    @classmethod
    def from_config(cls, config: AdConfig) -> "VideoPipeline":
        """Create pipeline from AdConfig."""
        return cls(
            backend=config.backend,
            video_model=config.video_model,
            speaker_model=config.speaker_model,
            image_model=config.image_model,
        )


def run_pipeline(config_path: str, output_dir: Optional[str] = None) -> str:
    """
    Convenience function to run pipeline.
    
    Args:
        config_path: Path to config file
        output_dir: Optional output directory
    
    Returns:
        Path to final video
    """
    pipeline = VideoPipeline()
    return pipeline.run(config_path, output_dir)


if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "example_ad.yaml"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        final_video = run_pipeline(config_path, output_dir)
        print(f"\n🎉 Success! Final video: {final_video}")
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
