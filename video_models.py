"""
Video Model Interface - Unified API for video generation models.

Supports:
- Replicate-based models (Veo, etc.)
- Self-hosted models (LTX-Video via diffusers)

This allows easy switching between models without changing pipeline code.
"""

import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch


@dataclass
class VideoOutput:
    """Output from video generation."""
    path: str
    duration_s: float
    width: int
    height: int
    fps: int = 24


class VideoModelInterface(ABC):
    """Abstract interface for video generation models."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for logging."""
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
    
    @abstractmethod
    def generate_video(
        self,
        prompt: str,
        duration: int = 8,
        width: int = 768,
        height: int = 512,
        start_image: Optional[str] = None,
        end_image: Optional[str] = None,
        generate_audio: bool = False,
        output_path: Optional[str] = None,
    ) -> VideoOutput:
        """
        Generate video from prompt.
        
        Args:
            prompt: Text description of the video
            duration: Duration in seconds
            width: Video width
            height: Video height
            start_image: Optional path to start frame image
            end_image: Optional path to end frame image
            generate_audio: Whether to generate audio (if supported)
            output_path: Optional output path (auto-generated if None)
        
        Returns:
            VideoOutput with path and metadata
        """
        pass


class ReplicateVideoModel(VideoModelInterface):
    """Replicate-based video models (Veo, etc.)."""
    
    def __init__(self, model_name: str = "veo-3.1-fast"):
        from replicate_client import get_model, MODEL_REGISTRY
        
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")
        
        self._model_name = model_name
        self._config = get_model(model_name)
    
    @property
    def name(self) -> str:
        return self._model_name
    
    @property
    def supports_start_frame(self) -> bool:
        return self._config.supports_start_frame
    
    @property
    def supports_end_frame(self) -> bool:
        return self._config.supports_end_frame
    
    @property
    def supports_audio(self) -> bool:
        return self._config.supports_audio
    
    def generate_video(
        self,
        prompt: str,
        duration: int = 8,
        width: int = 768,
        height: int = 512,
        start_image: Optional[str] = None,
        end_image: Optional[str] = None,
        generate_audio: bool = False,
        output_path: Optional[str] = None,
    ) -> VideoOutput:
        from replicate_client import (
            generate_video,
            generate_video_with_image,
            save_output,
        )
        
        # Determine aspect ratio from dimensions
        if width > height:
            aspect_ratio = "16:9"
        elif height > width:
            aspect_ratio = "9:16"
        else:
            aspect_ratio = "1:1"
        
        # Generate video
        if start_image:
            output = generate_video_with_image(
                prompt=prompt,
                image=start_image,
                duration=duration,
                generate_audio=generate_audio and self.supports_audio,
                last_frame=end_image if self.supports_end_frame else None,
                aspect_ratio=aspect_ratio,
                model=self._model_name,
            )
        else:
            output = generate_video(
                prompt=prompt,
                duration=duration,
                generate_audio=generate_audio and self.supports_audio,
                aspect_ratio=aspect_ratio,
                model=self._model_name,
            )
        
        # Save output
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".mp4")
        
        save_output(output, output_path)
        
        return VideoOutput(
            path=output_path,
            duration_s=duration,
            width=width,
            height=height,
            fps=24,
        )


class LTXVideoModel(VideoModelInterface):
    """
    Self-hosted LTX-Video model via diffusers.
    
    Requires: diffusers, torch, accelerate
    VRAM: ~10GB with memory optimizations, ~24GB without
    """
    
    VARIANTS = {
        "ltx-video": "Lightricks/LTX-Video",
        "ltx-video-dev": "Lightricks/LTX-Video-0.9.7-dev",
        "ltx-video-distilled": "Lightricks/LTX-Video-0.9.7-distilled",
    }
    
    def __init__(
        self,
        variant: str = "ltx-video",
        use_memory_optimization: bool = True,
        device: str = "cuda",
    ):
        """
        Initialize LTX-Video model.
        
        Args:
            variant: Model variant (ltx-video, ltx-video-dev, ltx-video-distilled)
            use_memory_optimization: Use fp8 + group offloading (~10GB VRAM)
            device: Device to run on (cuda, cpu)
        """
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Available: {list(self.VARIANTS.keys())}")
        
        self._variant = variant
        self._model_id = self.VARIANTS[variant]
        self._use_memory_optimization = use_memory_optimization
        self._device = device
        self._pipeline = None
        self._is_distilled = "distilled" in variant
    
    @property
    def name(self) -> str:
        return self._variant
    
    @property
    def supports_start_frame(self) -> bool:
        return True  # Via LTXConditionPipeline
    
    @property
    def supports_end_frame(self) -> bool:
        return False  # Not directly supported
    
    @property
    def supports_audio(self) -> bool:
        return False  # LTX-Video doesn't generate audio
    
    def _load_pipeline(self):
        """Lazy load the pipeline."""
        if self._pipeline is not None:
            return
        
        print(f"🔄 Loading LTX-Video pipeline: {self._model_id}")
        print(f"   Memory optimization: {self._use_memory_optimization}")
        
        from diffusers import LTXPipeline, LTXConditionPipeline, AutoModel
        from diffusers.utils import export_to_video
        
        # Determine dtype - MPS works better with float32, CUDA with bfloat16
        if self._device == "mps":
            dtype = torch.float32  # MPS has limited bfloat16 support
            print("   Using float32 for MPS compatibility")
        else:
            dtype = torch.bfloat16
        
        if self._use_memory_optimization and self._device == "cuda":
            # fp8 layerwise weight-casting for ~10GB VRAM (CUDA only)
            transformer = AutoModel.from_pretrained(
                self._model_id,
                subfolder="transformer",
                torch_dtype=dtype,
            )
            transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn,
                compute_dtype=dtype,
            )
            
            # Use condition pipeline for image-to-video support
            if "dev" in self._variant or "distilled" in self._variant:
                self._pipeline = LTXConditionPipeline.from_pretrained(
                    self._model_id,
                    transformer=transformer,
                    torch_dtype=dtype,
                )
            else:
                self._pipeline = LTXPipeline.from_pretrained(
                    self._model_id,
                    transformer=transformer,
                    torch_dtype=dtype,
                )
            
            # Group offloading
            from diffusers.hooks import apply_group_offloading
            
            onload_device = torch.device(self._device)
            offload_device = torch.device("cpu")
            
            self._pipeline.transformer.enable_group_offload(
                onload_device=onload_device,
                offload_device=offload_device,
                offload_type="leaf_level",
                use_stream=True,
            )
            apply_group_offloading(
                self._pipeline.text_encoder,
                onload_device=onload_device,
                offload_type="block_level",
                num_blocks_per_group=2,
            )
            apply_group_offloading(
                self._pipeline.vae,
                onload_device=onload_device,
                offload_type="leaf_level",
            )
        else:
            # Standard loading for MPS or high-VRAM CUDA
            if "dev" in self._variant or "distilled" in self._variant:
                self._pipeline = LTXConditionPipeline.from_pretrained(
                    self._model_id,
                    torch_dtype=dtype,
                )
            else:
                self._pipeline = LTXPipeline.from_pretrained(
                    self._model_id,
                    torch_dtype=dtype,
                )
            self._pipeline.to(self._device)
        
        # Enable VAE tiling for memory efficiency
        self._pipeline.vae.enable_tiling()
        
        print(f"✅ LTX-Video pipeline loaded")
    
    def generate_video(
        self,
        prompt: str,
        duration: int = 8,
        width: int = 768,
        height: int = 512,
        start_image: Optional[str] = None,
        end_image: Optional[str] = None,
        generate_audio: bool = False,
        output_path: Optional[str] = None,
    ) -> VideoOutput:
        from diffusers.utils import export_to_video
        
        self._load_pipeline()
        
        # Calculate frames from duration (24 fps)
        fps = 24
        num_frames = int(duration * fps)
        # LTX-Video works best with specific frame counts
        # Round to nearest valid frame count
        num_frames = max(25, min(161, num_frames))
        
        print(f"🎬 Generating video with LTX-Video")
        print(f"   Prompt: {prompt[:80]}...")
        print(f"   Resolution: {width}x{height}, Frames: {num_frames}")
        
        negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
        
        # Build generation kwargs
        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "decode_timestep": 0.03 if not self._is_distilled else 0.05,
            "decode_noise_scale": 0.025,
        }
        
        # Distilled model uses different settings
        if self._is_distilled:
            gen_kwargs["timesteps"] = [1000, 993, 987, 981, 975, 909, 725, 0.03]
            gen_kwargs["guidance_scale"] = 1.0
        else:
            gen_kwargs["num_inference_steps"] = 50
            gen_kwargs["guidance_scale"] = 5.0
        
        # Add image conditioning if provided and using condition pipeline
        if start_image and hasattr(self._pipeline, "LTXVideoCondition"):
            from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
            from diffusers.utils import load_image
            
            image = load_image(start_image)
            condition = LTXVideoCondition(video=[image], frame_index=0)
            gen_kwargs["conditions"] = [condition]
        
        # Generate video
        video = self._pipeline(**gen_kwargs).frames[0]
        
        # Save output
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".mp4")
        
        export_to_video(video, output_path, fps=fps)
        
        actual_duration = num_frames / fps
        print(f"✅ Video saved: {output_path} ({actual_duration:.1f}s)")
        
        return VideoOutput(
            path=output_path,
            duration_s=actual_duration,
            width=width,
            height=height,
            fps=fps,
        )


def get_video_model(
    model_name: str,
    **kwargs,
) -> VideoModelInterface:
    """
    Factory function to get a video model by name.
    
    Args:
        model_name: Model name (veo-3.1-fast, ltx-video, etc.)
        **kwargs: Additional arguments for model initialization
    
    Returns:
        VideoModelInterface implementation
    """
    # LTX-Video variants
    if model_name.startswith("ltx-video"):
        return LTXVideoModel(variant=model_name, **kwargs)
    
    # Replicate models
    return ReplicateVideoModel(model_name=model_name, **kwargs)


# =============================================================================
# Test function
# =============================================================================

def test_ltx_video():
    """Test LTX-Video model loading and generation."""
    print("=" * 60)
    print("Testing LTX-Video Model")
    print("=" * 60)
    
    # Check hardware
    import platform
    print(f"\n📊 Hardware Info:")
    print(f"   Platform: {platform.system()} {platform.machine()}")
    print(f"   Python: {platform.python_version()}")
    
    # Check GPU availability
    device = None
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"✅ MPS (Apple Silicon) available")
        print("   ⚠️  Experimental - may be slower than CUDA")
    else:
        print("❌ No GPU available. LTX-Video requires GPU.")
        print("\n💡 To run LTX-Video:")
        print("   1. Deploy on RunPod with RTX 4090 or A100 (~$0.40-$1.50/hr)")
        print("   2. Use Lambda Labs with A10G (~$0.60/hr)")
        print("   3. Use Replicate's hosted version (pay per run)")
        return False
    
    try:
        # Initialize model - use memory optimization only for CUDA
        model = LTXVideoModel(
            variant="ltx-video",
            use_memory_optimization=(device == "cuda"),
            device=device,
        )
        
        # Generate a short test video
        output = model.generate_video(
            prompt="A beautiful sunset over the ocean, waves gently rolling, golden hour lighting",
            duration=3,  # Short test
            width=512,
            height=768,  # 9:16 vertical
            output_path="test_ltx_output.mp4",
        )
        
        print(f"\n✅ Test successful!")
        print(f"   Output: {output.path}")
        print(f"   Duration: {output.duration_s}s")
        print(f"   Resolution: {output.width}x{output.height}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_ltx_video()
