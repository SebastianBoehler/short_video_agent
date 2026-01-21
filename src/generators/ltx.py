"""
Local LTX-2 video generator using diffusers.

Supports:
- Text-to-video generation
- Image-to-video generation
- Audio-to-video (when available in diffusers)

Requires GPU with ~10GB VRAM (with optimizations) or ~24GB without.
"""

import tempfile
from pathlib import Path
from typing import Optional

import torch

from .base import VideoGenerator, GeneratorOutput


class LTX2VideoGenerator(VideoGenerator):
    """
    Local LTX-2 video generator via diffusers.
    
    Memory requirements:
    - With fp8 + group offloading: ~10GB VRAM
    - Without optimizations: ~24GB VRAM
    """
    
    VARIANTS = {
        "ltx-2": "Lightricks/LTX-2",
        "ltx-2-distilled": "Lightricks/LTX-Video-0.9.7-distilled",
        "ltx-video": "Lightricks/LTX-Video",
    }
    
    def __init__(
        self,
        variant: str = "ltx-2",
        use_memory_optimization: bool = True,
        device: str = "auto",
    ):
        """
        Initialize LTX-2 video generator.
        
        Args:
            variant: Model variant (ltx-2, ltx-2-distilled, ltx-video)
            use_memory_optimization: Use fp8 + group offloading (~10GB VRAM)
            device: Device to run on (auto, cuda, mps, cpu)
        """
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Available: {list(self.VARIANTS.keys())}")
        
        self._variant = variant
        self._model_id = self.VARIANTS[variant]
        self._use_memory_optimization = use_memory_optimization
        self._pipeline = None
        self._is_distilled = "distilled" in variant
        self._is_ltx2 = "ltx-2" in variant.lower() or "LTX-2" in self._model_id
        
        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        else:
            self._device = device
    
    @property
    def name(self) -> str:
        return self._variant
    
    @property
    def supports_start_frame(self) -> bool:
        return True
    
    @property
    def supports_end_frame(self) -> bool:
        return False
    
    @property
    def supports_audio(self) -> bool:
        # LTX-2 supports audio-to-video, but pipeline not yet in diffusers
        return self._is_ltx2
    
    @property
    def supports_audio_input(self) -> bool:
        # Audio-to-video support (when available)
        return self._is_ltx2
    
    def _load_pipeline(self):
        """Lazy load the pipeline."""
        if self._pipeline is not None:
            return
        
        print(f"🔄 Loading LTX pipeline: {self._model_id}")
        print(f"   Device: {self._device}")
        print(f"   Memory optimization: {self._use_memory_optimization}")
        
        # Determine dtype
        if self._device == "mps":
            dtype = torch.float32
            print("   Using float32 for MPS compatibility")
        else:
            dtype = torch.bfloat16
        
        # Import diffusers components
        from diffusers import AutoModel
        from diffusers.utils import export_to_video
        
        # Try to import LTX2Pipeline for audio support, fall back to LTXConditionPipeline
        try:
            from diffusers import LTX2Pipeline
            pipeline_class = LTX2Pipeline
            print("   Using LTX2Pipeline (audio support available)")
        except ImportError:
            try:
                from diffusers import LTXConditionPipeline
                pipeline_class = LTXConditionPipeline
                print("   Using LTXConditionPipeline (no audio support yet)")
            except ImportError:
                from diffusers import LTXPipeline
                pipeline_class = LTXPipeline
                print("   Using LTXPipeline (basic)")
        
        if self._use_memory_optimization and self._device == "cuda":
            # fp8 layerwise weight-casting for ~10GB VRAM
            transformer = AutoModel.from_pretrained(
                self._model_id,
                subfolder="transformer",
                torch_dtype=dtype,
            )
            transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn,
                compute_dtype=dtype,
            )
            
            self._pipeline = pipeline_class.from_pretrained(
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
            # Standard loading
            self._pipeline = pipeline_class.from_pretrained(
                self._model_id,
                torch_dtype=dtype,
            )
            self._pipeline.to(self._device)
        
        # Enable VAE tiling for memory efficiency
        self._pipeline.vae.enable_tiling()
        
        print(f"✅ LTX pipeline loaded")
    
    def generate(
        self,
        prompt: str,
        duration: int = 8,
        width: int = 768,
        height: int = 512,
        start_image: Optional[str] = None,
        end_image: Optional[str] = None,
        audio_input: Optional[str] = None,
        generate_audio: bool = False,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> GeneratorOutput:
        """Generate video using local LTX model."""
        from diffusers.utils import export_to_video
        
        self._load_pipeline()
        
        # Calculate frames from duration (24 fps)
        fps = 24
        num_frames = int(duration * fps)
        # LTX works best with specific frame counts
        num_frames = max(25, min(161, num_frames))
        
        print(f"🎬 Generating video with {self._variant}")
        print(f"   Prompt: {prompt[:80]}...")
        print(f"   Resolution: {width}x{height}, Frames: {num_frames}")
        
        negative_prompt = kwargs.get(
            "negative_prompt",
            "worst quality, inconsistent motion, blurry, jittery, distorted"
        )
        
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
            gen_kwargs["num_inference_steps"] = kwargs.get("num_inference_steps", 50)
            gen_kwargs["guidance_scale"] = kwargs.get("guidance_scale", 5.0)
        
        # Add image conditioning if provided
        if start_image and hasattr(self._pipeline, "__class__") and "Condition" in self._pipeline.__class__.__name__:
            try:
                from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
                from diffusers.utils import load_image
                
                image = load_image(start_image)
                condition = LTXVideoCondition(video=[image], frame_index=0)
                gen_kwargs["conditions"] = [condition]
                print(f"   Using start image conditioning")
            except ImportError:
                print(f"   ⚠️ Image conditioning not available in this diffusers version")
        
        # Add audio conditioning if provided (when available)
        if audio_input and self.supports_audio_input:
            # TODO: Add audio conditioning when LTX2Pipeline supports it
            print(f"   ⚠️ Audio conditioning not yet implemented in diffusers")
        
        # Generate video
        video = self._pipeline(**gen_kwargs).frames[0]
        
        # Save output
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".mp4")
        
        export_to_video(video, output_path, fps=fps)
        
        actual_duration = num_frames / fps
        print(f"✅ Video saved: {output_path} ({actual_duration:.1f}s)")
        
        return GeneratorOutput(
            path=output_path,
            duration_s=actual_duration,
            width=width,
            height=height,
            fps=fps,
        )


def check_ltx_availability() -> dict:
    """Check if LTX-2 can run on this system."""
    result = {
        "available": False,
        "device": None,
        "vram_gb": None,
        "recommendations": [],
    }
    
    if torch.cuda.is_available():
        result["device"] = "cuda"
        result["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if result["vram_gb"] >= 24:
            result["available"] = True
            result["recommendations"].append("Full quality available (24GB+ VRAM)")
        elif result["vram_gb"] >= 10:
            result["available"] = True
            result["recommendations"].append("Use memory optimization (fp8 + offloading)")
        else:
            result["recommendations"].append(f"Insufficient VRAM ({result['vram_gb']:.1f}GB). Need 10GB+")
            result["recommendations"].append("Consider using Replicate API or cloud GPU")
    
    elif torch.backends.mps.is_available():
        result["device"] = "mps"
        result["available"] = True
        result["recommendations"].append("Apple Silicon detected - experimental support")
        result["recommendations"].append("May be slower than CUDA, use float32")
    
    else:
        result["recommendations"].append("No GPU available")
        result["recommendations"].append("Deploy on RunPod/Lambda with RTX 4090 or A100")
        result["recommendations"].append("Or use Replicate API for cloud inference")
    
    return result
