"""
Scene schema definitions for the short video pipeline.

Supports loading from YAML or JSON files, with validation and duration estimation.
"""

import json
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal

from replicate_client import estimate_speech_duration, CHARS_PER_SECOND, WORDS_PER_SECOND, get_word_range_for_duration


# Valid durations for Veo video models (4, 6, or 8 seconds)
VALID_VEO_DURATIONS = [4, 6, 8]


# =============================================================================
# Scene Types
# =============================================================================

@dataclass
class SceneConfig:
    """Configuration for a single scene."""
    id: str
    type: Literal["speaker", "speaker_in_scene", "speaker_angle_change", "broll", "product", "transition"]
    duration_s: float
    
    # Video generation
    video_prompt: str
    start_image: Optional[str] = None  # Path or URL to start frame
    end_image: Optional[str] = None    # Path or URL to end frame (for constrained generation)
    
    # Speaker/voiceover
    script: Optional[str] = None       # Text to speak
    speaker_image: Optional[str] = None  # Reference image for speaker
    voice_id: Optional[str] = None     # Voice ID for TTS
    emotion: Optional[str] = None      # TTS emotion
    
    # Product
    product_image: Optional[str] = None  # Single product image for background generation
    product_dir: Optional[str] = None    # Directory with product images (uses up to 4 as references)
    
    # Scene transformation (for speaker_in_scene type)
    scene_prompt: Optional[str] = None  # Prompt describing scene to put speaker into
    scene_model: str = "nano-banana-pro"  # Model for scene transformation
    
    # Angle change (for speaker_angle_change type - uses previous scene's last frame)
    angle_prompt: Optional[str] = None  # Prompt describing camera angle change (e.g., "30 degrees left rotation")
    
    # Compositing
    background: Optional[str] = None   # Background image/video path
    background_prompt: Optional[str] = None  # Prompt for generating background
    background_type: str = "image_to_video"  # Type: "image", "video", "image_to_video"
    overlay_position: str = "bottom_right"   # Position: top_left, top_right, bottom_left, bottom_right, center
    overlay_scale: float = 0.35        # Scale factor for speaker overlay (0.35 = 35% of frame)
    
    # Audio
    generate_video_audio: bool = False  # Use Veo's built-in audio generation
    background_audio: Optional[str] = None  # Audio prompt for background video (None = no audio)
    background_music: Optional[str] = None  # Path to background music file
    
    # Captions
    add_captions: bool = False  # Add TikTok-style animated captions to this scene
    caption_color: str = "#FFFFFF"  # Highlight color for captions (hex)
    caption_language: str = "auto"  # Language for caption detection
    
    # Models to use (override defaults)
    video_model: str = "wan-2.5-i2v"      # For background video generation
    speaker_model: str = "veo-3.1-fast"   # For speaker video generation
    image_model: str = "flux-2-pro"
    tts_model: str = "speech-02-hd"
    
    def validate_script_duration(self) -> tuple[bool, float]:
        """
        Check if script fits within scene duration.
        
        Returns:
            Tuple of (is_valid, estimated_duration)
        """
        if not self.script:
            return True, 0.0
        
        estimated = estimate_speech_duration(self.script)
        return estimated <= self.duration_s, estimated
    
    def get_max_script_chars(self) -> int:
        """Get maximum characters that fit in this scene's duration."""
        return int(self.duration_s * CHARS_PER_SECOND * 0.9)  # 10% buffer
    
    def get_word_range(self) -> tuple[int, int]:
        """Get recommended word count range for this scene's duration."""
        return get_word_range_for_duration(self.duration_s)
    
    def get_script_word_count(self) -> int:
        """Get current script word count."""
        if not self.script:
            return 0
        return len(self.script.split())


@dataclass
class AdConfig:
    """Configuration for a complete ad/video."""
    title: str
    aspect_ratio: str = "9:16"
    resolution: str = "720p"
    
    # Scenes
    scenes: list[SceneConfig] = field(default_factory=list)
    
    # Global voice settings
    default_voice_id: Optional[str] = None
    voice_clone_source: Optional[str] = None  # Path to audio for voice cloning
    
    # Global model defaults
    video_model: str = "wan-2.5-i2v"
    image_model: str = "flux-2-pro"
    tts_model: str = "speech-02-hd"
    
    # Branding
    brand_colors: list[str] = field(default_factory=list)
    logo_path: Optional[str] = None
    
    # Global caption settings (can be overridden per scene)
    add_captions: bool = False  # Add captions to all scenes by default
    caption_color: str = "#FFFFFF"  # Default caption highlight color
    caption_language: str = "auto"  # Default language for captions
    caption_final_video: bool = False  # Add captions to final stitched video instead of per-scene
    
    @property
    def total_duration(self) -> float:
        """Total duration of all scenes."""
        return sum(s.duration_s for s in self.scenes)
    
    def validate(self) -> list[str]:
        """
        Validate the ad configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not self.scenes:
            errors.append("No scenes defined")
        
        for scene in self.scenes:
            min_words, max_words = scene.get_word_range()
            word_count = scene.get_script_word_count()
            
            # Check script duration
            is_valid, estimated = scene.validate_script_duration()
            if not is_valid:
                errors.append(
                    f"Scene '{scene.id}': script too long ({estimated:.1f}s) "
                    f"for duration ({scene.duration_s}s). "
                    f"Use {min_words}-{max_words} words (currently {word_count})."
                )
            
            # Warn if script is too short
            if scene.script and word_count < min_words:
                errors.append(
                    f"Scene '{scene.id}': script may be too short ({word_count} words). "
                    f"Recommend {min_words}-{max_words} words for {scene.duration_s}s."
                )
            
            # Check Veo duration constraints
            if scene.video_model.startswith("veo") and int(scene.duration_s) not in VALID_VEO_DURATIONS:
                errors.append(
                    f"Scene '{scene.id}': duration {scene.duration_s}s invalid for Veo. "
                    f"Must be one of: {VALID_VEO_DURATIONS}"
                )
        
        return errors


# =============================================================================
# Loading Functions
# =============================================================================

def _resolve_path(path: Optional[str], base_dir: Optional[Path] = None) -> Optional[str]:
    """Resolve a relative path to absolute, checking if file exists."""
    if not path:
        return None
    
    p = Path(path)
    
    # Already absolute
    if p.is_absolute():
        return str(p) if p.exists() else None
    
    # Try relative to base_dir first
    if base_dir:
        resolved = base_dir / p
        if resolved.exists():
            return str(resolved.resolve())
    
    # Try relative to current working directory
    if p.exists():
        return str(p.resolve())
    
    return None


def load_scene_config(
    data: dict,
    default_video_model: str = "wan-2.5-i2v",
    default_speaker_model: str = "veo-3.1-fast",
    default_image_model: str = "flux-2-pro",
    default_tts_model: str = "speech-02-hd",
    base_dir: Optional[Path] = None,
) -> SceneConfig:
    """Load a SceneConfig from a dictionary, inheriting global defaults."""
    return SceneConfig(
        id=data["id"],
        type=data["type"],
        duration_s=data["duration_s"],
        video_prompt=data["video_prompt"],
        start_image=_resolve_path(data.get("start_image"), base_dir),
        end_image=_resolve_path(data.get("end_image"), base_dir),
        script=data.get("script"),
        speaker_image=_resolve_path(data.get("speaker_image"), base_dir),
        voice_id=data.get("voice_id"),
        emotion=data.get("emotion"),
        product_image=_resolve_path(data.get("product_image"), base_dir),
        product_dir=_resolve_path(data.get("product_dir"), base_dir),
        scene_prompt=data.get("scene_prompt"),
        scene_model=data.get("scene_model", "nano-banana-pro"),
        angle_prompt=data.get("angle_prompt"),
        background=_resolve_path(data.get("background"), base_dir),
        background_prompt=data.get("background_prompt"),
        background_type=data.get("background_type", "image_to_video"),
        overlay_position=data.get("overlay_position", "bottom_right"),
        overlay_scale=data.get("overlay_scale", 0.35),
        generate_video_audio=data.get("generate_video_audio", False),
        background_audio=data.get("background_audio"),
        background_music=_resolve_path(data.get("background_music"), base_dir),
        add_captions=data.get("add_captions", False),
        caption_color=data.get("caption_color", "#FFFFFF"),
        caption_language=data.get("caption_language", "auto"),
        video_model=data.get("video_model", default_video_model),
        speaker_model=data.get("speaker_model", default_speaker_model),
        image_model=data.get("image_model", default_image_model),
        tts_model=data.get("tts_model", default_tts_model),
    )


def load_ad_config(data: dict, base_dir: Optional[Path] = None) -> AdConfig:
    """Load an AdConfig from a dictionary."""
    # Get global model defaults
    global_video_model = data.get("video_model", "wan-2.5-i2v")
    global_speaker_model = data.get("speaker_model", "veo-3.1-fast")
    global_image_model = data.get("image_model", "flux-2-pro")
    global_tts_model = data.get("tts_model", "speech-02-hd")
    
    # Load scenes with global defaults
    scenes = [
        load_scene_config(s, global_video_model, global_speaker_model, global_image_model, global_tts_model, base_dir) 
        for s in data.get("scenes", [])
    ]
    
    return AdConfig(
        title=data["title"],
        aspect_ratio=data.get("aspect_ratio", "9:16"),
        resolution=data.get("resolution", "720p"),
        scenes=scenes,
        default_voice_id=data.get("default_voice_id"),
        voice_clone_source=data.get("voice_clone_source"),
        video_model=data.get("video_model", "wan-2.5-i2v"),
        image_model=data.get("image_model", "flux-2-pro"),
        tts_model=data.get("tts_model", "speech-02-hd"),
        brand_colors=data.get("brand_colors", []),
        logo_path=data.get("logo_path"),
        add_captions=data.get("add_captions", False),
        caption_color=data.get("caption_color", "#FFFFFF"),
        caption_language=data.get("caption_language", "auto"),
        caption_final_video=data.get("caption_final_video", False),
    )


def load_from_yaml(path: str | Path) -> AdConfig:
    """Load ad configuration from a YAML file."""
    path = Path(path).resolve()
    with open(path) as f:
        data = yaml.safe_load(f)
    
    # Use parent directory of YAML file as base for relative paths
    # But also check current working directory
    config = load_ad_config(data, base_dir=Path.cwd())
    errors = config.validate()
    
    if errors:
        print("⚠️ Validation warnings:")
        for error in errors:
            print(f"   - {error}")
    
    return config


def load_from_json(path: str | Path) -> AdConfig:
    """Load ad configuration from a JSON file."""
    path = Path(path).resolve()
    with open(path) as f:
        data = json.load(f)
    
    # Use parent directory of JSON file as base for relative paths
    config = load_ad_config(data, base_dir=Path.cwd())
    errors = config.validate()
    
    if errors:
        print("⚠️ Validation warnings:")
        for error in errors:
            print(f"   - {error}")
    
    return config


def load_config(path: str | Path) -> AdConfig:
    """Load ad configuration from YAML or JSON file (auto-detect)."""
    path = Path(path)
    
    if path.suffix in (".yaml", ".yml"):
        return load_from_yaml(path)
    elif path.suffix == ".json":
        return load_from_json(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .yaml, .yml, or .json")


# =============================================================================
# Example / Template Generation
# =============================================================================

def generate_example_yaml() -> str:
    """Generate an example YAML configuration."""
    return '''# Short Video Ad Configuration
# ================================

title: "Product Demo Ad"
aspect_ratio: "9:16"
resolution: "720p"

# Global model defaults (can be overridden per scene)
video_model: "veo-3.1-fast"
image_model: "flux-2-pro"
tts_model: "speech-02-hd"

# Voice settings
# voice_clone_source: "path/to/voice_sample.mp3"  # Clone a voice
# default_voice_id: "cloned_voice_id"  # Or use a preset

# Branding
brand_colors:
  - "#FF6B6B"
  - "#4ECDC4"
# logo_path: "assets/logo.png"

scenes:
  # Scene 1: Speaker introduction (6 seconds)
  - id: "intro"
    type: "speaker"
    duration_s: 6
    
    # Script must fit in duration (~84 chars max for 6s)
    script: "Hey! Want to know the secret to amazing skin? Let me show you!"
    
    video_prompt: >
      Close-up of a friendly person talking to camera,
      bright green screen background, studio lighting,
      natural gestures, engaging expression, vertical 9:16
    
    speaker_image: "speakers/speaker1.png"
    emotion: "happy"
    generate_video_audio: false  # We'll use TTS instead

  # Scene 2: Product showcase (5 seconds)
  - id: "product_reveal"
    type: "product"
    duration_s: 5
    
    script: "This serum changed everything for me."
    
    video_prompt: >
      Elegant product bottle on marble surface,
      soft natural lighting, slow camera push-in,
      water droplets, luxury aesthetic, vertical 9:16
    
    # start_image: "assets/product_start.png"  # Optional start frame
    emotion: "neutral"

  # Scene 3: Results/testimonial (5 seconds)
  - id: "results"
    type: "speaker"
    duration_s: 5
    
    script: "Two weeks later, my skin is glowing!"
    
    video_prompt: >
      Person showing clear, glowing skin to camera,
      bright natural lighting, happy expression,
      before/after energy, vertical 9:16
    
    speaker_image: "speakers/speaker1.png"
    emotion: "happy"

  # Scene 4: Call to action (4 seconds)
  - id: "cta"
    type: "broll"
    duration_s: 4
    
    script: "Try it now. Link in bio!"
    
    video_prompt: >
      Product with "Shop Now" text overlay aesthetic,
      dynamic motion graphics style, brand colors,
      urgency and excitement, vertical 9:16
    
    emotion: "excited"
    generate_video_audio: true  # Use Veo's audio for this scene
'''


if __name__ == "__main__":
    # Generate example YAML
    print(generate_example_yaml())
