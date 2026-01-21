"""
Scene and speaker configuration schema.

Supports:
- Multi-image speaker folders
- Product image folders
- YAML/JSON configuration loading
"""

import json
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal


@dataclass
class SpeakerConfig:
    """Configuration for a speaker with multi-image support."""
    id: str
    name: str
    image_dir: Optional[str] = None  # Directory with multiple speaker images
    images: list[str] = field(default_factory=list)  # Resolved image paths
    description: Optional[str] = None  # Physical description for prompts
    voice_id: Optional[str] = None  # TTS voice ID
    language: str = "en"
    
    def load_images(self, base_dir: Optional[Path] = None) -> list[str]:
        """Load all images from speaker directory."""
        if not self.image_dir:
            return self.images
        
        dir_path = Path(self.image_dir)
        if base_dir and not dir_path.is_absolute():
            dir_path = base_dir / dir_path
        
        if not dir_path.exists():
            return self.images
        
        # Load all images from directory
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']
        images = []
        for ext in extensions:
            images.extend(sorted(dir_path.glob(ext)))
        
        self.images = [str(p.resolve()) for p in images]
        return self.images
    
    def get_reference_images(self, max_images: int = 4) -> list[str]:
        """Get reference images for model input (up to max_images)."""
        return self.images[:max_images]


@dataclass
class ProductConfig:
    """Configuration for product images."""
    id: str
    name: str
    image_dir: Optional[str] = None  # Directory with product images
    images: list[str] = field(default_factory=list)  # Resolved image paths
    description: Optional[str] = None
    
    def load_images(self, base_dir: Optional[Path] = None) -> list[str]:
        """Load all images from product directory."""
        if not self.image_dir:
            return self.images
        
        dir_path = Path(self.image_dir)
        if base_dir and not dir_path.is_absolute():
            dir_path = base_dir / dir_path
        
        if not dir_path.exists():
            return self.images
        
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']
        images = []
        for ext in extensions:
            images.extend(sorted(dir_path.glob(ext)))
        
        self.images = [str(p.resolve()) for p in images]
        return self.images
    
    def get_reference_images(self, max_images: int = 4) -> list[str]:
        """Get reference images for model input."""
        return self.images[:max_images]


@dataclass
class SceneConfig:
    """Configuration for a single scene."""
    id: str
    type: Literal["speaker", "speaker_in_scene", "speaker_angle_change", "broll", "product", "transition"]
    duration_s: float
    
    # Video generation
    video_prompt: str
    start_image: Optional[str] = None
    end_image: Optional[str] = None
    
    # Speaker/voiceover
    script: Optional[str] = None
    speaker_id: Optional[str] = None  # Reference to SpeakerConfig
    speaker_image: Optional[str] = None  # Single image (legacy)
    speaker_images: list[str] = field(default_factory=list)  # Multiple images
    voice_id: Optional[str] = None
    emotion: Optional[str] = None
    
    # Product
    product_id: Optional[str] = None  # Reference to ProductConfig
    product_image: Optional[str] = None  # Single image (legacy)
    product_images: list[str] = field(default_factory=list)  # Multiple images
    product_dir: Optional[str] = None
    
    # Scene transformation
    scene_prompt: Optional[str] = None
    scene_model: str = "nano-banana-pro"
    
    # Angle change
    angle_prompt: Optional[str] = None
    
    # Compositing
    background: Optional[str] = None
    background_prompt: Optional[str] = None
    background_type: str = "image_to_video"
    overlay_position: str = "bottom_right"
    overlay_scale: float = 0.35
    
    # Audio
    generate_video_audio: bool = False
    background_audio: Optional[str] = None
    background_music: Optional[str] = None
    
    # Music generation
    music_prompt: Optional[str] = None  # Text prompt for music generation
    music_duration: Optional[int] = None  # Duration in seconds (defaults to scene duration)
    music_chords: Optional[str] = None  # Chord progression (e.g., "C G Am F")
    music_bpm: Optional[int] = None  # Beats per minute
    
    # Captions
    add_captions: bool = False
    caption_color: str = "#FFFFFF"
    caption_language: str = "auto"
    
    # Models (override defaults)
    video_model: str = "wan-2.5-i2v"
    speaker_model: str = "veo-3.1-fast"
    image_model: str = "seedream-4.5"
    tts_model: str = "speech-02-hd"
    music_model: str = "musicgen"  # musicgen or musicgen-stereo-chord
    
    def get_all_speaker_images(self) -> list[str]:
        """Get all speaker reference images."""
        images = []
        if self.speaker_image:
            images.append(self.speaker_image)
        images.extend(self.speaker_images)
        return images
    
    def get_all_product_images(self) -> list[str]:
        """Get all product reference images."""
        images = []
        if self.product_image:
            images.append(self.product_image)
        images.extend(self.product_images)
        return images


@dataclass
class AdConfig:
    """Configuration for a complete ad/video."""
    title: str
    aspect_ratio: str = "9:16"
    resolution: str = "720p"
    
    # Scenes
    scenes: list[SceneConfig] = field(default_factory=list)
    
    # Speakers (multi-image support)
    speakers: dict[str, SpeakerConfig] = field(default_factory=dict)
    
    # Products (multi-image support)
    products: dict[str, ProductConfig] = field(default_factory=dict)
    
    # Global voice settings
    default_voice_id: Optional[str] = None
    voice_clone_source: Optional[str] = None
    
    # Global model defaults
    video_model: str = "wan-2.5-i2v"
    speaker_model: str = "veo-3.1-fast"
    image_model: str = "seedream-4.5"
    tts_model: str = "speech-02-hd"
    
    # Backend configuration
    backend: str = "replicate"  # replicate, local, hybrid
    
    # Branding
    brand_colors: list[str] = field(default_factory=list)
    logo_path: Optional[str] = None
    
    # Caption settings
    add_captions: bool = False
    caption_color: str = "#FFFFFF"
    caption_language: str = "auto"
    caption_final_video: bool = False
    
    @property
    def total_duration(self) -> float:
        """Total duration of all scenes."""
        return sum(s.duration_s for s in self.scenes)
    
    def get_speaker(self, speaker_id: str) -> Optional[SpeakerConfig]:
        """Get speaker config by ID."""
        return self.speakers.get(speaker_id)
    
    def get_product(self, product_id: str) -> Optional[ProductConfig]:
        """Get product config by ID."""
        return self.products.get(product_id)
    
    def resolve_scene_references(self) -> None:
        """Resolve speaker/product references in scenes."""
        for scene in self.scenes:
            # Resolve speaker images
            if scene.speaker_id and scene.speaker_id in self.speakers:
                speaker = self.speakers[scene.speaker_id]
                scene.speaker_images = speaker.get_reference_images()
                if not scene.voice_id:
                    scene.voice_id = speaker.voice_id
            
            # Resolve product images
            if scene.product_id and scene.product_id in self.products:
                product = self.products[scene.product_id]
                scene.product_images = product.get_reference_images()
    
    def validate(self) -> list[str]:
        """Validate the configuration."""
        errors = []
        
        if not self.scenes:
            errors.append("No scenes defined")
        
        for scene in self.scenes:
            # Check speaker references
            if scene.speaker_id and scene.speaker_id not in self.speakers:
                errors.append(f"Scene '{scene.id}': unknown speaker_id '{scene.speaker_id}'")
            
            # Check product references
            if scene.product_id and scene.product_id not in self.products:
                errors.append(f"Scene '{scene.id}': unknown product_id '{scene.product_id}'")
            
            # Check scene type requirements
            if scene.type == "speaker_in_scene" and not scene.scene_prompt:
                errors.append(f"Scene '{scene.id}': speaker_in_scene requires scene_prompt")
            
            if scene.type == "speaker_angle_change" and not scene.angle_prompt:
                errors.append(f"Scene '{scene.id}': speaker_angle_change requires angle_prompt")
        
        return errors


def _resolve_path(path: Optional[str], base_dir: Optional[Path] = None) -> Optional[str]:
    """Resolve a relative path to absolute."""
    if not path:
        return None
    
    p = Path(path)
    if p.is_absolute():
        return str(p) if p.exists() else None
    
    if base_dir:
        resolved = base_dir / p
        if resolved.exists():
            return str(resolved.resolve())
    
    if p.exists():
        return str(p.resolve())
    
    return path  # Return as-is, might be URL


def _load_speaker_config(data: dict, base_dir: Optional[Path] = None) -> SpeakerConfig:
    """Load a SpeakerConfig from dictionary."""
    config = SpeakerConfig(
        id=data["id"],
        name=data.get("name", data["id"]),
        image_dir=data.get("image_dir"),
        description=data.get("description"),
        voice_id=data.get("voice_id"),
        language=data.get("language", "en"),
    )
    config.load_images(base_dir)
    return config


def _load_product_config(data: dict, base_dir: Optional[Path] = None) -> ProductConfig:
    """Load a ProductConfig from dictionary."""
    config = ProductConfig(
        id=data["id"],
        name=data.get("name", data["id"]),
        image_dir=data.get("image_dir"),
        description=data.get("description"),
    )
    config.load_images(base_dir)
    return config


def _load_scene_config(
    data: dict,
    default_video_model: str = "wan-2.5-i2v",
    default_speaker_model: str = "veo-3.1-fast",
    default_image_model: str = "seedream-4.5",
    default_tts_model: str = "speech-02-hd",
    base_dir: Optional[Path] = None,
) -> SceneConfig:
    """Load a SceneConfig from dictionary."""
    return SceneConfig(
        id=data["id"],
        type=data["type"],
        duration_s=data["duration_s"],
        video_prompt=data["video_prompt"],
        start_image=_resolve_path(data.get("start_image"), base_dir),
        end_image=_resolve_path(data.get("end_image"), base_dir),
        script=data.get("script"),
        speaker_id=data.get("speaker_id"),
        speaker_image=_resolve_path(data.get("speaker_image"), base_dir),
        voice_id=data.get("voice_id"),
        emotion=data.get("emotion"),
        product_id=data.get("product_id"),
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
        music_prompt=data.get("music_prompt"),
        music_duration=data.get("music_duration"),
        music_chords=data.get("music_chords"),
        music_bpm=data.get("music_bpm"),
        add_captions=data.get("add_captions", False),
        caption_color=data.get("caption_color", "#FFFFFF"),
        caption_language=data.get("caption_language", "auto"),
        video_model=data.get("video_model", default_video_model),
        speaker_model=data.get("speaker_model", default_speaker_model),
        image_model=data.get("image_model", default_image_model),
        tts_model=data.get("tts_model", default_tts_model),
        music_model=data.get("music_model", "musicgen"),
    )


def load_config(path: str | Path) -> AdConfig:
    """Load ad configuration from YAML or JSON file."""
    path = Path(path)
    base_dir = Path.cwd()
    
    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        elif path.suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
    
    # Load speakers
    speakers = {}
    for speaker_data in data.get("speakers", []):
        speaker = _load_speaker_config(speaker_data, base_dir)
        speakers[speaker.id] = speaker
    
    # Load products
    products = {}
    for product_data in data.get("products", []):
        product = _load_product_config(product_data, base_dir)
        products[product.id] = product
    
    # Get global model defaults
    global_video_model = data.get("video_model", "wan-2.5-i2v")
    global_speaker_model = data.get("speaker_model", "veo-3.1-fast")
    global_image_model = data.get("image_model", "seedream-4.5")
    global_tts_model = data.get("tts_model", "speech-02-hd")
    
    # Load scenes
    scenes = [
        _load_scene_config(
            s, global_video_model, global_speaker_model, 
            global_image_model, global_tts_model, base_dir
        )
        for s in data.get("scenes", [])
    ]
    
    config = AdConfig(
        title=data["title"],
        aspect_ratio=data.get("aspect_ratio", "9:16"),
        resolution=data.get("resolution", "720p"),
        scenes=scenes,
        speakers=speakers,
        products=products,
        default_voice_id=data.get("default_voice_id"),
        voice_clone_source=data.get("voice_clone_source"),
        video_model=global_video_model,
        speaker_model=global_speaker_model,
        image_model=global_image_model,
        tts_model=global_tts_model,
        backend=data.get("backend", "replicate"),
        brand_colors=data.get("brand_colors", []),
        logo_path=data.get("logo_path"),
        add_captions=data.get("add_captions", False),
        caption_color=data.get("caption_color", "#FFFFFF"),
        caption_language=data.get("caption_language", "auto"),
        caption_final_video=data.get("caption_final_video", False),
    )
    
    # Resolve references
    config.resolve_scene_references()
    
    # Validate
    errors = config.validate()
    if errors:
        print("⚠️ Validation warnings:")
        for error in errors:
            print(f"   - {error}")
    
    return config
