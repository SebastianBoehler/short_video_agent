"""Scene processing logic."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config.schema import SceneConfig, AdConfig
from ..generators.base import VideoGenerator, ImageGenerator, GeneratorOutput
from ..processors.matting import BackgroundRemover
from ..processors.compositor import VideoCompositor
from ..utils.video import extract_last_frame
from ..utils.files import ensure_dir, get_output_path


@dataclass
class SceneOutput:
    """Output from processing a scene."""
    scene_id: str
    video_path: str
    final_frame_path: Optional[str] = None


class SceneProcessor:
    """
    Processes individual scenes.
    
    Handles different scene types:
    - speaker: Person talking with background overlay
    - speaker_in_scene: Transform speaker into environment
    - speaker_angle_change: Continue with new camera angle
    - broll: Pure video without speaker
    - product: Product-focused scene
    """
    
    def __init__(
        self,
        video_generator: VideoGenerator,
        image_generator: ImageGenerator,
        background_remover: BackgroundRemover,
        compositor: VideoCompositor,
    ):
        self._video_gen = video_generator
        self._image_gen = image_generator
        self._bg_remover = background_remover
        self._compositor = compositor
    
    def process(
        self,
        scene: SceneConfig,
        output_dir: Path,
        previous_frame: Optional[str] = None,
        config: Optional[AdConfig] = None,
    ) -> SceneOutput:
        """
        Process a single scene.
        
        Args:
            scene: Scene configuration
            output_dir: Output directory
            previous_frame: Path to previous scene's last frame (for continuity)
            config: Full ad config (for speaker/product references)
        
        Returns:
            SceneOutput with paths to generated files
        """
        print(f"\n{'='*60}")
        print(f"🎬 Processing scene: {scene.id}")
        print(f"   Type: {scene.type}")
        print(f"   Duration: {scene.duration_s}s")
        if scene.script:
            print(f"   Script: {scene.script[:50]}...")
        print(f"{'='*60}")
        
        scene_dir = ensure_dir(output_dir / scene.id)
        
        # Route to appropriate handler
        if scene.type == "broll":
            return self._process_broll(scene, scene_dir)
        elif scene.type == "speaker_in_scene":
            return self._process_speaker_in_scene(scene, scene_dir)
        elif scene.type == "speaker_angle_change":
            return self._process_speaker_angle_change(scene, scene_dir, previous_frame)
        else:
            # Default: speaker with background overlay
            return self._process_speaker(scene, scene_dir, previous_frame)
    
    def _process_broll(
        self,
        scene: SceneConfig,
        scene_dir: Path,
    ) -> SceneOutput:
        """Process broll scene - pure video without speaker overlay."""
        print(f"\n🎬 Generating broll scene (pure video)...")
        
        # Generate video directly from prompt
        video_output = self._video_gen.generate(
            prompt=scene.video_prompt,
            duration=int(scene.duration_s),
            generate_audio=scene.generate_video_audio,
        )
        
        # Save to scene directory
        video_path = scene_dir / f"{scene.id}_final.mp4"
        Path(video_output.path).rename(video_path)
        
        # Extract final frame for continuity
        final_frame_path = scene_dir / f"{scene.id}_final_frame.png"
        extract_last_frame(str(video_path), str(final_frame_path))
        
        print(f"✅ Broll scene saved: {video_path}")
        return SceneOutput(
            scene_id=scene.id,
            video_path=str(video_path),
            final_frame_path=str(final_frame_path),
        )
    
    def _process_speaker_in_scene(
        self,
        scene: SceneConfig,
        scene_dir: Path,
    ) -> SceneOutput:
        """Process speaker_in_scene - transform speaker into environment."""
        print(f"\n🎭 Generating speaker-in-scene (scene transformation)...")
        
        # Get all speaker reference images
        speaker_images = scene.get_all_speaker_images()
        if not speaker_images:
            raise ValueError(f"Scene {scene.id} needs speaker images for scene transformation")
        if not scene.scene_prompt:
            raise ValueError(f"Scene {scene.id} needs scene_prompt for scene transformation")
        
        # Get product images if available
        product_images = scene.get_all_product_images()
        
        # Step 1: Generate scene image with speaker using multi-image support
        print(f"🖼️ Step 1: Transforming speaker into scene...")
        print(f"   Using {len(speaker_images)} speaker image(s)")
        if product_images:
            print(f"   Using {len(product_images)} product image(s)")
        
        scene_image_output = self._image_gen.generate_with_speaker(
            prompt=scene.scene_prompt,
            speaker_images=speaker_images,
            product_images=product_images,
        )
        scene_image_path = scene_dir / f"{scene.id}_scene_frame.png"
        Path(scene_image_output.path).rename(scene_image_path)
        
        # Step 2: Animate the scene image
        video_prompt = scene.video_prompt
        if scene.script:
            video_prompt = f'{scene.video_prompt} The person is saying: "{scene.script}"'
        
        print(f"📹 Step 2: Animating scene with audio...")
        video_output = self._video_gen.generate(
            prompt=video_prompt,
            start_image=str(scene_image_path),
            duration=int(scene.duration_s),
            generate_audio=True,
        )
        
        video_path = scene_dir / f"{scene.id}_final.mp4"
        Path(video_output.path).rename(video_path)
        
        # Extract final frame for continuity
        final_frame_path = scene_dir / f"{scene.id}_final_frame.png"
        extract_last_frame(str(video_path), str(final_frame_path))
        
        print(f"✅ Speaker-in-scene saved: {video_path}")
        return SceneOutput(
            scene_id=scene.id,
            video_path=str(video_path),
            final_frame_path=str(final_frame_path),
        )
    
    def _process_speaker_angle_change(
        self,
        scene: SceneConfig,
        scene_dir: Path,
        previous_frame: Optional[str],
    ) -> SceneOutput:
        """Process speaker_angle_change - continue with new camera angle."""
        print(f"\n🎬 Generating speaker angle change (camera rotation)...")
        
        if not previous_frame:
            raise ValueError(f"Scene {scene.id} needs previous scene's last frame for angle change")
        
        speaker_images = scene.get_all_speaker_images()
        if not speaker_images:
            raise ValueError(f"Scene {scene.id} needs speaker images for angle change")
        
        # Build angle prompt
        angle_direction = scene.angle_prompt or "camera rotated 30 degrees to the left, different perspective"
        base_prompt = scene.scene_prompt or scene.video_prompt
        full_angle_prompt = f"{base_prompt}, {angle_direction}, same person same setting same lighting, natural continuation"
        
        # Step 1: Generate new angle frame using previous frame + speaker as references
        print(f"🖼️ Step 1: Generating new camera angle...")
        print(f"   Angle: {angle_direction}")
        print(f"   Using previous frame + {len(speaker_images)} speaker image(s)")
        
        # Combine previous frame with speaker images
        reference_images = [previous_frame] + speaker_images
        
        angle_image_output = self._image_gen.generate(
            prompt=full_angle_prompt,
            reference_images=reference_images,
        )
        angle_image_path = scene_dir / f"{scene.id}_angle_frame.png"
        Path(angle_image_output.path).rename(angle_image_path)
        
        # Step 2: Animate the new angle
        video_prompt = scene.video_prompt
        if scene.script:
            video_prompt = f'{scene.video_prompt} The person is saying: "{scene.script}"'
        
        print(f"📹 Step 2: Animating new angle with audio...")
        video_output = self._video_gen.generate(
            prompt=video_prompt,
            start_image=str(angle_image_path),
            duration=int(scene.duration_s),
            generate_audio=True,
        )
        
        video_path = scene_dir / f"{scene.id}_final.mp4"
        Path(video_output.path).rename(video_path)
        
        # Extract final frame
        final_frame_path = scene_dir / f"{scene.id}_final_frame.png"
        extract_last_frame(str(video_path), str(final_frame_path))
        
        print(f"✅ Angle change saved: {video_path}")
        return SceneOutput(
            scene_id=scene.id,
            video_path=str(video_path),
            final_frame_path=str(final_frame_path),
        )
    
    def _process_speaker(
        self,
        scene: SceneConfig,
        scene_dir: Path,
        previous_frame: Optional[str],
    ) -> SceneOutput:
        """Process standard speaker scene with background overlay."""
        
        # Build video prompt with script
        video_prompt = scene.video_prompt
        if scene.script:
            video_prompt = f'{scene.video_prompt} The person is saying: "{scene.script}"'
        
        # Get speaker images
        speaker_images = scene.get_all_speaker_images()
        input_image = speaker_images[0] if speaker_images else previous_frame
        
        if not input_image:
            raise ValueError(f"Scene {scene.id} needs speaker_image or previous scene frame")
        
        # Step 1: Generate speaker video
        print(f"\n📹 Step 1: Generating speaker video with audio...")
        print(f"   Using {len(speaker_images)} speaker reference image(s)")
        
        video_output = self._video_gen.generate(
            prompt=video_prompt,
            start_image=input_image,
            duration=int(scene.duration_s),
            generate_audio=True,
        )
        
        raw_video_path = scene_dir / f"{scene.id}_speaker.mp4"
        Path(video_output.path).rename(raw_video_path)
        
        # Step 2: Generate alpha mask
        print(f"\n🎭 Step 2: Generating alpha mask...")
        alpha_output = self._bg_remover.remove_background(str(raw_video_path))
        alpha_path = scene_dir / f"{scene.id}_alpha.mp4"
        Path(alpha_output.path).rename(alpha_path)
        
        # Step 3: Generate or use background
        print(f"\n🖼️ Step 3: Preparing background...")
        background_path = self._prepare_background(scene, scene_dir)
        
        # Step 4: Composite
        print(f"\n🎨 Step 4: Compositing (scale: {scene.overlay_scale:.0%}, pos: {scene.overlay_position})...")
        
        final_video_path = scene_dir / f"{scene.id}_final.mp4"
        self._compositor.composite(
            speaker_video=str(raw_video_path),
            alpha_mask=str(alpha_path),
            background=background_path,
            output_path=str(final_video_path),
            scale=scene.overlay_scale,
            position=scene.overlay_position,
            mix_background_audio=scene.generate_video_audio,
        )
        
        # Extract last frame
        final_frame_path = scene_dir / f"{scene.id}_final_frame.png"
        extract_last_frame(str(final_video_path), str(final_frame_path))
        
        return SceneOutput(
            scene_id=scene.id,
            video_path=str(final_video_path),
            final_frame_path=str(final_frame_path),
        )
    
    def _prepare_background(
        self,
        scene: SceneConfig,
        scene_dir: Path,
    ) -> str:
        """Prepare background image or video for compositing."""
        import os
        
        # Use provided background file
        if scene.background and os.path.exists(scene.background):
            print(f"   Using provided background: {scene.background}")
            return scene.background
        
        # Generate background from prompt
        if scene.background_prompt:
            product_images = scene.get_all_product_images()
            
            if scene.background_type == "video":
                # Generate video directly
                print(f"   Generating background video...")
                bg_output = self._video_gen.generate(
                    prompt=scene.background_prompt,
                    duration=int(scene.duration_s),
                    generate_audio=scene.background_audio is not None,
                )
                background_path = scene_dir / f"{scene.id}_background.mp4"
                Path(bg_output.path).rename(background_path)
                return str(background_path)
            
            elif scene.background_type == "image":
                # Generate static image
                print(f"   Generating background image...")
                bg_output = self._image_gen.generate(
                    prompt=scene.background_prompt,
                    reference_images=product_images if product_images else None,
                )
                background_path = scene_dir / f"{scene.id}_background.png"
                Path(bg_output.path).rename(background_path)
                return str(background_path)
            
            else:  # image_to_video (default)
                # Generate image first, then animate
                print(f"   Generating background image with {len(product_images)} product reference(s)...")
                bg_image_output = self._image_gen.generate(
                    prompt=scene.background_prompt,
                    reference_images=product_images if product_images else None,
                )
                bg_image_path = scene_dir / f"{scene.id}_background_frame.png"
                Path(bg_image_output.path).rename(bg_image_path)
                
                # Animate
                print(f"   Animating background image...")
                bg_video_output = self._video_gen.generate(
                    prompt=scene.background_prompt,
                    start_image=str(bg_image_path),
                    duration=int(scene.duration_s),
                    generate_audio=scene.background_audio is not None,
                )
                background_path = scene_dir / f"{scene.id}_background.mp4"
                Path(bg_video_output.path).rename(background_path)
                return str(background_path)
        
        # Fallback: generate generic background
        print(f"   Generating generic background...")
        bg_prompt = "Beautiful background for video ad, aesthetic scene, no people, clean modern look, vertical 9:16"
        bg_output = self._image_gen.generate(prompt=bg_prompt)
        background_path = scene_dir / f"{scene.id}_background.png"
        Path(bg_output.path).rename(background_path)
        return str(background_path)
