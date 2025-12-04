"""
Pipeline v2: Speaker-centric video generation.

Each scene:
1. Generate speaker video with Veo (with audio - speaker talks)
2. Create alpha mask for background removal
3. Composite speaker onto background image/video
4. Stitch all scenes together (each keeps its own audio)
"""

import os
import subprocess
from pathlib import Path
from dataclasses import dataclass

from scene_schema import load_config, AdConfig, SceneConfig
from replicate_client import (
    generate_video_with_image,
    generate_video,
    generate_image,
    remove_background,
    save_output,
)
from utils import extract_last_frame


@dataclass
class SceneOutput:
    """Output files for a processed scene."""
    scene_id: str
    video_path: str  # Final composited video with audio
    final_frame_path: str | None = None


def ensure_output_dir(base_dir: str = "output") -> Path:
    """Create output directory if it doesn't exist."""
    output_dir = Path(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_video_properties(video_path: str) -> dict:
    """Get video properties using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,duration,r_frame_rate',
            '-of', 'csv=p=0',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            if len(parts) >= 4:
                fps_parts = parts[3].split('/')
                return {
                    'width': int(parts[0]),
                    'height': int(parts[1]),
                    'duration': float(parts[2]) if parts[2] != 'N/A' else 8.0,
                    'fps': int(fps_parts[0]) / int(fps_parts[1])
                }
    except Exception:
        pass
    return {'width': 720, 'height': 1280, 'duration': 8.0, 'fps': 24.0}


def calculate_overlay_position(
    bg_width: int,
    bg_height: int,
    overlay_width: int,
    overlay_height: int,
    position: str,
    padding: int = 20,
) -> tuple[int, int]:
    """Calculate x, y position for overlay based on position string."""
    positions = {
        "top_left": (padding, padding),
        "top_right": (bg_width - overlay_width - padding, padding),
        "bottom_left": (padding, bg_height - overlay_height - padding),
        "bottom_right": (bg_width - overlay_width - padding, bg_height - overlay_height - padding),
        "center": ((bg_width - overlay_width) // 2, (bg_height - overlay_height) // 2),
    }
    return positions.get(position, positions["bottom_right"])


def composite_speaker_on_background(
    speaker_video: str,
    alpha_mask: str,
    background: str,
    output_video: str,
    scale: float = 0.35,
    position: str = "bottom_right",
) -> str:
    """
    Composite scaled speaker onto background using alpha mask.
    Preserves audio from speaker video.
    
    Args:
        speaker_video: Path to speaker video
        alpha_mask: Path to alpha mask video
        background: Path to background image or video
        output_video: Output path
        scale: Scale factor for speaker (0.35 = 35% of frame width)
        position: Position string (top_left, top_right, bottom_left, bottom_right, center)
    """
    print(f"🎨 Compositing speaker onto background...")
    print(f"   Scale: {scale:.0%}, Position: {position}")
    
    # Get video properties
    props = get_video_properties(speaker_video)
    bg_width, bg_height = props['width'], props['height']
    
    # Calculate scaled speaker dimensions
    speaker_width = int(bg_width * scale)
    speaker_height = int(bg_height * scale)
    
    # Calculate position
    pos_x, pos_y = calculate_overlay_position(
        bg_width, bg_height, speaker_width, speaker_height, position
    )
    
    # Check if background is image or video
    bg_ext = Path(background).suffix.lower()
    is_image_bg = bg_ext in ['.png', '.jpg', '.jpeg', '.webp']
    
    if is_image_bg:
        # For image background - loop it and composite
        filter_complex = (
            f'[0:v]scale={bg_width}:{bg_height},format=rgba[bg];'
            f'[1:v]scale={speaker_width}:{speaker_height},format=rgba[fg_scaled];'
            f'[2:v]scale={speaker_width}:{speaker_height},format=gray[alpha_scaled];'
            f'[fg_scaled][alpha_scaled]alphamerge[masked];'
            f'[bg][masked]overlay={pos_x}:{pos_y}:format=auto:shortest=1[out]'
        )
        cmd = [
            'ffmpeg',
            '-loop', '1',
            '-i', background,
            '-i', speaker_video,
            '-i', alpha_mask,
            '-filter_complex', filter_complex,
            '-map', '[out]',
            '-map', '1:a?',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-shortest',
            '-y',
            output_video
        ]
    else:
        # For video background
        filter_complex = (
            f'[1:v]scale={speaker_width}:{speaker_height}[fg_scaled];'
            f'[2:v]scale={speaker_width}:{speaker_height},format=gray[alpha_scaled];'
            f'[fg_scaled][alpha_scaled]alphamerge[masked];'
            f'[0:v][masked]overlay={pos_x}:{pos_y}:format=auto:shortest=1[out]'
        )
        cmd = [
            'ffmpeg',
            '-i', background,
            '-i', speaker_video,
            '-i', alpha_mask,
            '-filter_complex', filter_complex,
            '-map', '[out]',
            '-map', '1:a?',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-shortest',
            '-y',
            output_video
        ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"⚠️ Compositing error: {result.stderr[:500]}")
        raise Exception(f"Compositing failed")
    
    print(f"✅ Composited video saved: {output_video}")
    return output_video


def process_scene(
    scene: SceneConfig,
    output_dir: Path,
    previous_frame: str | None = None,
) -> SceneOutput:
    """
    Process a single scene:
    1. Generate speaker video with Veo (audio enabled)
    2. Create alpha mask
    3. Composite onto background
    """
    print(f"\n{'='*60}")
    print(f"🎬 Processing scene: {scene.id}")
    print(f"   Type: {scene.type}")
    print(f"   Duration: {scene.duration_s}s")
    print(f"   Script: {scene.script[:50] if scene.script else 'N/A'}...")
    print(f"{'='*60}")
    
    scene_dir = output_dir / scene.id
    scene_dir.mkdir(parents=True, exist_ok=True)
    
    # Build the video prompt - include script for Veo to generate speech
    video_prompt = scene.video_prompt
    if scene.script:
        # Veo will generate the speaker saying this
        video_prompt = f'{scene.video_prompt} The person is saying: "{scene.script}"'
    
    # Step 1: Generate speaker video with Veo (with audio!)
    print(f"\n📹 Step 1: Generating speaker video with audio...")
    
    # Use speaker image or previous frame for continuity
    input_image = scene.speaker_image or previous_frame
    if not input_image:
        raise ValueError(f"Scene {scene.id} needs speaker_image or previous scene frame")
    
    video_output = generate_video_with_image(
        prompt=video_prompt,
        image=input_image,
        duration=int(scene.duration_s),
        generate_audio=True,  # Veo generates the speaker's voice!
        model=scene.video_model,
    )
    
    raw_video_path = scene_dir / f"{scene.id}_speaker.mp4"
    save_output(video_output, raw_video_path)
    
    # Step 2: Generate alpha mask
    print(f"\n🎭 Step 2: Generating alpha mask...")
    
    alpha_output = remove_background(str(raw_video_path))
    alpha_path = scene_dir / f"{scene.id}_alpha.mp4"
    save_output(alpha_output, alpha_path)
    
    # Step 3: Generate or use background (image or video)
    print(f"\n🖼️ Step 3: Preparing background...")
    
    if scene.background and os.path.exists(scene.background):
        # Use provided background file
        background_path = scene.background
        print(f"   Using provided background: {background_path}")
    elif scene.background_prompt:
        # Generate background VIDEO from prompt (with optional product image)
        if scene.product_image and os.path.exists(scene.product_image):
            print(f"   Generating background video with product image...")
            bg_output = generate_video_with_image(
                prompt=scene.background_prompt,
                image=scene.product_image,
                duration=int(scene.duration_s),
                generate_audio=False,
                model=scene.video_model,
            )
        else:
            # For I2V models without a product image, generate an image first then animate
            print(f"   Generating background image first...")
            bg_image_output = generate_image(
                prompt=scene.background_prompt,
                model=scene.image_model,
            )
            bg_image_path = scene_dir / f"{scene.id}_background_frame.png"
            save_output(bg_image_output, bg_image_path)
            
            print(f"   Animating background image...")
            bg_output = generate_video_with_image(
                prompt=scene.background_prompt,
                image=str(bg_image_path),
                duration=int(scene.duration_s),
                generate_audio=False,
                model=scene.video_model,
            )
        background_path = scene_dir / f"{scene.id}_background.mp4"
        save_output(bg_output, background_path)
        background_path = str(background_path)
    else:
        # Generate a background IMAGE (with optional product image)
        if scene.product_image and os.path.exists(scene.product_image):
            print(f"   Generating background image with product...")
            bg_prompt = f"Product showcase, elegant presentation, {scene.product_image}, clean modern aesthetic, vertical 9:16"
            bg_output = generate_video_with_image(
                prompt=bg_prompt,
                image=scene.product_image,
                duration=int(scene.duration_s),
                generate_audio=False,
                model=scene.video_model,
            )
            background_path = scene_dir / f"{scene.id}_background.mp4"
        else:
            print(f"   Generating background image...")
            bg_prompt = f"Beautiful background for video ad, aesthetic scene, no people, clean modern look, vertical 9:16"
            bg_output = generate_image(
                prompt=bg_prompt,
                model=scene.image_model,
            )
            background_path = scene_dir / f"{scene.id}_background.png"
        save_output(bg_output, background_path)
        background_path = str(background_path)
    
    # Step 4: Composite scaled speaker onto background
    print(f"\n🎨 Step 4: Compositing (scale: {scene.overlay_scale:.0%}, pos: {scene.overlay_position})...")
    
    final_video_path = scene_dir / f"{scene.id}_final.mp4"
    composite_speaker_on_background(
        speaker_video=str(raw_video_path),
        alpha_mask=str(alpha_path),
        background=background_path,
        output_video=str(final_video_path),
        scale=scene.overlay_scale,
        position=scene.overlay_position,
    )
    
    # Extract last frame for next scene continuity
    final_frame_path = scene_dir / f"{scene.id}_final_frame.png"
    extract_last_frame(str(final_video_path), str(final_frame_path))
    
    return SceneOutput(
        scene_id=scene.id,
        video_path=str(final_video_path),
        final_frame_path=str(final_frame_path),
    )


def stitch_scenes(scene_outputs: list[SceneOutput], output_path: str) -> str:
    """
    Stitch all scene videos together.
    Each scene keeps its own audio track from Veo.
    """
    print(f"\n{'='*60}")
    print(f"🎬 Stitching {len(scene_outputs)} scenes together...")
    print(f"{'='*60}")
    
    if len(scene_outputs) == 1:
        # Just copy single scene
        import shutil
        shutil.copy(scene_outputs[0].video_path, output_path)
        print(f"✅ Final video saved: {output_path}")
        return output_path
    
    # Create concat file for ffmpeg
    concat_file = Path(output_path).parent / "concat_list.txt"
    
    with open(concat_file, "w") as f:
        for scene in scene_outputs:
            abs_path = str(Path(scene.video_path).resolve())
            escaped_path = abs_path.replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")
    
    # Concatenate with re-encoding to handle different codecs
    cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(concat_file),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'aac',
        '-y',
        output_path
    ]
    
    print(f"Running ffmpeg concat...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"⚠️ Concat error: {result.stderr[:500]}")
        raise Exception(f"FFmpeg concat failed")
    
    # Cleanup
    os.remove(concat_file)
    
    print(f"✅ Final video saved: {output_path}")
    return output_path


def run_pipeline(config_path: str, output_dir: str | None = None) -> str:
    """
    Run the complete video generation pipeline.
    
    Args:
        config_path: Path to YAML/JSON config file
        output_dir: Output directory (default: outputs/{scheme_name}/)
    """
    # Generate output directory from scheme name if not specified
    if output_dir is None:
        # Extract scheme name from config path (e.g., schemes/red_sports_car.yaml -> red_sports_car)
        scheme_name = Path(config_path).stem
        output_dir = f"outputs/{scheme_name}"
    
    print(f"\n{'#'*60}")
    print(f"# Short Video Pipeline")
    print(f"# Config: {config_path}")
    print(f"# Output: {output_dir}")
    print(f"{'#'*60}")
    
    # Load configuration
    config = load_config(config_path)
    print(f"\n📋 Loaded: {config.title}")
    print(f"   Scenes: {len(config.scenes)}")
    print(f"   Total duration: {config.total_duration}s")
    
    # Validate
    errors = config.validate()
    if errors:
        print("\n⚠️ Validation warnings:")
        for error in errors:
            print(f"   - {error}")
    
    # Setup output directory
    output_path = ensure_output_dir(output_dir)
    
    # Process each scene
    scene_outputs: list[SceneOutput] = []
    previous_frame: str | None = None
    
    for i, scene in enumerate(config.scenes):
        print(f"\n\n{'*'*60}")
        print(f"* Scene {i+1}/{len(config.scenes)}: {scene.id}")
        print(f"{'*'*60}")
        
        output = process_scene(scene, output_path, previous_frame)
        scene_outputs.append(output)
        previous_frame = output.final_frame_path
    
    # Stitch everything together
    final_video = output_path / f"{config.title.replace(' ', '_').lower()}_final.mp4"
    stitch_scenes(scene_outputs, str(final_video))
    
    print(f"\n\n{'#'*60}")
    print(f"# Pipeline Complete!")
    print(f"# Output: {final_video}")
    print(f"{'#'*60}\n")
    
    return str(final_video)


if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "example_ad.yaml"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None  # Default: outputs/{scheme_name}/
    
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
