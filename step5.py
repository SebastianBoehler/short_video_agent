import os
import subprocess
import json

def create_pip_composite(background_video, speaker_video, alpha_mask, output_video="step5_final.mp4", 
                         scale_factor=0.3, position_x=20, position_y=20):
    """
    Create picture-in-picture composite with speaker overlay in corner.
    
    Args:
        background_video (str): Path to background video (step4_scene.mp4)
        speaker_video (str): Path to original speaker video (step1.mp4)
        alpha_mask (str): Path to alpha mask (step2_alpha_mask.mp4)
        output_video (str): Output video path
        scale_factor (float): Scale factor for speaker (0.3 = 30% of original size)
        position_x (int): X position from left edge
        position_y (int): Y position from top edge
        
    Returns:
        str: Path to the final composited video
    """
    
    print(f"🎬 Creating picture-in-picture composite...")
    print(f"📐 Speaker scale: {scale_factor}, Position: ({position_x}, {position_y})")
    
    # Check if all video files exist
    for video in [background_video, speaker_video, alpha_mask]:
        if not os.path.exists(video):
            raise FileNotFoundError(f"Video file not found: {video}")
    
    try:
        # Get background video dimensions
        bg_props = get_video_properties(background_video)
        bg_width = bg_props['width']
        bg_height = bg_props['height']
        
        # Calculate speaker dimensions after scaling
        speaker_width = int(bg_width * scale_factor)
        speaker_height = int(bg_height * scale_factor)
        
        # Use ffmpeg to create PIP composite
        cmd = [
            'ffmpeg',
            '-i', background_video,
            '-i', speaker_video,
            '-i', alpha_mask,
            '-filter_complex',
            f'[1:v][2:v]alphamerge[fg];'  # Apply alpha mask to speaker
            f'[fg]scale={speaker_width}:{speaker_height}[scaled];'  # Scale speaker
            f'[0:v][scaled]overlay={position_x}:{position_y}:format=auto',  # Overlay speaker on background
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-y',
            output_video
        ]
        
        print(f"Running: {' '.join(cmd[:10])}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ PIP compositing error: {result.stderr}")
            raise Exception(f"PIP compositing failed: {result.stderr}")
        
        print(f"✅ PIP composite video created: {output_video}")
        return output_video
        
    except Exception as e:
        print(f"❌ Error creating PIP composite: {str(e)}")
        raise

def get_video_properties(video_path):
    """
    Get video properties using ffprobe.
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        dict: Video properties (width, height, duration, fps)
    """
    
    try:
        # Get width and height
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,duration,r_frame_rate',
            '-of', 'csv=p=0',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            # Return defaults if we can't get properties
            return {'width': 720, 'height': 1280, 'duration': 8.0, 'fps': 24.0}
        
        parts = result.stdout.strip().split(',')
        if len(parts) >= 4:
            width = int(parts[0])
            height = int(parts[1])
            duration = float(parts[2])
            
            # Parse fps (e.g., "24/1" -> 24.0)
            fps_parts = parts[3].split('/')
            fps = int(fps_parts[0]) / int(fps_parts[1])
            
            return {
                'width': width,
                'height': height,
                'duration': duration,
                'fps': fps
            }
        else:
            return {'width': 720, 'height': 1280, 'duration': 8.0, 'fps': 24.0}
        
    except Exception:
        return {'width': 720, 'height': 1280, 'duration': 8.0, 'fps': 24.0}

def process_pip_compositing(background_video="step4_scene.mp4", 
                           speaker_video="step1.mp4", 
                           alpha_mask="step2_alpha_mask.mp4",
                           output_video="step5_final.mp4",
                           scale_factor=0.3,
                           position="top_right"):
    """
    Complete picture-in-picture compositing process.
    
    Args:
        background_video (str): Path to background video
        speaker_video (str): Path to speaker video
        alpha_mask (str): Path to alpha mask
        output_video (str): Output video path
        scale_factor (float): Scale factor for speaker
        position (str): Position ("top_left", "top_right", "bottom_left", "bottom_right")
        
    Returns:
        str: Path to the final composited video
    """
    
    print(f"🎬 Starting picture-in-picture compositing...")
    
    try:
        # Get background dimensions for positioning
        bg_props = get_video_properties(background_video)
        bg_width = bg_props['width']
        bg_height = bg_props['height']
        
        # Calculate position based on placement
        speaker_width = int(bg_width * scale_factor)
        speaker_height = int(bg_height * scale_factor)
        
        if position == "top_right":
            pos_x = bg_width - speaker_width - 20
            pos_y = 20
        elif position == "top_left":
            pos_x = 20
            pos_y = 20
        elif position == "bottom_right":
            pos_x = bg_width - speaker_width - 20
            pos_y = bg_height - speaker_height - 20
        elif position == "bottom_left":
            pos_x = 20
            pos_y = bg_height - speaker_height - 20
        else:
            pos_x = 20
            pos_y = 20
        
        # Create PIP composite
        result = create_pip_composite(
            background_video=background_video,
            speaker_video=speaker_video,
            alpha_mask=alpha_mask,
            output_video=output_video,
            scale_factor=scale_factor,
            position_x=pos_x,
            position_y=pos_y
        )
        
        return result
        
    except Exception as e:
        print(f"❌ Error in PIP compositing process: {str(e)}")
        raise

if __name__ == "__main__":
    # Configuration
    BACKGROUND_VIDEO = "step4_scene.mp4"  # Living room scene
    SPEAKER_VIDEO = "step1.mp4"          # Original speaker video
    ALPHA_MASK = "step2_alpha_mask.mp4"  # Alpha mask
    OUTPUT_VIDEO = "step5_final.mp4"
    
    # PIP settings
    SCALE_FACTOR = 0.3  # Speaker at 30% of background size
    POSITION = "top_right"  # Position in top-right corner
    
    # Process PIP compositing
    try:
        result = process_pip_compositing(
            background_video=BACKGROUND_VIDEO,
            speaker_video=SPEAKER_VIDEO,
            alpha_mask=ALPHA_MASK,
            output_video=OUTPUT_VIDEO,
            scale_factor=SCALE_FACTOR,
            position=POSITION
        )
        
        print(f"🎉 Step 5 completed successfully! Generated: {result}")
        print("📋 Next: Add audio (voiceover, background music, sound effects) (step 6)")
        
    except Exception as e:
        print(f"💥 Step 5 failed: {e}")
        exit(1)
