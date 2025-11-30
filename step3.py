import os
import subprocess

def create_solid_color_background(width, height, duration, fps, color="black", output_path="temp_background.mp4"):
    """
    Create a solid color background video using ffmpeg.
    
    Args:
        width (int): Video width
        height (int): Video height  
        duration (int): Duration in seconds
        fps (int): Frames per second
        color (str): Background color
        output_path (str): Output video path
    """
    
    print(f"🎬 Creating solid {color} background...")
    
    try:
        cmd = [
            'ffmpeg',
            '-f', 'lavfi',
            '-i', f'color=c={color}:s={width}x{height}:d={duration}:r={fps}',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-y',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Background creation error: {result.stderr}")
            raise Exception(f"Background creation failed: {result.stderr}")
        
        print(f"✅ Background created: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Error creating background: {str(e)}")
        raise

def composite_speaker_on_background(original_video, alpha_mask, background, output_video="step3_composited.mp4"):
    """
    Composite speaker onto background using ffmpeg with alpha mask.
    
    Args:
        original_video (str): Path to original speaker video
        alpha_mask (str): Path to alpha mask video
        background (str): Path to background video/image
        output_video (str): Output composited video path
    """
    
    print(f"🎬 Compositing speaker onto background...")
    
    try:
        # Use ffmpeg to composite with alpha mask
        cmd = [
            'ffmpeg',
            '-i', background,
            '-i', original_video,
            '-i', alpha_mask,
            '-filter_complex',
            '[1:v][2:v]alphamerge[fg];[0:v][fg]overlay=format=auto',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-y',
            output_video
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Compositing error: {result.stderr}")
            raise Exception(f"Compositing failed: {result.stderr}")
        
        print(f"✅ Composited video saved: {output_video}")
        return output_video
        
    except Exception as e:
        print(f"❌ Error compositing: {str(e)}")
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
        cmd_size = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,duration,r_frame_rate',
            '-of', 'csv=p=0',
            video_path
        ]
        
        result = subprocess.run(cmd_size, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Failed to get video properties: {result.stderr}")
        
        parts = result.stdout.strip().split(',')
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
        
    except Exception as e:
        print(f"❌ Error getting video properties: {str(e)}")
        # Return defaults if we can't get properties
        return {
            'width': 720,
            'height': 1280,
            'duration': 8.0,
            'fps': 24.0
        }

def process_compositing(original_video, alpha_mask, background_input=None, output_video="step3_composited.mp4"):
    """
    Complete compositing process: create background (if needed) and composite speaker.
    
    Args:
        original_video (str): Path to original speaker video
        alpha_mask (str): Path to alpha mask video
        background_input (str): Path to custom background (None for solid color)
        output_video (str): Output composited video path
    """
    
    # Check if input files exist
    if not os.path.exists(original_video):
        raise FileNotFoundError(f"Original video not found: {original_video}")
    if not os.path.exists(alpha_mask):
        raise FileNotFoundError(f"Alpha mask not found: {alpha_mask}")
    
    print(f"Starting compositing process...")
    
    try:
        # Get video properties from original video
        props = get_video_properties(original_video)
        print(f"Video properties: {props}")
        
        # Create or use background
        if background_input and os.path.exists(background_input):
            background_path = background_input
            print(f"Using custom background: {background_path}")
        else:
            # Create solid color background
            background_path = "temp_background.mp4"
            create_solid_color_background(
                width=props['width'],
                height=props['height'],
                duration=int(props['duration']),
                fps=int(props['fps']),
                color="blue",  # You can change this color
                output_path=background_path
            )
        
        # Composite speaker onto background
        result = composite_speaker_on_background(
            original_video=original_video,
            alpha_mask=alpha_mask,
            background=background_path,
            output_video=output_video
        )
        
        # Clean up temporary background if we created one
        if not background_input and os.path.exists(background_path):
            os.remove(background_path)
            print("🧹 Cleaned up temporary background file")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in compositing process: {str(e)}")
        raise

if __name__ == "__main__":
    # Configuration
    ORIGINAL_VIDEO = "step1.mp4"
    ALPHA_MASK = "step2_alpha_mask.mp4"
    BACKGROUND_INPUT = None  # Set to path of background video/image, or None for solid color
    OUTPUT_VIDEO = "step3_composited.mp4"
    
    # Process compositing
    try:
        result = process_compositing(
            original_video=ORIGINAL_VIDEO,
            alpha_mask=ALPHA_MASK,
            background_input=BACKGROUND_INPUT,
            output_video=OUTPUT_VIDEO
        )
        print(f"🎉 Step 3 completed successfully! Generated: {result}")
        print("📋 Next: Generate start images for scenes (step 4)")
        
    except Exception as e:
        print(f"💥 Step 3 failed: {e}")
        exit(1)
