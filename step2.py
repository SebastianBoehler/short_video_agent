import os
import subprocess
import tempfile
from replicate_client import remove_background

def apply_alpha_mask_ffmpeg(original_video_path, alpha_mask_path, output_video_path="step2_no_bg.mp4"):
    """
    Apply alpha mask to original video using ffmpeg for reliable output.
    Composites onto black background for MP4 compatibility.
    
    Args:
        original_video_path (str): Path to original video with green screen
        alpha_mask_path (str): Path to alpha mask video
        output_video_path (str): Path to save final video with black background
    """
    
    print(f"🎬 Applying alpha mask using ffmpeg...")
    
    try:
        # Use ffmpeg to composite the videos
        cmd = [
            'ffmpeg',
            '-i', original_video_path,
            '-i', alpha_mask_path,
            '-filter_complex', 
            '[1:v]format=gray,colorchannelmixer=aa=0.5[alpha];[0:v][alpha]alphamerge,format=yuv420p',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-y',  # Overwrite output file
            output_video_path
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ FFmpeg error: {result.stderr}")
            raise Exception(f"FFmpeg failed: {result.stderr}")
        
        print(f"✅ Background removed video saved as: {output_video_path}")
        return output_video_path
        
    except Exception as e:
        print(f"❌ Error applying alpha mask: {str(e)}")
        raise

def process_green_screen_video(input_video_path, output_video_path="step2_alpha_mask.mp4"):
    """
    Generate alpha mask from green screen video.
    Saves the alpha mask for use in step 3 compositing.
    
    Args:
        input_video_path (str): Path to input video with green screen
        output_video_path (str): Path to save alpha mask video
    """
    
    # Check if input video exists
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")
    
    print(f"Processing green screen video: {input_video_path}")
    
    try:
        # Generate alpha mask using Replicate
        print("🎬 Generating alpha mask...")
        alpha_video = remove_background(input_video_path)
        
        # Save alpha mask for step 3
        with open(output_video_path, "wb") as video_file:
            video_file.write(alpha_video.read())
        
        print(f"✅ Alpha mask saved as: {output_video_path}")
        print("📋 This alpha mask will be used in step 3 for background compositing")
        return output_video_path
        
    except Exception as e:
        print(f"❌ Error processing video: {str(e)}")
        raise

if __name__ == "__main__":
    # Configuration
    INPUT_VIDEO = "step1.mp4"
    OUTPUT_VIDEO = "step2_alpha_mask.mp4"
    
    # Process the green screen video
    try:
        result = process_green_screen_video(
            input_video_path=INPUT_VIDEO,
            output_video_path=OUTPUT_VIDEO
        )
        print(f"🎉 Step 2 completed successfully! Generated: {result}")
        print("📋 Next: Use original video + alpha mask for compositing onto backgrounds (step 3)")
        
    except Exception as e:
        print(f"💥 Step 2 failed: {e}")
        exit(1)
