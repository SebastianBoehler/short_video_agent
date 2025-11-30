import os
import cv2
from dotenv import load_dotenv
from replicate_client import generate_video_with_image
from utils import extract_last_frame

# Load environment variables from .env file
load_dotenv()

def generate_speaker_video(speaker_image_path, prompt, output_filename="step1.mp4", duration=8):
    """
    Generate a speaker video using Replicate's Veo model with a speaker image input.
    
    Args:
        speaker_image_path (str): Path to the speaker image file
        prompt (str): Text prompt describing the speaker action/scene
        output_filename (str): Output video filename
        duration (int): Video duration in seconds
    
    Returns:
        tuple: (video_filename, frame_filename) - paths to generated video and extracted frame
    """
    
    # Check if speaker image exists
    if not os.path.exists(speaker_image_path):
        raise FileNotFoundError(f"Speaker image not found: {speaker_image_path}")
    
    print(f"Generating speaker video using {speaker_image_path}...")
    print(f"Prompt: {prompt}")
    
    try:
        # Generate video using the replicate_client module
        output = generate_video_with_image(
            prompt=prompt,
            image_path=speaker_image_path,
            duration=duration,
            generate_audio=False
        )
        
        # Save the generated video
        with open(output_filename, "wb") as video_file:
            video_file.write(output.read())
        
        print(f"✅ Speaker video saved as: {output_filename}")
        
        # Extract the last frame for use as start frame in next scene
        frame_filename = output_filename.replace('.mp4', '_final_frame.png')
        extract_last_frame(output_filename, frame_filename)
        
        return output_filename, frame_filename
        
    except Exception as e:
        print(f"❌ Error generating video: {str(e)}")
        raise

if __name__ == "__main__":
    # Configuration
    SPEAKER_IMAGE = "speakers/speaker3.png"
    OUTPUT_VIDEO = "step1.mp4"
    
    # Prompt for speaker video with green screen background
    SPEAKER_PROMPT = (
        "Professional speaker talking enthusiastically to camera, natural gestures and facial expressions, "
        "standing in front of bright green chroma key background, studio lighting, vertical 9:16 format, "
        "engaging presentation style, clear speech, friendly demeanor, high quality video"
    )
    
    # Generate the speaker video
    try:
        video_result, frame_result = generate_speaker_video(
            speaker_image_path=SPEAKER_IMAGE,
            prompt=SPEAKER_PROMPT,
            output_filename=OUTPUT_VIDEO,
            duration=8
        )
        print(f"🎉 Step 1 completed successfully!")
        print(f"📹 Generated video: {video_result}")
        print(f"🖼️ Extracted final frame: {frame_result}")
        
    except Exception as e:
        print(f"💥 Step 1 failed: {e}")
        exit(1)
