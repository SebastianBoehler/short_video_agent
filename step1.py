import replicate
import os
from dotenv import load_dotenv

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
    """
    
    # Check if speaker image exists
    if not os.path.exists(speaker_image_path):
        raise FileNotFoundError(f"Speaker image not found: {speaker_image_path}")
    
    print(f"Generating speaker video using {speaker_image_path}...")
    print(f"Prompt: {prompt}")
    
    try:
        # Open and read the speaker image
        with open(speaker_image_path, "rb") as image_file:
            # Generate video using Veo model
            output = replicate.run(
                "google/veo-3.1-fast",
                input={
                    "image": image_file,
                    "prompt": prompt,
                    "duration": duration,
                    "resolution": "720p",
                    "aspect_ratio": "9:16",
                    "generate_audio": False  # We'll add audio later
                }
            )
        
        # Save the generated video
        with open(output_filename, "wb") as video_file:
            video_file.write(output.read())
        
        print(f"✅ Speaker video saved as: {output_filename}")
        return output_filename
        
    except Exception as e:
        print(f"❌ Error generating video: {str(e)}")
        raise

if __name__ == "__main__":
    # Configuration
    SPEAKER_IMAGE = "speakers/speaker1.png"
    OUTPUT_VIDEO = "step1.mp4"
    
    # Prompt for speaker video with green screen background
    SPEAKER_PROMPT = (
        "Professional speaker talking enthusiastically to camera, natural gestures and facial expressions, "
        "standing in front of bright green chroma key background, studio lighting, vertical 9:16 format, "
        "engaging presentation style, clear speech, friendly demeanor, high quality video"
    )
    
    # Generate the speaker video
    try:
        result = generate_speaker_video(
            speaker_image_path=SPEAKER_IMAGE,
            prompt=SPEAKER_PROMPT,
            output_filename=OUTPUT_VIDEO,
            duration=8
        )
        print(f"🎉 Step 1 completed successfully! Generated: {result}")
        
    except Exception as e:
        print(f"💥 Step 1 failed: {e}")
        exit(1)
