import os
import replicate
from dotenv import load_dotenv
from replicate_client import generate_image, generate_video

load_dotenv()

def generate_scene_start_image(prompt, output_image="step4_start_scene.png"):
    """
    Generate a start scene image using Flux model.
    
    Args:
        prompt (str): Text prompt for scene generation
        output_image (str): Path to save the generated image
        
    Returns:
        str: Path to the saved image
    """
    
    print(f"🎨 Generating start scene image...")
    print(f"Prompt: {prompt}")
    
    try:
        # Generate image using the existing replicate client function
        image_result = generate_image(prompt)
        
        # Save the image
        with open(output_image, "wb") as image_file:
            image_file.write(image_result.read())
        
        print(f"✅ Start scene image saved: {output_image}")
        return output_image
        
    except Exception as e:
        print(f"❌ Error generating start image: {str(e)}")
        raise

def generate_scene_video_from_image(image_path, video_prompt, duration=8, output_video="step4_scene.mp4"):
    """
    Generate a video scene using the start image as reference frame.
    
    Args:
        image_path (str): Path to the start image
        video_prompt (str): Text prompt for video generation
        duration (int): Video duration in seconds
        output_video (str): Path to save the generated video
        
    Returns:
        str: Path to the saved video
    """
    
    print(f"🎬 Generating scene video from start image...")
    print(f"Video prompt: {video_prompt}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Start image not found: {image_path}")
    
    try:
        # Open image for Replicate upload
        with open(image_path, "rb") as image_file:
            # Generate video using Veo model with start image
            output = replicate.run(
                "google/veo-3.1-fast",
                input={
                    "image": image_file,
                    "prompt": video_prompt,
                    "duration": duration,
                    "resolution": "720p",
                    "aspect_ratio": "9:16",
                    "generate_audio": False  # We'll add audio later
                }
            )
        
        # Save the generated video
        with open(output_video, "wb") as video_file:
            video_file.write(output.read())
        
        print(f"✅ Scene video saved: {output_video}")
        return output_video
        
    except Exception as e:
        print(f"❌ Error generating scene video: {str(e)}")
        raise

def process_scene_generation(scene_prompt, video_prompt, duration=8, prefix="step4"):
    """
    Complete scene generation: create start image then generate video from it.
    
    Args:
        scene_prompt (str): Prompt for generating the start scene image
        video_prompt (str): Prompt for video generation/motion
        duration (int): Video duration in seconds
        prefix (str): Prefix for output files
        
    Returns:
        dict: Paths to generated image and video
    """
    
    print(f"🎬 Starting scene generation process...")
    
    try:
        # Step 1: Generate start scene image
        start_image = generate_scene_start_image(
            prompt=scene_prompt,
            output_image=f"{prefix}_start_scene.png"
        )
        
        # Step 2: Generate video from start image
        scene_video = generate_scene_video_from_image(
            image_path=start_image,
            video_prompt=video_prompt,
            duration=duration,
            output_video=f"{prefix}_scene.mp4"
        )
        
        return {
            "start_image": start_image,
            "scene_video": scene_video
        }
        
    except Exception as e:
        print(f"❌ Error in scene generation process: {str(e)}")
        raise

if __name__ == "__main__":
    # Configuration
    SCENE_PROMPT = (
        "Beautiful modern living room with soft natural lighting, "
        "cozy sofa, plants, warm atmosphere, vertical smartphone shot, "
        "photorealistic, high quality, interior design"
    )
    
    VIDEO_PROMPT = (
        "Gentle camera movement showing the living room, "
        "soft lighting changes, peaceful atmosphere, "
        "slow pan across the room, vertical 9:16 format"
    )
    
    DURATION = 6  # seconds
    
    # Generate scene
    try:
        result = process_scene_generation(
            scene_prompt=SCENE_PROMPT,
            video_prompt=VIDEO_PROMPT,
            duration=DURATION,
            prefix="step4"
        )
        
        print(f"🎉 Step 4 completed successfully!")
        print(f"📸 Start image: {result['start_image']}")
        print(f"🎬 Scene video: {result['scene_video']}")
        print("📋 Next: Stitch scenes together into final video (step 5)")
        
    except Exception as e:
        print(f"💥 Step 4 failed: {e}")
        exit(1)
