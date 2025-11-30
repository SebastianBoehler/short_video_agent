import cv2

def extract_last_frame(video_path, output_frame_path):
    """
    Extract the last frame from a video file and save it as an image.
    
    Args:
        video_path (str): Path to the input video file
        output_frame_path (str): Path to save the extracted frame as image
    
    Returns:
        str: Path to the saved frame image
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError(f"Video file has no frames: {video_path}")
        
        # Seek to the last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        
        # Read the last frame
        ret, frame = cap.read()
        
        if not ret:
            raise ValueError(f"Could not read the last frame from: {video_path}")
        
        # Save the frame as an image
        cv2.imwrite(output_frame_path, frame)
        
        # Release the video capture
        cap.release()
        
        print(f"✅ Last frame extracted and saved as: {output_frame_path}")
        return output_frame_path
        
    except Exception as e:
        print(f"❌ Error extracting last frame: {str(e)}")
        raise
