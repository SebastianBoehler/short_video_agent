"""Video utility functions."""

import subprocess
from pathlib import Path
from typing import Optional


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


def extract_last_frame(video_path: str, output_path: str) -> str:
    """Extract the last frame from a video."""
    props = get_video_properties(video_path)
    duration = props['duration']
    
    # Seek to near the end
    seek_time = max(0, duration - 0.1)
    
    cmd = [
        'ffmpeg',
        '-ss', str(seek_time),
        '-i', video_path,
        '-vframes', '1',
        '-q:v', '2',
        '-y',
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"Failed to extract frame: {result.stderr[:200]}")
    
    return output_path


def extract_first_frame(video_path: str, output_path: str) -> str:
    """Extract the first frame from a video."""
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vframes', '1',
        '-q:v', '2',
        '-y',
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"Failed to extract frame: {result.stderr[:200]}")
    
    return output_path


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    return get_video_properties(video_path)['duration']


def resize_video(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
) -> str:
    """Resize video to specified dimensions."""
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-vf', f'scale={width}:{height}',
        '-c:a', 'copy',
        '-y',
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"Failed to resize video: {result.stderr[:200]}")
    
    return output_path
