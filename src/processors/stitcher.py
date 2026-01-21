"""Video stitching processor."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


class VideoStitcher:
    """
    Video stitching processor.
    
    Concatenates multiple video clips into a single video.
    """
    
    def stitch(
        self,
        video_paths: list[str],
        output_path: str,
        transition: Optional[str] = None,
    ) -> str:
        """
        Stitch multiple videos together.
        
        Args:
            video_paths: List of video paths to concatenate
            output_path: Output path for final video
            transition: Optional transition type (future: fade, dissolve, etc.)
        
        Returns:
            Path to stitched video
        """
        print(f"\n{'='*60}")
        print(f"🎬 Stitching {len(video_paths)} scenes together...")
        print(f"{'='*60}")
        
        if len(video_paths) == 0:
            raise ValueError("No videos to stitch")
        
        if len(video_paths) == 1:
            # Just copy single video
            shutil.copy(video_paths[0], output_path)
            print(f"✅ Final video saved: {output_path}")
            return output_path
        
        # Create concat file for ffmpeg
        concat_file = Path(output_path).parent / "concat_list.txt"
        
        with open(concat_file, "w") as f:
            for video_path in video_paths:
                abs_path = str(Path(video_path).resolve())
                escaped_path = abs_path.replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")
        
        # Concatenate with re-encoding
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
            raise Exception("FFmpeg concat failed")
        
        # Cleanup
        os.remove(concat_file)
        
        print(f"✅ Final video saved: {output_path}")
        return output_path
    
    def stitch_with_audio(
        self,
        video_paths: list[str],
        audio_path: str,
        output_path: str,
        audio_volume: float = 0.3,
    ) -> str:
        """
        Stitch videos and add background audio track.
        
        Args:
            video_paths: List of video paths
            audio_path: Path to background audio
            output_path: Output path
            audio_volume: Volume for background audio (0.0-1.0)
        
        Returns:
            Path to final video
        """
        # First stitch videos
        temp_stitched = str(Path(output_path).parent / "temp_stitched.mp4")
        self.stitch(video_paths, temp_stitched)
        
        # Then add background audio
        cmd = [
            'ffmpeg',
            '-i', temp_stitched,
            '-i', audio_path,
            '-filter_complex',
            f'[0:a][1:a]amix=inputs=2:weights="1.0 {audio_volume}":duration=first[aout]',
            '-map', '0:v',
            '-map', '[aout]',
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-y',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Cleanup temp file
        os.remove(temp_stitched)
        
        if result.returncode != 0:
            print(f"⚠️ Audio mixing error: {result.stderr[:500]}")
            raise Exception("Audio mixing failed")
        
        print(f"✅ Final video with audio saved: {output_path}")
        return output_path
