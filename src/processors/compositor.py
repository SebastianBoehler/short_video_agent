"""Video compositing processor."""

import subprocess
from pathlib import Path
from typing import Optional

from ..utils.video import get_video_properties


class VideoCompositor:
    """
    Video compositing processor.
    
    Composites speaker video onto background using alpha mask.
    """
    
    POSITIONS = {
        "top_left": lambda bg_w, bg_h, ov_w, ov_h, pad: (pad, pad),
        "top_center": lambda bg_w, bg_h, ov_w, ov_h, pad: ((bg_w - ov_w) // 2, pad),
        "top_right": lambda bg_w, bg_h, ov_w, ov_h, pad: (bg_w - ov_w - pad, pad),
        "center_left": lambda bg_w, bg_h, ov_w, ov_h, pad: (pad, (bg_h - ov_h) // 2),
        "center": lambda bg_w, bg_h, ov_w, ov_h, pad: ((bg_w - ov_w) // 2, (bg_h - ov_h) // 2),
        "center_right": lambda bg_w, bg_h, ov_w, ov_h, pad: (bg_w - ov_w - pad, (bg_h - ov_h) // 2),
        "bottom_left": lambda bg_w, bg_h, ov_w, ov_h, pad: (pad, bg_h - ov_h - pad),
        "bottom_center": lambda bg_w, bg_h, ov_w, ov_h, pad: ((bg_w - ov_w) // 2, bg_h - ov_h - pad),
        "bottom_right": lambda bg_w, bg_h, ov_w, ov_h, pad: (bg_w - ov_w - pad, bg_h - ov_h - pad),
    }
    
    def composite(
        self,
        speaker_video: str,
        alpha_mask: str,
        background: str,
        output_path: str,
        scale: float = 0.35,
        position: str = "bottom_right",
        padding: int = 20,
        mix_background_audio: bool = False,
    ) -> str:
        """
        Composite speaker onto background using alpha mask.
        
        Args:
            speaker_video: Path to speaker video
            alpha_mask: Path to alpha mask video
            background: Path to background image or video
            output_path: Output path
            scale: Scale factor for speaker (0.35 = 35% of frame width)
            position: Position string (top_left, bottom_right, center, etc.)
            padding: Padding from edges in pixels
            mix_background_audio: Whether to mix background audio with speaker
        
        Returns:
            Path to composited video
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
        pos_func = self.POSITIONS.get(position, self.POSITIONS["bottom_right"])
        pos_x, pos_y = pos_func(bg_width, bg_height, speaker_width, speaker_height, padding)
        
        # Check if background is image or video
        bg_ext = Path(background).suffix.lower()
        is_image_bg = bg_ext in ['.png', '.jpg', '.jpeg', '.webp']
        
        # Build ffmpeg command
        if is_image_bg:
            cmd = self._build_image_bg_command(
                speaker_video, alpha_mask, background, output_path,
                bg_width, bg_height, speaker_width, speaker_height,
                pos_x, pos_y, mix_background_audio
            )
        else:
            cmd = self._build_video_bg_command(
                speaker_video, alpha_mask, background, output_path,
                speaker_width, speaker_height, pos_x, pos_y,
                mix_background_audio
            )
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"⚠️ Compositing error: {result.stderr[:500]}")
            raise Exception("Compositing failed")
        
        print(f"✅ Composited video saved: {output_path}")
        return output_path
    
    def _build_image_bg_command(
        self,
        speaker_video: str,
        alpha_mask: str,
        background: str,
        output_path: str,
        bg_width: int,
        bg_height: int,
        speaker_width: int,
        speaker_height: int,
        pos_x: int,
        pos_y: int,
        mix_background_audio: bool,
    ) -> list[str]:
        """Build ffmpeg command for image background."""
        filter_complex = (
            f'[0:v]scale={bg_width}:{bg_height},format=rgba[bg];'
            f'[1:v]scale={speaker_width}:{speaker_height},format=rgba[fg_scaled];'
            f'[2:v]scale={speaker_width}:{speaker_height},format=gray[alpha_scaled];'
            f'[fg_scaled][alpha_scaled]alphamerge[masked];'
            f'[bg][masked]overlay={pos_x}:{pos_y}:format=auto:shortest=1[out]'
        )
        
        return [
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
            output_path
        ]
    
    def _build_video_bg_command(
        self,
        speaker_video: str,
        alpha_mask: str,
        background: str,
        output_path: str,
        speaker_width: int,
        speaker_height: int,
        pos_x: int,
        pos_y: int,
        mix_background_audio: bool,
    ) -> list[str]:
        """Build ffmpeg command for video background."""
        if mix_background_audio:
            filter_complex = (
                f'[1:v]scale={speaker_width}:{speaker_height}[fg_scaled];'
                f'[2:v]scale={speaker_width}:{speaker_height},format=gray[alpha_scaled];'
                f'[fg_scaled][alpha_scaled]alphamerge[masked];'
                f'[0:v][masked]overlay={pos_x}:{pos_y}:format=auto:shortest=1[out];'
                f'[0:a][1:a]amix=inputs=2:weights="0.3 1.0":duration=shortest[aout]'
            )
            return [
                'ffmpeg',
                '-i', background,
                '-i', speaker_video,
                '-i', alpha_mask,
                '-filter_complex', filter_complex,
                '-map', '[out]',
                '-map', '[aout]',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-shortest',
                '-y',
                output_path
            ]
        else:
            filter_complex = (
                f'[1:v]scale={speaker_width}:{speaker_height}[fg_scaled];'
                f'[2:v]scale={speaker_width}:{speaker_height},format=gray[alpha_scaled];'
                f'[fg_scaled][alpha_scaled]alphamerge[masked];'
                f'[0:v][masked]overlay={pos_x}:{pos_y}:format=auto:shortest=1[out]'
            )
            return [
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
                output_path
            ]
