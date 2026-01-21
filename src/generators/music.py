"""
Music generation via Replicate API.

Supports:
- MusicGen by Meta - text to music
- MusicGen Stereo with chord control
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Union

import replicate
from dotenv import load_dotenv

from .base import GeneratorOutput
from ..config.models import ModelRegistry

load_dotenv()


class MusicGenerator:
    """
    Music generation via Replicate API.
    
    Supports text-to-music generation with optional melody conditioning.
    """
    
    def __init__(self, model_name: str = "musicgen"):
        self._model_name = model_name
        self._config = ModelRegistry.get(model_name)
        self._client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))
    
    @property
    def name(self) -> str:
        return self._model_name
    
    def generate(
        self,
        prompt: str,
        duration: int = 8,
        melody_audio: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> GeneratorOutput:
        """
        Generate music from text prompt.
        
        Args:
            prompt: Text description of the music (e.g., "upbeat electronic dance music")
            duration: Duration in seconds (max ~30s for MusicGen)
            melody_audio: Optional path to audio file for melody conditioning
            output_path: Optional output path
            **kwargs: Additional model-specific parameters
        
        Returns:
            GeneratorOutput with path to generated audio
        """
        import base64
        import mimetypes
        import requests
        
        def _prepare_audio(path: str) -> str:
            """Prepare audio file for upload."""
            p = Path(path)
            mime_type, _ = mimetypes.guess_type(str(p))
            if mime_type is None:
                mime_type = 'audio/mpeg'
            with open(p, 'rb') as f:
                data = base64.b64encode(f.read()).decode('utf-8')
            return f"data:{mime_type};base64,{data}"
        
        input_data = {
            **self._config.default_params,
            "prompt": prompt,
            "duration": min(duration, 30),  # MusicGen max is ~30s
        }
        
        # Add melody conditioning if provided
        if melody_audio:
            if melody_audio.startswith(('http://', 'https://')):
                input_data["input_audio"] = melody_audio
            else:
                input_data["input_audio"] = _prepare_audio(melody_audio)
        
        # Override with kwargs
        for key, value in kwargs.items():
            if key in ["model_version", "temperature", "top_k", "top_p", 
                       "classifier_free_guidance", "output_format"]:
                input_data[key] = value
        
        print(f"🎵 Generating music with {self._model_name}")
        print(f"   Prompt: {prompt[:80]}...")
        print(f"   Duration: {duration}s")
        
        output = self._client.run(self._config.model_id, input=input_data)
        
        # Save output
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".mp3")
        
        # Handle output (could be URL or file object)
        if hasattr(output, 'read'):
            with open(output_path, 'wb') as f:
                f.write(output.read())
        elif isinstance(output, str) and output.startswith(('http://', 'https://')):
            response = requests.get(output)
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                f.write(response.content)
        else:
            raise ValueError(f"Unknown output type: {type(output)}")
        
        print(f"✅ Music saved: {output_path}")
        
        return GeneratorOutput(
            path=output_path,
            duration_s=duration,
        )
    
    def generate_with_chords(
        self,
        prompt: str,
        chords: str,
        bpm: int = 120,
        duration: int = 8,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> GeneratorOutput:
        """
        Generate music with chord progression control.
        
        Requires musicgen-stereo-chord model.
        
        Args:
            prompt: Text description of the music
            chords: Chord progression (e.g., "C G Am F")
            bpm: Beats per minute
            duration: Duration in seconds
            output_path: Optional output path
            **kwargs: Additional parameters
        
        Returns:
            GeneratorOutput with path to generated audio
        """
        if self._model_name != "musicgen-stereo-chord":
            # Switch to chord model
            self._config = ModelRegistry.get("musicgen-stereo-chord")
            self._model_name = "musicgen-stereo-chord"
        
        input_data = {
            **self._config.default_params,
            "prompt": prompt,
            "chords": chords,
            "bpm": bpm,
            "duration": min(duration, 30),
        }
        
        print(f"🎵 Generating music with chord control")
        print(f"   Prompt: {prompt[:60]}...")
        print(f"   Chords: {chords}, BPM: {bpm}")
        
        output = self._client.run(self._config.model_id, input=input_data)
        
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".mp3")
        
        import requests
        if hasattr(output, 'read'):
            with open(output_path, 'wb') as f:
                f.write(output.read())
        elif isinstance(output, str) and output.startswith(('http://', 'https://')):
            response = requests.get(output)
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                f.write(response.content)
        
        print(f"✅ Music saved: {output_path}")
        
        return GeneratorOutput(
            path=output_path,
            duration_s=duration,
        )
