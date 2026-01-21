"""
Replicate API-based generators.

Implements video, image, and audio generation via Replicate API.
"""

import os
import re
import time
import base64
import mimetypes
import tempfile
from pathlib import Path
from typing import Optional, Union

import replicate
from dotenv import load_dotenv

from .base import (
    VideoGenerator,
    ImageGenerator,
    AudioGenerator,
    MattingGenerator,
    CaptionGenerator,
    GeneratorOutput,
)
from ..config.models import ModelRegistry, ModelType, ModelBackend

load_dotenv()


def _prepare_file_input(path: Union[str, Path]) -> str:
    """Prepare a file for upload to Replicate as a data URI."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        ext = path.suffix.lower()
        mime_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp',
            '.mp4': 'video/mp4',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
        }
        mime_type = mime_map.get(ext, 'application/octet-stream')
    
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    
    return f"data:{mime_type};base64,{data}"


def _retry_call(func, max_retries=3, delay=1):
    """Retry a Replicate API call with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_str = str(e)
            if "status: 429" in error_str and "rate limit resets in" in error_str:
                reset_match = re.search(r'resets in ~(\d+)s', error_str)
                if reset_match:
                    wait_time = int(reset_match.group(1)) + 2
                    print(f"⏳ Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            if attempt == max_retries - 1:
                raise
            wait = delay * (2 ** attempt)
            print(f"⚠️ Attempt {attempt + 1} failed, retrying in {wait}s...")
            time.sleep(wait)


def _save_output(output, path: Union[str, Path]) -> str:
    """Save Replicate output to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if hasattr(output, 'read'):
        with open(path, 'wb') as f:
            f.write(output.read())
    elif isinstance(output, bytes):
        with open(path, 'wb') as f:
            f.write(output)
    elif isinstance(output, str) and output.startswith(('http://', 'https://')):
        import requests
        response = requests.get(output)
        response.raise_for_status()
        with open(path, 'wb') as f:
            f.write(response.content)
    else:
        raise ValueError(f"Unknown output type: {type(output)}")
    
    return str(path)


class ReplicateVideoGenerator(VideoGenerator):
    """Replicate-based video generator."""
    
    def __init__(self, model_name: str = "wan-2.5-i2v"):
        self._model_name = model_name
        self._config = ModelRegistry.get(model_name)
        self._client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))
    
    @property
    def name(self) -> str:
        return self._model_name
    
    @property
    def supports_start_frame(self) -> bool:
        return self._config.supports_start_frame
    
    @property
    def supports_end_frame(self) -> bool:
        return self._config.supports_end_frame
    
    @property
    def supports_audio(self) -> bool:
        return self._config.supports_audio
    
    def generate(
        self,
        prompt: str,
        duration: int = 8,
        width: int = 768,
        height: int = 1280,
        start_image: Optional[str] = None,
        end_image: Optional[str] = None,
        audio_input: Optional[str] = None,
        generate_audio: bool = False,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> GeneratorOutput:
        """Generate video via Replicate API."""
        
        # Determine aspect ratio
        if width > height:
            aspect_ratio = "16:9"
        elif height > width:
            aspect_ratio = "9:16"
        else:
            aspect_ratio = "1:1"
        
        # Clamp duration for specific models
        if self._model_name.startswith("wan"):
            duration = 5 if duration <= 7 else 10
        elif self._model_name.startswith("veo"):
            if duration <= 5:
                duration = 4
            elif duration <= 7:
                duration = 6
            else:
                duration = 8
        
        def call_api():
            input_data = {
                **self._config.default_params,
                "prompt": prompt,
                "duration": duration,
            }
            
            # Handle start image
            if start_image:
                if start_image.startswith(('http://', 'https://', 'data:')):
                    input_data["image"] = start_image
                else:
                    input_data["image"] = _prepare_file_input(start_image)
            
            # Handle end image
            if end_image and self.supports_end_frame:
                if end_image.startswith(('http://', 'https://', 'data:')):
                    input_data["last_frame"] = end_image
                else:
                    input_data["last_frame"] = _prepare_file_input(end_image)
            
            # Model-specific parameters
            if self._model_name.startswith("veo"):
                input_data["aspect_ratio"] = aspect_ratio
                input_data["resolution"] = kwargs.get("resolution", "720p")
                if self.supports_audio:
                    input_data["generate_audio"] = generate_audio
            elif self._model_name.startswith("wan") and "t2v" in self._model_name:
                size_map = {"9:16": "720*1280", "16:9": "1280*720", "1:1": "720*720"}
                input_data["size"] = size_map.get(aspect_ratio, "720*1280")
            
            print(f"🎬 Generating video with {self._model_name}")
            print(f"   Prompt: {prompt[:80]}...")
            
            return self._client.run(self._config.model_id, input=input_data)
        
        output = _retry_call(call_api)
        
        # Save output
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".mp4")
        
        _save_output(output, output_path)
        
        return GeneratorOutput(
            path=output_path,
            duration_s=duration,
            width=width,
            height=height,
            fps=24,
        )


class ReplicateImageGenerator(ImageGenerator):
    """Replicate-based image generator with multi-image support."""
    
    def __init__(self, model_name: str = "seedream-4.5"):
        self._model_name = model_name
        self._config = ModelRegistry.get(model_name)
        self._client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))
    
    @property
    def name(self) -> str:
        return self._model_name
    
    @property
    def supports_multi_image(self) -> bool:
        return self._config.supports_multi_image
    
    @property
    def max_reference_images(self) -> int:
        return self._config.max_reference_images
    
    def generate(
        self,
        prompt: str,
        width: int = 768,
        height: int = 1280,
        reference_images: Optional[list[str]] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> GeneratorOutput:
        """Generate image via Replicate API with multi-image support."""
        
        # Determine aspect ratio
        if width > height:
            aspect_ratio = "16:9"
        elif height > width:
            aspect_ratio = "9:16"
        else:
            aspect_ratio = "1:1"
        
        def call_api():
            input_data = {
                **self._config.default_params,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
            }
            
            # Prepare reference images
            if reference_images:
                refs = []
                for ref in reference_images[:self.max_reference_images]:
                    if ref.startswith(('http://', 'https://', 'data:')):
                        refs.append(ref)
                    else:
                        refs.append(_prepare_file_input(ref))
                
                # Model-specific reference image parameter
                if self._model_name in ("nano-banana-pro", "seedream-4.5"):
                    input_data["image_input"] = refs
                elif self._model_name == "flux-2-pro":
                    input_data["input_images"] = refs
                
                print(f"   Using {len(refs)} reference image(s)")
            
            print(f"🖼️ Generating image with {self._model_name}")
            print(f"   Prompt: {prompt[:80]}...")
            
            return self._client.run(self._config.model_id, input=input_data)
        
        output = _retry_call(call_api)
        
        # Save output
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".png")
        
        _save_output(output, output_path)
        
        return GeneratorOutput(
            path=output_path,
            width=width,
            height=height,
        )


class ReplicateMattingGenerator(MattingGenerator):
    """Replicate-based video matting/background removal."""
    
    def __init__(self, model_name: str = "robust-video-matting"):
        self._model_name = model_name
        self._config = ModelRegistry.get(model_name)
        self._client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))
    
    @property
    def name(self) -> str:
        return self._model_name
    
    def remove_background(
        self,
        video_path: str,
        output_type: str = "alpha-mask",
        output_path: Optional[str] = None,
        **kwargs,
    ) -> GeneratorOutput:
        """Remove background from video."""
        
        def call_api():
            if video_path.startswith(('http://', 'https://')):
                video_input = video_path
            else:
                video_input = _prepare_file_input(video_path)
            
            input_data = {
                **self._config.default_params,
                "input_video": video_input,
                "output_type": output_type,
            }
            
            print(f"🎭 Removing background with {self._model_name}")
            
            return self._client.run(self._config.model_id, input=input_data)
        
        output = _retry_call(call_api)
        
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".mp4")
        
        _save_output(output, output_path)
        
        return GeneratorOutput(path=output_path)


class ReplicateCaptionGenerator(CaptionGenerator):
    """Replicate-based video captioning."""
    
    def __init__(self, model_name: str = "tiktok-captions"):
        self._model_name = model_name
        self._config = ModelRegistry.get(model_name)
        self._client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))
    
    @property
    def name(self) -> str:
        return self._model_name
    
    def add_captions(
        self,
        video_path: str,
        language: str = "auto",
        highlight_color: str = "#FFFFFF",
        output_path: Optional[str] = None,
        **kwargs,
    ) -> GeneratorOutput:
        """Add captions to video."""
        
        def call_api():
            # Upload video for large files
            if video_path.startswith(('http://', 'https://')):
                video_url = video_path
            else:
                # Upload to Replicate's file hosting
                with open(video_path, 'rb') as f:
                    video_url = replicate.files.upload(f)
            
            input_data = {
                **self._config.default_params,
                "video": video_url,
                "language": language,
                "highlight_color": highlight_color,
            }
            
            print(f"📝 Adding captions with {self._model_name}")
            
            return self._client.run(self._config.model_id, input=input_data)
        
        output = _retry_call(call_api)
        
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".mp4")
        
        _save_output(output, output_path)
        
        return GeneratorOutput(path=output_path)


class ReplicateAudioGenerator(AudioGenerator):
    """Replicate-based TTS and voice cloning."""
    
    def __init__(self, model_name: str = "speech-02-hd"):
        self._model_name = model_name
        self._config = ModelRegistry.get(model_name)
        self._client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))
    
    @property
    def name(self) -> str:
        return self._model_name
    
    @property
    def supports_voice_cloning(self) -> bool:
        return self._config.model_type == ModelType.VOICE_CLONING
    
    def generate_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "en",
        emotion: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> GeneratorOutput:
        """Generate speech from text."""
        
        def call_api():
            input_data = {
                **self._config.default_params,
                "text": text,
            }
            
            if voice_id:
                input_data["voice_id"] = voice_id
            
            print(f"🔊 Generating speech with {self._model_name}")
            print(f"   Text: {text[:50]}...")
            
            return self._client.run(self._config.model_id, input=input_data)
        
        output = _retry_call(call_api)
        
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".mp3")
        
        _save_output(output, output_path)
        
        return GeneratorOutput(path=output_path)
    
    def clone_voice(
        self,
        audio_sample: str,
        voice_name: str,
        **kwargs,
    ) -> str:
        """Clone a voice from audio sample."""
        voice_clone_config = ModelRegistry.get("voice-cloning")
        
        def call_api():
            if audio_sample.startswith(('http://', 'https://')):
                audio_input = audio_sample
            else:
                audio_input = _prepare_file_input(audio_sample)
            
            input_data = {
                "audio": audio_input,
                "voice_name": voice_name,
            }
            
            print(f"🎤 Cloning voice: {voice_name}")
            
            return self._client.run(voice_clone_config.model_id, input=input_data)
        
        result = _retry_call(call_api)
        
        # Return voice ID from result
        if isinstance(result, dict) and "voice_id" in result:
            return result["voice_id"]
        return str(result)
