"""
Chroma-4B local TTS generator.

FlashLabs Chroma 1.0 - multimodal model for speech generation with voice cloning.
Excellent for German and other languages.

Requires:
- transformers
- torch
- HuggingFace access to FlashLabs/Chroma-4B (gated model)
"""

import tempfile
from pathlib import Path
from typing import Optional

import torch

from .base import AudioGenerator, GeneratorOutput


class ChromaTTSGenerator(AudioGenerator):
    """
    Local Chroma-4B TTS generator.
    
    Features:
    - Multimodal speech generation
    - Voice cloning from reference audio
    - Excellent German/multilingual support
    - Real-time capable
    
    Requirements:
    - ~10GB VRAM
    - HuggingFace access to gated model
    """
    
    def __init__(
        self,
        model_id: str = "FlashLabs/Chroma-4B",
        device: str = "auto",
    ):
        self._model_id = model_id
        self._model = None
        self._processor = None
        
        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        else:
            self._device = device
    
    @property
    def name(self) -> str:
        return "chroma-4b"
    
    @property
    def supports_voice_cloning(self) -> bool:
        return True
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return
        
        print(f"🔄 Loading Chroma-4B model...")
        print(f"   Device: {self._device}")
        
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            trust_remote_code=True,
            device_map="auto" if self._device == "cuda" else None,
            torch_dtype=torch.bfloat16 if self._device != "mps" else torch.float32,
        )
        
        if self._device != "cuda":
            self._model = self._model.to(self._device)
        
        self._processor = AutoProcessor.from_pretrained(
            self._model_id,
            trust_remote_code=True,
        )
        
        print(f"✅ Chroma-4B loaded")
    
    def generate_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "en",
        emotion: Optional[str] = None,
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> GeneratorOutput:
        """
        Generate speech from text.
        
        Args:
            text: Text to speak
            voice_id: Not used (use reference_audio instead)
            language: Language code (en, de, etc.)
            emotion: Optional emotion/style description
            reference_audio: Path to reference audio for voice cloning
            reference_text: Text spoken in reference audio (helps with cloning)
            output_path: Optional output path
            **kwargs: Additional parameters (max_new_tokens, temperature, top_p)
        
        Returns:
            GeneratorOutput with path to generated audio
        """
        import scipy.io.wavfile as wavfile
        
        self._load_model()
        
        # Build system prompt
        system_prompt = (
            "You are Chroma, an advanced virtual human created by FlashLabs. "
            "You possess the ability to understand auditory inputs and generate both text and speech."
        )
        
        if emotion:
            system_prompt += f" Speak with a {emotion} tone."
        
        if language == "de":
            system_prompt += " Respond in German with native German pronunciation."
        elif language != "en":
            system_prompt += f" Respond in {language}."
        
        # Build conversation
        conversation = [[
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Please say: {text}"}],
            },
        ]]
        
        # Prepare reference audio for voice cloning
        prompt_audio = []
        prompt_text = []
        
        if reference_audio:
            prompt_audio = [reference_audio]
            if reference_text:
                prompt_text = [reference_text]
            else:
                prompt_text = [text[:50]]  # Use part of target text as fallback
        
        # Process inputs
        inputs = self._processor(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
            prompt_audio=prompt_audio if prompt_audio else None,
            prompt_text=prompt_text if prompt_text else None,
        )
        
        # Move to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        # Generate
        print(f"🔊 Generating speech with Chroma-4B")
        print(f"   Text: {text[:50]}...")
        if reference_audio:
            print(f"   Voice cloning from: {reference_audio}")
        
        max_new_tokens = kwargs.get("max_new_tokens", 200)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        
        output = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
        )
        
        # Decode audio
        audio_values = self._model.codec_model.decode(
            output.permute(0, 2, 1)
        ).audio_values
        
        # Save output
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")
        
        audio_data = audio_values[0].cpu().detach().numpy()
        wavfile.write(output_path, 24000, audio_data)
        
        # Estimate duration
        duration = len(audio_data) / 24000
        
        print(f"✅ Speech saved: {output_path} ({duration:.1f}s)")
        
        return GeneratorOutput(
            path=output_path,
            duration_s=duration,
        )
    
    def clone_voice(
        self,
        audio_sample: str,
        voice_name: str,
        **kwargs,
    ) -> str:
        """
        Chroma uses reference audio directly, no separate cloning step needed.
        
        Returns the audio sample path as the "voice_id" for use in generate_speech.
        """
        # Verify file exists
        if not Path(audio_sample).exists():
            raise FileNotFoundError(f"Audio sample not found: {audio_sample}")
        
        print(f"🎤 Voice reference registered: {voice_name}")
        print(f"   Use reference_audio='{audio_sample}' in generate_speech()")
        
        # Return the path as the "voice_id"
        return audio_sample


def check_chroma_availability() -> dict:
    """Check if Chroma-4B can run on this system."""
    result = {
        "available": False,
        "device": None,
        "vram_gb": None,
        "recommendations": [],
    }
    
    if torch.cuda.is_available():
        result["device"] = "cuda"
        result["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if result["vram_gb"] >= 10:
            result["available"] = True
            result["recommendations"].append("Chroma-4B ready (~10GB VRAM)")
        else:
            result["recommendations"].append(f"Insufficient VRAM ({result['vram_gb']:.1f}GB). Need 10GB+")
    
    elif torch.backends.mps.is_available():
        result["device"] = "mps"
        result["available"] = True
        result["recommendations"].append("Apple Silicon - experimental support")
    
    else:
        result["recommendations"].append("No GPU available")
        result["recommendations"].append("Use Bark via Replicate API instead")
    
    # Check HuggingFace access
    result["recommendations"].append("Requires HuggingFace access to FlashLabs/Chroma-4B (gated model)")
    
    return result
