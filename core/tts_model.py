"""
TTS Model wrapper integrating NeuTTSAir for the service.
"""

import io
import time
import torch
import torchaudio
import numpy as np
from typing import Tuple, List, Iterator
from dataclasses import dataclass

from core.logging import app_logger
from core.exceptions import ModelLoadingError, SynthesisError
from core.settings import settings
from core.neutts_wrapper import NeuTTSAirWrapper
from utils.audio_converter import convert_audio_format


@dataclass
class TTSModelConfig:
    """Configuration for the TTS model."""
    model_checkpoint_path: str
    codec_repo: str
    backbone_device: str
    codec_device: str
    use_vllm: bool
    use_gguf: bool
    use_async_engine: bool
    max_context: int
    gpu_memory_utilization: float
    seed: int

    @classmethod
    def from_settings(cls):
        """Create TTSModelConfig from application settings."""
        return cls(
            model_checkpoint_path=settings.model_checkpoint_path,
            codec_repo=settings.codec_repo,
            backbone_device=settings.backbone_device,
            codec_device=settings.codec_device,
            use_vllm=settings.use_vllm,
            use_gguf=settings.use_gguf,
            use_async_engine=settings.use_async_engine,
            max_context=settings.max_context,
            gpu_memory_utilization=settings.gpu_memory_utilization,
            seed=settings.seed
        )


class TTSModel:
    """TTS model wrapper using NeuTTSAir."""
    
    def __init__(self, config: TTSModelConfig):
        self.config = config
        self.neutts_wrapper = None
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
        self._initialize_model()
        app_logger.info("TTS model initialized successfully")
    
    def _initialize_model(self) -> None:
        try:
            app_logger.info("Initializing NeuTTSAir wrapper...")
            self.neutts_wrapper = NeuTTSAirWrapper(
                backbone_repo=self.config.model_checkpoint_path,
                backbone_device=self.config.backbone_device,
                codec_repo=self.config.codec_repo,
                codec_device=self.config.codec_device,
                use_vllm=self.config.use_vllm,
                use_gguf=self.config.use_gguf,
                use_async_engine=self.config.use_async_engine,
                max_context=self.config.max_context,
                gpu_memory_utilization=self.config.gpu_memory_utilization
            )
        except Exception as e:
            raise ModelLoadingError(f"Failed to initialize TTS model: {e}")
    
    def synthesize_speech_with_tokens(
        self, text: str, speech_codes: List[int], reference_text: str = "", 
        response_format: str = "mp3", **kwargs
    ) -> Tuple[bytes, str, float]:
        """Synchronous synthesis for non-AsyncLLMEngine backends."""
        try:
            start_time = time.time()
            if not speech_codes:
                raise ValueError("Speech codes required for voice cloning")
            app_logger.info(f"Synthesis | codes={len(speech_codes)} | text={len(text)} chars | format={response_format}")
            ref_codes = np.array(speech_codes, dtype=np.int64)
            wav = self.neutts_wrapper.infer(text=text, ref_codes=ref_codes, ref_text=reference_text)
            
            # wav is [1, samples] numpy array from _decode()
            wav_tensor = torch.from_numpy(wav)
            
            # Convert to requested format (MP3 or WAV) at 24kHz
            audio_bytes, content_type = convert_audio_format(
                wav_tensor=wav_tensor,
                orig_sr=settings.sample_rate,  # 24kHz from NeuCodec
                target_format=response_format,
                target_sr=settings.sample_rate,  # Keep at 24kHz (no resample)
                bitrate="128k"
            )
            
            total_time = time.time() - start_time
            duration = wav.shape[1] / settings.sample_rate
            app_logger.info(f"Synth done | audio={duration:.2f}s | total={total_time:.2f}s | size={len(audio_bytes)/1024:.1f}KB")
            return (audio_bytes, content_type, duration)
        except Exception as e:
            app_logger.error(f"Failed to synthesize speech: {e}")
            raise SynthesisError(f"Failed to synthesize speech: {e}")
    
    async def synthesize_speech_with_tokens_async(
        self, text: str, speech_codes: List[int], reference_text: str = "", 
        response_format: str = "mp3", **kwargs
    ) -> Tuple[bytes, str, float]:
        """Async synthesis for AsyncLLMEngine backend."""
        try:
            start_time = time.time()
            if not speech_codes:
                raise ValueError("Speech codes required for voice cloning")
            app_logger.info(f"Async Synthesis | codes={len(speech_codes)} | text={len(text)} chars | format={response_format}")
            ref_codes = np.array(speech_codes, dtype=np.int64)
            wav = await self.neutts_wrapper.infer_async(text=text, ref_codes=ref_codes, ref_text=reference_text)
            
            # wav is [1, samples] numpy array from _decode()
            wav_tensor = torch.from_numpy(wav)
            
            # Convert to requested format (MP3 or WAV) at 24kHz
            audio_bytes, content_type = convert_audio_format(
                wav_tensor=wav_tensor,
                orig_sr=settings.sample_rate,  # 24kHz from NeuCodec
                target_format=response_format,
                target_sr=settings.sample_rate,  # Keep at 24kHz (no resample)
                bitrate="128k"
            )
            
            total_time = time.time() - start_time
            duration = wav.shape[1] / settings.sample_rate
            app_logger.info(f"Async Synth done | audio={duration:.2f}s | total={total_time:.2f}s | size={len(audio_bytes)/1024:.1f}KB")
            return (audio_bytes, content_type, duration)
        except Exception as e:
            app_logger.error(f"Failed to synthesize speech (async): {e}")
            raise SynthesisError(f"Failed to synthesize speech: {e}")
    
    def stream_speech_with_tokens(
        self, text: str, speech_codes: List[int], reference_text: str = "", **kwargs
    ) -> Iterator[bytes]:
        """Stream synthesis by chunking the full audio output."""
        try:
            audio_bytes, _, _ = self.synthesize_speech_with_tokens(
                text=text, speech_codes=speech_codes, reference_text=reference_text, **kwargs
            )
            chunk_size = 4096
            for i in range(0, len(audio_bytes), chunk_size):
                yield audio_bytes[i:i + chunk_size]
        except Exception as e:
            raise SynthesisError(f"Failed to stream speech: {e}")
