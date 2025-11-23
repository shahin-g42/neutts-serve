"""
Settings management for the NeuTTS-Air TTS service using Pydantic Settings.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # Model paths
    model_checkpoint_path: str = Field(
        default="neuphonic/neutts-air",
        description="Path to the NeuTTS-Air model checkpoint directory"
    )
    codec_repo: str = Field(
        default="neuphonic/neucodec",
        description="Path to the NeuCodec model repository"
    )

    # Device configuration
    backbone_device: str = Field(
        default="cuda",
        description="Device for backbone model (cpu/cuda)"
    )
    codec_device: str = Field(
        default="cuda",
        description="Device for codec model (cpu/cuda)"
    )

    # Backend configuration
    use_vllm: bool = Field(
        default=True,
        description="Use vLLM for inference acceleration"
    )
    use_gguf: bool = Field(
        default=False,
        description="Use GGUF quantized model (requires llama-cpp-python)"
    )
    use_async_engine: bool = Field(
        default=True,
        description="Use AsyncLLMEngine for streaming (vLLM only)"
    )

    # Server configuration
    host: str = Field(
        default="127.0.0.1",
        description="Host address for the server"
    )
    port: int = Field(
        default=8002,
        description="Port number for the server"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )

    # Model configuration
    max_context: int = Field(
        default=2048,
        description="Maximum context length for the model"
    )
    gpu_memory_utilization: float = Field(
        default=0.7,
        description="GPU memory utilization for vLLM (0.0-1.0)"
    )
    seed: int = Field(
        default=42,
        description="Random seed for generation"
    )

    # Synthesis parameters
    temperature: float = Field(
        default=1.0,
        description="Sampling temperature for synthesis"
    )
    max_tokens: int = Field(
        default=2048,
        description="Maximum number of tokens to generate"
    )
    min_tokens: int = Field(
        default=50,
        description="Minimum number of tokens to generate"
    )
    top_p: float = Field(
        default=1.0,
        description="Top-p (nucleus) sampling parameter"
    )
    top_k: int = Field(
        default=50,
        description="Top-k sampling parameter"
    )

    # Streaming parameters
    streaming_frames_per_chunk: int = Field(
        default=25,
        description="Number of tokens per audio chunk for streaming"
    )
    streaming_overlap_frames: int = Field(
        default=1,
        description="Overlap frames between chunks"
    )
    streaming_lookforward: int = Field(
        default=5,
        description="Future context tokens for streaming"
    )
    streaming_lookback: int = Field(
        default=50,
        description="Past context tokens for streaming"
    )

    # Audio parameters
    sample_rate: int = Field(
        default=24000,
        description="Output audio sample rate"
    )
    hop_length: int = Field(
        default=480,
        description="Codec hop length in samples"
    )

    # Phonemizer configuration
    phonemizer_language: str = Field(
        default="en-us",
        description="Language for phonemizer (e.g., en-us, en-gb, es, fr, de)"
    )
    enable_phonemization: bool = Field(
        default=True,
        description="Enable phonemization (set to False to pass text directly without phoneme conversion)"
    )
    phonemizer_preserve_punctuation: bool = Field(
        default=True,
        description="Preserve punctuation in phonemization"
    )
    phonemizer_with_stress: bool = Field(
        default=True,
        description="Include stress markers in phonemization"
    )
    phonemizer_espeak_path: Optional[str] = Field(
        default=None,
        description="Path to espeak-ng binary (for macOS/custom installs)"
    )
    phonemizer_espeak_library: Optional[str] = Field(
        default=None,
        description="Path to espeak-ng library (for macOS/custom installs)"
    )

    use_template_v2: bool = Field(
        default=False,
        description="Use template v2"
    )

    # Logging configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    log_file: str = Field(
        default="logs/tts_service.log",
        description="Path to the log file"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


# Global settings instance
settings = Settings()
