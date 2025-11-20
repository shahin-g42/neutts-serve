"""
Settings management for the NeuTTS-Air TTS service using Pydantic Settings.
"""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""
    
    # Model paths
    model_checkpoint_path: str = Field(
        default="neuphonic/neutts-air",
        description="Path to the NeuTTS-Air model checkpoint directory",
        env="MODEL_CHECKPOINT_PATH"
    )
    codec_repo: str = Field(
        default="neuphonic/neucodec",
        description="Path to the NeuCodec model repository",
        env="CODEC_REPO"
    )
    
    # Device configuration
    backbone_device: str = Field(
        default="cuda",
        description="Device for backbone model (cpu/cuda)",
        env="BACKBONE_DEVICE"
    )
    codec_device: str = Field(
        default="cuda",
        description="Device for codec model (cpu/cuda)",
        env="CODEC_DEVICE"
    )
    
    # Backend configuration
    use_vllm: bool = Field(
        default=True,
        description="Use vLLM for inference acceleration",
        env="USE_VLLM"
    )
    use_gguf: bool = Field(
        default=False,
        description="Use GGUF quantized model (requires llama-cpp-python)",
        env="USE_GGUF"
    )
    use_async_engine: bool = Field(
        default=True,
        description="Use AsyncLLMEngine for streaming (vLLM only)",
        env="USE_ASYNC_ENGINE"
    )
    
    # Server configuration
    host: str = Field(
        default="127.0.0.1",
        description="Host address for the server",
        env="HOST"
    )
    port: int = Field(
        default=8002,
        description="Port number for the server",
        env="PORT"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
        env="DEBUG"
    )
    
    # Model configuration
    max_context: int = Field(
        default=2048,
        description="Maximum context length for the model",
        env="MAX_CONTEXT"
    )
    gpu_memory_utilization: float = Field(
        default=0.7,
        description="GPU memory utilization for vLLM (0.0-1.0)",
        env="GPU_MEMORY_UTILIZATION"
    )
    seed: int = Field(
        default=42,
        description="Random seed for generation",
        env="SEED"
    )
    
    # Synthesis parameters
    temperature: float = Field(
        default=1.0,
        description="Sampling temperature for synthesis",
        env="TEMPERATURE"
    )
    max_tokens: int = Field(
        default=2048,
        description="Maximum number of tokens to generate",
        env="MAX_TOKENS"
    )
    min_tokens: int = Field(
        default=50,
        description="Minimum number of tokens to generate",
        env="MIN_TOKENS"
    )
    top_p: float = Field(
        default=1.0,
        description="Top-p (nucleus) sampling parameter",
        env="TOP_P"
    )
    top_k: int = Field(
        default=50,
        description="Top-k sampling parameter",
        env="TOP_K"
    )
    
    # Streaming parameters
    streaming_frames_per_chunk: int = Field(
        default=25,
        description="Number of tokens per audio chunk for streaming",
        env="STREAMING_FRAMES_PER_CHUNK"
    )
    streaming_overlap_frames: int = Field(
        default=1,
        description="Overlap frames between chunks",
        env="STREAMING_OVERLAP_FRAMES"
    )
    streaming_lookforward: int = Field(
        default=5,
        description="Future context tokens for streaming",
        env="STREAMING_LOOKFORWARD"
    )
    streaming_lookback: int = Field(
        default=50,
        description="Past context tokens for streaming",
        env="STREAMING_LOOKBACK"
    )
    
    # Audio parameters
    sample_rate: int = Field(
        default=24000,
        description="Output audio sample rate",
        env="SAMPLE_RATE"
    )
    hop_length: int = Field(
        default=480,
        description="Codec hop length in samples",
        env="HOP_LENGTH"
    )
    
    # Phonemizer configuration
    phonemizer_espeak_path: Optional[str] = Field(
        default=None,
        description="Path to espeak-ng binary (for macOS/custom installs)",
        env="PHONEMIZER_ESPEAK_PATH"
    )
    phonemizer_espeak_library: Optional[str] = Field(
        default=None,
        description="Path to espeak-ng library (for macOS/custom installs)",
        env="PHONEMIZER_ESPEAK_LIBRARY"
    )
    
    # Logging configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        env="LOG_LEVEL"
    )
    log_file: str = Field(
        default="logs/tts_service.log",
        description="Path to the log file",
        env="LOG_FILE"
    )
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
