"""
Error codes for the TTS service.
"""

from enum import Enum


class ErrorCode(str, Enum):
    """Enumeration of error codes used in the TTS service."""
    
    # Model errors
    MODEL_LOADING_ERROR = "model_loading_error"
    MODEL_NOT_AVAILABLE = "model_not_available"
    
    # Voice profile errors
    VOICE_PROFILE_MISSING = "voice_profile_missing"
    VOICE_PROFILE_NOT_FOUND = "voice_profile_not_found"
    VOICE_PROFILE_EMPTY_TOKENS = "voice_profile_empty_tokens"
    VOICE_PROFILE_CREATION_FAILED = "voice_profile_creation_failed"
    VOICE_PROFILE_INVALID_NAME = "voice_profile_invalid_name"
    
    # Synthesis errors
    SYNTHESIS_ERROR = "synthesis_error"
    AUDIO_PROCESSING_ERROR = "audio_processing_error"
    AUDIO_DECODING_ERROR = "audio_decoding_error"
    AUDIO_ENCODING_ERROR = "audio_encoding_error"
    
    # Validation errors
    VALIDATION_ERROR = "validation_error"
    INVALID_INPUT = "invalid_input"
    INVALID_AUDIO_FILE = "invalid_audio_file"
    
    # Internal errors
    INTERNAL_ERROR = "internal_error"
    TIMEOUT_ERROR = "timeout_error"
