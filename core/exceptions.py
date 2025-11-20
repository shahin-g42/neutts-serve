"""
Custom exceptions for the TTS service.
"""

from typing import Any, Dict
from fastapi import HTTPException, status
from core.error_codes import ErrorCode


class TTSServiceError(Exception):
    """Base exception for all TTS service errors."""
    
    def __init__(self, message: str, error_code: ErrorCode, status_code: int = 500):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format for API responses."""
        return {
            "error": {
                "code": self.error_code.value,
                "message": self.message,
                "detail": self.message
            }
        }
    
    def to_http_exception(self) -> HTTPException:
        """Convert to FastAPI HTTPException."""
        return HTTPException(
            status_code=self.status_code,
            detail=self.to_dict()["error"]
        )


class ModelLoadingError(TTSServiceError):
    """Raised when model fails to load."""
    
    def __init__(self, message: str = "Failed to load TTS model"):
        super().__init__(
            message=message,
            error_code=ErrorCode.MODEL_LOADING_ERROR,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class VoiceProfileMissingError(TTSServiceError):
    """Raised when voice profile parameter is missing or empty."""
    
    def __init__(self, message: str = "Voice parameter is required"):
        super().__init__(
            message=message,
            error_code=ErrorCode.VOICE_PROFILE_MISSING,
            status_code=status.HTTP_400_BAD_REQUEST
        )


class VoiceProfileNotFoundError(TTSServiceError):
    """Raised when requested voice profile doesn't exist."""
    
    def __init__(self, voice_name: str):
        super().__init__(
            message=f"Voice profile '{voice_name}' not found",
            error_code=ErrorCode.VOICE_PROFILE_NOT_FOUND,
            status_code=status.HTTP_404_NOT_FOUND
        )


class VoiceProfileEmptyTokensError(TTSServiceError):
    """Raised when voice profile has no speech tokens."""
    
    def __init__(self, voice_name: str):
        super().__init__(
            message=f"Voice profile '{voice_name}' has no speech tokens (profile may be corrupted)",
            error_code=ErrorCode.VOICE_PROFILE_EMPTY_TOKENS,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )


class SynthesisError(TTSServiceError):
    """Raised when speech synthesis fails."""
    
    def __init__(self, message: str = "Speech synthesis failed"):
        super().__init__(
            message=message,
            error_code=ErrorCode.SYNTHESIS_ERROR,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class AudioProcessingError(TTSServiceError):
    """Raised when audio processing fails."""
    
    def __init__(self, message: str = "Audio processing failed"):
        super().__init__(
            message=message,
            error_code=ErrorCode.AUDIO_PROCESSING_ERROR,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class ValidationError(TTSServiceError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str = "Validation failed"):
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            status_code=status.HTTP_400_BAD_REQUEST
        )
