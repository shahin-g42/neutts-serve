"""
Pydantic schemas for TTS service requests and responses.
"""

from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class TTSModel(str, Enum):
    """Supported TTS models."""
    TTS_1 = "tts-1"
    TTS_1_HD = "tts-1-hd"


class TTSResponseFormat(str, Enum):
    """Supported audio response formats."""
    MP3 = "mp3"
    OPUS = "opus"
    AAC = "aac"
    FLAC = "flac"
    WAV = "wav"
    PCM = "pcm"


class TTSRequest(BaseModel):
    """Request schema for TTS generation."""
    model: TTSModel = Field(
        default=TTSModel.TTS_1,
        description="The TTS model to use."
    )
    input: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="The text to generate audio for. Maximum 4096 characters."
    )
    voice: str = Field(
        ...,
        description="The voice profile name to use for generation. Must be a valid profile from data/voice_profiles.json."
    )
    response_format: TTSResponseFormat = Field(
        default=TTSResponseFormat.WAV,
        description="The format to return audio in."
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the generated audio. 1.0 is normal speed."
    )
    
    @field_validator('input')
    @classmethod
    def validate_input(cls, v):
        if not v.strip():
            raise ValueError('Input text cannot be empty or whitespace only')
        return v
    
    @field_validator('voice')
    @classmethod
    def validate_voice(cls, v):
        if not v or not v.strip():
            raise ValueError('Voice parameter cannot be empty. Please specify a voice profile name.')
        return v.strip()
