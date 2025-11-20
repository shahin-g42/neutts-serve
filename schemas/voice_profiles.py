"""
Pydantic schemas for voice profile management.
"""

from typing import List
from pydantic import BaseModel, Field


class VoiceProfileCreate(BaseModel):
    """Schema for creating a voice profile from audio."""
    name: str = Field(..., min_length=1, max_length=50)
    description: str = Field(..., min_length=1, max_length=200)
    reference_text: str = Field(..., min_length=1, max_length=1000)


class VoiceProfileResponse(BaseModel):
    """Schema for voice profile response."""
    name: str
    description: str
    reference_text: str
    speech_codes: List[int]  # NeuCodec FSQ codes as integers
