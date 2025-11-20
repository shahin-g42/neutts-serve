"""
Audio processor for extracting speech codes from audio files using NeuCodec.
"""

import os
import torch
import numpy as np
from typing import List, Dict
from pathlib import Path

from core.logging import app_logger
from utils.audio_utils import load_wav


def extract_speech_codes_from_audio(audio_path: str, codec_model) -> List[int]:
    """
    Extract speech codes from audio using NeuCodec.
    
    Args:
        audio_path: Path to audio file
        codec_model: NeuCodec model instance
        
    Returns:
        List of FSQ codes as integers
    """
    app_logger.info(f"Extracting speech codes from: {audio_path}")
    
    # Load audio at 16kHz (NeuCodec encoding rate)
    waveform, _ = load_wav(audio_path, target_sample_rate=16000)
    
    # Add batch dimension: [1, 1, T]
    waveform = waveform.unsqueeze(0)
    
    # Encode with NeuCodec
    with torch.no_grad():
        codes = codec_model.encode_code(waveform)
        # codes shape: [1, 1, T_codes]
        codes = codes.squeeze(0).squeeze(0)  # Remove batch and channel dims
    
    # Convert to list of integers
    codes_list = codes.cpu().numpy().tolist()
    
    app_logger.info(f"Extracted {len(codes_list)} speech codes")
    
    return codes_list


def validate_speech_codes(speech_codes: List[int]) -> bool:
    """
    Validate speech codes.
    
    Args:
        speech_codes: List of speech codes
        
    Returns:
        True if valid
    """
    if not speech_codes:
        return False
    
    # Check all are integers
    if not all(isinstance(code, int) for code in speech_codes):
        return False
    
    return True


def process_reference_audio(audio_path: str, reference_text: str, codec_model) -> Dict:
    """
    Process reference audio to create voice profile data.
    
    Args:
        audio_path: Path to reference audio
        reference_text: Transcription of the audio
        codec_model: NeuCodec model instance
        
    Returns:
        Dictionary with reference_text and speech_codes
    """
    app_logger.info(f"Processing reference audio: {audio_path}")
    
    # Extract codes
    speech_codes = extract_speech_codes_from_audio(audio_path, codec_model)
    
    # Validate
    if not validate_speech_codes(speech_codes):
        raise ValueError("Invalid speech codes extracted")
    
    app_logger.info(f"Processed {len(speech_codes)} codes")
    
    return {
        "reference_text": reference_text,
        "speech_codes": speech_codes
    }
