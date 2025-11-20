"""
Audio format conversion utilities.
Handles conversion between different audio formats and sample rates.
"""

import io
import tempfile
import os
import torch
import torchaudio
import numpy as np
from typing import Tuple
from core.logging import app_logger


def resample_audio(
    wav_tensor: torch.Tensor, 
    orig_sr: int, 
    target_sr: int
) -> torch.Tensor:
    """
    Resample audio tensor to target sample rate.
    
    Args:
        wav_tensor: Audio tensor with shape [channels, samples]
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio tensor
    """
    if orig_sr == target_sr:
        return wav_tensor
    
    app_logger.info(f"Resampling from {orig_sr}Hz to {target_sr}Hz")
    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_sr,
        new_freq=target_sr
    )
    return resampler(wav_tensor)


def convert_to_mp3(
    wav_tensor: torch.Tensor,
    sample_rate: int,
    bitrate: str = "128k"
) -> bytes:
    """
    Convert audio tensor to MP3 format using torchaudio's FFmpeg backend.
    
    Args:
        wav_tensor: Audio tensor with shape [channels, samples]
        sample_rate: Sample rate in Hz
        bitrate: MP3 bitrate (e.g., "128k", "192k", "320k")
        
    Returns:
        MP3 audio as bytes
    """
    try:
        # Use temporary files for MP3 encoding (torchaudio's FFmpeg backend)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
            temp_mp3_path = tmp_mp3.name
        
        try:
            # Save as MP3 with specified bitrate
            torchaudio.save(
                temp_mp3_path,
                wav_tensor,
                sample_rate,
                format="mp3",
                bits_per_sample=16,
                encoding="mp3",
                compression=bitrate
            )
            
            # Read back the MP3 bytes
            with open(temp_mp3_path, 'rb') as f:
                mp3_bytes = f.read()
            
            app_logger.info(f"Converted to MP3 ({bitrate}) - Size: {len(mp3_bytes)/1024:.1f}KB")
            return mp3_bytes
            
        finally:
            # Cleanup
            if os.path.exists(temp_mp3_path):
                os.remove(temp_mp3_path)
                
    except Exception as e:
        app_logger.error(f"MP3 conversion failed: {e}")
        raise RuntimeError(f"Failed to convert to MP3: {e}")


def convert_audio_format(
    wav_tensor: torch.Tensor,
    orig_sr: int,
    target_format: str = "mp3",
    target_sr: int = 44100,
    bitrate: str = "128k"
) -> Tuple[bytes, str]:
    """
    Convert audio to specified format and sample rate.
    
    Args:
        wav_tensor: Audio tensor with shape [channels, samples]
        orig_sr: Original sample rate
        target_format: Target format ('wav', 'mp3', 'flac', etc.)
        target_sr: Target sample rate (default: 44100 for MP3)
        bitrate: MP3 bitrate (only for MP3 format)
        
    Returns:
        Tuple of (audio_bytes, content_type)
    """
    app_logger.info(
        f"Converting audio: {orig_sr}Hz -> {target_sr}Hz, format: {target_format}"
    )
    
    # Step 1: Resample to target sample rate
    resampled = resample_audio(wav_tensor, orig_sr, target_sr)
    
    # Step 2: Convert to target format
    if target_format.lower() == "mp3":
        audio_bytes = convert_to_mp3(resampled, target_sr, bitrate)
        content_type = "audio/mpeg"
    
    elif target_format.lower() in ["wav", "wave"]:
        # Save as WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            temp_wav_path = tmp_wav.name
        
        try:
            torchaudio.save(temp_wav_path, resampled, target_sr)
            with open(temp_wav_path, 'rb') as f:
                audio_bytes = f.read()
            content_type = "audio/wav"
        finally:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
    
    elif target_format.lower() == "flac":
        # Save as FLAC
        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp_flac:
            temp_flac_path = tmp_flac.name
        
        try:
            torchaudio.save(temp_flac_path, resampled, target_sr, format="flac")
            with open(temp_flac_path, 'rb') as f:
                audio_bytes = f.read()
            content_type = "audio/flac"
        finally:
            if os.path.exists(temp_flac_path):
                os.remove(temp_flac_path)
    
    else:
        raise ValueError(f"Unsupported audio format: {target_format}")
    
    app_logger.info(
        f"Conversion complete - Format: {target_format}, "
        f"Size: {len(audio_bytes)/1024:.1f}KB"
    )
    
    return audio_bytes, content_type
