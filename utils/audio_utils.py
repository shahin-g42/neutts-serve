"""
Audio utility functions for loading and processing audio files.
"""

import torch
import soundfile as sf
import numpy as np
from torchaudio import transforms as T
from pathlib import Path
from typing import Tuple

from core.logging import app_logger


def load_wav(audio_path: str | Path, target_sample_rate: int = 16000) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        audio_path: Path to audio file
        target_sample_rate: Target sample rate (default: 16000 for NeuCodec encoding)
        
    Returns:
        Tuple of (waveform tensor [1, T], sample_rate)
    """
    app_logger.info(f"Loading audio from: {audio_path}")
    
    # Load audio using soundfile (avoids torchcodec dependency in torchaudio 2.9)
    audio_data, sample_rate = sf.read(str(audio_path), dtype='float32')
    
    # Convert to torch tensor
    waveform = torch.from_numpy(audio_data)
    
    # Ensure shape is [C, T] (channels, time)
    if waveform.ndim == 1:
        # Mono: [T] -> [1, T]
        waveform = waveform.unsqueeze(0)
    else:
        # Stereo: [T, C] -> [C, T]
        waveform = waveform.transpose(0, 1)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        app_logger.debug("Converted stereo to mono")
    
    # Resample if needed
    if sample_rate != target_sample_rate:
        resampler = T.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
        app_logger.debug(f"Resampled from {sample_rate}Hz to {target_sample_rate}Hz")
    
    app_logger.info(f"Audio loaded: shape={waveform.shape}, sr={target_sample_rate}Hz")
    
    return waveform, target_sample_rate
