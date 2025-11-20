"""Voice profile management API endpoints."""

import json
import os
from typing import List, Dict, Any
from functools import lru_cache
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form

from core.logging import app_logger
from schemas.voice_profiles import VoiceProfileResponse
from utils.audio_processor import process_reference_audio

router = APIRouter(prefix="/v1", tags=["voice-profiles"])

VOICE_PROFILES_FILE = "data/voice_profiles.json"
os.makedirs("data", exist_ok=True)


class VoiceProfileManager:
    """Voice profile manager with LRU cache."""
    
    @staticmethod
    def _ensure_profiles_file_exists():
        if not os.path.exists(VOICE_PROFILES_FILE):
            with open(VOICE_PROFILES_FILE, 'w') as f:
                json.dump({}, f)
            app_logger.info(f"Created voice profiles file: {VOICE_PROFILES_FILE}")
    
    @staticmethod
    @lru_cache(maxsize=100)
    def _get_profile_cached(name: str) -> Dict[str, Any]:
        VoiceProfileManager._ensure_profiles_file_exists()
        try:
            with open(VOICE_PROFILES_FILE, 'r') as f:
                profiles = json.load(f)
            return profiles.get(name, None)
        except Exception as e:
            app_logger.error(f"Error loading voice profile {name}: {e}")
            return None
    
    @classmethod
    def _load_profiles_from_file(cls) -> Dict[str, Any]:
        cls._ensure_profiles_file_exists()
        try:
            with open(VOICE_PROFILES_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            app_logger.error(f"Error loading voice profiles: {e}")
            return {}
    
    @classmethod
    def get_all_profiles(cls) -> List[VoiceProfileResponse]:
        try:
            profiles = cls._load_profiles_from_file()
            return [VoiceProfileResponse(**profile) for profile in profiles.values()]
        except Exception as e:
            app_logger.error(f"Error loading profiles: {e}")
            return []
    
    @classmethod
    def get_profile(cls, name: str) -> Dict[str, Any]:
        return cls._get_profile_cached(name)
    
    @classmethod
    def add_profile(cls, name: str, description: str, reference_text: str, speech_codes: List[int]) -> VoiceProfileResponse:
        try:
            profiles = cls._load_profiles_from_file()
            
            if name in profiles:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Voice profile '{name}' already exists"
                )
            
            profile_dict = {
                "name": name,
                "description": description,
                "reference_text": reference_text,
                "speech_codes": speech_codes
            }
            profiles[name] = profile_dict
            
            with open(VOICE_PROFILES_FILE, 'w') as f:
                json.dump(profiles, f, indent=2)
            
            cls._get_profile_cached.cache_clear()
            app_logger.info(f"Created profile '{name}' with {len(speech_codes)} codes")
            
            return VoiceProfileResponse(**profile_dict)
        except HTTPException:
            raise
        except Exception as e:
            app_logger.error(f"Error adding profile: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to add voice profile"
            )
    
    @classmethod
    def delete_profile(cls, name: str) -> bool:
        try:
            profiles = cls._load_profiles_from_file()
            if name not in profiles:
                return False
            del profiles[name]
            with open(VOICE_PROFILES_FILE, 'w') as f:
                json.dump(profiles, f, indent=2)
            cls._get_profile_cached.cache_clear()
            app_logger.info(f"Deleted profile: {name}")
            return True
        except Exception as e:
            app_logger.error(f"Error deleting profile: {e}")
            return False


@router.post("/voice-profiles", response_model=VoiceProfileResponse)
async def create_voice_profile(
    name: str = Form(..., min_length=1, max_length=50),
    description: str = Form(..., min_length=1, max_length=200),
    reference_text: str = Form(..., min_length=1, max_length=1000),
    audio_file: UploadFile = File(...)
):
    try:
        app_logger.info(f"Creating voice profile: {name}")
        
        if '..' in name or '/' in name or '\\' in name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid voice profile name"
            )
        
        temp_audio_path = f"data/temp_{audio_file.filename}"
        with open(temp_audio_path, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        
        # Get codec from TTS model
        from api.tts import get_tts_model
        tts_model = get_tts_model()
        if tts_model is None or tts_model.neutts_wrapper is None:
            raise HTTPException(status_code=500, detail="TTS model not available")
        
        codec_model = tts_model.neutts_wrapper.codec
        
        processed_data = process_reference_audio(temp_audio_path, reference_text, codec_model)
        os.remove(temp_audio_path)
        
        return VoiceProfileManager.add_profile(
            name=name,
            description=description,
            reference_text=processed_data["reference_text"],
            speech_codes=processed_data["speech_codes"]
        )
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error creating profile: {e}")
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create voice profile: {str(e)}"
        )


@router.get("/voice-profiles", response_model=List[VoiceProfileResponse])
async def list_voice_profiles():
    try:
        return VoiceProfileManager.get_all_profiles()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voice-profiles/{name}", response_model=VoiceProfileResponse)
async def get_voice_profile(name: str):
    if '..' in name or '/' in name or '\\' in name:
        raise HTTPException(status_code=400, detail="Invalid name")
    
    profile = VoiceProfileManager.get_profile(name)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
    
    return VoiceProfileResponse(**profile)


@router.delete("/voice-profiles/{name}")
async def delete_voice_profile(name: str):
    if '..' in name or '/' in name or '\\' in name:
        raise HTTPException(status_code=400, detail="Invalid name")
    
    if not VoiceProfileManager.delete_profile(name):
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
    
    return {"message": f"Profile '{name}' deleted successfully"}
