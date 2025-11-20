"""TTS API endpoints following OpenAI TTS standards."""

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse

from core.logging import app_logger
from core.exceptions import (
    VoiceProfileMissingError,
    VoiceProfileNotFoundError,
    VoiceProfileEmptyTokensError,
    SynthesisError
)
from schemas.tts import TTSRequest
from core.tts_model import TTSModel, TTSModelConfig
from core.settings import settings
from api.voice_profiles import VoiceProfileManager

_tts_model = None


def get_tts_model():
    """Get or create the TTS model instance (lazy initialization)."""
    global _tts_model
    if _tts_model is None:
        try:
            model_config = TTSModelConfig.from_settings()
            _tts_model = TTSModel(model_config)
            app_logger.info("TTS model initialized successfully")
        except Exception as e:
            app_logger.error(f"Failed to initialize TTS model: {e}")
            _tts_model = None
    return _tts_model


router = APIRouter(prefix="/v1", tags=["text-to-speech"])


@router.post("/audio/speech", response_class=Response)
async def create_speech(request: TTSRequest):
    """Creates audio speech from the input text."""
    tts_model = get_tts_model()
    if tts_model is None:
        raise HTTPException(status_code=500, detail="TTS model is not available")
    
    try:
        app_logger.info(f"TTS | voice={request.voice} | text_len={len(request.input)}")
        
        if not request.voice or not request.voice.strip():
            raise VoiceProfileMissingError()
        
        voice_profile = VoiceProfileManager.get_profile(request.voice)
        
        if not voice_profile:
            raise VoiceProfileNotFoundError(request.voice)
        
        speech_codes = voice_profile.get("speech_codes", [])
        reference_text = voice_profile.get("reference_text", "")
        
        if not speech_codes or len(speech_codes) == 0:
            raise VoiceProfileEmptyTokensError(request.voice)
        
        app_logger.info(f"Loaded | codes={len(speech_codes)} | ref_text={len(reference_text)} chars")
        
        # Use async synthesis for AsyncLLMEngine, sync otherwise
        if tts_model.config.use_async_engine and tts_model.config.use_vllm:
            audio_bytes, content_type, duration = await tts_model.synthesize_speech_with_tokens_async(
                text=request.input,
                speech_codes=speech_codes,
                reference_text=reference_text
            )
        else:
            audio_bytes, content_type, duration = tts_model.synthesize_speech_with_tokens(
                text=request.input,
                speech_codes=speech_codes,
                reference_text=reference_text
            )
        
        app_logger.info(f"Complete | duration={duration:.2f}s | size={len(audio_bytes)/1024:.1f}KB")
        
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"inline; filename=speech.{request.response_format.value}"
            }
        )
    
    except (VoiceProfileMissingError, VoiceProfileNotFoundError, VoiceProfileEmptyTokensError):
        raise
    except Exception as e:
        app_logger.error(f"Synthesis error: {str(e)[:100]}")
        raise SynthesisError(str(e))


@router.post("/audio/speech/stream")
async def stream_speech(request: TTSRequest):
    """Streams audio speech from the input text."""
    tts_model = get_tts_model()
    if tts_model is None:
        raise HTTPException(status_code=500, detail="TTS model is not available")
    
    async def audio_stream():
        try:
            app_logger.info(f"Stream | voice={request.voice} | text_len={len(request.input)}")
            
            if not request.voice or not request.voice.strip():
                raise VoiceProfileMissingError()
            
            voice_profile = VoiceProfileManager.get_profile(request.voice)
            if not voice_profile:
                raise VoiceProfileNotFoundError(request.voice)
            
            speech_codes = voice_profile.get("speech_codes", [])
            reference_text = voice_profile.get("reference_text", "")
            
            if not speech_codes or len(speech_codes) == 0:
                raise VoiceProfileEmptyTokensError(request.voice)
            
            app_logger.info(f"Streaming | codes={len(speech_codes)}")
            
            for audio_chunk in tts_model.stream_speech_with_tokens(
                text=request.input,
                speech_codes=speech_codes,
                reference_text=reference_text
            ):
                yield audio_chunk
            
            app_logger.info("Stream complete")
        
        except Exception as exc:
            app_logger.error(f"Stream error: {str(exc)[:100]}")
            yield str.encode(f"Error: {str(exc)[:200]}")
    
    content_type = f"audio/{request.response_format.value}"
    return StreamingResponse(audio_stream(), media_type=content_type)
