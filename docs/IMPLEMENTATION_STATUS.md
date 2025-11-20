# NeuTTS-Air Implementation Status

## Completed ✓

### Core Infrastructure
1. **requirements.txt** - All dependencies specified including:
   - neucodec, vllm, transformers, phonemizer
   - FastAPI, Pydantic for API layer
   - torch, torchaudio for model operations

2. **core/settings.py** - Complete configuration management:
   - Model paths and device settings
   - Backend selection (vLLM/GGUF/Transformers)
   - Streaming parameters
   - All synthesis parameters

3. **core/exceptions.py** & **core/error_codes.py** - Error handling:
   - TTSServiceError base class
   - Model loading, voice profile, synthesis errors
   - HTTP status code mapping

4. **core/logging.py** - Logging setup with log

uru:
   - Console and file handlers
   - Rotation and compression
   - Configurable log levels

5. **core/error_handler.py** - FastAPI error handlers

6. **schemas/tts.py** & **schemas/voice_profiles.py** - Request/response schemas:
   - TTSRequest with validation
   - VoiceProfileCreate and VoiceProfileResponse
   - Uses integer codes (not token strings)

7. **core/neutts_wrapper.py** (417 lines) - NeuTTSAir wrapper:
   - Multi-backend support (Transformers/vLLM/GGUF)
   - NeuCodec integration for encoding/decoding
   - Phonemizer integration
   - Basic inference methods (_infer_torch, _infer_ggml)
   - Audio encoding (encode_reference)
   - Watermarking support
   - Linear overlap-add for streaming

8. **core/tts_model.py** - TTSModel service wrapper:
   - TTSModelConfig dataclass
   - synthesize_speech_with_tokens method
   - stream_speech_with_tokens (fallback chunking)
   - Integration with NeuTTSAirWrapper

## Remaining Tasks

### API Layer (High Priority)
1. **api/voice_profiles.py** - Voice profile management API:
   - GET /v1/voice-profiles (list all)
   - POST /v1/voice-profiles (create from audio)
   - GET /v1/voice-profiles/{name}
   - DELETE /v1/voice-profiles/{name}
   - VoiceProfileManager class
   - NeuCodec encoding integration

2. **api/tts.py** - TTS synthesis endpoints:
   - POST /v1/audio/speech
   - POST /v1/audio/speech/stream
   - Integration with TTSModel

3. **main.py** - FastAPI application:
   - App initialization
   - Lifespan manager for model loading
   - Router registration
   - CORS middleware
   - Health endpoints

### Utility Modules
4. **utils/audio_processor.py** - Audio processing:
   - extract_speech_codes_from_audio (NeuCodec encoding)
   - validate_speech_codes
   - process_reference_audio

5. **utils/audio_utils.py** - Audio utilities:
   - load_wav function
   - Sample rate conversion helpers

### Configuration
6. **.env** - Environment configuration file
7. **data/** directory initialization
8. **__init__.py** files for packages

### Advanced Features (Lower Priority)
9. **Streaming Implementation** - True token-by-token streaming:
   - AsyncLLMEngine integration in neutts_wrapper.py
   - _infer_stream_vllm method
   - _infer_stream_ggml method (GGUF streaming)
   - Chunked audio decoding with overlap-add
   - Async generator support in API endpoints

10. **Testing**:
   - Unit tests for core modules
   - Integration tests for API endpoints
   - Voice profile creation tests

11. **Documentation**:
   - README.md with setup instructions
   - API documentation
   - Example usage

## Architecture Notes

### Voice Profile Format Change
- **Old (ALMAS v3)**: Stored as `<|s_{id}|>` token strings
- **New (NeuTTS-Air)**: Stored as integer codes directly from NeuCodec

### Model Loading Strategy
The system supports three backends:
1. **Standard Transformers**: Development/debugging
2. **vLLM (LLM or AsyncLLMEngine)**: Production, high-throughput
3. **GGUF/llama-cpp**: Resource-constrained, CPU inference

### Critical Design Decisions
1. **No Batching**: NeuCodec processes audio sequentially (known limitation)
2. **No torch.compile**: NeuCodec uses numpy operations incompatible with compilation
3. **Streaming**: Basic chunking implemented, advanced streaming requires AsyncLLMEngine
4. **Phonemization**: Required preprocessing step for NeuTTS-Air
5. **Watermarking**: Optional perth watermarker integration

## Next Steps

### Immediate (to get working system):
1. Create API layer (voice_profiles.py, tts.py)
2. Create main.py with FastAPI app
3. Create audio processing utilities
4. Create .env configuration
5. Test basic synthesis pipeline

### Short-term (enhancements):
1. Implement true streaming with AsyncLLMEngine
2. Add comprehensive error handling
3. Performance optimization
4. Add logging/monitoring

### Long-term (production):
1. Docker containerization
2. Load testing and benchmarking
3. Multi-worker deployment
4. CI/CD pipeline
5. Monitoring and alerting

## Known Issues/Limitations

1. Streaming currently uses fallback (synthesize-then-chunk)
2. Need to install perth for watermarking (optional)
3. Need espeak-ng system package for phonemizer
4. vLLM requires CUDA (no CPU support)
5. Voice profiles from old system need re-encoding

## File Structure

```
neutts-serve/
├── core/
│   ├── settings.py ✓
│   ├── exceptions.py ✓
│   ├── error_codes.py ✓
│   ├── error_handler.py ✓
│   ├── logging.py ✓
│   ├── neutts_wrapper.py ✓
│   └── tts_model.py ✓
├── schemas/
│   ├── tts.py ✓
│   └── voice_profiles.py ✓
├── api/
│   ├── tts.py ⏳
│   └── voice_profiles.py ⏳
├── utils/
│   ├── audio_processor.py ⏳
│   └── audio_utils.py ⏳
├── data/
│   └── voice_profiles.json ⏳
├── logs/
│   └── (created dynamically)
├── requirements.txt ✓
├── main.py ⏳
└── .env ⏳
```

Legend: ✓ = Complete, ⏳ = Pending

## Testing Checklist

- [ ] Model loading (all backends)
- [ ] Voice profile creation
- [ ] Voice profile listing
- [ ] Basic synthesis
- [ ] Streaming synthesis
- [ ] Error handling
- [ ] API endpoint compatibility
- [ ] Performance benchmarks
