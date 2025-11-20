# NeuTTS-Air TTS Service

A production-ready Text-to-Speech service using **NeuTTS-Air** and **NeuCodec**, following OpenAI TTS API standards.

## Features

- ✅ **Multi-Backend Support**: Transformers, vLLM, or GGUF quantization
- ✅ **NeuCodec Integration**: Unified audio encoding/decoding
- ✅ **Voice Cloning**: Create custom voice profiles from reference audio
- ✅ **Streaming Support**: Real-time audio generation
- ✅ **OpenAI-Compatible API**: Drop-in replacement for OpenAI TTS
- ✅ **Production Ready**: FastAPI with async support, error handling, logging

## Architecture

### Core Components

- **NeuTTS-Air Backbone**: Text-to-speech model (neuphonic/neutts-air)
- **NeuCodec**: Audio codec for encoding/decoding (neuphonic/neucodec)
- **Phonemizer**: Text-to-phoneme conversion (espeak backend)
- **FastAPI**: High-performance async web framework

### Backend Options

| Backend | Use Case | Requirements |
|---------|----------|--------------|
| **vLLM** | High-throughput production | CUDA GPU |
| **GGUF** | Resource-constrained deployment | llama-cpp-python |
| **Transformers** | Development/debugging | CPU/CUDA |

## Installation

### Prerequisites

```bash
# System dependencies
sudo apt-get install espeak-ng  # Ubuntu/Debian
# or
brew install espeak  # macOS

# Python 3.10+
python --version
```

### Install Dependencies

```bash
cd /Users/shahin.konadath/Documents/project/v7/neutts-serve

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install Python packages
pip install -r requirements.txt

# Optional: Install GGUF support
# pip install llama-cpp-python

# Optional: Install watermarking
# pip install perth
```

## Configuration

Edit `.env` file to configure the service:

```bash
# Model paths
MODEL_CHECKPOINT_PATH=neuphonic/neutts-air
CODEC_REPO=neuphonic/neucodec

# Device selection
BACKBONE_DEVICE=cuda  # or 'cpu'
CODEC_DEVICE=cuda

# Backend (choose one)
USE_VLLM=true
USE_GGUF=false

# Server
HOST=127.0.0.1
PORT=8002
```

## Usage

### Start the Service

```bash
python main.py
```

The service will be available at `http://localhost:8002`

API documentation: `http://localhost:8002/docs`

### Create a Voice Profile

```bash
curl -X POST "http://localhost:8002/v1/voice-profiles" \
  -F "name=my_voice" \
  -F "description=My custom voice" \
  -F "reference_text=This is a sample of my voice." \
  -F "audio_file=@/path/to/reference_audio.wav"
```

### Generate Speech

```bash
curl -X POST "http://localhost:8002/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello, this is a test of the NeuTTS-Air TTS system.",
    "voice": "my_voice",
    "response_format": "wav"
  }' \
  --output output.wav
```

### List Voice Profiles

```bash
curl "http://localhost:8002/v1/voice-profiles"
```

### Stream Audio

```bash
curl -X POST "http://localhost:8002/v1/audio/speech/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Streaming audio generation test.",
    "voice": "my_voice"
  }' \
  --output stream.wav
```

## API Endpoints

### TTS Endpoints

- `POST /v1/audio/speech` - Generate audio from text
- `POST /v1/audio/speech/stream` - Stream audio generation

### Voice Profile Endpoints

- `GET /v1/voice-profiles` - List all profiles
- `POST /v1/voice-profiles` - Create new profile
- `GET /v1/voice-profiles/{name}` - Get specific profile
- `DELETE /v1/voice-profiles/{name}` - Delete profile

### System Endpoints

- `GET /` - Service information
- `GET /health` - Health check

## Project Structure

```
neutts-serve/
├── api/                    # API endpoints
│   ├── tts.py             # TTS synthesis endpoints
│   └── voice_profiles.py  # Voice profile management
├── core/                   # Core business logic
│   ├── neutts_wrapper.py  # NeuTTSAir wrapper (417 lines)
│   ├── tts_model.py       # Service-layer model wrapper
│   ├── settings.py        # Configuration management
│   ├── exceptions.py      # Custom exceptions
│   ├── error_handler.py   # FastAPI error handlers
│   └── logging.py         # Logging configuration
├── schemas/                # Pydantic models
│   ├── tts.py
│   └── voice_profiles.py
├── utils/                  # Utility functions
│   ├── audio_utils.py     # Audio loading/processing
│   └── audio_processor.py # NeuCodec encoding
├── data/                   # Runtime data
│   └── voice_profiles.json
├── logs/                   # Application logs
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
└── .env                    # Configuration file
```

## Voice Profile Format

Voice profiles are stored as JSON with NeuCodec FSQ codes:

```json
{
  "profile_name": {
    "name": "profile_name",
    "description": "Description of the voice",
    "reference_text": "Transcription of reference audio",
    "speech_codes": [123, 456, 789, ...]  // Integer codes from NeuCodec
  }
}
```

**Note**: Voice profiles from the old ALMAS v3 system (using `<|s_{id}|>` format) must be re-created.

## Advanced Configuration

### vLLM Optimization

```bash
# .env
USE_VLLM=true
USE_ASYNC_ENGINE=true
GPU_MEMORY_UTILIZATION=0.7
MAX_CONTEXT=2048
```

### GGUF Quantization

```bash
# .env
USE_GGUF=true
MODEL_CHECKPOINT_PATH=neuphonic/neutts-air-q4-gguf
BACKBONE_DEVICE=cpu
```

### Streaming Parameters

```bash
STREAMING_FRAMES_PER_CHUNK=25  # Lower = more responsive, higher = smoother
STREAMING_OVERLAP_FRAMES=1
STREAMING_LOOKFORWARD=5
STREAMING_LOOKBACK=50
```

## Known Limitations

1. **No Batching**: NeuCodec processes audio sequentially (design constraint)
2. **No torch.compile**: Incompatible with NeuCodec preprocessing
3. **Streaming**: Currently uses fallback (synthesize-then-chunk). True async streaming with AsyncLLMEngine is planned.
4. **vLLM**: Requires CUDA (no CPU support)
5. **Phonemizer**: Requires espeak-ng system package

## Troubleshooting

### Model Loading Issues

```python
# Check logs
tail -f logs/tts_service.log

# Test model loading
python -c "from core.tts_model import TTSModel, TTSModelConfig; config = TTSModelConfig.from_settings(); model = TTSModel(config)"
```

### Audio Issues

- Ensure reference audio is clear and high-quality
- Use WAV format for best results
- Reference audio should be 3-10 seconds long
- Provide accurate transcription for best voice cloning

### Performance

- Use vLLM backend for production
- Enable CUDA for 10-100x speedup
- Consider GGUF quantization for CPU deployment
- Use multiple worker processes for concurrency

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov httpx

# Run tests (when implemented)
pytest tests/
```

### Code Quality

```bash
# Format code
black .

# Lint
ruff check .
```

## Contributing

This is based on the design document at `.qoder/quests/tts-model-replacement.md`.

## License

Follow the licenses of the underlying models:
- NeuTTS-Air: Check neuphonic/neutts-air license
- NeuCodec: Check neuphonic/neucodec license

## Acknowledgments

- **Neuphonic** for NeuTTS-Air and NeuCodec models
- **vLLM team** for high-performance inference
- **Reference project** at `ref_project/` for API design inspiration

## Support

- Documentation: `http://localhost:8002/docs`
- Logs: `logs/tts_service.log`
- Status: `.qoder/quests/IMPLEMENTATION_STATUS.md`
