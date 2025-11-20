# Linux Setup Instructions for NeuTTS-Air TTS Service

This guide is specifically for Ubuntu/Debian Linux systems with CUDA/GPU support.

## Prerequisites

Your system shows espeak-ng is already installed at:
- Binary: `/usr/bin/espeak-ng`
- Data: `/usr/lib/x86_64-linux-gnu/espeak-ng-data`
- Library: `/usr/lib/x86_64-linux-gnu/libespeak-ng.so`

## Quick Setup Steps

### 1. Copy the Linux-specific configuration

On your remote machine:

```bash
cd /space/neutts-serve
cp .env.linux .env
```

### 2. Verify espeak-ng paths (already done on your system)

```bash
which espeak-ng
# Should output: /usr/bin/espeak-ng

ls -la /usr/lib/x86_64-linux-gnu/libespeak-ng.so*
# Should show the library files
```

### 3. Install Python dependencies (if not already done)

```bash
pip3 install phonemizer neucodec neuttsair librosa soundfile torch torchaudio \
    transformers fastapi uvicorn pydantic pydantic-settings python-multipart \
    loguru httpx pytest pytest-asyncio vllm
```

### 4. Start the server

```bash
python3 main.py
```

## Expected Output

The server should now start successfully:

```
2025-11-20 XX:XX:XX | INFO     | __main__:<module>:97 - Starting NeuTTS-Air TTS service on 0.0.0.0:8002
INFO:     Started server process [XXXXX]
INFO:     Waiting for application startup.
2025-11-20 XX:XX:XX | INFO     | main:lifespan:27 - Starting NeuTTS-Air TTS service initialization...
2025-11-20 XX:XX:XX | INFO     | main:lifespan:31 - Loading TTS model...
2025-11-20 XX:XX:XX | INFO     | core.neutts_wrapper:_load_phonemizer:125 - Loading phonemizer...
2025-11-20 XX:XX:XX | INFO     | core.neutts_wrapper:_load_phonemizer:138 - Phonemizer loaded successfully
2025-11-20 XX:XX:XX | INFO     | core.neutts_wrapper:_load_backbone:145 - Loading backbone from: neuphonic/neutts-air on cuda ...
...
2025-11-20 XX:XX:XX | INFO     | main:lifespan:38 - TTS model loaded successfully in XX.XX seconds
```

## Troubleshooting

### If phonemizer still cannot find espeak-ng:

1. **Verify the library path exists:**
   ```bash
   ls -la /usr/lib/x86_64-linux-gnu/libespeak-ng.so*
   ```

2. **If the library is in a different location, find it:**
   ```bash
   find /usr/lib -name "libespeak-ng.so*" 2>/dev/null
   ```

3. **Update .env with the correct path:**
   ```bash
   # Edit .env and set:
   PHONEMIZER_ESPEAK_LIBRARY=/path/to/your/libespeak-ng.so
   ```

### If espeak-ng is not installed (shouldn't be your case):

```bash
sudo apt-get update
sudo apt-get install espeak-ng
```

### Check environment variables are loaded:

```bash
python3 -c "from core.settings import settings; print(f'espeak path: {settings.phonemizer_espeak_path}'); print(f'library: {settings.phonemizer_espeak_library}')"
```

## Configuration Differences from macOS

The main difference is the espeak-ng paths:

| Platform | Binary Path | Library Path |
|----------|-------------|--------------|
| **Linux (Ubuntu/Debian)** | `/usr/bin/espeak-ng` | `/usr/lib/x86_64-linux-gnu/libespeak-ng.so` |
| **macOS (Homebrew)** | `/opt/homebrew/bin/espeak-ng` | `/opt/homebrew/lib/libespeak-ng.dylib` |

## GPU Configuration

Your `.env.linux` is configured for GPU (H100):

- `BACKBONE_DEVICE=cuda`
- `CODEC_DEVICE=cuda`
- `USE_VLLM=true` (for high-performance inference)
- `GPU_MEMORY_UTILIZATION=0.7` (adjust if needed)

## Server Access

Since `HOST=0.0.0.0`, the server will be accessible from:

- Localhost: `http://127.0.0.1:8002`
- Network: `http://<your-server-ip>:8002`
- Health check: `http://<your-server-ip>:8002/health`
- API docs: `http://<your-server-ip>:8002/docs`

## Next Steps

1. **Test the server:**
   ```bash
   curl http://127.0.0.1:8002/health
   ```

2. **Create voice profiles:**
   ```bash
   # Upload audio files to create profiles
   scripts/create_voice_profiles_from_folder.sh /path/to/audio/folder http://127.0.0.1:8002
   ```

3. **Generate audio:**
   ```bash
   scripts/generate_audio_from_text.py http://127.0.0.1:8002 data/target_texts.txt
   ```

## Performance Notes for H100

Your H100 GPU should provide excellent performance:

- Expected model loading time: 20-40 seconds
- Expected synthesis latency: < 1 second per request
- vLLM with PagedAttention will optimize memory usage
- Consider increasing `GPU_MEMORY_UTILIZATION` to 0.8-0.9 for better throughput
