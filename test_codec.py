import torch
from neucodec import NeuCodec
import torchaudio

print("Loading codec...")
codec = NeuCodec.from_pretrained('neuphonic/neucodec').to('cpu')
print("Codec loaded!")

# Create a simple test waveform
waveform = torch.randn(1, 1, 16000)  # 1 second at 16kHz
print(f"Test waveform shape: {waveform.shape}")

# Try encoding
try:
    codes = codec.encode_code(waveform)
    print(f'✓ Success! Codes shape: {codes.shape}')
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()
