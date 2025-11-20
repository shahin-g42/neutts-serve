#!/bin/bash
cd /Users/shahin.konadath/Documents/project/v7/neutts-serve
source .venv/bin/activate

# Start server in background
python3 main.py > /tmp/server.log 2>&1 &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server to start..."
sleep 35

# Check if server is running
if ! ps -p $SERVER_PID > /dev/null; then
   echo "ERROR: Server failed to start!"
   cat /tmp/server.log
   exit 1
fi

echo "Testing health endpoint..."
curl -s http://127.0.0.1:8002/health | python3 -m json.tool

echo -e "\n\nTesting voice profile creation..."
curl -s -X POST http://127.0.0.1:8002/v1/voice-profiles \
  -F "name=test_voice_manual" \
  -F "description=Test voice profile" \
  -F "reference_text=Hello world, this is a test" \
  -F "audio_file=@tests/test_audio/test_reference.wav" | python3 -m json.tool

echo -e "\n\nTesting voice profile list..."
curl -s http://127.0.0.1:8002/v1/voice-profiles | python3 -m json.tool

# Kill server
kill $SERVER_PID 2>/dev/null
echo -e "\n\nServer stopped"
