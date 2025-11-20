#!/bin/bash

# Script to generate audio from text using voice profiles
# Usage: ./generate_audio_from_text.sh [api_url] [texts_file] [output_dir] [num_generations]

API_URL=${1:-"http://127.0.0.1:8002"}
TEXTS_FILE=${2:-"data/target_texts.txt"}
OUTPUT_DIR=${3:-"generated_audio"}
NUM_GENERATIONS=${4:-10}

echo "============================================================"
echo "🔊 NeuTTS-Air Audio Generator"
echo "============================================================"
echo "🌐 API URL: $API_URL"
echo "📄 Texts file: $TEXTS_FILE"
echo "📁 Output directory: $OUTPUT_DIR"
echo "🔢 Number of generations: $NUM_GENERATIONS"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Get voice profiles
echo "📋 Getting voice profiles..."
VOICE_PROFILES=$(curl -s "$API_URL/v1/voice-profiles")
if [ $? -ne 0 ] || [ "$VOICE_PROFILES" = "[]" ]; then
    echo "❌ No voice profiles found or error getting voice profiles."
    echo "   Please create some voice profiles first using:"
    echo "   scripts/create_voice_profiles_from_folder.sh"
    exit 1
fi

# Count voice profiles
VOICE_COUNT=$(echo "$VOICE_PROFILES" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))")
echo "✓ Found $VOICE_COUNT voice profiles:"
echo "$VOICE_PROFILES" | python3 -c "import sys, json; profiles = json.load(sys.stdin); [print(f\"   • {p['name']}\") for p in profiles]"
echo ""

# Check if texts file exists
if [ ! -f "$TEXTS_FILE" ]; then
    echo "❌ Texts file not found: $TEXTS_FILE"
    exit 1
fi

# Count texts (non-empty lines)
TEXT_COUNT=$(grep -c '[^[:space:]]' "$TEXTS_FILE")
if [ "$TEXT_COUNT" -eq 0 ]; then
    echo "❌ No texts found in $TEXTS_FILE"
    exit 1
fi

echo "📝 Loading texts from $TEXTS_FILE..."
echo "✓ Loaded $TEXT_COUNT texts."
echo ""

# Generate audio files
echo "🎵 Generating audio files..."
echo "============================================================"
GENERATED_COUNT=0

# Generate random combinations
for i in $(seq 1 $NUM_GENERATIONS); do
    # Select random voice profile (get a random index)
    RANDOM_VOICE_INDEX=$((RANDOM % VOICE_COUNT))
    VOICE_NAME=$(echo "$VOICE_PROFILES" | python3 -c "import sys, json; print(json.load(sys.stdin)[$RANDOM_VOICE_INDEX]['name'])")
    
    # Select random text (get a random non-empty line from the file)
    # First get all non-empty line numbers
    TEXT_LINE_NUM=$((RANDOM % TEXT_COUNT + 1))
    TEXT=$(awk 'NF {print; count++} count=='"$TEXT_LINE_NUM"' {exit}' "$TEXTS_FILE" | tail -1)
    
    # Skip if text is empty
    if [ -z "$TEXT" ]; then
        continue
    fi
    
    # Create output filename
    OUTPUT_FILE="$OUTPUT_DIR/generated_$(printf "%03d" $i)_${VOICE_NAME}.wav"
    
    echo ""
    echo "[$i/$NUM_GENERATIONS] Generating audio:"
    echo "   Voice: $VOICE_NAME"
    echo "   Text: ${TEXT:0:70}$([ ${#TEXT} -gt 70 ] && echo '...' || echo '')"
    
    # Generate audio using curl
    RESPONSE=$(curl -s -w "\n%{http_code}" -o "$OUTPUT_FILE" -X POST \
        "$API_URL/v1/audio/speech" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"tts-1\",
            \"input\": $(echo "$TEXT" | python3 -c "import sys, json; print(json.dumps(sys.stdin.read().strip()))"),
            \"voice\": \"$VOICE_NAME\",
            \"response_format\": \"wav\"
        }" \
        --max-time 60)
    
    # Extract HTTP code (last line)
    HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)
    
    if [ "$HTTP_CODE" -eq 200 ]; then
        # Get file size
        FILE_SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
        echo "✅ Successfully generated audio: $OUTPUT_FILE ($FILE_SIZE)"
        GENERATED_COUNT=$((GENERATED_COUNT + 1))
    else
        echo "❌ Failed to generate audio: HTTP $HTTP_CODE"
        rm -f "$OUTPUT_FILE"
    fi
done

echo ""
echo "============================================================"
echo "✅ Completed! Generated $GENERATED_COUNT/$NUM_GENERATIONS audio files in $OUTPUT_DIR"
echo "============================================================"
