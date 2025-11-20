#!/bin/bash

# Script to automatically create voice profiles from audio/text pairs using curl
# Usage: ./create_voice_profiles_from_folder.sh [folder_path] [api_url]

FOLDER_PATH=${1:-"data/voice_samples"}
API_URL=${2:-"http://127.0.0.1:8002"}

echo "============================================================"
echo "🎤 NeuTTS-Air Voice Profile Batch Creator"
echo "============================================================"
echo "📁 Processing audio/text pairs from: $FOLDER_PATH"
echo "🌐 API URL: $API_URL"
echo ""

# Check if folder exists
if [ ! -d "$FOLDER_PATH" ]; then
    echo "❌ Error: Folder not found: $FOLDER_PATH"
    exit 1
fi

# Count WAV files
WAV_COUNT=$(find "$FOLDER_PATH" -name "*.wav" | wc -l | tr -d ' ')
if [ "$WAV_COUNT" -eq 0 ]; then
    echo "❌ No .wav files found in the specified folder."
    exit 0
fi

echo "✓ Found $WAV_COUNT .wav files to process."
echo ""

SUCCESS_COUNT=0
TOTAL_COUNT=0

# Process each WAV file
for WAV_FILE in "$FOLDER_PATH"/*.wav; do
    # Get the base name (without path and extension)
    BASE_NAME=$(basename "$WAV_FILE" .wav)
    
    # Look for corresponding TXT file
    TXT_FILE="$FOLDER_PATH/$BASE_NAME.txt"
    
    if [ ! -f "$TXT_FILE" ]; then
        echo "⚠️  Warning: No corresponding .txt file found for $BASE_NAME.wav"
        continue
    fi
    
    # Read the reference text
    REFERENCE_TEXT=$(cat "$TXT_FILE" | tr '\n' ' ' | sed 's/ $//')
    
    if [ -z "$REFERENCE_TEXT" ]; then
        echo "⚠️  Skipping $BASE_NAME due to empty text file."
        continue
    fi
    
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    echo "[$TOTAL_COUNT/$WAV_COUNT] Processing: $BASE_NAME"
    echo "   Reference text: ${REFERENCE_TEXT:0:60}$([ ${#REFERENCE_TEXT} -gt 60 ] && echo '...' || echo '')"
    
    # Create voice profile using curl
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
        "$API_URL/v1/voice-profiles" \
        -F "name=$BASE_NAME" \
        -F "description=Voice profile for $BASE_NAME" \
        -F "reference_text=$REFERENCE_TEXT" \
        -F "audio_file=@$WAV_FILE" \
        --max-time 120)
    
    # Extract HTTP code (last line)
    HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)
    # Extract response body (all but last line)
    RESPONSE_BODY=$(echo "$RESPONSE" | sed '$d')
    
    if [ "$HTTP_CODE" -eq 200 ]; then
        SPEECH_CODES=$(echo "$RESPONSE_BODY" | grep -o '"speech_codes":\[[^]]*\]' | grep -o '\[.*\]' | grep -o ',' | wc -l | tr -d ' ')
        SPEECH_CODES=$((SPEECH_CODES + 1))
        echo "✅ Successfully created voice profile: $BASE_NAME"
        echo "   Speech codes: $SPEECH_CODES tokens"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ Failed to create voice profile $BASE_NAME: HTTP $HTTP_CODE"
        echo "   Response: $(echo "$RESPONSE_BODY" | head -c 200)"
    fi
    
    echo ""
done

echo "============================================================"
echo "✅ Completed! Successfully created $SUCCESS_COUNT out of $TOTAL_COUNT voice profiles."
echo "============================================================"
