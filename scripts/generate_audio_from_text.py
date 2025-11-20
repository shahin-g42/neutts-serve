#!/usr/bin/env python3
"""
Script to generate audio from text using voice profiles.
This script:
1. Gets all available voice profiles from the NeuTTS-Air API
2. Loads texts from a file
3. Generates audio using random voice profile and text combinations
"""

import os
import sys
import json
import random
import requests
from pathlib import Path
from typing import List, Dict, Any

def get_voice_profiles(api_url: str) -> List[Dict[str, Any]]:
    """
    Get all available voice profiles from the API.
    
    Args:
        api_url (str): Base URL of the API
        
    Returns:
        List[Dict[str, Any]]: List of voice profiles
    """
    try:
        response = requests.get(f"{api_url}/v1/voice-profiles")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error getting voice profiles: {e}")
        return []

def load_texts_from_file(file_path: str) -> List[str]:
    """
    Load texts from a file, one text per line.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        List[str]: List of texts
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read lines and strip whitespace, filter out empty lines
            texts = [line.strip() for line in f.readlines() if line.strip()]
        return texts
    except Exception as e:
        print(f"❌ Error loading texts from file {file_path}: {e}")
        return []

def generate_audio(api_url: str, text: str, voice_profile_name: str, output_file: str, response_format: str = "wav") -> bool:
    """
    Generate audio from text using a specific voice profile.
    
    Args:
        api_url (str): Base URL of the API
        text (str): Text to convert to speech
        voice_profile_name (str): Name of the voice profile to use
        output_file (str): Path to save the generated audio file
        response_format (str): Audio format (wav, mp3, etc.)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Prepare the request data
        data = {
            "model": "tts-1",
            "input": text,
            "voice": voice_profile_name,
            "response_format": response_format
        }
        
        # Make the API request
        response = requests.post(
            f"{api_url}/v1/audio/speech",
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            # Save the audio content to file
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            # Get file size
            file_size = len(response.content) / 1024  # KB
            print(f"✅ Successfully generated audio: {output_file} ({file_size:.1f} KB)")
            return True
        else:
            print(f"❌ Failed to generate audio: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   Error: {error_detail}")
            except:
                print(f"   Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Error generating audio: {e}")
        return False

def main():
    """
    Main function to generate audio from text using voice profiles.
    """
    # Configuration
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8002"
    texts_file = sys.argv[2] if len(sys.argv) > 2 else "data/target_texts.txt"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "generated_audio"
    num_generations = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    response_format = sys.argv[5] if len(sys.argv) > 5 else "mp3"  # Default to MP3
    
    print("=" * 60)
    print("🔊 NeuTTS-Air Audio Generator")
    print("=" * 60)
    print(f"🌐 API URL: {api_url}")
    print(f"📄 Texts file: {texts_file}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔢 Number of generations: {num_generations}")
    print(f"🎵 Audio format: {response_format}")
    print()
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get voice profiles
    print("📋 Getting voice profiles...")
    voice_profiles = get_voice_profiles(api_url)
    
    if not voice_profiles:
        print("❌ No voice profiles found. Please create some voice profiles first.")
        print("   Use: scripts/create_voice_profiles_from_folder.py")
        return
    
    print(f"✓ Found {len(voice_profiles)} voice profiles:")
    for vp in voice_profiles:
        print(f"   • {vp['name']}")
    print()
    
    # Load texts
    print(f"📝 Loading texts from {texts_file}...")
    texts = load_texts_from_file(texts_file)
    
    if not texts:
        print(f"❌ No texts found in {texts_file}. Please check the file.")
        return
    
    print(f"✓ Loaded {len(texts)} texts.\n")
    
    # Generate audio files
    print("🎵 Generating audio files...")
    print("=" * 60)
    generated_count = 0
    
    # Generate random combinations
    for i in range(num_generations):
        # Select random voice profile and text
        voice_profile = random.choice(voice_profiles)
        text = random.choice(texts)
        
        # Create output filename with correct extension
        output_file = os.path.join(
            output_dir, 
            f"generated_{i+1:03d}_{voice_profile['name']}.{response_format}"
        )
        
        print(f"\n[{i+1}/{num_generations}] Generating audio:")
        print(f"   Voice: {voice_profile['name']}")
        print(f"   Text: {text[:70]}{'...' if len(text) > 70 else ''}")
        
        # Generate audio with specified format
        if generate_audio(api_url, text, voice_profile['name'], output_file, response_format):
            generated_count += 1
    
    print()
    print("=" * 60)
    print(f"✅ Completed! Generated {generated_count}/{num_generations} audio files in {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
