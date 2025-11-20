#!/usr/bin/env python3
"""
Script to automatically create voice profiles from audio/text pairs in a folder.
This script processes all .wav/.txt file pairs and creates voice profiles using the NeuTTS-Air API.
"""

import os
import sys
import json
import requests
from pathlib import Path

def find_audio_text_pairs(folder_path):
    """
    Find all .wav/.txt file pairs in the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing audio/text pairs
        
    Returns:
        list: List of tuples (wav_file_path, txt_file_path, profile_name)
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Get all .wav files
    wav_files = list(folder.glob("*.wav"))
    pairs = []
    
    for wav_file in wav_files:
        # Look for corresponding .txt file
        txt_file = wav_file.with_suffix('.txt')
        if txt_file.exists():
            # Use the base name (without extension) as the profile name
            profile_name = wav_file.stem
            pairs.append((str(wav_file), str(txt_file), profile_name))
        else:
            print(f"⚠️  Warning: No corresponding .txt file found for {wav_file.name}")
    
    return pairs

def read_text_file(file_path):
    """
    Read text content from a file.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        str: Content of the text file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"❌ Error reading text file {file_path}: {e}")
        return ""

def create_voice_profile(api_url, wav_path, text_content, profile_name, description=""):
    """
    Create a voice profile using the NeuTTS-Air API endpoint.
    
    Args:
        api_url (str): Base URL of the API
        wav_path (str): Path to the audio file
        text_content (str): Reference text for the audio
        profile_name (str): Name for the voice profile
        description (str): Description for the voice profile
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Prepare the files and data for the POST request
        with open(wav_path, 'rb') as audio_file:
            files = {
                'audio_file': (os.path.basename(wav_path), audio_file, 'audio/wav')
            }
            
            data = {
                'name': profile_name,
                'description': description or f"Voice profile for {profile_name}",
                'reference_text': text_content
            }
            
            # Make the API request
            response = requests.post(
                f"{api_url}/v1/voice-profiles",
                files=files,
                data=data,
                timeout=120  # 2 minute timeout for encoding
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Successfully created voice profile: {profile_name}")
                print(f"   Speech codes: {len(result.get('speech_codes', []))} tokens")
                return True
            else:
                print(f"❌ Failed to create voice profile {profile_name}: {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"   Error: {error_detail}")
                except:
                    print(f"   Response: {response.text[:200]}")
                return False
                
    except Exception as e:
        print(f"❌ Error creating voice profile {profile_name}: {e}")
        return False

def main():
    """
    Main function to process all audio/text pairs and create voice profiles.
    """
    # Configuration
    folder_path = sys.argv[1] if len(sys.argv) > 1 else "data/tts/.dset/vp"
    api_url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8002"
    
    print("=" * 60)
    print("🎤 NeuTTS-Air Voice Profile Batch Creator")
    print("=" * 60)
    print(f"📁 Processing audio/text pairs from: {folder_path}")
    print(f"🌐 API URL: {api_url}")
    print()
    
    try:
        # Find all audio/text pairs
        pairs = find_audio_text_pairs(folder_path)
        
        if not pairs:
            print("❌ No audio/text pairs found in the specified folder.")
            print("   Please ensure your folder contains .wav files with corresponding .txt files")
            return
        
        print(f"✓ Found {len(pairs)} audio/text pairs to process.\n")
        
        # Process each pair
        success_count = 0
        for idx, (wav_path, txt_path, profile_name) in enumerate(pairs, 1):
            print(f"[{idx}/{len(pairs)}] Processing: {profile_name}")
            
            # Read the reference text
            text_content = read_text_file(txt_path)
            if not text_content:
                print(f"⚠️  Skipping {profile_name} due to empty or unreadable text file.\n")
                continue
            
            print(f"   Reference text: {text_content[:60]}{'...' if len(text_content) > 60 else ''}")
            
            # Create the voice profile
            if create_voice_profile(api_url, wav_path, text_content, profile_name):
                success_count += 1
            print()
        
        print("=" * 60)
        print(f"✅ Completed! Successfully created {success_count} out of {len(pairs)} voice profiles.")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
