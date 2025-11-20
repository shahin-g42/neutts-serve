#!/usr/bin/env python3
"""Verification script to check if all components are properly installed."""

import sys
import os

def check_file(path, description):
    if os.path.exists(path):
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}: {path} NOT FOUND")
        return False

def main():
    print("=" * 60)
    print("NeuTTS-Air TTS Service - Installation Verification")
    print("=" * 60)
    
    base_path = "/Users/shahin.konadath/Documents/project/v7/neutts-serve"
    os.chdir(base_path)
    
    checks = [
        ("requirements.txt", "Dependencies file"),
        (".env", "Configuration file"),
        ("main.py", "Main application"),
        ("core/settings.py", "Settings module"),
        ("core/exceptions.py", "Exceptions module"),
        ("core/logging.py", "Logging module"),
        ("core/neutts_wrapper.py", "NeuTTSAir wrapper"),
        ("core/tts_model.py", "TTS model wrapper"),
        ("api/tts.py", "TTS API"),
        ("api/voice_profiles.py", "Voice profiles API"),
        ("schemas/tts.py", "TTS schemas"),
        ("schemas/voice_profiles.py", "Voice profile schemas"),
        ("utils/audio_utils.py", "Audio utilities"),
        ("utils/audio_processor.py", "Audio processor"),
        ("README.md", "Documentation"),
    ]
    
    results = []
    for path, desc in checks:
        results.append(check_file(path, desc))
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Verification Result: {passed}/{total} checks passed")
    
    if passed == total:
        print("✅ All components installed successfully!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Install espeak-ng system package")
        print("3. Start service: python main.py")
    else:
        print("❌ Some components are missing. Please check the installation.")
        return 1
    
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
