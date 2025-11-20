#!/usr/bin/env python3
"""
Verify espeak-ng installation and phonemizer configuration.
Run this on your Linux system to diagnose phonemizer issues.
"""

import sys
import os
from pathlib import Path

def check_espeak_binary():
    """Check if espeak-ng binary is available."""
    print("=" * 60)
    print("1. Checking espeak-ng binary...")
    print("=" * 60)
    
    # Try to find espeak-ng
    import shutil
    espeak_path = shutil.which('espeak-ng')
    
    if espeak_path:
        print(f"✅ espeak-ng found at: {espeak_path}")
        return espeak_path
    else:
        print("❌ espeak-ng not found in PATH")
        print("   Install with: sudo apt-get install espeak-ng")
        return None

def check_espeak_library():
    """Check if libespeak-ng library exists."""
    print("\n" + "=" * 60)
    print("2. Checking libespeak-ng library...")
    print("=" * 60)
    
    # Common library paths on Linux
    common_paths = [
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so",
        "/usr/lib/libespeak-ng.so",
        "/usr/lib64/libespeak-ng.so",
        "/usr/local/lib/libespeak-ng.so",
    ]
    
    found_libraries = []
    for lib_path in common_paths:
        if Path(lib_path).exists():
            found_libraries.append(lib_path)
            print(f"✅ Found library: {lib_path}")
    
    if not found_libraries:
        print("❌ libespeak-ng.so not found in common locations")
        print("   Searching...")
        import subprocess
        try:
            result = subprocess.run(
                ['find', '/usr/lib', '-name', 'libespeak-ng.so*'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.stdout.strip():
                print(f"   Found at non-standard location:\n{result.stdout}")
                found_libraries = result.stdout.strip().split('\n')
        except Exception as e:
            print(f"   Search failed: {e}")
    
    return found_libraries[0] if found_libraries else None

def test_phonemizer():
    """Test if phonemizer can use espeak-ng."""
    print("\n" + "=" * 60)
    print("3. Testing phonemizer with espeak-ng...")
    print("=" * 60)
    
    try:
        from phonemizer.backend import EspeakBackend
        
        # Try to initialize
        backend = EspeakBackend(
            language='en-us',
            preserve_punctuation=True,
            with_stress=True
        )
        
        # Try to phonemize
        test_text = "Hello world"
        result = backend.phonemize([test_text])
        
        print(f"✅ Phonemizer works!")
        print(f"   Input: '{test_text}'")
        print(f"   Output: '{result[0]}'")
        return True
        
    except Exception as e:
        print(f"❌ Phonemizer failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

def check_environment_variables():
    """Check if environment variables are set."""
    print("\n" + "=" * 60)
    print("4. Checking environment variables...")
    print("=" * 60)
    
    espeak_path = os.environ.get('PHONEMIZER_ESPEAK_PATH')
    espeak_lib = os.environ.get('PHONEMIZER_ESPEAK_LIBRARY')
    
    if espeak_path:
        print(f"✅ PHONEMIZER_ESPEAK_PATH set to: {espeak_path}")
    else:
        print("⚠️  PHONEMIZER_ESPEAK_PATH not set")
    
    if espeak_lib:
        print(f"✅ PHONEMIZER_ESPEAK_LIBRARY set to: {espeak_lib}")
    else:
        print("⚠️  PHONEMIZER_ESPEAK_LIBRARY not set")
    
    return espeak_path, espeak_lib

def test_with_env_vars(binary_path, library_path):
    """Test phonemizer with explicit environment variables."""
    print("\n" + "=" * 60)
    print("5. Testing with explicit paths...")
    print("=" * 60)
    
    if not binary_path or not library_path:
        print("⚠️  Skipping - paths not available")
        return
    
    # Set environment variables
    os.environ['PHONEMIZER_ESPEAK_PATH'] = binary_path
    os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = library_path
    
    print(f"Setting PHONEMIZER_ESPEAK_PATH={binary_path}")
    print(f"Setting PHONEMIZER_ESPEAK_LIBRARY={library_path}")
    
    try:
        # Reload phonemizer to pick up new env vars
        import importlib
        import phonemizer.backend.espeak
        importlib.reload(phonemizer.backend.espeak)
        
        from phonemizer.backend import EspeakBackend
        
        backend = EspeakBackend(
            language='en-us',
            preserve_punctuation=True,
            with_stress=True
        )
        
        test_text = "Testing with environment variables"
        result = backend.phonemize([test_text])
        
        print(f"✅ Success with explicit paths!")
        print(f"   Output: '{result[0]}'")
        
    except Exception as e:
        print(f"❌ Still failed: {e}")

def generate_env_config(binary_path, library_path):
    """Generate .env configuration."""
    print("\n" + "=" * 60)
    print("6. Recommended .env configuration")
    print("=" * 60)
    
    if binary_path and library_path:
        print("\nAdd these lines to your .env file:")
        print("-" * 60)
        print(f"PHONEMIZER_ESPEAK_PATH={binary_path}")
        print(f"PHONEMIZER_ESPEAK_LIBRARY={library_path}")
        print("-" * 60)
    else:
        print("⚠️  Cannot generate config - paths not found")

def main():
    print("\n🔍 espeak-ng and Phonemizer Diagnostic Tool")
    print("=" * 60)
    
    # Run checks
    binary_path = check_espeak_binary()
    library_path = check_espeak_library()
    
    # Check current environment
    env_binary, env_library = check_environment_variables()
    
    # Test basic phonemizer
    if not test_phonemizer():
        # If basic test failed, try with explicit paths
        test_with_env_vars(
            env_binary or binary_path,
            env_library or library_path
        )
    
    # Generate config
    generate_env_config(binary_path, library_path)
    
    print("\n" + "=" * 60)
    print("Diagnostic complete!")
    print("=" * 60)
    
    if binary_path and library_path:
        print("\n✅ All components found. Update your .env file and restart the server.")
    else:
        print("\n❌ Missing components. Please install espeak-ng:")
        print("   sudo apt-get update && sudo apt-get install espeak-ng")

if __name__ == "__main__":
    main()
