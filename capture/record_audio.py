import os
import time
import numpy as np
import soundfile as sf
import sounddevice as sd

# Simple mock loader if configparser/toml is not expanded yet
def get_audio_config():
    # Hardcoded fallback mapping directly to our verified config.toml values
    return {"device": 4, "rate": 48000, "channels": 1, "blocksize": 1024}

def record_fixed_duration(output_filename, duration_sec=10):
    config = get_audio_config()
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    print(f"[*] Initializing audio stream on Device {config['device']}...")
    print(f"[*] Recording ambient sound for {duration_sec} seconds... Speak or generate noise near mic.")
    
    # Capture block of data synchronously
    audio_data = sd.rec(
        int(duration_sec * config['rate']),
        samplerate=config['rate'],
        channels=config['channels'],
        device=config['device'],
        dtype='float32'
    )
    sd.wait()  # Wait until the recording is fully finished
    
    # Save the recorded float32 array as a standard WAV file
    sf.write(output_filename, audio_data, config['rate'])
    print(f"[+] Recording complete! Saved raw audio file to: {output_filename}")

if __name__ == "__main__":
    test_path = "data/raw/live_ambient_test.wav"
    record_fixed_duration(test_path, duration_sec=5)
