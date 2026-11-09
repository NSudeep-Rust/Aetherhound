import os
import numpy as np
import scipy.io.wavfile as wav

def generate_synthetic_data(duration_sec=10, sample_rate=48000, infected=False):
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    
    # 1. Simulate Baseline Server Room Noise (White Noise + 50Hz/60Hz Main Hum)
    white_noise = np.random.normal(0, 0.005, size=t.shape)
    electrical_hum = 0.002 * np.sin(2 * np.pi * 50 * t) + 0.001 * np.sin(2 * np.pi * 100 * t)
    
    signal = white_noise + electrical_hum
    
    # 2. If infected, overlay a covert channel signal (e.g., a pulsing 4000Hz frequency)
    if infected:
        pulse_frequency = 4000  # 4 kHz covert channel transmission
        pulse_train = np.zeros_like(t)
        
        # Create a simple rhythmic on-off pulse pattern (Morse-code style)
        # Pulse turns on every alternate second
        for sec in range(duration_sec):
            if sec % 2 == 1:
                start_idx = int(sec * sample_rate)
                end_idx = int((sec + 0.5) * sample_rate)
                pulse_train[start_idx:end_idx] = 1.0
                
        covert_signal = 0.008 * np.sin(2 * np.pi * pulse_frequency * t) * pulse_train
        signal += covert_signal
        
    # Prevent clipping and normalize to 16-bit PCM scale
    signal = np.clip(signal, -1.0, 1.0)
    return sample_rate, (signal * 32767).astype(np.int16)

if __name__ == "__main__":
    os.makedirs("data/baseline", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)
    
    print("[*] Generating clean baseline server room audio...")
    sr, baseline_audio = generate_synthetic_data(duration_sec=20, infected=False)
    wav.write("data/baseline/normal_room.wav", sr, baseline_audio)
    print("[+] Saved data/baseline/normal_room.wav")
    
    print("[*] Generating infected covert channel attack audio...")
    sr, infected_audio = generate_synthetic_data(duration_sec=20, infected=True)
    wav.write("data/test/infected_attack.wav", sr, infected_audio)
    print("[+] Saved data/test/infected_attack.wav")
