import os
import numpy as np
import scipy.io.wavfile as wav

_RNG = np.random.default_rng(42)

def generate_synthetic_data(duration_sec=10, sample_rate=48000, infected=False, add_silence=False, gain_range=(0.2, 5.0)):
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    
    # 1. Simulate Baseline Server Room Noise (White Noise + 50Hz/60Hz Main Hum)
    white_noise = _RNG.normal(0, 0.005, size=t.shape)
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
    
    # 3. Add silence segments (for training the NO_SIGNAL state)
    if add_silence:
        # Add 1-2 seconds of pure silence at random positions
        silence_duration = _RNG.uniform(1.0, 2.0)
        silence_samples = int(silence_duration * sample_rate)
        silence_start = _RNG.integers(0, max(1, len(signal) - silence_samples))
        signal[silence_start:silence_start + silence_samples] = 0.0
    
    # 4. Apply random gain augmentation (simulates different mic levels / distances)
    gain = _RNG.uniform(gain_range[0], gain_range[1])
    signal = signal * gain
        
    # Prevent clipping and normalize to 16-bit PCM scale
    signal = np.clip(signal, -1.0, 1.0)
    return sample_rate, (signal * 32767).astype(np.int16)

def generate_training_dataset(output_dir="data/baseline", num_samples=100, sample_rate=48000, duration_sec=2):
    """Generate diverse training patches with silence and gain augmentation."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_patches = []
    
    for i in range(num_samples):
        # Mix of clean, silence, and varying gains
        add_silence = _RNG.random() < 0.1  # 10% have silence segments
        gain_range = (0.2, 5.0)
        
        sr, audio = generate_synthetic_data(
            duration_sec=duration_sec,
            sample_rate=sample_rate,
            infected=False,
            add_silence=add_silence,
            gain_range=gain_range
        )
        
        # Convert to float
        audio_float = audio.astype(np.float32) / 32767.0
        
        # Extract spectrogram patches
        from dsp.pipeline import spectrogram_from_array, apply_filters
        from config_loader import load_config
        
        cfg = load_config()
        filt = apply_filters(audio_float, sample_rate, cfg.band)
        spec = spectrogram_from_array(filt, sample_rate, cfg.dsp.nfft, cfg.dsp.hop)
        
        # Split into patches
        patch_size = cfg.dsp.patch
        for j in range(0, spec.shape[0] - patch_size + 1, patch_size // 2):
            patch = spec[j:j+patch_size]
            if patch.shape[0] == patch_size:
                all_patches.append(patch)
    
    if all_patches:
        patches_array = np.stack(all_patches)
        np.save(os.path.join(output_dir, "baseline_patches.npy"), patches_array)
        print(f"[+] Generated {len(patches_array)} training patches with augmentation")
        print(f"[+] Saved to {output_dir}/baseline_patches.npy")
    
    return all_patches

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
    
    print("[*] Generating augmented training dataset...")
    generate_training_dataset(num_samples=200)
