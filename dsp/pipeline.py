import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav

def extract_spectrogram(file_path, nfft=512, hop_size=32):
    # 1. Load the wav audio file
    sample_rate, audio_data = wav.read(file_path)
    
    # If audio is stereo, convert to mono by taking the first channel
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
        
    # Convert integer PCM data back to float scale (-1.0 to 1.0)
    audio_float = audio_data.astype(np.float32) / 32767.0
    
    # 2. Compute the Short-Time Fourier Transform (STFT)
    frequencies, times, spec_matrix = signal.stft(
        audio_float, 
        fs=sample_rate, 
        nperseg=nfft, 
        noverlap=nfft - hop_size
    )
    
    # 3. Convert complex voltage values to absolute power magnitude scale
    magnitude_spec = np.abs(spec_matrix)
    
    return magnitude_spec.T  # Transpose so time frames are rows, frequencies are columns

if __name__ == "__main__":
    import os
    print("[*] Testing DSP Pipeline extraction on mock files...")
    
    # Process the normal room baseline file we generated
    baseline_matrix = extract_spectrogram("data/baseline/normal_room.wav")
    print(f"[+] Baseline Spectrogram Matrix Shape: {baseline_matrix.shape}")
    
    # Verify shape dimensions are correct
    assert len(baseline_matrix.shape) == 2, "DSP output must be a 2D matrix!"
    print("[+] DSP Pipeline verified successfully!")
