import os
import numpy as np
from dsp.pipeline import extract_spectrogram

def generate_and_save_patches(audio_file_path, output_npy_path, patch_size=64):
    # 1. Extract the frequency matrix from the DSP pipeline
    spectrogram = extract_spectrogram(audio_file_path)
    num_frames, num_frequencies = spectrogram.shape
    
    patches = []
    
    # 2. Slice the matrix timeline into discrete patches of fixed length
    # Step through the timeline jumping by the length of a patch
    for i in range(0, num_frames - patch_size + 1, patch_size):
        patch = spectrogram[i:i + patch_size, :]
        patches.append(patch)
        
    patches_array = np.array(patches, dtype=np.float32)
    
    # Ensure output directory exists and write out the binary file
    os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
    np.save(output_npy_path, patches_array)
    
    return patches_array.shape

if __name__ == "__main__":
    print("[*] Slicing spectrogram data into structured training tensors...")
    
    # Process the clean baseline data
    baseline_shape = generate_and_save_patches(
        "data/baseline/normal_room.wav", 
        "data/baseline_patches.npy"
    )
    print(f"[+] Created clean baseline tensors: {baseline_shape}")
    
    # Process the infected attack data
    attack_shape = generate_and_save_patches(
        "data/test/infected_attack.wav", 
        "data/test_patches.npy"
    )
    print(f"[+] Created infected test tensors: {attack_shape}")
    print("[+] Phase 4: Patch generation completed successfully!")
