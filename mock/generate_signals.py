import os
import numpy as np
import scipy.io.wavfile as wav

_RNG = np.random.default_rng(42)

def generate_synthetic_data(duration_sec=10, sample_rate=48000, infected=False, add_silence=False,
                            gain_range=(0.2, 5.0), morse_message="SOS", morse_dot_s=0.15):
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)

    # 1. Simulate Baseline Server Room Noise (White Noise + 50Hz/60Hz Main Hum)
    white_noise = _RNG.normal(0, 0.005, size=t.shape)
    electrical_hum = 0.002 * np.sin(2 * np.pi * 50 * t) + 0.001 * np.sin(2 * np.pi * 100 * t)

    signal = white_noise + electrical_hum

    # 2. If infected, overlay a covert channel signal: a 4 kHz tone keyed with
    # real Morse timing (dsp/morse.py), not just a flat on-every-other-second
    # toggle. mock/fan_morse.py already does real Morse timing for the
    # hardware/fan demo; this brings the virtual/simulated attack path (what
    # the test suite and training actually exercise) into line with it, so
    # "Morse fan pulse" in the brief is genuinely true end to end: the
    # generator here encodes `morse_message`, and dsp.morse.decode_timing
    # (fed from per-patch tone presence, see presence_to_segments) recovers
    # it back from the detector's observations.
    if infected:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from dsp.morse import encode_timing

        pulse_frequency = 4000  # 4 kHz covert channel transmission
        pulse_train = np.zeros_like(t)

        message_segments = encode_timing(morse_message, dot_s=morse_dot_s, lead_in_s=morse_dot_s * 2)
        # A clearly-longer pause between repeats than the standard 7-unit
        # inter-word gap, so repeats of the message don't read as one long
        # run-on word.
        repeat_gap_s = morse_dot_s * 14
        if message_segments:
            cursor = 0.0
            while cursor < duration_sec:
                for is_on, duration in message_segments:
                    if cursor >= duration_sec:
                        break
                    start_idx = int(cursor * sample_rate)
                    end_idx = min(int((cursor + duration) * sample_rate), len(t))
                    if is_on and end_idx > start_idx:
                        pulse_train[start_idx:end_idx] = 1.0
                    cursor += duration
                cursor += repeat_gap_s

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
    """Generate diverse training patches with silence and gain augmentation.

    This writes a richer, augmented patch set (many short clips across a
    range of gains, some with silence) to data/baseline_patches_augmented.npy.
    train/run_training.py trains on this set (not the single-clip
    data/baseline_patches.npy produced by dsp/patch_generator.py from
    data/baseline/normal_room.wav) specifically so the autoencoder sees the
    x0.2-x5.0 gain range below during training -- that's what stops it
    tripping on a louder room at inference time.

    Patches are sliced with a 50%-overlap sliding window (step =
    patch_size // 2), so neighbouring patches from the same `i`-th synthetic
    clip are near-duplicates of each other. A plain per-patch shuffle would
    therefore leak validation information into training. To let
    train/run_training.py hold out whole recordings instead, this also
    writes data/baseline_patches_augmented_groups.npy: one integer recording
    id per patch (the `i` loop index below), aligned with the patches array.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    os.makedirs(output_dir, exist_ok=True)

    all_patches = []
    all_groups = []

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
                all_groups.append(i)

    if all_patches:
        patches_array = np.stack(all_patches)
        groups_array = np.asarray(all_groups, dtype=np.int32)
        out_path = "data/baseline_patches_augmented.npy"
        groups_path = "data/baseline_patches_augmented_groups.npy"
        np.save(out_path, patches_array)
        np.save(groups_path, groups_array)
        print(f"[+] Generated {len(patches_array)} training patches with augmentation across {num_samples} recordings")
        print(f"[+] Saved to {out_path}")
        print(f"[+] Saved per-patch recording ids to {groups_path} (used for a recording-independent held-out split)")

    return all_patches

if __name__ == "__main__":
    os.makedirs("data/baseline", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)
    
    # These two clips are the canonical "normal room" / "attack" fixtures:
    # dsp/patch_generator.py slices them into data/baseline_patches.npy and
    # data/test_patches.npy, which tests/test_virtual_detection.py and
    # train/calibrate_threshold.py's attack-detection sanity check treat as
    # a ~1x-loudness reference and then scale up to simulate a louder room
    # (see test_autoencoder_tolerates_louder_room_after_augmented_training).
    # Leaving gain_range at its default here let it fall through to
    # generate_synthetic_data's random `_RNG.uniform(0.2, 5.0)` draw, so
    # "normal_room.wav" silently baked in an arbitrary gain (it worked out to
    # ~2.27x) that depends on the _RNG call order of everything before it in
    # this script -- any unrelated edit upstream (e.g. to the training-set
    # loop) reshuffles it again. A scaled-by-3 "normal" room was then
    # actually ~6.8x louder than the unscaled clip: well outside the
    # x0.2-x5.0 range mock/generate_signals.py's own augmentation covers, so
    # the AE had never seen anything like it. Pin both fixtures to a fixed,
    # documented gain of 1.0 instead so "the room got Nx louder" in a test
    # means exactly Nx, independent of RNG draw order elsewhere.
    print("[*] Generating clean baseline server room audio...")
    sr, baseline_audio = generate_synthetic_data(duration_sec=20, infected=False, gain_range=(1.0, 1.0))
    wav.write("data/baseline/normal_room.wav", sr, baseline_audio)
    print("[+] Saved data/baseline/normal_room.wav")

    print("[*] Generating infected covert channel attack audio...")
    sr, infected_audio = generate_synthetic_data(duration_sec=20, infected=True, gain_range=(1.0, 1.0))
    wav.write("data/test/infected_attack.wav", sr, infected_audio)
    print("[+] Saved data/test/infected_attack.wav")
    
    # duration_sec=4 (was the function default of 2) roughly doubles the
    # underlying normal-hum audio the model trains on, to 200*4s = 800s
    # (~13.3 min) -- up from ~6.7 min, and a long way up from the ~1 minute
    # (1,404 patches at the old single-clip 43 ms/patch) the review flagged.
    # Literal "hours" was considered and rejected: patches scale with total
    # audio duration regardless of how it's chunked into recordings, and the
    # augmented set is already ~2.4 GB on disk at the current size, so an
    # hour (9x more audio) would be tens of GB on disk plus a large, risky
    # jump in peak RAM while np.stack()-ing the in-memory patch list during
    # generation (see train/run_training.py). Doubling is a safer, still
    # honest improvement; bump num_samples and/or duration_sec further here
    # if your machine has the RAM/disk for it, and update the README's
    # Section 4 note to match.
    print("[*] Generating augmented training dataset...")
    generate_training_dataset(num_samples=200, duration_sec=4)
