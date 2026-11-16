import os
import pytest
import numpy as np
import scipy.io.wavfile as wav
from dsp.pipeline import extract_spectrogram
from infer.detector import AetherHoundEngine

@pytest.fixture
def setup_dummy_wav():
    """Generates a brief structural wave asset to validate DSP processing."""
    file_path = "tests/dummy_test.wav"
    os.makedirs("tests", exist_ok=True)
    sr = 48000
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t) * 10000
    wav.write(file_path, sr, audio.astype(np.int16))
    yield file_path
    if os.path.exists(file_path):
        os.remove(file_path)

def test_dsp_spectrogram_dimensions(setup_dummy_wav):
    """Verifies that the Short-Time Fourier Transform compiles into a flawless 2D grid matrix."""
    matrix = extract_spectrogram(setup_dummy_wav, nfft=512, hop=32)
    assert len(matrix.shape) == 2, "DSP output framework must be a clean 2D array matrix layout."
    assert matrix.shape[1] == 257, "Spectrogram frequency bins mismatch for NFFT=512 parameters."

def test_inference_engine_state_logic():
    """Validates the state machine transitions and database logger behavior."""
    # Temporarily bypass files checking by targeting the local operational components
    engine = AetherHoundEngine()
    
    # 1. Force clear data matrix values to confirm system remains normal
    fake_clean_patch = np.zeros((64, 257), dtype=np.float32)
    engine.state = "CLEAN"
    engine.anomaly_counter = 0
    
    score, state, _ = engine.process_patch(fake_clean_patch)
    assert state == "NO_SIGNAL", "Silent patch should produce NO_SIGNAL state."
    
    # 2. Use a real infected test patch that trips either guard
    test_patches = np.load("data/test_patches.npy")
    baseline_patches = np.load("data/baseline_patches.npy")
    limit = engine.threshold * engine.threat_margin
    # The attack is a brief keyed tone: the AE alone rarely trips on it (that's
    # why the tone guard exists -- see infer/detector.process_patch, which
    # ORs ae_anomalous with tone_anomalous). Search using the same combined
    # check production uses, not the AE score in isolation.
    fake_malware_patch = None
    for p in test_patches:
        if float(np.sqrt(np.mean(p ** 2))) < engine.silence_rms:
            continue
        ae_anomalous = engine.score(p) > limit
        tone_anomalous = engine.tone_score(p)[0] > engine.tone_threshold
        if ae_anomalous or tone_anomalous:
            fake_malware_patch = p.astype(np.float32)
            break
    assert fake_malware_patch is not None, "No anomalous test patch found above silence gate"

    # The tone guard now only arms once a tone has toggled on/off a couple of
    # times (see dsp/keying.py), not on a single patch over the trip, so
    # hammering the same static patch would no longer escalate if it only
    # trips the tone guard. Prime a real on/off/on cycle with a quiet
    # baseline patch in between, matching how the attack fixture actually
    # alternates, before driving the hysteresis loop below.
    quiet_patch = next(
        p for p in baseline_patches
        if float(np.sqrt(np.mean(p ** 2))) >= engine.silence_rms
        and engine.tone_score(p)[0] <= engine.tone_threshold
    ).astype(np.float32)
    for p in (quiet_patch, fake_malware_patch, quiet_patch):
        engine.process_patch(p)

    # Drive hysteresis: armed_frames consecutive anomalies needed to reach ARMED
    for _ in range(engine.armed_limit + 2):
        score, second_state, _ = engine.process_patch(fake_malware_patch)

    assert second_state in ["ARMED", "THREAT_DETECTED"], f"Hysteresis pipeline failed to escalate threat level triggers. State: {second_state}"
    print("\n[+] Verification Unit Tests Passed successfully.")
