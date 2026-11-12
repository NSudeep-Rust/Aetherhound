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
    
    score, state = engine.process_patch(fake_clean_patch)
    assert state == "NO_SIGNAL", "Silent patch should produce NO_SIGNAL state."
    
    # 2. Use real infected test patch that produces high MSE after per-patch peak normalization
    test_patches = np.load("data/test_patches.npy")
    limit = engine.threshold * engine.threat_margin
    # Find a patch that is both above silence gate and anomalous
    fake_malware_patch = None
    for p in test_patches:
        if float(np.sqrt(np.mean(p ** 2))) >= engine.silence_rms and engine.score(p) > limit:
            fake_malware_patch = p.astype(np.float32)
            break
    assert fake_malware_patch is not None, "No anomalous test patch found above silence gate"
    
    # Drive hysteresis: armed_frames=4 consecutive anomalies needed to reach ARMED
    for _ in range(engine.armed_limit + 2):
        score, second_state = engine.process_patch(fake_malware_patch)
    
    assert second_state in ["ARMED", "THREAT_DETECTED"], f"Hysteresis pipeline failed to escalate threat level triggers. State: {second_state}"
    print("\n[+] Verification Unit Tests Passed successfully.")
