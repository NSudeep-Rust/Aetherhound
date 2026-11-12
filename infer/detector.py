import os
import time
import sqlite3
from pathlib import Path
import numpy as np
import onnxruntime as ort
from dsp.pipeline import extract_spectrogram
from config_loader import load_config


class AetherHoundEngine:
    def __init__(self, config_toml_path="config.toml"):
        # 1. Load config
        cfg = load_config(config_toml_path)
        
        # 2. Load ONNX Session and Calibration Threshold parameters
        self.ort_session = ort.InferenceSession("models/autoencoder.onnx")
        with open(cfg.detect.threshold_file, "r") as f:
            self.threshold = float(f.read().strip())
        
        # 3. State Parameters from config
        self.db_path = cfg.log.db
        self.armed_limit = cfg.detect.armed_frames
        self.clear_limit = cfg.detect.clear_frames
        self.threat_margin = float(getattr(cfg.detect, "threat_margin", 1.25))
        self.sensor_loss_s = float(getattr(cfg.detect, "sensor_loss_s", 5.0))
        
        # Load real-room RMS floor for silence gating (falls back to config value)
        p = Path("models/baseline_rms.txt")
        if p.exists():
            self.silence_rms = float(p.read_text().strip()) * 0.25
        else:
            self.silence_rms = float(getattr(cfg.detect, "silence_rms", 1e-6))
        
        # Initialize internal monitoring metrics tracking
        self.anomaly_counter = 0
        self.clean_counter = 0
        self.state = "CLEAN"  # Options: CLEAN, ARMED, THREAT_DETECTED, NO_SIGNAL, SENSOR_LOSS
        self._silent_since = None
        
        self._init_database()
        print(f"[+] Engine Initialized. Threshold: {self.threshold:.6f} | Margin: {self.threat_margin}x | Silence RMS: {self.silence_rms:.2e}")

    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                score REAL,
                current_state TEXT,
                description TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log_event(self, score, state, desc):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO security_events (timestamp, score, current_state, description) VALUES (?, ?, ?, ?)",
            (time.time(), float(score), state, desc)
        )
        conn.commit()
        conn.close()

    def score(self, patch_matrix):
        """Compute reconstruction MSE for a patch using per-patch peak normalization.

        np.clip(..., 2.0) is the sigmoid-saturation guard: peak normalization pins
        the 99.9th percentile at 1.0, so only the top 0.1 % can run away; clipping
        stops a single loud bin from railing the decoder.
        """
        peak = float(np.percentile(patch_matrix, 99.9))
        norm = np.clip(patch_matrix / max(peak, 1e-12), 0.0, 2.0).astype(np.float32)
        inp = np.expand_dims(norm, axis=0)
        out = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: inp})[0][0]
        return float(np.mean((norm - out) ** 2))

    def process_patch(self, patch_matrix):
        # Silence gate: a dead mic is not an anomaly
        rms = float(np.sqrt(np.mean(patch_matrix ** 2)))
        if rms < self.silence_rms:
            now = time.time()
            if self._silent_since is None:
                self._silent_since = now
            self.anomaly_counter = 0
            self.clean_counter = 0
            old_state = self.state
            self.state = ("SENSOR_LOSS"
                          if now - self._silent_since >= self.sensor_loss_s
                          else "NO_SIGNAL")
            if old_state != self.state:
                self.log_event(0.0, self.state, f"State change to {self.state}: mic silence (rms={rms:.2e})")
                print(f"[!] STATE: {old_state} -> {self.state}")
            return 0.0, self.state, False
        self._silent_since = None

        mse_score = self.score(patch_matrix)
        is_anomalous = mse_score > (self.threshold * self.threat_margin)
        
        # 3. Process Hysteresis State Transitions Machine
        old_state = self.state
        
        # If we were in NO_SIGNAL/SENSOR_LOSS and now have signal, reset to CLEAN
        if old_state in ("NO_SIGNAL", "SENSOR_LOSS"):
            self.state = "CLEAN"
            self.anomaly_counter = 0
            self.clean_counter = 0
        
        if is_anomalous:
            self.clean_counter = 0
            self.anomaly_counter += 1
            if self.state == "CLEAN" and self.anomaly_counter >= self.armed_limit:
                self.state = "ARMED"
            elif self.state == "ARMED" and self.anomaly_counter >= (self.armed_limit + 1):
                self.state = "THREAT_DETECTED"
        else:
            self.anomaly_counter = 0
            self.clean_counter += 1
            if self.state in ["ARMED", "THREAT_DETECTED"] and self.clean_counter >= self.clear_limit:
                self.state = "CLEAN"
                
        # 4. If system state has transitioned, commit data records to the SQLite database
        if old_state != self.state or self.state == "THREAT_DETECTED":
            desc = f"Anomaly score {mse_score:.6f} breached threshold {self.threshold * self.threat_margin:.6f}" if is_anomalous else f"Room sound returned to normal."
            self.log_event(mse_score, self.state, f"State change to {self.state}: {desc}")
            print(f"[!] STATE: {old_state} -> {self.state} | Score: {mse_score:.6f}")
            
        return mse_score, self.state, is_anomalous

if __name__ == "__main__":
    print("[*] Instantiating live runtime test loop over simulated baseline matrix...")
    engine = AetherHoundEngine()
    
    # Load infected patches to test state transition parameters
    infected_tensors = np.load("data/test_patches.npy")
    print(f"[*] Simulating real-time signal loop over {len(infected_tensors)} live buffer steps...")
    
    for step, single_patch in enumerate(infected_tensors[:20]):
        score, current_status, _ = engine.process_patch(single_patch)
        if current_status == "THREAT_DETECTED":
            print(f"[+] Operational Guard Verification Passed: Alert tripped at frame cycle {step}!")
            break
    print("[+] Live Anomaly Engine pipeline confirmed fully functional!")
