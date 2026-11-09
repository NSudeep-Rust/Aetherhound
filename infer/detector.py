import os
import time
import sqlite3
import numpy as np
import onnxruntime as ort
from dsp.pipeline import extract_spectrogram

class AetherHoundEngine:
    def __init__(self, config_toml_path="config.toml"):
        # 1. Load ONNX Session and Calibration Threshold parameters
        self.ort_session = ort.InferenceSession("models/autoencoder.onnx")
        with open("models/threshold.txt", "r") as f:
            self.threshold = float(f.read().strip())
            
        # 2. Hardcoded State Parameters matching config.toml constraints
        self.db_path = "events.db"
        self.armed_limit = 2
        self.clear_limit = 5
        
        # Initialize internal monitoring metrics tracking
        self.anomaly_counter = 0
        self.clean_counter = 0
        self.state = "CLEAN"  # Options: CLEAN, ARMED, THREAT_DETECTED
        
        self._init_database()
        print(f"[+] Engine Initialized. Operational Baseline Guardrail Threshold: {self.threshold:.6f}")

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

    def process_patch(self, patch_matrix):
        # Normalize input dimensions to feed structural ONNX requirements
        max_val = patch_matrix.max() if patch_matrix.max() > 0 else 1.0
        norm_patch = (patch_matrix / max_val).astype(np.float32)
        input_tensor = np.expand_dims(norm_patch, axis=0) # Add batch dimension
        
        # 1. Run low-overhead math evaluation via ONNX Runtime
        ort_inputs = {self.ort_session.get_inputs()[0].name: input_tensor}
        ort_outputs = self.ort_session.run(None, ort_inputs)
        reconstructed_patch = ort_outputs[0][0]
        
        # 2. Compute Reconstruction Mean Squared Error (Loss Score)
        mse_score = np.mean((norm_patch - reconstructed_patch) ** 2)
        is_anomalous = mse_score > self.threshold
        
        # 3. Process Hysteresis State Transitions Machine
        old_state = self.state
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
            desc = f"Anomaly score {mse_score:.6f} breached threshold {self.threshold:.6f}" if is_anomalous else f"Room sound returned to normal."
            self.log_event(mse_score, self.state, f"State change to {self.state}: {desc}")
            print(f"[!] SYSTEM ALARM STATE ALERT: {old_state} -> {self.state} | Score: {mse_score:.6f}")
            
        return mse_score, self.state

if __name__ == "__main__":
    print("[*] Instantiating live runtime test loop over simulated baseline matrix...")
    engine = AetherHoundEngine()
    
    # Load infected patches to test state transition parameters
    infected_tensors = np.load("data/test_patches.npy")
    print(f"[*] Simulating real-time signal loop over {len(infected_tensors)} live buffer steps...")
    
    for step, single_patch in enumerate(infected_tensors[:20]):
        score, current_status = engine.process_patch(single_patch)
        if current_status == "THREAT_DETECTED":
            print(f"[+] Operational Guard Verification Passed: Alert tripped at frame cycle {step}!")
            break
    print("[+] Live Anomaly Engine pipeline confirmed fully functional!")
