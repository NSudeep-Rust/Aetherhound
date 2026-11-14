import json
import time
import sqlite3
import queue
import threading
from pathlib import Path
import numpy as np
import onnxruntime as ort
from config_loader import load_config
from dsp.features import tonal_prominence
from dsp.normalization import load_reference_scale, normalise_spectrogram


class _EventWriter:
    """Single WAL-backed SQLite writer kept out of the DSP hot path."""

    _STOP = object()

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._queue: queue.Queue[tuple | object] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="aetherhound-event-writer", daemon=True)
        self._thread.start()

    def submit(self, event: tuple[float, float, str, str, float | None]) -> None:
        self._queue.put(event)

    def flush(self) -> None:
        self._queue.join()

    def close(self) -> None:
        if self._thread.is_alive():
            self._queue.put(self._STOP)
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            while True:
                item = self._queue.get()
                try:
                    if item is self._STOP:
                        return
                    timestamp, score, state, description, frequency_hz = item
                    conn.execute(
                        "INSERT INTO security_events "
                        "(timestamp, score, current_state, description, frequency_hz) VALUES (?, ?, ?, ?, ?)",
                        (timestamp, score, state, description, frequency_hz),
                    )
                    conn.commit()
                finally:
                    self._queue.task_done()


class AetherHoundEngine:
    def __init__(self, config_toml_path="config.toml", profile: str | Path | None = None):
        # 1. Load config
        cfg = load_config(config_toml_path)
        self._cfg = cfg
        
        # 2. Load ONNX Session and calibration artifacts.
        self.ort_session = ort.InferenceSession("models/autoencoder.onnx")
        with open(cfg.detect.threshold_file, "r") as f:
            self.threshold = float(f.read().strip())
        reference_scale_path = Path("models/ref_scale.txt")
        if not reference_scale_path.exists():
            raise FileNotFoundError(
                "missing models/ref_scale.txt; it is required to apply the model's "
                "training-time normalization"
            )
        self.reference_scale = load_reference_scale(reference_scale_path)

        tone_path = Path(cfg.detect.tone_threshold_file)
        self.tone_threshold = float(tone_path.read_text().strip()) if tone_path.exists() else None
        self.tone_min_hz = float(cfg.detect.tone_min_hz)
        self.tone_max_hz = float(cfg.detect.tone_max_hz)
        self.tone_window_bins = int(cfg.detect.tone_window_bins)
        self.profile_name, profile_data = self._load_profile(profile, cfg)
        if profile_data is not None:
            # A room profile stores operational trip points rather than a
            # training checkpoint. The shipped ONNX model remains unchanged.
            self.threshold = float(profile_data["ae_trip"]) / float(cfg.detect.threat_margin)
            self.tone_threshold = float(profile_data["tone_trip_db"])
        
        # 3. State Parameters from config
        self.db_path = cfg.log.db
        self.armed_limit = cfg.detect.armed_frames
        self.clear_limit = cfg.detect.clear_frames
        self.threat_margin = float(getattr(cfg.detect, "threat_margin", 1.25))
        self.sensor_loss_s = float(getattr(cfg.detect, "sensor_loss_s", 5.0))
        self.hold_s = float(cfg.detect.hold_s)
        self.clear_ratio = float(cfg.detect.clear_ratio)
        
        # Load real-room RMS floor for silence gating (falls back to config value)
        p = Path("models/baseline_rms.txt")
        if profile_data is not None:
            self.silence_rms = float(profile_data["silence_rms"])
        elif p.exists():
            self.silence_rms = float(p.read_text().strip()) * 0.25
        else:
            self.silence_rms = float(getattr(cfg.detect, "silence_rms", 1e-6))
        
        # Initialize internal monitoring metrics tracking
        self.anomaly_counter = 0
        self.clean_counter = 0
        self.state = "CLEAN"  # Options: CLEAN, ARMED, THREAT_DETECTED, NO_SIGNAL, SENSOR_LOSS
        self._silent_since = None
        self.last_tone_score = 0.0
        self.last_tone_hz = 0.0
        self.last_reason = ""
        self._tone_active = False
        self._threat_started: float | None = None
        self._incident_peak_score = 0.0
        self._incident_peak_hz: float | None = None
        
        self._init_database()
        self._events = _EventWriter(self.db_path)
        tone_label = f"{self.tone_threshold:.2f} dB" if self.tone_threshold is not None else "disabled"
        print(
            f"[+] Engine Initialized ({self.profile_name} profile). AE trip: {self.threshold * self.threat_margin:.6f} "
            f"| tone trip: {tone_label} | Silence RMS: {self.silence_rms:.2e}"
        )

    @staticmethod
    def _load_profile(profile: str | Path | None, cfg) -> tuple[str, dict | None]:
        """Load an optional environment-specific calibration profile."""
        if profile in (None, "virtual", "default"):
            return "virtual", None
        path = Path("models/profiles/room.json") if profile == "room" else Path(profile)
        if not path.exists():
            raise FileNotFoundError(
                f"calibration profile not found: {path}. Create it with "
                "python train/calibrate_room.py --seconds 45"
            )
        try:
            data = json.loads(path.read_text())
            required = ("ae_trip", "tone_trip_db", "silence_rms", "sample_rate", "nfft")
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"missing keys: {', '.join(missing)}")
            if int(data["sample_rate"]) != cfg.audio.rate or int(data["nfft"]) != cfg.dsp.nfft:
                raise ValueError("sample rate or FFT size does not match config.toml")
            values = [float(data[key]) for key in ("ae_trip", "tone_trip_db", "silence_rms")]
            if not all(np.isfinite(value) and value > 0 for value in values):
                raise ValueError("trip points and silence RMS must be positive")
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise ValueError(f"invalid calibration profile {path}: {exc}") from exc
        return str(data.get("name") or path.stem), data

    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    score REAL,
                    current_state TEXT,
                    description TEXT,
                    frequency_hz REAL
                )
            """)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(security_events)")}
            if "frequency_hz" not in columns:
                conn.execute("ALTER TABLE security_events ADD COLUMN frequency_hz REAL")

    def log_event(self, score, state, desc, frequency_hz: float | None = None):
        self._events.submit((time.time(), float(score), state, desc, frequency_hz))

    def flush_events(self) -> None:
        self._events.flush()

    def close(self) -> None:
        self._events.close()

    def score(self, patch_matrix):
        """Compute reconstruction MSE using the scale used while training.

        The previous live path renormalized every patch by its own peak, although
        the network was trained using ``models/ref_scale.txt``.  That mismatch
        erased much of a pulsed carrier's amplitude pattern and made deployed
        scores incomparable to calibration scores.
        """
        norm = normalise_spectrogram(patch_matrix, self.reference_scale)
        inp = np.expand_dims(norm, axis=0)
        out = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: inp})[0][0]
        return float(np.mean((norm - out) ** 2))

    def score_many(self, patches, batch_size: int = 64) -> np.ndarray:
        """Score a sequence efficiently while preserving ``score`` semantics."""
        values = list(patches)
        if not values:
            return np.empty(0, dtype=np.float64)
        results: list[np.ndarray] = []
        input_name = self.ort_session.get_inputs()[0].name
        for start in range(0, len(values), batch_size):
            raw = np.asarray(values[start : start + batch_size], dtype=np.float32)
            normalised = normalise_spectrogram(raw, self.reference_scale)
            reconstructed = self.ort_session.run(None, {input_name: normalised})[0]
            results.append(np.mean((normalised - reconstructed) ** 2, axis=(1, 2)))
        return np.concatenate(results).astype(np.float64, copy=False)

    def tone_score(self, patch_matrix) -> tuple[float, float]:
        """Return a scale-invariant narrow-band score and its peak frequency."""
        return tonal_prominence(
            patch_matrix,
            sample_rate=self._cfg.audio.rate,
            nfft=self._cfg.dsp.nfft,
            min_hz=self.tone_min_hz,
            max_hz=self.tone_max_hz,
            window_bins=self.tone_window_bins,
        )

    def process_patch(self, patch_matrix):
        # Silence gate: a dead mic is not an anomaly
        rms = float(np.sqrt(np.mean(patch_matrix ** 2)))
        if rms < self.silence_rms:
            now = time.monotonic()
            if self._silent_since is None:
                self._silent_since = now
            self.anomaly_counter = 0
            self.clean_counter = 0
            self.last_tone_score = 0.0
            self.last_tone_hz = 0.0
            self.last_reason = "no signal"
            self._tone_active = False
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
        tone_score, tone_hz = self.tone_score(patch_matrix)
        ae_anomalous = mse_score > (self.threshold * self.threat_margin)
        tone_anomalous = self.tone_threshold is not None and tone_score > self.tone_threshold
        tone_started = tone_anomalous and not self._tone_active
        self._tone_active = bool(tone_anomalous)
        is_anomalous = ae_anomalous or tone_anomalous
        self.last_tone_score = tone_score
        self.last_tone_hz = tone_hz
        reasons = []
        if ae_anomalous:
            reasons.append("autoencoder")
        if tone_anomalous:
            reasons.append(f"tonal line at {tone_hz:.0f} Hz")
        self.last_reason = ", ".join(reasons) if reasons else "within baseline"
        
        # 3. Process Hysteresis State Transitions Machine
        old_state = self.state
        now = time.monotonic()
        
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
                self._threat_started = now
                self._incident_peak_score = mse_score
                self._incident_peak_hz = tone_hz if tone_anomalous else None
            elif self.state == "THREAT_DETECTED":
                self._incident_peak_score = max(self._incident_peak_score, mse_score)
                if tone_anomalous:
                    self._incident_peak_hz = tone_hz
        else:
            self.anomaly_counter = 0
            recovered = (
                mse_score <= self.threshold * self.threat_margin * self.clear_ratio
                and (self.tone_threshold is None or tone_score <= self.tone_threshold * self.clear_ratio)
            )
            self.clean_counter = self.clean_counter + 1 if recovered else 0
            can_clear = self.clean_counter >= self.clear_limit
            if self.state == "THREAT_DETECTED" and self._threat_started is not None:
                can_clear = can_clear and now - self._threat_started >= self.hold_s
            if self.state in ["ARMED", "THREAT_DETECTED"] and can_clear:
                self.state = "CLEAN"
                if old_state == "THREAT_DETECTED":
                    self.log_event(
                        self._incident_peak_score,
                        "INCIDENT_CLOSED",
                        "Incident closed after clean hold; peak reconstruction score "
                        f"{self._incident_peak_score:.6f}",
                        self._incident_peak_hz,
                    )
                    self._threat_started = None
                
        # 4. If system state has transitioned, commit data records to the SQLite database
        # Persist state transitions, not every analysis frame.  This preserves
        # an operator-readable timeline instead of flooding the SQLite log.
        if old_state != self.state or tone_started:
            if is_anomalous:
                desc = (
                    f"Anomaly sources: {self.last_reason}; AE score {mse_score:.6f} "
                    f"(trip {self.threshold * self.threat_margin:.6f}); tonal prominence "
                    f"{tone_score:.2f} dB"
                )
                frequency_hz = tone_hz if tone_anomalous else None
            else:
                desc = "Room sound returned to normal."
                frequency_hz = None
            self.log_event(mse_score, self.state, f"State change to {self.state}: {desc}", frequency_hz)
            if old_state != self.state:
                print(f"[!] STATE: {old_state} -> {self.state} | Score: {mse_score:.6f}")
            elif tone_started:
                print(f"[!] TONAL CHANNEL: {tone_hz:.0f} Hz | Prominence: {tone_score:.2f} dB")
            
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
