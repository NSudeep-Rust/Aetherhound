"""Regression checks for the hardware-free acoustic detection path."""
from __future__ import annotations

import sqlite3
import numpy as np

from capture.simulated_audio import SimulatedAudioSource
from config_loader import load_config
from dsp.features import tonal_prominence
from infer.detector import AetherHoundEngine


def test_config_exposes_runtime_detection_controls():
    cfg = load_config()
    # Calibration now writes the operational trip point directly, so the
    # margin stays at 1.0 instead of stacking an undocumented extra factor.
    assert cfg.detect.threat_margin == 1.0
    assert cfg.detect.stream_warmup_s > 0
    assert cfg.detect.tone_min_hz < cfg.detect.tone_max_hz


def test_tonal_prominence_localises_a_virtual_carrier():
    rng = np.random.default_rng(7)
    patch = np.abs(rng.normal(0.01, 0.002, size=(64, 257))).astype(np.float32)
    carrier_bin = 43  # 43 * 48_000 / 512 = 4031.25 Hz
    patch[:, carrier_bin] += 0.20

    prominence_db, frequency_hz = tonal_prominence(patch, 48_000, 512)

    assert prominence_db > 15.0
    assert abs(frequency_hz - 4031.25) < 48_000 / 512


def test_virtual_source_needs_no_audio_device():
    cfg = load_config()
    source = SimulatedAudioSource(cfg.audio, ["data/baseline/normal_room.wav"], loop=False)
    source.start()
    block = source.read()
    source.stop()

    assert block is not None
    assert block.shape == (cfg.audio.blocksize,)
    assert block.dtype == np.float32


def test_trained_model_separates_saved_virtual_attack_fixture():
    # The virtual attack is a brief keyed tone (25% duty-cycle burst), which
    # the AE reconstructs too well to trip on its own -- that's exactly why
    # the tone guard exists alongside it (see infer/detector.score/tone_score
    # and the state machine in process_patch). Check the combined guard that
    # production actually uses, not the AE in isolation, and only require a
    # detection rate consistent with the ~25% duty cycle rather than 100%.
    engine = AetherHoundEngine()
    baseline = np.load("data/baseline_patches.npy", mmap_mode="r")[:128]
    attack = np.load("data/test_patches.npy", mmap_mode="r")
    trip = engine.threshold * engine.threat_margin

    baseline_scores = np.array([engine.score(patch) for patch in baseline])
    attack_ae_scores = engine.score_many(attack)
    attack_tone_scores = np.array([engine.tone_score(patch)[0] for patch in attack])
    detected = (attack_ae_scores > trip) | (attack_tone_scores > engine.tone_threshold)
    detection_rate = float(detected.mean())

    assert baseline_scores.max() < trip
    assert detection_rate > 0.15, f"only {detection_rate:.1%} of attack patches tripped either guard"


def test_batched_scores_match_single_patch_scores():
    engine = AetherHoundEngine()
    patches = np.load("data/baseline_patches.npy", mmap_mode="r")[:4]

    batched = engine.score_many(patches)
    single = np.array([engine.score(patch) for patch in patches])

    assert np.allclose(batched, single, rtol=1e-6, atol=1e-8)


def test_room_profile_overrides_virtual_calibration(tmp_path):
    profile_path = tmp_path / "room.json"
    database_path = tmp_path / "room-events.db"
    config_path = tmp_path / "room.toml"
    profile_path.write_text(
        '{"name":"test room","sample_rate":48000,"nfft":512,'
        '"ae_trip":0.5,"tone_trip_db":9.0,"silence_rms":0.0001}'
    )
    config_path.write_text(f'[log]\ndb = "{database_path}"\n')

    engine = AetherHoundEngine(config_path, profile=profile_path)

    assert engine.profile_name == "test room"
    assert engine.threshold * engine.threat_margin == 0.5
    assert engine.tone_threshold == 9.0
    assert engine.silence_rms == 0.0001


def test_tonal_channel_event_records_its_frequency(tmp_path):
    config_path = tmp_path / "test_config.toml"
    database_path = tmp_path / "events.db"
    config_path.write_text(f'[log]\ndb = "{database_path}"\n')
    engine = AetherHoundEngine(config_path)
    attack = np.load("data/test_patches.npy", mmap_mode="r")
    patch = next(
        candidate for candidate in attack
        if engine.tone_score(candidate)[0] > engine.tone_threshold
    )

    engine.process_patch(patch)
    # Events are written on a background thread (see infer.detector._EventWriter)
    # since the hardening round; wait for the queue to drain before reading the
    # DB, or this races and intermittently sees no row yet.
    engine.flush_events()
    with sqlite3.connect(database_path) as conn:
        frequency_hz = conn.execute(
            "SELECT frequency_hz FROM security_events WHERE frequency_hz IS NOT NULL"
        ).fetchone()[0]

    assert abs(frequency_hz - 4031.25) < 48_000 / 512
