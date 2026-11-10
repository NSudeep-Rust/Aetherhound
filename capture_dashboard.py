#!/usr/bin/env python3
"""Capture dashboard frames as text for documentation."""
import os
import time
import numpy as np
from rich.console import Console
from rich.live import Live
from ui.dashboard import generate_dashboard_content, make_layout
from infer.detector import AetherHoundEngine

# Use a file-based console to capture output
console = Console(record=True, width=120, height=40)

engine = AetherHoundEngine()
test_patches = np.load("data/test_patches.npy")

print("[*] Capturing dashboard frames...")

with Live(make_layout(), console=console, refresh_per_second=4, screen=False) as live:
    for step, patch in enumerate(test_patches[:8]):
        score, state = engine.process_patch(patch)
        updated_content = generate_dashboard_content(engine, step, score, state)
        live.update(updated_content)
        time.sleep(0.3)
        
        # Save each frame
        frame_text = console.export_text()
        with open(f"docs/dashboard_frame_{step}.txt", "w") as f:
            f.write(frame_text)

# Save final frame as HTML for viewing
console.save_html("docs/dashboard_final.html")
print("[+] Dashboard frames captured to docs/")
