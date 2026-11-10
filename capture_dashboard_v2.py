#!/usr/bin/env python3
"""Capture dashboard frames as text/HTML for documentation."""
import os
import time
import numpy as np
from rich.console import Console
from ui.dashboard import generate_dashboard_content, make_layout
from infer.detector import AetherHoundEngine

# Create docs directory
os.makedirs("docs", exist_ok=True)

engine = AetherHoundEngine()
test_patches = np.load("data/test_patches.npy")

print("[*] Capturing dashboard frames...")

# Capture frames at key state transitions
states_to_capture = ["CLEAN", "ARMED", "THREAT_DETECTED"]
captured_states = set()

for step, patch in enumerate(test_patches[:15]):
    score, state = engine.process_patch(patch)
    
    if state not in captured_states:
        # Generate dashboard for this state
        layout = generate_dashboard_content(engine, step, score, state)
        
        # Save as HTML
        console = Console(record=True, width=120, height=40)
        console.print(layout)
        console.save_html(f"docs/dashboard_{state.lower()}.html")
        
        # Save as text
        with open(f"docs/dashboard_{state.lower()}.txt", "w") as f:
            f.write(console.export_text())
        
        captured_states.add(state)
        print(f"  Captured {state} at step {step}")
    
    if len(captured_states) == 3:
        break

# Also capture a final frame
layout = generate_dashboard_content(engine, step, score, state)
console = Console(record=True, width=120, height=40)
console.print(layout)
console.save_html("docs/dashboard_final.html")
with open("docs/dashboard_final.txt", "w") as f:
    f.write(console.export_text())

print("[+] Dashboard frames captured to docs/")
