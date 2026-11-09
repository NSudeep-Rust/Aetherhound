import time
import sqlite3
import numpy as np
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from rich.console import Console
from rich.text import Text
from infer.detector import AetherHoundEngine

console = Console()

def make_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1)
    )
    layout["body"].split_row(
        Layout(name="metrics", ratio=1),
        Layout(name="logs", ratio=1)
    )
    return layout

def get_latest_logs(db_path="events.db", limit=5):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, current_state, score FROM security_events ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception:
        return []

def generate_dashboard_content(engine, step, score, current_state) -> Layout:
    layout = make_layout()
    
    # 1. Generate Header Banner with Conditional Color-Coding
    if current_state == "THREAT_DETECTED":
        status_text = Text("🚨 SYSTEM STATUS: THREAT DETECTED (MALWARE ACTIVE) 🚨", style="bold white on red", justify="center")
    elif current_state == "ARMED":
        status_text = Text("⚠️ SYSTEM STATUS: ARMED (ANOMALY DETECTED) ⚠️", style="bold black on yellow", justify="center")
    else:
        status_text = Text("✅ SYSTEM STATUS: CLEAN (MONITORING BACKGROUND) ✅", style="bold white on green", justify="center")
        
    layout["header"].update(Panel(status_text, border_style="blue"))
    
    # 2. Build Metrics Panel Table Layout
    metrics_table = Table(title="Live Extraction Performance metrics", expand=True)
    metrics_table.add_column("Parameter", style="cyan")
    metrics_table.add_column("Current Metric Value", style="magenta")
    
    metrics_table.add_row("Buffer Step Sequence", str(step))
    metrics_table.add_row("Current Anomaly Score", f"{score:.6f}")
    metrics_table.add_row("Baseline Target Guardrail", f"{engine.threshold:.6f}")
    metrics_table.add_row("Anomaly Counter State", str(engine.anomaly_counter))
    metrics_table.add_row("Hysteresis Reset Pacing", str(engine.clean_counter))
    
    layout["metrics"].update(Panel(metrics_table, title="Signal Processing Statistics", border_style="cyan"))
    
    # 3. Build Scrolling History Log Panel from SQLite Records
    logs_table = Table(title="Persistent Security Event Repository (SQLite)", expand=True)
    logs_table.add_column("Timestamp Offset", style="dim")
    logs_table.add_column("State Flag", style="bold")
    logs_table.add_column("Evaluated Loss", style="red")
    
    db_rows = get_latest_logs()
    for row in db_rows:
        t_stamp, state_flag, loss_val = row
        time_str = time.strftime('%H:%M:%S', time.localtime(t_stamp))
        state_style = "red" if "THREAT" in state_flag else ("yellow" if "ARMED" in state_flag else "green")
        logs_table.add_row(time_str, f"[{state_style}]{state_flag}[/{state_style}]", f"{loss_val:.6f}")
        
    layout["logs"].update(Panel(logs_table, title="Security Event Log Stream", border_style="magenta"))
    
    return layout

def run_dashboard_simulation():
    engine = AetherHoundEngine()
    test_patches = np.load("data/test_patches.npy")
    
    console.print("[*] Starting AetherHound Live Terminal UI Framework...", style="bold green")
    time.sleep(1.5)
    
    # Execute full layout interface updates within a Rich Live context worker loop
    with Live(make_layout(), refresh_per_second=4, screen=True) as live:
        for step, current_patch in enumerate(test_patches):
            # Run background sample tensor through the engine computation pipeline
            score, current_state = engine.process_patch(current_patch)
            
            # Feed state metrics back to display renderer matrix templates
            updated_content = generate_dashboard_content(engine, step, score, current_state)
            live.update(updated_content)
            
            # Control rendering playback step interval delay to emulate real-world pacing
            time.sleep(0.4)
            
            # Stop simulation once threat verification alerts clear out the execution pool
            if current_state == "THREAT_DETECTED" and step > 5:
                break
                
    console.print("[+] Terminal UI monitoring framework verified successfully!", style="bold green")

if __name__ == "__main__":
    run_dashboard_simulation()
