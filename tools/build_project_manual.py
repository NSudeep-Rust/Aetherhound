"""Build the AetherHound operating and presentation guide."""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "docs" / "AetherHound_Project_Guide.docx"
SCREENSHOT = Path("/tmp/codex-clipboard-17085251-6ab5-4fb4-8a3d-c2c903d45d30.png")

NAVY = "18324A"
MID_BLUE = "DDEBF7"
PALE_BLUE = "F3F8FC"
LIGHT_GRAY = "D9D9D9"
CODE_GRAY = "F2F2F2"
BLACK = RGBColor(0, 0, 0)


def shade(cell, fill: str) -> None:
    props = cell._tc.get_or_add_tcPr()
    element = OxmlElement("w:shd")
    element.set(qn("w:fill"), fill)
    props.append(element)


def set_cell_border(cell, color: str = LIGHT_GRAY) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        element = borders.find(tag)
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "6")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def set_cell_margins(cell, top: int = 90, start: int = 110, bottom: int = 90, end: int = 110) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    mar = tc_pr.first_child_found_in("w:tcMar")
    if mar is None:
        mar = OxmlElement("w:tcMar")
        tc_pr.append(mar)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = mar.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def format_run(run, size: float | None = None, bold: bool | None = None, color: RGBColor | None = None, font: str = "Liberation Sans"):
    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:ascii"), font)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), font)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    run.font.color.rgb = color or BLACK


def add_text(doc: Document, text: str, *, style: str | None = None, size: float | None = None, bold: bool | None = None, after: float = 6.0):
    paragraph = doc.add_paragraph(style=style)
    paragraph.paragraph_format.space_after = Pt(after)
    paragraph.paragraph_format.line_spacing = 1.15
    run = paragraph.add_run(text)
    format_run(run, size=size, bold=bold)
    return paragraph


def add_bullet(doc: Document, text: str, level: int = 0):
    paragraph = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.line_spacing = 1.1
    run = paragraph.add_run(text)
    format_run(run, size=10.25)
    return paragraph


def add_code(doc: Document, code: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(7)
    paragraph.paragraph_format.left_indent = Inches(0.18)
    p_pr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), CODE_GRAY)
    p_pr.append(shd)
    run = paragraph.add_run(code)
    format_run(run, size=9.1, font="Courier New")


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    paragraph = doc.add_paragraph(style=f"Heading {level}")
    paragraph.paragraph_format.space_before = Pt(13 if level == 1 else 9)
    paragraph.paragraph_format.space_after = Pt(5)
    run = paragraph.add_run(text)
    format_run(run, size=15 if level == 1 else 12, bold=True)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = False
    header = table.rows[0]
    for idx, value in enumerate(headers):
        cell = header.cells[idx]
        cell.width = Inches(widths[idx])
        shade(cell, NAVY)
        set_cell_border(cell)
        set_cell_margins(cell)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = paragraph.add_run(value)
        format_run(run, size=9.2, bold=True, color=RGBColor(255, 255, 255))
    for row_index, values in enumerate(rows):
        row = table.add_row()
        for idx, value in enumerate(values):
            cell = row.cells[idx]
            cell.width = Inches(widths[idx])
            if row_index % 2:
                shade(cell, PALE_BLUE)
            set_cell_border(cell)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.05
            run = paragraph.add_run(value)
            format_run(run, size=8.8)
    doc.add_paragraph().paragraph_format.space_after = Pt(3)


def configure(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.68)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Liberation Sans"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Liberation Sans")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Liberation Sans")
    normal.font.size = Pt(10.3)
    for name in ("Title", "Heading 1", "Heading 2"):
        style = styles[name]
        style.font.name = "Liberation Sans"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Liberation Sans")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Liberation Sans")
        style.font.color.rgb = BLACK
    # The stock Word Title style can include a coloured paragraph rule.  This
    # guide deliberately uses a clean report title instead.
    title_props = styles["Title"]._element.find(qn("w:pPr"))
    if title_props is not None:
        title_border = title_props.find(qn("w:pBdr"))
        if title_border is not None:
            title_props.remove(title_border)
    styles["Title"].font.size = Pt(25)
    styles["Heading 1"].font.size = Pt(15)
    styles["Heading 2"].font.size = Pt(12)


def build() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    configure(doc)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(12)
    run = title.add_run("AetherHound Project Guide and Final Verification")
    format_run(run, size=25, bold=True)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(24)
    run = subtitle.add_run("Software only covert channel detector with virtual and real room demonstrations")
    format_run(run, size=12)

    add_text(
        doc,
        "This guide explains how to present, operate, test, and extend AetherHound. "
        "The default demonstration is fully virtual and needs no Raspberry Pi, SDR, or microphone. "
        "An optional separate room profile supports a laptop microphone without changing the virtual model or its calibration.",
        size=11,
        after=14,
    )
    add_heading(doc, "Final Status", 1)
    add_table(
        doc,
        ["Area", "Final status", "Meaning"],
        [
            ["Virtual dashboard", "Ready", "Rich terminal interface is configured at 30 refreshes per second and shows acoustic state, scores, tonal evidence, and waterfalls."],
            ["Virtual validation", "Passed", "All 1,404 saved baseline patches stayed below calibrated guards. All 936 saved attack patches were detected by the combined detector."],
            ["Automated checks", "Passed", "23 project tests pass, including virtual source, detector separation, calibration profile, RF, sweep, and bearing maths checks."],
            ["Real room mode", "Ready to calibrate", "A separate room profile learns normal microphone conditions. It must be calibrated on the actual room before presenting it."],
            ["Hardware deployment", "Integration path available", "Audio and RF source abstractions exist, but real Raspberry Pi and SDR operation have not been validated in this software-only project."],
        ],
        [1.28, 1.18, 4.0],
    )

    add_heading(doc, "Project Purpose", 1)
    add_text(
        doc,
        "AetherHound is a defensive monitoring prototype for a room containing an air-gapped system. "
        "It looks for unusual structured acoustic or RF activity that may resemble a covert channel. "
        "The supplied demonstration represents normal room audio and a simulated pulsed carrier. The objective is to show that normal baseline conditions remain clean while the structured attack is escalated.")
    add_text(
        doc,
        "The project brief asks for DSP, a PyTorch autoencoder trained on normal emissions, and a fast hacker-style command line interface that displays spectrum activity. "
        "AetherHound implements those software requirements with Python, SciPy, PyTorch/ONNX Runtime, SQLite, and Rich. Rich is permitted by the brief as an alternative to curses.")

    # Keep this dense requirements table together rather than leaving only its
    # header at the foot of the title page.
    doc.add_page_break()
    add_heading(doc, "Requirements Check", 1)
    add_table(
        doc,
        ["Brief requirement", "Current implementation", "Assessment"],
        [
            ["DSP and spectrograms", "Bandpass and mains notch filtering feed a 512 point STFT. Audio becomes 64 by 257 spectrogram patches.", "Met for virtual and microphone software paths."],
            ["Autoencoder trained on normal data", "PyTorch autoencoder is exported to ONNX and loaded for live scoring.", "Met. Calibration and inference now use the same fixed reference scale."],
            ["CLI or curses spectrum display", "Rich live terminal UI has acoustic and optional RF waterfall panels, status, scores, and frequency evidence.", "Met. It is a CLI dashboard, not a graphical desktop app."],
            ["30 plus FPS UI", "UI refresh configuration is 30 Hz.", "Configured and exercised in the demo. Formal performance testing on a Raspberry Pi is still required."],
            ["Alert and local buzzer", "Red threat state is implemented. The software buzzer is optional with --buzzer.", "Met for the software alert. Hardware buzzer wiring is not part of the current build."],
            ["Timestamp and frequency logs", "SQLite security events log state transitions; tonal channel starts include frequency_hz.", "Met for acoustic events."],
            ["RF hardware and rack localisation", "Simulated RF works. RTL SDR and HackRF backends are adapters only.", "Not proven on hardware. Single sensor output is not rack-level location."],
        ],
        [1.55, 3.1, 1.82],
    )

    add_heading(doc, "Dashboard Guide", 1)
    add_text(doc, "The dashboard is accurate for its stated purpose: it presents the internal detector state and signal measurements. It must not be interpreted as proof that every yellow pixel is a threat or that it can locate a physical rack without additional hardware.")
    if SCREENSHOT.exists():
        image_paragraph = doc.add_paragraph()
        image_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        image_paragraph.add_run().add_picture(str(SCREENSHOT), width=Inches(3.75))
        caption = doc.add_paragraph()
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = caption.add_run("Figure 1  Acoustic dashboard during a threat state")
        format_run(run, size=9)
        caption.paragraph_format.space_after = Pt(10)
    add_heading(doc, "What the Screenshot Shows", 2)
    add_bullet(doc, "The red header and THREAT DETECTED state mean the decision logic has escalated the current patch sequence.")
    add_bullet(doc, "AE score is the autoencoder reconstruction error. AE trip is the calibrated error level that begins anomaly handling.")
    add_bullet(doc, "Tone prominence measures how much a narrow frequency line rises above its nearby spectrum. Tone trip is the room or virtual threshold in decibels.")
    add_bullet(doc, "Tone peak is the strongest narrow-band frequency at that instant. In the screenshot, the prominence is below the tone trip, so the autoencoder is the evidence source.")
    add_bullet(doc, "RMS is the signal level in the current spectrogram patch. It helps distinguish silence from active input.")
    add_heading(doc, "Yellow Bars", 2)
    add_text(doc, "The yellow horizontal bars are the acoustic waterfall. Each row is one recent spectrogram patch, and each horizontal position represents a frequency region from approximately 200 Hz to 8 kHz. Yellow means high signal energy after the fixed display scale; black means low energy. Rows scroll as new patches arrive, so older history moves upward. They are an intensity visualisation, not an alert by themselves. A loud song or broad noise can look yellow without being a confirmed covert channel.")
    add_heading(doc, "Mic Off or Silent Room", 2)
    add_table(
        doc,
        ["Condition", "Dashboard state", "Expected display"],
        [
            ["Microphone sends zeros or room is below silence floor", "NO SIGNAL", "Grey header, score 0, tone 0, and evidence no signal."],
            ["No signal continues for about 5 seconds", "SENSOR LOSS", "Orange header and red acoustic border. This means the source is silent or unavailable, not that an attacker was found."],
            ["Input stream cannot open because a microphone is disconnected", "Startup error or fallback", "The program tries the system default device. Reconnect or select a valid input device before relying on monitoring."],
        ],
        [2.0, 1.55, 2.92],
    )

    add_heading(doc, "Virtual Demonstration Runbook", 1)
    add_text(doc, "Run these commands from the AetherHound project folder. The virtual paths are the primary project demonstration because they use only saved WAV data and generated RF signals.")
    add_heading(doc, "Prepare and Verify", 2)
    add_code(doc, "source .venv/bin/activate\npython deploy/run_app.py --self-test\npytest -q")
    add_text(doc, "Expected result: three PASS lines from the self-test and 23 passed from pytest.")
    add_heading(doc, "Clean Virtual Room", 2)
    add_code(doc, "python deploy/run_app.py --scenario baseline")
    add_text(doc, "Expected result: a green CLEAN dashboard. The normal_room WAV loops as a virtual microphone. There should be no red threat state.")
    add_heading(doc, "Virtual Acoustic Attack", 2)
    add_code(doc, "python deploy/run_app.py --scenario attack")
    add_text(doc, "Expected result: the state progresses through ARMED to THREAT DETECTED. The tonal detector later reports a channel near 4031 Hz. Press Ctrl C to stop the dashboard.")
    add_heading(doc, "Virtual RF and Acoustic Attack", 2)
    add_code(doc, "python deploy/run_app.py --scenario attack --simulated-rf --inject")
    add_text(doc, "Expected result: the acoustic attack is visible and the RF waterfall receives an injected simulated carrier. This does not access an SDR.")
    add_heading(doc, "Mixed Scenario", 2)
    add_code(doc, "python deploy/run_app.py --scenario mixed --duration 40")
    add_text(doc, "Expected result: the dashboard starts clean, then reaches the attack fixture after the baseline fixture has played. A 40 second duration is suitable for a repeatable demonstration.")
    # Keep the multi-line WAV example and its result together on the next page.
    doc.add_page_break()
    add_heading(doc, "Specific Supplied WAV Files", 2)
    add_code(doc, "python deploy/run_app.py --audio-file \"$(pwd)/data/baseline/live_room.wav\" --duration 10\npython deploy/run_app.py --audio-file \"$(pwd)/data/baseline/normal_room.wav\" --duration 10\npython deploy/run_app.py --audio-file \"$(pwd)/data/test/infected_attack.wav\" --duration 10")
    add_text(doc, "The first two are expected to remain clean. The final WAV is expected to reach a threat state. Custom WAV input must be mono or stereo 48 kHz audio.")

    add_heading(doc, "Real Room Profile Runbook", 1)
    add_text(doc, "Real microphone mode is optional and intentionally separate from the virtual demonstration. The base ONNX model is not retrained when a room profile is created. The profile only saves thresholds for the actual microphone environment.")
    add_heading(doc, "Choose a Microphone if Needed", 2)
    add_code(doc, "python deploy/run_app.py --list-devices")
    add_text(doc, "If the default microphone is not correct, note the input-device number and include it during room calibration.")
    add_heading(doc, "Calibrate Normal Conditions", 2)
    add_code(doc, "python train/calibrate_room.py --seconds 45")
    add_text(doc, "During recording, keep the sounds you want to regard as normal present: ordinary room/fan noise, speech, and a short sample of normal music. The command writes models/profiles/room.json. It does not overwrite the virtual threshold files.")
    add_heading(doc, "Start the Real Room Dashboard", 2)
    add_code(doc, "python deploy/run_app.py --source microphone --profile room")
    add_text(doc, "Expected result: normal conditions that were included in calibration should remain CLEAN. Absolute silence becomes NO SIGNAL and later SENSOR LOSS. If a normal repeated sound produces an alert, recalibrate with that sound present; do not alter the virtual profile.")
    add_heading(doc, "Optional Named Device", 2)
    add_code(doc, "python train/calibrate_room.py --seconds 45 --device 3\npython deploy/run_app.py --source microphone --profile room")
    add_text(doc, "Replace 3 with the input-device number shown by --list-devices. The dashboard itself uses the configured/default microphone after calibration.")
    add_heading(doc, "Important Real Room Limit", 2)
    add_text(doc, "The saved validation result is a synthetic-fixture result, not a guarantee for every real song, voice, fan, or room. A room profile reduces false positives for conditions it has observed. It should be recalibrated after a microphone, room layout, or normal noise pattern changes.")

    # Keep the accuracy table as one coherent piece of evidence.
    doc.add_page_break()
    add_heading(doc, "ML Model Status and Accuracy", 1)
    add_table(
        doc,
        ["Item", "Current value", "Interpretation"],
        [
            ["Deployment model", "models/autoencoder.onnx, about 65 MB", "ONNX Runtime model used by the detector."],
            ["Training checkpoint", "models/autoencoder.pt, about 65 MB", "PyTorch training artifact."],
            ["Input", "64 by 257 magnitude spectrogram", "A short time-frequency patch from the audio DSP path."],
            ["Architecture", "16,448 to 512 to 64 bottleneck to 512 to 16,448", "Fully connected autoencoder with ReLU layers and sigmoid decoder."],
            ["Training", "Adam, learning rate 0.001, MSE, 15 epochs", "Trained only on baseline patches using a fixed reference scale."],
            ["Virtual AE trip", "0.029356", "Operational reconstruction-error threshold after applying the configured margin."],
            ["Virtual tone trip", "7.275 dB", "Secondary narrow-band guard calibrated from baseline patches."],
            ["Saved fixture result", "0 of 1,404 baseline breaches; 936 of 936 attack patches detected", "Strong in-project validation, but not a general real-world accuracy percentage."],
        ],
        [1.35, 2.15, 3.0],
    )
    add_text(doc, "The detector is hybrid. The autoencoder finds patterns that reconstruct poorly compared with normal data. A separate tonal-prominence measurement looks for narrow spectral lines commonly associated with pulsed acoustic channels. A threat can be raised by either guard, and a tonal-channel event includes its frequency in the SQLite log.")
    add_text(doc, "The 100 percent attack-fixture detection figure should be presented carefully: it applies to the supplied 936 synthetic attack patches and the saved baseline set used for calibration. It does not prove 100 percent accuracy against unknown real attacks or all possible music and environmental noise. Real-world evaluation needs separately labelled recordings, multiple rooms, multiple microphones, false-positive measurement, and hardware performance tests.")

    add_heading(doc, "How the Project Was Built", 1)
    add_bullet(doc, "Synthetic normal and infected WAV fixtures were generated for a safe reproducible demonstration.")
    add_bullet(doc, "The DSP path applies bandpass and mains-harmonic filtering, then uses a short-time Fourier transform to create spectrogram patches.")
    add_bullet(doc, "The autoencoder learns to reconstruct normal patches. Large reconstruction error is anomaly evidence.")
    add_bullet(doc, "The ONNX export keeps inference lightweight and avoids requiring PyTorch during normal monitoring.")
    add_bullet(doc, "The detector uses hysteresis states CLEAN, ARMED, THREAT DETECTED, NO SIGNAL, and SENSOR LOSS to avoid treating a single frame as a final decision.")
    add_bullet(doc, "The Rich terminal dashboard renders acoustic and optional RF waterfalls, while SQLite retains event state, timestamps, scores, descriptions, and tonal frequency where available.")
    add_bullet(doc, "Virtual and real-room calibration profiles are kept separate so a physical microphone experiment never changes the verified virtual demonstration.")

    doc.add_page_break()
    add_heading(doc, "Optional Raspberry Pi and Hardware Integration", 1)
    add_text(doc, "The current requested deliverable is software-first and does not require hardware. The repository nevertheless contains an integration path for a later edge-device build. Treat the following as an engineering plan, not as completed or validated deployment work.")
    add_heading(doc, "Existing Integration Points", 2)
    add_table(
        doc,
        ["Hardware role", "Existing software point", "What remains"],
        [
            ["USB microphone", "LiveAudioSource through sounddevice and --source microphone", "Select stable device, calibrate room profile, measure CPU load and latency."],
            ["RTL SDR", "RTLSDRBackend in capture/rf_source.py", "Install radio driver and pyrtlsdr, set RF configuration, create an RF baseline."],
            ["HackRF", "HackRFBackend in capture/rf_source.py", "Install driver/binding, set gain and sample rate, create an RF baseline."],
            ["Edge computer or Raspberry Pi", "Python runner and ONNX Runtime inference", "Benchmark 30 Hz UI and RF sweep throughput on the chosen board."],
            ["Multiple antennas or sensors", "Bearing and sweep modules", "Design sensor geometry, synchronisation, and a validated localisation method."],
        ],
        [1.35, 2.35, 2.8],
    )
    add_heading(doc, "Suggested Hardware Bring Up", 2)
    add_bullet(doc, "Use a supported Linux edge device with a 48 kHz USB microphone and either an RTL SDR or HackRF. Connect only hardware authorised for the deployment environment.")
    add_bullet(doc, "Install the base project dependencies and, only for real RF use, the optional RF dependencies and vendor drivers.")
    add_code(doc, "pip install -r requirements.txt\npip install -r requirements-rf.txt")
    add_bullet(doc, "Set audio.device to the correct microphone input number. Set rf.enabled to true and rf.backend to rtlsdr or hackrf in config.toml. Keep the virtual configuration as a separate copy for presentations.")
    add_bullet(doc, "Create a room profile with calibrate_room.py. Then create an RF baseline in an authorised quiet environment.")
    add_code(doc, "python train/calibrate_room.py --seconds 45\npython capture/record_rf.py --sweeps 100\npython deploy/run_app.py --source microphone --profile room")
    add_bullet(doc, "Run controlled tests with benign simulated carriers only. Record false positives, missed detections, temperature, CPU use, UI refresh rate, and RF sweep time before making any security claim.")
    add_heading(doc, "Hardware Limits", 2)
    add_text(doc, "A single microphone or single RF receiver can report anomaly evidence and a frequency, but it cannot reliably identify an infected server rack. Rack-level localisation needs multiple calibrated sensors or directional/steered antennas, controlled geometry, synchronisation, and testing against multipath reflections. This capability is outside the validated scope of the current project.")

    # Keep the demo heading with its checklist for presentation use.
    doc.add_page_break()
    add_heading(doc, "Demo Day Script", 1)
    add_text(doc, "Use this short sequence when presenting the project. It separates proven virtual behaviour from the optional calibrated real-room experiment.")
    add_table(
        doc,
        ["Step", "Command or action", "What to explain"],
        [
            ["1", "python deploy/run_app.py --scenario baseline", "This is the normal virtual room. The dashboard remains CLEAN."],
            ["2", "python deploy/run_app.py --scenario attack --simulated-rf --inject", "The structured virtual signal drives ARMED then THREAT DETECTED; the dashboard exposes AE and tonal evidence."],
            ["3", "Show Figure 1 metrics", "Yellow waterfall energy is visual context. The red state comes from detection logic, not colour alone."],
            ["4", "Optional: python deploy/run_app.py --source microphone --profile room", "This is separately calibrated for the current room. Normal calibrated conditions should remain CLEAN."],
            ["5", "Show pytest -q and --self-test output", "The software has automated checks and fixture validation; real-world accuracy still needs field evaluation."],
        ],
        [0.48, 2.65, 3.37],
    )

    add_heading(doc, "Quick Troubleshooting", 1)
    add_table(
        doc,
        ["Symptom", "Cause", "Action"],
        [
            ["File not found for a custom WAV", "Example path was typed literally or file does not exist", "Use one of the supplied data paths or give the full path to an existing 48 kHz WAV."],
            ["Normal music shows threat in microphone mode", "Virtual calibration was used or music was not represented during room calibration", "Create or refresh the room profile while the normal music is present."],
            ["Mic off shows NO SIGNAL or SENSOR LOSS", "The silence gate sees no usable input", "This is correct behaviour. Check the microphone only if monitoring should be active."],
            ["Room profile cannot be found", "Calibration has not yet created models/profiles/room.json", "Run python train/calibrate_room.py --seconds 45."],
            ["RF hardware does not start", "Driver, permission, device, or optional package is missing", "Return to simulated RF for the demo; troubleshoot the hardware separately before claiming deployment."],
        ],
        [1.45, 2.35, 2.7],
    )

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
