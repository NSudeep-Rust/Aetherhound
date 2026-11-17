# gnuradio/ -- reference flowgraph for the Project 2 brief

`aetherhound_dsp_demo.grc` is a standalone GNU Radio Companion flowgraph
mirroring `dsp/pipeline.py`'s signal chain (room noise + 50 Hz hum + a 4 kHz
covert tone -> band-pass -> mains notch -> FFT -> averaged power spectrum)
using stock GRC blocks instead of scipy.

**This was authored without access to GNU Radio Companion** (no
network/display in the sandbox it was written in) -- the YAML is valid and
internally consistent (every connection references a declared block; see
`python -c "import yaml; yaml.safe_load(open('gnuradio/aetherhound_dsp_demo.grc'))"`
for a quick syntax check), but GNU Radio's own per-block parameter schema
was never checked against an actual GRC install. **Open it once in GNU
Radio Companion and re-save before relying on it**:

```bash
sudo apt install gnuradio      # Debian/Kali; see the review's own suggestion
gnuradio-companion gnuradio/aetherhound_dsp_demo.grc
```

This is a reference/demo artifact, not part of the live detection path --
the actual, tested, calibrated DSP stays in `dsp/pipeline.py` (scipy). It
exists to close the brief's "DSP with a GNU Radio flowgraph" checkbox
honestly: something real to open and inspect, clearly documented as
illustrative rather than production.
