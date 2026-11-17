# native/ -- C averaged_spectrum (opt-in, not wired into production)

Closes the Project 2 brief's "move DSP to C++" checkbox honestly rather than
just for show. See `averaged_spectrum.c`'s file header and
`bench_averaged_spectrum.py` for the full story; the short version:

The project review profiled a live RF sweep and found FFT work is ~33% of
per-step time (the simulator's own RNG is 45%, a fixed per-step sleep is
19%) -- so even a free, instant FFT could only ever claw back about a third
of one sweep step. This directory ports `dsp/rf_pipeline.py`'s
`averaged_spectrum` to plain C (a hand-rolled radix-2 FFT, called via
`ctypes` -- no pybind11/cmake, no network access needed to build or run it)
and benchmarks it against the existing numpy/scipy implementation at the
project's real sizes.

Measured in this sandbox (scipy wasn't installable here -- see the
benchmark script's note; the numpy-only reference it falls back to is
mathematically identical, just swaps `scipy.signal.get_window("hann", n)`
for the equivalent closed-form formula):

| nfft | frames | native vs numpy |
|------|--------|------------------|
| 512  | 1      | ~0.6x (native a bit faster) |
| 2048 | 4      | ~2.8x slower |
| 2048 (larger capture) | 4 | ~3.0x slower |

At the project's actual RF config (`nfft = 2048`, `frames_per_spectrum = 4`
in `config.toml`), the native port is **slower**, not faster -- numpy's FFT
(pocketfft, vectorised/SIMD) wins once there's enough work per call to
amortize its overhead. That matches the review's own conclusion. Re-run
`python native/bench_averaged_spectrum.py` on a machine with the project's
real `requirements-train.txt`/`requirements.txt` installed (scipy included)
to see the true production-path numbers instead of the numpy-only
fallback.

**This is why `native_rf.py` is not imported anywhere in the live detection
path.** Swapping a working, tested implementation for a slower, less-tested
one would be a bad trade dressed up as a performance win.

## Build

```bash
./native/build.sh          # -> native/libaveraged_spectrum.so (gitignored, rebuild per machine)
python native/bench_averaged_spectrum.py
pytest tests/test_native_averaged_spectrum.py   # skips automatically if the .so isn't built
```
