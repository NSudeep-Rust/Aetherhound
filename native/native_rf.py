"""ctypes wrapper around native/averaged_spectrum.c's averaged_spectrum_c().

Mirrors dsp.rf_pipeline.averaged_spectrum's signature and semantics (same
Welch-averaged, dB, fftshifted output) so it can be swapped in for direct
comparison -- see native/bench_averaged_spectrum.py. Not imported by
anything in the live detection path; this is a benchmark/demo artifact (see
averaged_spectrum.c's file header for why).

Build the library first:
    ./native/build.sh
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).resolve().parent / "libaveraged_spectrum.so"
_lib = None


class NativeLibraryNotBuilt(RuntimeError):
    pass


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    if not _LIB_PATH.exists():
        raise NativeLibraryNotBuilt(
            f"{_LIB_PATH} not found -- build it first with ./native/build.sh"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))

    c_double_p = ctypes.POINTER(ctypes.c_double)
    lib.averaged_spectrum_c.argtypes = [
        c_double_p, c_double_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, c_double_p,
    ]
    lib.averaged_spectrum_c.restype = ctypes.c_int
    _lib = lib
    return lib


def averaged_spectrum_native(frames, nfft: int, sample_rate: float) -> np.ndarray:
    """Drop-in native equivalent of dsp.rf_pipeline.averaged_spectrum.

    ``sample_rate`` is accepted for signature parity (the upstream Python
    function also never uses it in the actual computation) but is unused.
    """
    if not frames:
        raise ValueError("no frames supplied")
    lib = _load_lib()

    frames = [np.asarray(f, dtype=np.complex128) for f in frames]
    n_samples_per_frame = frames[0].size
    for f in frames:
        if f.size != n_samples_per_frame:
            raise ValueError("all frames must have the same number of samples")
        if f.size < nfft:
            raise ValueError(f"need >= {nfft} samples per frame, got {f.size}")

    stacked = np.vstack(frames)
    re = np.ascontiguousarray(stacked.real, dtype=np.float64)
    im = np.ascontiguousarray(stacked.imag, dtype=np.float64)
    out = np.empty(nfft, dtype=np.float64)

    c_double_p = ctypes.POINTER(ctypes.c_double)
    rc = lib.averaged_spectrum_c(
        re.ctypes.data_as(c_double_p),
        im.ctypes.data_as(c_double_p),
        ctypes.c_int(len(frames)),
        ctypes.c_int(n_samples_per_frame),
        ctypes.c_int(nfft),
        out.ctypes.data_as(c_double_p),
    )
    if rc != 0:
        raise ValueError(
            "native averaged_spectrum failed -- check nfft is a power of two "
            "and each frame has >= nfft samples"
        )
    return out
