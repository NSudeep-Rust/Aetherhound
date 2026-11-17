/*
 * Native (C) port of dsp/rf_pipeline.py's power_spectrum_db / averaged_spectrum.
 *
 * This exists to "tick the box honestly" on the Project 2 brief's "move DSP
 * to C++" ask: the project review profiled the RF sweep and found FFT work
 * is ~33% of sweep time, with the simulator's own RNG (45%) and a fixed
 * per-step sleep (19%) dominating instead -- i.e. moving this one function
 * to native code was never going to move the needle much. See
 * native/bench_averaged_spectrum.py for the actual before/after numbers:
 * expect numpy's FFT (pocketfft, vectorised/SIMD) to stay competitive with
 * or beat this hand-rolled radix-2 FFT, especially at the project's actual
 * nfft (2048). This is therefore built and benchmarked as an opt-in
 * artifact, not wired into the live RF path by default -- replacing a
 * working, tested implementation with an unretested one for a performance
 * win that profiling says isn't there would not be a responsible trade.
 *
 * Plain C (not pybind11: no network access to install it in the sandbox
 * this was authored in, and a flat C ABI loaded via ctypes avoids coupling
 * the extension to one specific Python/CPython ABI version -- this repo's
 * own venv is 3.14, so a hand-built CPython extension for whatever headers
 * happen to be on a given machine is a worse bet than a ctypes shim that
 * works unchanged on any Python).
 *
 * Only supports nfft that is a power of two (true for every nfft used in
 * this project: 512 for audio, 2048 for RF).
 */
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct { double re, im; } cplx_t;

static int is_pow2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

/* In-place iterative radix-2 Cooley-Tukey FFT (forward transform only --
 * that's all averaged_spectrum needs). */
static void fft_radix2(cplx_t *a, int n) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            cplx_t tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * M_PI / (double)len;
        cplx_t wlen = { cos(ang), sin(ang) };
        for (int i = 0; i < n; i += len) {
            cplx_t w = { 1.0, 0.0 };
            int half = len / 2;
            for (int j = 0; j < half; j++) {
                cplx_t u = a[i + j];
                cplx_t t = a[i + j + half];
                cplx_t v = { t.re * w.re - t.im * w.im, t.re * w.im + t.im * w.re };
                a[i + j].re = u.re + v.re;
                a[i + j].im = u.im + v.im;
                a[i + j + half].re = u.re - v.re;
                a[i + j + half].im = u.im - v.im;
                double nw_re = w.re * wlen.re - w.im * wlen.im;
                double nw_im = w.re * wlen.im + w.im * wlen.re;
                w.re = nw_re;
                w.im = nw_im;
            }
        }
    }
}

/* Symmetric Hann window, matches scipy.signal.get_window("hann", n) with its
 * default sym=True: w[k] = 0.5 - 0.5*cos(2*pi*k/(n-1)). */
static void hann_window(double *w, int n) {
    if (n == 1) { w[0] = 1.0; return; }
    for (int k = 0; k < n; k++) {
        w[k] = 0.5 - 0.5 * cos(2.0 * M_PI * (double)k / (double)(n - 1));
    }
}

/*
 * Welch-style averaged periodogram in dB for one frame of complex IQ
 * samples, DC removed -- a faithful port of
 * dsp.rf_pipeline.power_spectrum_db.
 *
 * iq_re, iq_im: n_samples each.
 * out: nfft-length buffer, receives the fftshifted dB spectrum.
 * Returns 0 on success, -1 if n_samples < nfft or nfft is not a power of 2.
 */
int power_spectrum_db_c(const double *iq_re, const double *iq_im, int n_samples,
                        int nfft, double *out) {
    if (n_samples < nfft || !is_pow2(nfft)) return -1;

    double mean_re = 0.0, mean_im = 0.0;
    for (int i = 0; i < n_samples; i++) {
        mean_re += iq_re[i];
        mean_im += iq_im[i];
    }
    mean_re /= (double)n_samples;
    mean_im /= (double)n_samples;

    double *window = (double *)malloc(sizeof(double) * (size_t)nfft);
    double *acc = (double *)calloc((size_t)nfft, sizeof(double));
    cplx_t *buf = (cplx_t *)malloc(sizeof(cplx_t) * (size_t)nfft);
    if (!window || !acc || !buf) {
        free(window); free(acc); free(buf);
        return -1;
    }
    hann_window(window, nfft);

    int segments = n_samples / nfft;
    for (int s = 0; s < segments; s++) {
        int base = s * nfft;
        for (int k = 0; k < nfft; k++) {
            buf[k].re = (iq_re[base + k] - mean_re) * window[k];
            buf[k].im = (iq_im[base + k] - mean_im) * window[k];
        }
        fft_radix2(buf, nfft);
        for (int k = 0; k < nfft; k++) {
            acc[k] += buf[k].re * buf[k].re + buf[k].im * buf[k].im;
        }
    }
    for (int k = 0; k < nfft; k++) acc[k] /= (double)segments;

    /* fftshift (nfft is even: swap the two halves), floor, then dB. */
    int half = nfft / 2;
    for (int k = 0; k < nfft; k++) {
        double val = acc[(k + half) % nfft];
        if (val < 1e-20) val = 1e-20;
        out[k] = 10.0 * log10(val);
    }

    free(window);
    free(acc);
    free(buf);
    return 0;
}

/*
 * Multi-frame average: mirrors dsp.rf_pipeline.averaged_spectrum, which
 * computes power_spectrum_db per frame and then averages those *dB* values
 * across frames (not the linear power) -- reproduced exactly here, not
 * "corrected", to stay a faithful port.
 *
 * iq_re_frames, iq_im_frames: n_frames * n_samples_per_frame each, frames
 * laid out contiguously.
 * out: nfft-length buffer.
 * Returns 0 on success, -1 on error (including n_frames < 1).
 */
int averaged_spectrum_c(const double *iq_re_frames, const double *iq_im_frames,
                        int n_frames, int n_samples_per_frame, int nfft, double *out) {
    if (n_frames < 1) return -1;

    double *frame_out = (double *)malloc(sizeof(double) * (size_t)nfft);
    if (!frame_out) return -1;

    for (int k = 0; k < nfft; k++) out[k] = 0.0;

    for (int f = 0; f < n_frames; f++) {
        int base = f * n_samples_per_frame;
        int rc = power_spectrum_db_c(iq_re_frames + base, iq_im_frames + base,
                                     n_samples_per_frame, nfft, frame_out);
        if (rc != 0) { free(frame_out); return rc; }
        for (int k = 0; k < nfft; k++) out[k] += frame_out[k];
    }
    for (int k = 0; k < nfft; k++) out[k] /= (double)n_frames;

    free(frame_out);
    return 0;
}
