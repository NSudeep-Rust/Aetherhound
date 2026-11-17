#!/usr/bin/env bash
# Build native/averaged_spectrum.c into a shared library. No cmake/pybind11
# needed -- this is one file with a plain C ABI, loaded via ctypes.
set -euo pipefail
cd "$(dirname "$0")"

CC="${CC:-cc}"
OUT="libaveraged_spectrum.so"

echo "[*] Building $OUT with $CC -O3..."
"$CC" -O3 -shared -fPIC -o "$OUT" averaged_spectrum.c -lm
echo "[+] Built native/$OUT"
