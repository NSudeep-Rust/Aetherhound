"""Mock malware: burn CPU in a Morse pattern so the heatsink fan carries the message.

Fan inertia is 1-3 s, so this is slow CW, not real Morse timing. Symbols are
stretched accordingly. Calibrate first: run --dry-run and listen to whether the
fan audibly changes pitch at all before trusting a live demo.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..", "0": "-----", "1": ".----", "2": "..---",
    "3": "...--", "4": "....-", "5": ".....", "6": "-....", "7": "--...",
    "8": "---..", "9": "----.",
}


def to_morse(text: str) -> str:
    words = []
    for word in text.upper().split():
        words.append(" ".join(MORSE[c] for c in word if c in MORSE))
    return " / ".join(words)


def burn(stop: mp.Event) -> None:
    x = 0.0
    while not stop.is_set():
        for _ in range(20000):
            x = x * 1.0000001 + 0.5
        if x > 1e12:
            x = 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("message", nargs="?", default="SOS AETHERHOUND")
    ap.add_argument("--cores", type=int, default=0)
    ap.add_argument("--dot-ms", type=float, default=800)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cores = args.cores or mp.cpu_count()
    dot = args.dot_ms / 1000.0
    dash, intra, gap, word = dot * 3, dot, dot * 3, dot * 7

    code = to_morse(args.message)
    print(f"[*] {cores} cores, dot {dot*1000:.0f} ms")
    print(f"[*] {code}")

    total = sum(dash if s == "-" else dot if s == "." else
                gap if s == " " else word if s == "/" else intra
                for s in code)
    print(f"[*] estimated duration {total:.1f} s")

    if args.dry_run:
        return 0

    stop = mp.Event()
    workers = [mp.Process(target=burn, args=(stop,), daemon=True) for _ in range(cores)]
    for w in workers:
        w.start()

    print("[*] idling 4 s so the fan settles…")
    time.sleep(4.0)

    try:
        for sym in code:
            if sym == ".":
                stop.clear(); time.sleep(dot)
            elif sym == "-":
                stop.clear(); time.sleep(dash)
            elif sym == " ":
                stop.set(); time.sleep(gap)
            elif sym == "/":
                stop.set(); time.sleep(word)
            else:
                stop.set(); time.sleep(intra)
            print(sym, end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for w in workers:
            w.join(timeout=2.0)
        print("\n[+] done")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
