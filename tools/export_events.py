"""Export the security_events SQLite log to CSV and/or JSONL.

The brief asks to "log timestamp and frequency" -- SQLite already does that
(see infer/detector.py's _EventWriter), but a plain-text export is useful for
handing a report to someone who doesn't want to open the .db file, or for
feeding the data to another tool. This only reads events.db; it never
touches the live detection path.

Usage:
    python tools/export_events.py                       # events.csv + events.jsonl next to the db
    python tools/export_events.py --format csv
    python tools/export_events.py --format jsonl
    python tools/export_events.py --db events.db --out output/events
    python tools/export_events.py --since 2026-11-01T00:00:00 --state THREAT_DETECTED
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

COLUMNS = ("id", "timestamp", "timestamp_iso", "score", "current_state", "description", "frequency_hz")


def _parse_since(value: str | None) -> float | None:
    """Accept either a unix timestamp or an ISO-8601 string."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def fetch_events(db_path: str, since: float | None = None, state: str | None = None) -> list[dict]:
    if not Path(db_path).exists():
        raise FileNotFoundError(f"no such database: {db_path}")
    query = "SELECT id, timestamp, score, current_state, description, frequency_hz FROM security_events"
    clauses: list[str] = []
    params: list[object] = []
    if since is not None:
        clauses.append("timestamp >= ?")
        params.append(since)
    if state is not None:
        clauses.append("current_state = ?")
        params.append(state)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY id ASC"

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()

    events = []
    for row in rows:
        record = dict(row)
        record["timestamp_iso"] = (
            datetime.fromtimestamp(record["timestamp"], tz=timezone.utc).isoformat()
            if record["timestamp"] is not None
            else None
        )
        events.append(record)
    return events


def write_csv(events: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for event in events:
            writer.writerow({col: event.get(col) for col in COLUMNS})


def write_jsonl(events: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps({col: event.get(col) for col in COLUMNS}) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="events.db", help="path to the SQLite event log (default: events.db)")
    parser.add_argument("--out", default=None, help="output path without extension (default: <db-stem>_events next to the db)")
    parser.add_argument("--format", choices=("csv", "jsonl", "both"), default="both")
    parser.add_argument("--since", default=None, help="only export events at/after this unix timestamp or ISO-8601 datetime")
    parser.add_argument("--state", default=None, help="only export events matching this current_state (e.g. THREAT_DETECTED)")
    args = parser.parse_args(argv)

    try:
        since = _parse_since(args.since)
    except ValueError:
        print(f"error: --since value {args.since!r} is neither a unix timestamp nor ISO-8601", file=sys.stderr)
        return 2

    try:
        events = fetch_events(args.db, since=since, state=args.state)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_base = Path(args.out) if args.out else Path(args.db).with_suffix("").parent / f"{Path(args.db).stem}_events"

    wrote = []
    if args.format in ("csv", "both"):
        csv_path = out_base.with_suffix(".csv")
        write_csv(events, csv_path)
        wrote.append(csv_path)
    if args.format in ("jsonl", "both"):
        jsonl_path = out_base.with_suffix(".jsonl")
        write_jsonl(events, jsonl_path)
        wrote.append(jsonl_path)

    print(f"[+] Exported {len(events)} event(s) from {args.db}")
    for path in wrote:
        print(f"    -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
