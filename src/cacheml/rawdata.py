from __future__ import annotations
from .common import *


def convert_twitter(raw: str, out: str, max_rows: int = 0) -> int:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    kept, scanned, bad_rows = 0, 0, 0
    with Path(out).open("w", newline="", encoding="utf-8") as f, Path(raw).open(
        "r", encoding="utf-8", errors="strict"
    ) as text_file:
        w = csv.writer(f)
        w.writerow(["obj", "size", "type", "timestamp", "op", "ttl"])
        for line in text_file:
            scanned += 1
            if len(row := line.strip().split(",")) < 7:
                bad_rows += 1
                continue
            try:
                w.writerow(
                    [
                        row[1],
                        max(float(row[3]), float(row[2]), 1.0),
                        row[4],
                        float(row[0]),
                        row[5].strip().lower(),
                        row[6],
                    ]
                )
                kept += 1
                if 0 < max_rows <= kept:
                    break
            except Exception:
                bad_rows += 1
    print(f"scanned_rows={scanned} bad_rows={bad_rows} converted_rows={kept}")
    return kept
