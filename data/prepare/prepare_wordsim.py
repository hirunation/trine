#!/usr/bin/env python3
"""Convert WordSim-353 raw data to standardized JSONL format.

WordSim-353 format (TSV/CSV): Word 1 Word 2 Human (mean)
Score range: 0.0 - 10.0 (normalized to 0.0 - 1.0)
"""
import json
import os
import sys
import glob

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw", "wordsim")
OUT_FILE = os.path.join(os.path.dirname(__file__), "..", "prepared", "semantic", "wordsim_pairs.jsonl")

def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    candidates = glob.glob(os.path.join(RAW_DIR, "**", "*.txt"), recursive=True) + \
                 glob.glob(os.path.join(RAW_DIR, "**", "*.tsv"), recursive=True) + \
                 glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True) + \
                 glob.glob(os.path.join(RAW_DIR, "**", "*.tab"), recursive=True)
    candidates = sorted(set(candidates))

    if not candidates:
        print(f"ERROR: No WordSim files found in {RAW_DIR}", file=sys.stderr)
        sys.exit(1)

    total = 0
    seen = set()
    with open(OUT_FILE, "w") as out:
        for fpath in candidates:
            fname = os.path.basename(fpath)
            if os.path.isdir(fpath) or "readme" in fname.lower():
                continue
            print(f"Processing {fname}...")

            # Detect separator
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                first = f.readline()
            sep = "\t" if "\t" in first else ","

            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                header_skipped = False
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = [p.strip().strip('"') for p in line.split(sep)]

                    # Skip header
                    if not header_skipped:
                        if any(h.lower() in ["word", "word1", "word 1", "human"] for h in parts):
                            header_skipped = True
                            continue
                        # Check if first field looks like a word (not a number)
                        try:
                            float(parts[0])
                            # It's a number, probably not a header
                        except ValueError:
                            # Could be header or data word
                            if len(parts) >= 3:
                                try:
                                    float(parts[-1])
                                    # Last field is numeric = data row, don't skip
                                except ValueError:
                                    header_skipped = True
                                    continue

                    if len(parts) < 3:
                        continue

                    word1 = parts[0].strip()
                    word2 = parts[1].strip()

                    try:
                        score_raw = float(parts[-1])
                    except ValueError:
                        try:
                            score_raw = float(parts[2])
                        except ValueError:
                            continue

                    if not word1 or not word2:
                        continue

                    key = f"{word1.lower()}|{word2.lower()}"
                    if key in seen:
                        continue
                    seen.add(key)

                    # WordSim scores are 0-10, normalize to 0-1
                    score_norm = score_raw / 10.0
                    score_norm = max(0.0, min(1.0, score_norm))

                    label = "similar" if score_norm >= 0.6 else ("neutral" if score_norm >= 0.3 else "dissimilar")

                    record = {
                        "id": f"wordsim-{total:04d}",
                        "text_a": word1,
                        "text_b": word2,
                        "score": round(score_norm, 4),
                        "label": label,
                        "source": "wordsim"
                    }
                    out.write(json.dumps(record) + "\n")
                    total += 1

    print(f"Wrote {total} pairs to {OUT_FILE}")
    return total

if __name__ == "__main__":
    n = main()
    if n == 0:
        sys.exit(1)
