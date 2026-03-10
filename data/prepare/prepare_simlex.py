#!/usr/bin/env python3
"""Convert SimLex-999 raw data to standardized JSONL format.

SimLex-999 format (TSV): word1 word2 POS SimLex999 conc(w1) conc(w2) concQ Assoc1 Assoc2 SD(SimLex)
Score range: 0.0 - 10.0 (normalized to 0.0 - 1.0)
"""
import json
import os
import sys
import glob

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw", "simlex")
OUT_FILE = os.path.join(os.path.dirname(__file__), "..", "prepared", "semantic", "simlex_pairs.jsonl")

def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    candidates = glob.glob(os.path.join(RAW_DIR, "**", "SimLex*"), recursive=True) + \
                 glob.glob(os.path.join(RAW_DIR, "**", "simlex*"), recursive=True) + \
                 glob.glob(os.path.join(RAW_DIR, "*.txt"), recursive=False) + \
                 glob.glob(os.path.join(RAW_DIR, "*.tsv"), recursive=False) + \
                 glob.glob(os.path.join(RAW_DIR, "*.csv"), recursive=False)
    candidates = sorted(set(candidates))

    if not candidates:
        print(f"ERROR: No SimLex files found in {RAW_DIR}", file=sys.stderr)
        sys.exit(1)

    total = 0
    with open(OUT_FILE, "w") as out:
        for fpath in candidates:
            fname = os.path.basename(fpath)
            if os.path.isdir(fpath) or "readme" in fname.lower():
                continue
            print(f"Processing {fname}...")
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                header = None
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")

                    if header is None:
                        header = [h.strip().lower() for h in parts]
                        continue

                    row = {}
                    for i, h in enumerate(header):
                        if i < len(parts):
                            row[h] = parts[i].strip()

                    word1 = row.get("word1", row.get("w1", ""))
                    word2 = row.get("word2", row.get("w2", ""))

                    score_raw = None
                    for key in ["simlex999", "similarity", "score", "sim"]:
                        if key in row:
                            try:
                                score_raw = float(row[key])
                                break
                            except ValueError:
                                continue

                    if not word1 or not word2 or score_raw is None:
                        continue

                    # SimLex scores are 0-10, normalize to 0-1
                    score_norm = score_raw / 10.0
                    score_norm = max(0.0, min(1.0, score_norm))

                    label = "similar" if score_norm >= 0.6 else ("neutral" if score_norm >= 0.3 else "dissimilar")

                    record = {
                        "id": f"simlex-{total:04d}",
                        "text_a": word1,
                        "text_b": word2,
                        "score": round(score_norm, 4),
                        "label": label,
                        "source": "simlex"
                    }
                    out.write(json.dumps(record) + "\n")
                    total += 1

    print(f"Wrote {total} pairs to {OUT_FILE}")
    return total

if __name__ == "__main__":
    n = main()
    if n == 0:
        sys.exit(1)
