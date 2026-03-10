#!/usr/bin/env python3
"""Convert SICK raw data to standardized JSONL format.

SICK format (TSV): pair_ID sentence_A sentence_B entailment_label relatedness_score ...
Relatedness range: 1.0 - 5.0 (normalized to 0.0 - 1.0)
"""
import json
import os
import sys
import glob

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw", "sick")
OUT_FILE = os.path.join(os.path.dirname(__file__), "..", "prepared", "semantic", "sick_pairs.jsonl")

def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    # Find SICK file
    candidates = glob.glob(os.path.join(RAW_DIR, "SICK*.txt")) + \
                 glob.glob(os.path.join(RAW_DIR, "sick*.txt")) + \
                 glob.glob(os.path.join(RAW_DIR, "SICK*.tsv")) + \
                 glob.glob(os.path.join(RAW_DIR, "*.tsv")) + \
                 glob.glob(os.path.join(RAW_DIR, "*.txt"))
    candidates = sorted(set(candidates))

    if not candidates:
        print(f"ERROR: No SICK files found in {RAW_DIR}", file=sys.stderr)
        sys.exit(1)

    total = 0
    with open(OUT_FILE, "w") as out:
        for fpath in candidates:
            print(f"Processing {os.path.basename(fpath)}...")
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

                    # Map columns by header
                    row = {}
                    for i, h in enumerate(header):
                        if i < len(parts):
                            row[h] = parts[i].strip()

                    # Extract fields - try various column names
                    pair_id = row.get("pair_id", row.get("id", str(total)))
                    text_a = row.get("sentence_a", row.get("sent_a", row.get("sentence1", "")))
                    text_b = row.get("sentence_b", row.get("sent_b", row.get("sentence2", "")))

                    # Score column
                    score_raw = None
                    for key in ["relatedness_score", "score", "relatedness"]:
                        if key in row:
                            try:
                                score_raw = float(row[key])
                                break
                            except ValueError:
                                continue

                    if not text_a or not text_b or score_raw is None:
                        continue

                    # Normalize 1-5 -> 0-1
                    score_norm = (score_raw - 1.0) / 4.0
                    score_norm = max(0.0, min(1.0, score_norm))

                    label_str = row.get("entailment_label", row.get("entailment_judgment", row.get("label", "")))
                    if "entail" in label_str.lower():
                        label = "similar"
                    elif "contradict" in label_str.lower():
                        label = "dissimilar"
                    else:
                        label = "similar" if score_norm >= 0.6 else ("neutral" if score_norm >= 0.3 else "dissimilar")

                    record = {
                        "id": f"sick-{pair_id}",
                        "text_a": text_a,
                        "text_b": text_b,
                        "score": round(score_norm, 4),
                        "label": label,
                        "source": "sick"
                    }
                    out.write(json.dumps(record) + "\n")
                    total += 1

    print(f"Wrote {total} pairs to {OUT_FILE}")
    return total

if __name__ == "__main__":
    n = main()
    if n == 0:
        sys.exit(1)
