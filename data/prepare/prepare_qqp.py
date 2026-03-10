#!/usr/bin/env python3
"""Convert QQP raw data to standardized JSONL format (50K subset).

QQP format (TSV): id qid1 qid2 question1 question2 is_duplicate
Score: binary (1.0 or 0.0)
We take a balanced 50K subset (25K duplicates + 25K non-duplicates).
"""
import json
import os
import sys
import glob
import random

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw", "qqp")
OUT_FILE = os.path.join(os.path.dirname(__file__), "..", "prepared", "semantic", "qqp_pairs.jsonl")
SUBSET_SIZE = 50000

def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    random.seed(42)  # Reproducible subset

    # Find QQP files
    candidates = glob.glob(os.path.join(RAW_DIR, "*.tsv")) + \
                 glob.glob(os.path.join(RAW_DIR, "*.txt")) + \
                 glob.glob(os.path.join(RAW_DIR, "*.csv"))
    candidates = sorted(set(candidates))

    if not candidates:
        print(f"ERROR: No QQP files found in {RAW_DIR}", file=sys.stderr)
        sys.exit(1)

    positive = []
    negative = []

    for fpath in candidates:
        fname = os.path.basename(fpath)
        if "readme" in fname.lower():
            continue
        # Prefer train split for more data
        print(f"Processing {fname}...")
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            header = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")

                if header is None:
                    if "question" in line.lower() or "duplicate" in line.lower():
                        header = [h.strip().lower() for h in parts]
                        continue
                    else:
                        header = ["id", "qid1", "qid2", "question1", "question2", "is_duplicate"]

                # Map columns
                row = {}
                for i, h in enumerate(header):
                    if i < len(parts):
                        row[h] = parts[i].strip()

                text_a = row.get("question1", row.get("q1", ""))
                text_b = row.get("question2", row.get("q2", ""))
                is_dup_str = row.get("is_duplicate", row.get("label", row.get("duplicate", "")))

                if not text_a or not text_b:
                    continue
                try:
                    is_dup = int(is_dup_str)
                except (ValueError, TypeError):
                    continue

                pair_id = row.get("id", str(len(positive) + len(negative)))
                record = {
                    "id": f"qqp-{pair_id}",
                    "text_a": text_a,
                    "text_b": text_b,
                    "score": 1.0 if is_dup else 0.0,
                    "label": "similar" if is_dup else "dissimilar",
                    "source": "qqp"
                }
                if is_dup:
                    positive.append(record)
                else:
                    negative.append(record)

    print(f"Full dataset: {len(positive)} positive, {len(negative)} negative")

    # Balanced subset
    half = SUBSET_SIZE // 2
    pos_sample = random.sample(positive, min(half, len(positive)))
    neg_sample = random.sample(negative, min(half, len(negative)))
    subset = pos_sample + neg_sample
    random.shuffle(subset)

    # Re-id
    for i, rec in enumerate(subset):
        rec["id"] = f"qqp-{i:05d}"

    with open(OUT_FILE, "w") as out:
        for rec in subset:
            out.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(subset)} pairs to {OUT_FILE} (balanced 50K subset)")
    return len(subset)

if __name__ == "__main__":
    n = main()
    if n == 0:
        sys.exit(1)
