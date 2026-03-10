#!/usr/bin/env python3
"""Convert SNLI raw data to standardized JSONL format (100K subset).

SNLI format (JSONL or TSV): gold_label sentence1 sentence2 ...
Labels: entailment/neutral/contradiction
Score mapping: entailment=1.0, neutral=0.5, contradiction=0.0
We take a balanced 100K subset.
"""
import json
import os
import sys
import glob
import random

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw", "snli")
OUT_FILE = os.path.join(os.path.dirname(__file__), "..", "prepared", "semantic", "snli_pairs.jsonl")
SUBSET_SIZE = 100000

LABEL_MAP = {
    "entailment": ("similar", 1.0),
    "neutral": ("neutral", 0.5),
    "contradiction": ("dissimilar", 0.0),
}

def parse_jsonl(fpath):
    """Parse SNLI JSONL format."""
    records = {"entailment": [], "neutral": [], "contradiction": []}
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            gold = obj.get("gold_label", obj.get("label", "")).strip().lower()
            if gold not in LABEL_MAP:
                continue

            s1 = obj.get("sentence1", obj.get("premise", "")).strip()
            s2 = obj.get("sentence2", obj.get("hypothesis", "")).strip()
            if not s1 or not s2:
                continue

            label, score = LABEL_MAP[gold]
            records[gold].append({
                "text_a": s1,
                "text_b": s2,
                "score": score,
                "label": label,
                "source": "snli"
            })
    return records

def parse_tsv(fpath):
    """Parse SNLI TSV format."""
    records = {"entailment": [], "neutral": [], "contradiction": []}
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

            gold = row.get("gold_label", row.get("label", "")).lower()
            if gold not in LABEL_MAP:
                continue

            s1 = row.get("sentence1", row.get("premise", ""))
            s2 = row.get("sentence2", row.get("hypothesis", ""))
            if not s1 or not s2:
                continue

            label, score = LABEL_MAP[gold]
            records[gold].append({
                "text_a": s1,
                "text_b": s2,
                "score": score,
                "label": label,
                "source": "snli"
            })
    return records

def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    random.seed(42)

    candidates = glob.glob(os.path.join(RAW_DIR, "**", "*.jsonl"), recursive=True) + \
                 glob.glob(os.path.join(RAW_DIR, "**", "*.txt"), recursive=True) + \
                 glob.glob(os.path.join(RAW_DIR, "**", "*.tsv"), recursive=True)
    candidates = sorted(set(candidates))

    if not candidates:
        print(f"ERROR: No SNLI files found in {RAW_DIR}", file=sys.stderr)
        sys.exit(1)

    all_records = {"entailment": [], "neutral": [], "contradiction": []}

    for fpath in candidates:
        fname = os.path.basename(fpath)
        if "readme" in fname.lower() or fname.startswith("."):
            continue
        print(f"Processing {fname}...")

        # Detect format
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline().strip()

        if first_line.startswith("{"):
            records = parse_jsonl(fpath)
        else:
            records = parse_tsv(fpath)

        for key in all_records:
            all_records[key].extend(records[key])

    total_raw = sum(len(v) for v in all_records.values())
    print(f"Full dataset: {total_raw} pairs")
    for k, v in all_records.items():
        print(f"  {k}: {len(v)}")

    # Balanced 100K subset (33.3K per label)
    third = SUBSET_SIZE // 3
    subset = []
    for key in ["entailment", "neutral", "contradiction"]:
        sample = random.sample(all_records[key], min(third, len(all_records[key])))
        subset.extend(sample)

    # Fill remainder if any label was short
    remaining = SUBSET_SIZE - len(subset)
    if remaining > 0:
        all_flat = []
        for v in all_records.values():
            all_flat.extend(v)
        used = set(id(r) for r in subset)
        extras = [r for r in all_flat if id(r) not in used]
        subset.extend(random.sample(extras, min(remaining, len(extras))))

    random.shuffle(subset)

    # Assign IDs
    with open(OUT_FILE, "w") as out:
        for i, rec in enumerate(subset):
            rec["id"] = f"snli-{i:06d}"
            out.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(subset)} pairs to {OUT_FILE} (balanced 100K subset)")
    return len(subset)

if __name__ == "__main__":
    n = main()
    if n == 0:
        sys.exit(1)
