#!/usr/bin/env python3
"""Validate all prepared data files — schema, row counts, FNV-1a hashes.

Checks:
1. Every JSONL record has required fields: id, text_a, text_b, score, label, source
2. Score is float in [0.0, 1.0]
3. Label is one of: similar, dissimilar, neutral
4. No empty text fields
5. Row counts match expectations
6. FNV-1a hash of each file for integrity tracking
"""
import json
import os
import sys
from collections import defaultdict

PREPARED_DIR = os.path.join(os.path.dirname(__file__), "..", "prepared", "semantic")
SPLITS_DIR = os.path.join(os.path.dirname(__file__), "..", "splits")

REQUIRED_FIELDS = {"id", "text_a", "text_b", "score", "label", "source"}
VALID_LABELS = {"similar", "dissimilar", "neutral"}

# Expected minimum row counts per dataset
EXPECTED_MINS = {
    "sts": 5000,
    "sick": 4000,
    "mrpc": 3000,
    "qqp": 40000,
    "snli": 80000,
    "simlex": 900,
    "wordsim": 300,
}

def fnv1a_64(data: bytes) -> int:
    """FNV-1a 64-bit hash."""
    h = 0xcbf29ce484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h

def validate_jsonl(fpath):
    """Validate a single JSONL file. Returns (record_count, errors, source_counts)."""
    errors = []
    source_counts = defaultdict(int)
    count = 0

    with open(fpath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"  Line {line_num}: invalid JSON: {e}")
                continue

            count += 1

            # Check required fields
            missing = REQUIRED_FIELDS - set(record.keys())
            if missing:
                errors.append(f"  Line {line_num}: missing fields: {missing}")
                continue

            # Check types
            if not isinstance(record["text_a"], str) or not record["text_a"].strip():
                errors.append(f"  Line {line_num}: empty text_a")
            if not isinstance(record["text_b"], str) or not record["text_b"].strip():
                errors.append(f"  Line {line_num}: empty text_b")
            if not isinstance(record["score"], (int, float)):
                errors.append(f"  Line {line_num}: score not numeric: {record['score']}")
            elif not (0.0 <= record["score"] <= 1.0):
                errors.append(f"  Line {line_num}: score out of range: {record['score']}")
            if record["label"] not in VALID_LABELS:
                errors.append(f"  Line {line_num}: invalid label: {record['label']}")

            source_counts[record.get("source", "unknown")] += 1

    return count, errors, source_counts

def main():
    all_ok = True
    print("=== TRINE Data Validation ===\n")

    # Validate prepared files
    print("--- Prepared JSONL Files ---")
    if os.path.isdir(PREPARED_DIR):
        jsonl_files = sorted(f for f in os.listdir(PREPARED_DIR) if f.endswith(".jsonl"))
        for fname in jsonl_files:
            fpath = os.path.join(PREPARED_DIR, fname)
            count, errors, sources = validate_jsonl(fpath)

            # Compute hash
            with open(fpath, "rb") as f:
                h = fnv1a_64(f.read())

            status = "OK" if not errors else "ERRORS"
            print(f"  {fname}: {count} records, FNV-1a={h:016x} [{status}]")
            for src, cnt in sorted(sources.items()):
                print(f"    source={src}: {cnt}")
                # Check minimum counts
                if src in EXPECTED_MINS and cnt < EXPECTED_MINS[src]:
                    print(f"    WARNING: expected >= {EXPECTED_MINS[src]}, got {cnt}")
                    all_ok = False
            if errors:
                all_ok = False
                for e in errors[:10]:
                    print(e)
                if len(errors) > 10:
                    print(f"  ... and {len(errors) - 10} more errors")
    else:
        print(f"  WARNING: {PREPARED_DIR} does not exist")
        all_ok = False

    # Validate splits
    print("\n--- Split Files ---")
    total_split = 0
    if os.path.isdir(SPLITS_DIR):
        for split_name in ["train", "val", "test"]:
            fpath = os.path.join(SPLITS_DIR, f"{split_name}.jsonl")
            if not os.path.exists(fpath):
                print(f"  {split_name}.jsonl: MISSING")
                all_ok = False
                continue
            count, errors, sources = validate_jsonl(fpath)
            with open(fpath, "rb") as f:
                h = fnv1a_64(f.read())
            status = "OK" if not errors else "ERRORS"
            print(f"  {split_name}.jsonl: {count} records, FNV-1a={h:016x} [{status}]")
            total_split += count
            if errors:
                all_ok = False
                for e in errors[:5]:
                    print(e)
    else:
        print(f"  WARNING: {SPLITS_DIR} does not exist")
        all_ok = False

    if total_split > 0:
        print(f"\n  Total across splits: {total_split}")

    print(f"\n=== Validation {'PASSED' if all_ok else 'FAILED'} ===")
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
