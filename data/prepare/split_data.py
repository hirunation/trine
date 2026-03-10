#!/usr/bin/env python3
"""Create stratified train/val/test splits from prepared JSONL files.

Reads all prepared JSONL files, combines them, creates 70/15/15 splits
stratified by source dataset for balanced representation.
"""
import json
import os
import sys
import random
from collections import defaultdict

PREPARED_DIR = os.path.join(os.path.dirname(__file__), "..", "prepared", "semantic")
SPLITS_DIR = os.path.join(os.path.dirname(__file__), "..", "splits")
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def main():
    os.makedirs(SPLITS_DIR, exist_ok=True)
    random.seed(42)

    # Load all prepared JSONL files
    by_source = defaultdict(list)
    jsonl_files = [f for f in os.listdir(PREPARED_DIR) if f.endswith(".jsonl")]

    if not jsonl_files:
        print(f"ERROR: No JSONL files found in {PREPARED_DIR}", file=sys.stderr)
        sys.exit(1)

    total_loaded = 0
    for fname in sorted(jsonl_files):
        fpath = os.path.join(PREPARED_DIR, fname)
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                source = record.get("source", "unknown")
                by_source[source].append(record)
                total_loaded += 1

    print(f"Loaded {total_loaded} pairs from {len(jsonl_files)} files")
    for source, records in sorted(by_source.items()):
        print(f"  {source}: {len(records)} pairs")

    # Stratified split
    train = []
    val = []
    test = []

    for source, records in by_source.items():
        random.shuffle(records)
        n = len(records)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # Remaining goes to test
        train.extend(records[:n_train])
        val.extend(records[n_train:n_train + n_val])
        test.extend(records[n_train + n_val:])

    # Shuffle within splits
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    # Write splits
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        fpath = os.path.join(SPLITS_DIR, f"{split_name}.jsonl")
        with open(fpath, "w") as f:
            for record in split_data:
                f.write(json.dumps(record) + "\n")
        # Source distribution
        dist = defaultdict(int)
        for r in split_data:
            dist[r["source"]] += 1
        print(f"\n{split_name}: {len(split_data)} pairs")
        for src, cnt in sorted(dist.items()):
            print(f"  {src}: {cnt}")

    total_split = len(train) + len(val) + len(test)
    print(f"\nTotal: {total_split} pairs (train={len(train)}, val={len(val)}, test={len(test)})")
    print(f"Split ratios: train={len(train)/total_split:.2%}, val={len(val)/total_split:.2%}, test={len(test)/total_split:.2%}")

    if total_split != total_loaded:
        print(f"WARNING: {total_loaded - total_split} pairs lost during splitting!")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
