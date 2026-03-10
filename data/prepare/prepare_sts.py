#!/usr/bin/env python3
"""Convert STS-B raw files to standardized JSONL format.

STS-B format (TSV): genre filename year score sentence1 sentence2
Score range: 0.0 - 5.0 (normalized to 0.0 - 1.0)
"""
import json
import os
import sys
import glob

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw", "sts")
OUT_FILE = os.path.join(os.path.dirname(__file__), "..", "prepared", "semantic", "sts_pairs.jsonl")

def parse_sts_line(line, line_num, split_name):
    """Parse a single STS-B TSV line."""
    parts = line.strip().split("\t")
    # GLUE STS-B format (10 columns):
    #   index genre filename year old_index source1 source2 sentence1 sentence2 score
    # Test split has only 9 columns (no score).
    if len(parts) >= 10:
        # Full GLUE format with score
        text_a = parts[7]
        text_b = parts[8]
        try:
            score_raw = float(parts[9])
        except ValueError:
            return None
    elif len(parts) == 9:
        # Test split (no score) — skip, we can't use unlabeled data
        return None
    elif len(parts) >= 4:
        # Fallback: try last column as score, second/third-to-last as texts
        try:
            score_raw = float(parts[-1])
            text_a = parts[-3]
            text_b = parts[-2]
        except (ValueError, IndexError):
            return None
    else:
        return None

    if not text_a.strip() or not text_b.strip():
        return None
    # Filter out placeholder values like "none"
    if text_a.strip().lower() == "none" or text_b.strip().lower() == "none":
        return None

    score_norm = score_raw / 5.0  # Normalize 0-5 -> 0-1
    score_norm = max(0.0, min(1.0, score_norm))
    label = "similar" if score_norm >= 0.6 else ("neutral" if score_norm >= 0.3 else "dissimilar")

    return {
        "id": f"sts-{split_name}-{line_num:05d}",
        "text_a": text_a.strip(),
        "text_b": text_b.strip(),
        "score": round(score_norm, 4),
        "label": label,
        "source": "sts"
    }

def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    # Find STS files - try multiple naming conventions
    patterns = [
        os.path.join(RAW_DIR, "sts-*.tsv"),
        os.path.join(RAW_DIR, "sts-*.csv"),
        os.path.join(RAW_DIR, "*.tsv"),
        os.path.join(RAW_DIR, "*.csv"),
        os.path.join(RAW_DIR, "*.txt"),
    ]

    all_files = []
    for pat in patterns:
        all_files.extend(glob.glob(pat))
    all_files = sorted(set(all_files))

    if not all_files:
        print(f"ERROR: No STS files found in {RAW_DIR}", file=sys.stderr)
        print(f"  Searched patterns: {patterns}", file=sys.stderr)
        sys.exit(1)

    total = 0
    with open(OUT_FILE, "w") as out:
        for fpath in all_files:
            fname = os.path.basename(fpath)
            # Determine split name from filename
            if "train" in fname.lower():
                split_name = "train"
            elif "dev" in fname.lower() or "val" in fname.lower():
                split_name = "dev"
            elif "test" in fname.lower():
                split_name = "test"
            else:
                split_name = "unknown"

            print(f"Processing {fname} (split={split_name})...")
            line_num = 0
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Skip header lines
                    if line_num == 0 and ("sentence1" in line.lower() or "score" in line.lower().split("\t")[0:1]):
                        line_num += 1
                        continue
                    record = parse_sts_line(line, total, split_name)
                    if record:
                        out.write(json.dumps(record) + "\n")
                        total += 1
                    line_num += 1

    print(f"Wrote {total} pairs to {OUT_FILE}")
    return total

if __name__ == "__main__":
    n = main()
    if n == 0:
        sys.exit(1)
