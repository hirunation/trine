#!/usr/bin/env python3
"""Convert MRPC raw data to standardized JSONL format.

MRPC format (TSV): Quality #1 ID #2 ID #1 String #2 String
Quality: 1 = paraphrase, 0 = not paraphrase
Score: binary (normalized: 1.0 or 0.0)
"""
import json
import os
import sys
import glob

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw", "mrpc")
OUT_FILE = os.path.join(os.path.dirname(__file__), "..", "prepared", "semantic", "mrpc_pairs.jsonl")

def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    # Find MRPC files
    candidates = glob.glob(os.path.join(RAW_DIR, "msr_paraphrase_*.txt")) + \
                 glob.glob(os.path.join(RAW_DIR, "*.tsv")) + \
                 glob.glob(os.path.join(RAW_DIR, "*.txt"))
    candidates = sorted(set(candidates))

    if not candidates:
        print(f"ERROR: No MRPC files found in {RAW_DIR}", file=sys.stderr)
        sys.exit(1)

    total = 0
    seen_ids = set()
    with open(OUT_FILE, "w") as out:
        for fpath in candidates:
            fname = os.path.basename(fpath)
            if "readme" in fname.lower():
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
                        # Detect header
                        if "quality" in line.lower() or "string" in line.lower() or "label" in line.lower():
                            header = [h.strip().lower() for h in parts]
                            continue
                        elif len(parts) >= 5:
                            # No header, assume standard MRPC format
                            header = ["quality", "id1", "id2", "string1", "string2"]
                        else:
                            continue

                    # Parse based on column count
                    if len(parts) >= 5:
                        # Standard: Quality ID1 ID2 String1 String2
                        try:
                            quality = int(parts[0].strip())
                        except ValueError:
                            continue
                        text_a = parts[3].strip()
                        text_b = parts[4].strip()
                        pair_id = f"{parts[1].strip()}-{parts[2].strip()}"
                    elif len(parts) >= 3:
                        # Simplified: label s1 s2 or s1 s2 label
                        try:
                            quality = int(parts[0].strip())
                            text_a = parts[1].strip()
                            text_b = parts[2].strip()
                        except ValueError:
                            try:
                                quality = int(parts[-1].strip())
                                text_a = parts[0].strip()
                                text_b = parts[1].strip()
                            except ValueError:
                                continue
                        pair_id = str(total)
                    else:
                        continue

                    if not text_a or not text_b:
                        continue

                    uid = f"{text_a[:50]}|{text_b[:50]}"
                    if uid in seen_ids:
                        continue
                    seen_ids.add(uid)

                    record = {
                        "id": f"mrpc-{pair_id}",
                        "text_a": text_a,
                        "text_b": text_b,
                        "score": 1.0 if quality == 1 else 0.0,
                        "label": "similar" if quality == 1 else "dissimilar",
                        "source": "mrpc"
                    }
                    out.write(json.dumps(record) + "\n")
                    total += 1

    print(f"Wrote {total} pairs to {OUT_FILE}")
    return total

if __name__ == "__main__":
    n = main()
    if n == 0:
        sys.exit(1)
