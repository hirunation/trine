# Security

If you discover a vulnerability in TRINE, report it privately to [AiroNahiru@pm.me](mailto:AiroNahiru@pm.me) with a description of the issue, steps to reproduce, and your assessment of severity. Do not open a public issue for security vulnerabilities. You will receive acknowledgment within 72 hours.

Once a fix is developed and tested, the vulnerability will be disclosed transparently in the CHANGELOG with credit to the reporter unless anonymity is requested. Fixes ship as patch releases. There is no embargo beyond the time required to produce a correct fix.

TRINE's zero-dependency, deterministic architecture exists in part to minimize attack surface. But inspectability demands operational rigor — the fact that anyone can read the code means vulnerabilities are found faster, and that advantage only holds if the response is equally fast.

## v1.0.3 Safety Fixes

### Block-diagonal accumulator sign inversion (trine_accumulator.c)
`trine_block_accumulator_update()` and `trine_block_accumulator_update_weighted()` used `positive ? 1 : -1` to normalize the sign parameter. When the caller passed `-1` (dissimilar pair), C evaluates `-1` as truthy, producing `+1` instead of `-1`. All block-diagonal Hebbian training accumulated the wrong sign for negative pairs. Fixed to `(sign > 0) ? 1 : -1`, matching the full-matrix path.

### Buffer overflow: canon SUPPORT/POLICY presets (trine_canon.c)
`trine_canon_bucket_numbers()` replaces digit runs with `<N>` (3 bytes). A single-digit input expands from 1 to 3 bytes. The SUPPORT and POLICY presets allocated only `len+1` bytes for the working buffer, insufficient for worst-case input (e.g., alternating digits and non-digits). Fixed to allocate `3*len+1` for all presets that invoke bucket_numbers.

### Thread-safe RNG (trine_hebbian.c)
Removed static LCG state shared across threads. The previous implementation used a file-scope static variable for pseudo-random number generation during Hebbian weight updates. Concurrent training calls produced a data race on that state, yielding nondeterministic results and potential torn reads. RNG state is now caller-owned.

### Buffer overflow: encode depth bounds check
`encode_depths` now rejects depth values greater than 64. Previously, an unchecked depth parameter could write past the end of stack-allocated buffers during multi-depth encoding. Callers passing untrusted depth values (e.g., from CLI args or model files) were exposed.

### Buffer overflow: canon CODE preset
The CODE canonicalization preset now allocates a 3x output buffer and uses bounds-checked normalization. The prior fixed-size buffer was insufficient for worst-case expansion during code-specific canonicalization transforms, allowing a heap overflow on adversarial input.

### Trit validation on .trine2 model load
Model deserialization now validates that every weight byte is in {0, 1, 2} before accepting a .trine2 file. A corrupted or maliciously crafted model file previously loaded without validation, producing undefined comparison results and potential out-of-range array indexing in the projection path.

### Realloc atomicity in index growth
Index resize operations now allocate into temporary pointers. The previous pattern called `realloc` directly on the live pointer; if the second of two coupled reallocs failed, the index was left in an inconsistent state (one array resized, the other not). Neither pointer is now committed until both allocations succeed.

### OOM propagation from trine_encode_shingle
`trine_encode_shingle` now returns `int` (0 on success, -1 on allocation failure). The previous `void` return silently continued with NULL buffers after OOM, leading to null-pointer dereferences downstream. All callers have been updated to check the return value.

### JSON escaped quote handling (field parser)
The multi-field JSON parser now correctly handles escaped quotes (`\"`) inside string values. Previously, an escaped quote terminated the value early, causing field misalignment and incorrect indexing of subsequent fields in the same record.

### JSONL dynamic line reading
The JSONL ingest path now uses `getline` instead of a fixed `fgets` buffer. Lines exceeding the former fixed buffer size were silently truncated, producing corrupted records. `getline` dynamically allocates to fit the actual line length.
