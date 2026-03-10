# ── TrineDB Docker Image ─────────────────────────────────────────────
# Multi-stage build: Alpine builder + minimal runtime.
# Build context must be the project root directory.
#
# Build:
#   docker build -t trinedb -f deploy/docker/Dockerfile .
#
# Run:
#   docker run -p 7319:7319 -v trine_data:/data trinedb
# ─────────────────────────────────────────────────────────────────────

# ── Stage 1: Build ──────────────────────────────────────────────────
# Compile libtrine.a and the trinedb server binary against musl.
# Everything in this stage is discarded in the final image.
FROM alpine:3.19 AS builder

RUN apk add --no-cache gcc musl-dev make

WORKDIR /src

# Copy source tree structure needed for the build.
# Stage-1 core
COPY src/encode/trine_encode.h src/encode/trine_encode.c \
     src/encode/trine_idf.h src/encode/
COPY src/compare/trine_stage1.h src/compare/trine_stage1.c \
     src/compare/trine_csidf.h src/compare/trine_csidf.c src/compare/
COPY src/index/trine_route.h src/index/trine_route.c \
     src/index/trine_field.h src/index/trine_field.c src/index/
COPY src/canon/trine_canon.h src/canon/trine_canon.c src/canon/
COPY src/algebra/oicos.h src/algebra/trine_algebra.h \
     src/algebra/trine_format.h src/algebra/trine_format.c src/algebra/
COPY src/model/trine.h src/model/trine.c src/model/
COPY src/pack/trine_pack.c src/pack/
# Stage-2 semantic projection
COPY src/stage2/projection/trine_project.h src/stage2/projection/trine_project.c \
     src/stage2/projection/trine_majority.c src/stage2/projection/
COPY src/stage2/cascade/trine_learned_cascade.h src/stage2/cascade/trine_learned_cascade.c \
     src/stage2/cascade/trine_topology_gen.c src/stage2/cascade/
COPY src/stage2/inference/trine_stage2.h src/stage2/inference/trine_stage2.c \
     src/stage2/inference/
COPY src/stage2/hebbian/trine_hebbian.h src/stage2/hebbian/trine_hebbian.c \
     src/stage2/hebbian/trine_accumulator.h src/stage2/hebbian/trine_accumulator.c \
     src/stage2/hebbian/trine_freeze.h src/stage2/hebbian/trine_freeze.c \
     src/stage2/hebbian/trine_self_deepen.c src/stage2/hebbian/
COPY src/stage2/persist/trine_s2_persist.h src/stage2/persist/trine_s2_persist.c \
     src/stage2/persist/
COPY Makefile ./

# Copy TrineDB server source and its Makefile.
COPY deploy/trinedb/trinedb.c deploy/trinedb/Makefile deploy/trinedb/

# Build the static library (build/libtrine.a).
RUN make libtrine.a

# Build the TrineDB server binary.
# The trinedb Makefile compiles trinedb.c together with the library
# sources (not linking against the .a), producing a single static binary.
RUN cd deploy/trinedb && make

# ── Stage 2: Runtime ────────────────────────────────────────────────
# Minimal image containing only the server binary.
# musl-linked binaries on Alpine need no extra shared libraries beyond
# what the base image provides; libgcc is included for safety.
FROM alpine:3.19

RUN apk add --no-cache libgcc

# Copy the compiled server binary from the builder stage.
COPY --from=builder /src/deploy/trinedb/trinedb /usr/local/bin/trinedb

# Create the default data directory for index persistence.
RUN mkdir -p /data

# TrineDB listens on port 7319 by default.
EXPOSE 7319

# Persist index data across container restarts.
VOLUME ["/data"]

# The entrypoint is the trinedb binary itself.
# Default arguments set the port and data directory.
# Override CMD to change port, threshold, recall mode, etc.
#   e.g. docker run trinedb -p 8080 -t 0.70 --recall strict -d /data
ENTRYPOINT ["trinedb"]
CMD ["-p", "7319", "-d", "/data"]
