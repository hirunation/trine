# ── TrineDB Docker Image ─────────────────────────────────────────────
# Multi-stage build: Alpine builder + minimal runtime.
# Build context must be the hteb/ directory (parent of docker/).
#
# Build:
#   docker build -t trinedb -f docker/Dockerfile .
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

# Copy library sources and headers needed for the build.
# Listing files explicitly keeps the build context minimal and
# ensures Docker layer caching works when only unrelated files change.
COPY trine_encode.h trine_encode.c \
     trine_stage1.h trine_stage1.c \
     trine_route.h trine_route.c \
     trine_canon.h trine_canon.c \
     trine_idf.h \
     Makefile ./

# Copy TrineDB server source and its Makefile.
COPY trinedb/trinedb.c trinedb/Makefile trinedb/

# Build the static library (build/libtrine.a).
RUN make libtrine.a

# Build the TrineDB server binary.
# The trinedb Makefile compiles trinedb.c together with the library
# sources (not linking against the .a), producing a single static binary.
RUN cd trinedb && make

# ── Stage 2: Runtime ────────────────────────────────────────────────
# Minimal image containing only the server binary.
# musl-linked binaries on Alpine need no extra shared libraries beyond
# what the base image provides; libgcc is included for safety.
FROM alpine:3.19

RUN apk add --no-cache libgcc

# Copy the compiled server binary from the builder stage.
COPY --from=builder /src/trinedb/trinedb /usr/local/bin/trinedb

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
