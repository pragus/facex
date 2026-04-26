# FaceX — Face recognition REST API
# Build: docker build -t facex .
# Run:   docker run -p 8080:8080 facex
#
# API:
#   POST /detect  — detect faces + embeddings (multipart image)
#   POST /embed   — embed pre-aligned 112x112 face
#   POST /compare — compare two embeddings
#   GET  /health  — health check

FROM gcc:13 AS builder

WORKDIR /build
COPY src/ src/
COPY include/ include/
COPY Makefile .

# Build static library + CLI
RUN gcc -O3 -march=x86-64-v3 -mavx2 -mfma -funroll-loops -static -DFACEX_LIB \
    -Iinclude -o facex-server \
    src/facex.c src/detect.c src/align.c src/edgeface_engine.c \
    src/transformer_ops.c src/gemm_int8_4x8c8.c src/threadpool.c \
    src/weight_crypto.c \
    -lm -lpthread && \
    strip facex-server

# Runtime — minimal image
FROM alpine:3.19

RUN apk add --no-cache python3 py3-pip && \
    pip3 install --break-system-packages flask gunicorn pillow numpy

WORKDIR /app
COPY --from=builder /build/facex-server /app/
COPY docker/server.py /app/

# Weights must be mounted or baked in
# docker run -v /path/to/weights:/app/weights facex
EXPOSE 8080
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "server:app"]
