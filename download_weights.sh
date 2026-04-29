#!/bin/bash
# Download FaceX model weights from GitHub Release
set -e

WEIGHTS_DIR="data"
WEIGHTS_FILE="$WEIGHTS_DIR/edgeface_xs_fp32.bin"
RELEASE_URL="https://github.com/facex-engine/facex/releases/download/v1.0.0/edgeface_xs_fp32.bin"
EXPECTED_SIZE=7133584

if [ -f "$WEIGHTS_FILE" ]; then
    echo "Weights already exist: $WEIGHTS_FILE"
    exit 0
fi

mkdir -p "$WEIGHTS_DIR"

echo "Downloading EdgeFace-XS weights (7 MB)..."
if command -v curl &> /dev/null; then
    curl -L -o "$WEIGHTS_FILE" "$RELEASE_URL"
elif command -v wget &> /dev/null; then
    wget -O "$WEIGHTS_FILE" "$RELEASE_URL"
else
    echo "Error: curl or wget required"
    exit 1
fi

# Verify size
ACTUAL_SIZE=$(stat -c%s "$WEIGHTS_FILE" 2>/dev/null || stat -f%z "$WEIGHTS_FILE" 2>/dev/null)
if [ "$ACTUAL_SIZE" != "$EXPECTED_SIZE" ]; then
    echo "Warning: expected $EXPECTED_SIZE bytes, got $ACTUAL_SIZE"
else
    echo "OK: $WEIGHTS_FILE ($ACTUAL_SIZE bytes)"
fi
