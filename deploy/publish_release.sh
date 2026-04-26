#!/bin/bash
# Create GitHub Release with pre-built binaries.
# Requires: gh CLI authenticated.
# Usage: ./deploy/publish_release.sh v2.0.0
set -e

VERSION="${1:-v2.0.0}"
ARCHIVE="deploy/facex-${VERSION#v}-linux-x64.tar.gz"

if [ ! -f "$ARCHIVE" ]; then
    echo "ERROR: $ARCHIVE not found. Run deploy/build_release.sh first."
    exit 1
fi

echo "Creating GitHub Release ${VERSION}..."
gh release create "$VERSION" \
    --title "FaceX ${VERSION}" \
    --notes "$(cat <<EOF
## FaceX ${VERSION}

Face detection + recognition library. Detect → Align → Embed in one call.

### Performance
- **Embedding:** 3.0ms native (CPU), 7.0ms WASM (browser)
- **Detection:** 3.8ms native, 6.5ms WASM
- **Full pipeline:** ~13.5ms in browser = 74 fps
- **Accuracy:** 99.73% LFW

### What's included
- \`libfacex.a\` — static C library (detect + embed)
- \`facex-cli\` — CLI tool (stdin/stdout protocol)
- \`facex-encrypt\` — AES-256 weight encryption
- Headers: \`facex.h\`, \`detect.h\`

### Quick start
\`\`\`bash
tar xzf facex-*-linux-x64.tar.gz
cd facex-*/
# Test
echo "dummy" | ./bin/facex-cli weights.bin
# Link in your project
gcc -Iinclude myapp.c -Llib -lfacex -lm -lpthread
\`\`\`

### Browser (WASM)
\`\`\`
npm install facex-wasm
\`\`\`

### Python
\`\`\`
pip install facex
\`\`\`
EOF
)" \
    "$ARCHIVE"

echo "Done: https://github.com/facex-engine/facex/releases/tag/${VERSION}"
