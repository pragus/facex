<p align="center">
  <img src="docs/logo.jpg" alt="FaceX" width="480">
</p>

<p align="center"><em>Fast face embedding library. 3 ms inference, 7 MB binary, zero dependencies.</em></p>

<p align="center">

[![Language: C99](https://img.shields.io/badge/language-C99-blue.svg)](https://en.wikipedia.org/wiki/C99)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![Platform: x86-64](https://img.shields.io/badge/platform-x86__64-lightgrey.svg)](#limitations)
[![AVX2 / AVX-512](https://img.shields.io/badge/SIMD-AVX2_%2F_AVX--512-blueviolet.svg)](#architecture)
[![LFW](https://img.shields.io/badge/LFW-99.73%25-success.svg)](#benchmarks)
[![Latency](https://img.shields.io/badge/latency-3.0_ms%2Fface-brightgreen.svg)](#benchmarks)
[![Binary](https://img.shields.io/badge/binary-148_KB-informational.svg)](#footprint)
[![Deps](https://img.shields.io/badge/dependencies-zero-green.svg)](#architecture)

</p>

**A 148 KB face embedding library that runs EdgeFace-XS at 3.0 ms/face
on a consumer i5 -- faster than ONNX Runtime on the same model and CPU.**

No Python. No GPU. No ONNX Runtime. One header, one static library.

```c
#include "facex.h"

FaceX* fx = facex_init("edgeface_xs_fp32.bin", NULL);
float emb[512];
facex_embed(fx, rgb_112x112, emb);
```

---

## Why this exists

Face embedding is the core of every face recognition system -- turning
a face photo into a 512-dimensional vector. The standard way to run it
on CPU is ONNX Runtime (28 MB DLL + Python). That's fine for servers,
but too heavy for edge devices, kiosks, embedded systems, and anyone
who values simplicity.

FaceX is a single C99 file with handwritten AVX2/AVX-512 SIMD kernels
that **beats ONNX Runtime by 23%** with zero dependencies. Six months
of optimization: per-op profiling, custom GEMM microkernels, INT8
quantization, cache-tuned memory layout, thread pool -- every
millisecond fought for.

### Target users

- **Edge AI / embedded** -- kiosks, turnstiles, access control where
  you have a $40 CPU and no GPU.
- **Server-side at scale** -- 3 ms/face means 300+ faces/sec on a
  single core.
- **Privacy-first deployments** -- runs fully offline, no cloud,
  no telemetry.
- **Developers** -- `#include "facex.h"`, link, done. No package
  managers, no version conflicts.

---

## Benchmarks

Measured on Intel i5-11500 (6 cores, AVX-512 + VNNI):

### Speed

![Speed comparison](docs/speed_comparison.svg)

| Engine | Median | Min | vs FaceX |
|--------|-------:|----:|:--------:|
| **FaceX** | **3.0 ms** | **2.87 ms** | -- |
| ONNX Runtime 1.23 | 3.9 ms | 3.18 ms | 1.30x slower |
| InsightFace (R34) | 17 ms | -- | 5.7x slower |
| FaceNet (PyTorch) | 30 ms | -- | 10x slower |
| dlib | 50+ ms | -- | 17x slower |

### Accuracy

| Benchmark | Score |
|-----------|------:|
| **LFW verification** | **99.73%** |
| Model parameters | 1.77M |
| Embedding dim | 512 |

### Footprint

![Footprint comparison](docs/footprint.svg)

| Metric | FaceX | ONNX Runtime |
|--------|------:|-------------:|
| Library size | **148 KB** | 28 MB |
| Total deploy | **7 MB** | 157 MB |
| Dependencies | **none** | Python + onnxruntime |
| Cold start | **~100 ms** | ~350 ms |

---

## Quick start

### C

```c
#include "facex.h"

int main() {
    // Load engine (one-time, ~100ms)
    FaceX* fx = facex_init("edgeface_xs_fp32.bin", NULL);

    // Compute embedding (3ms per call)
    float face[112 * 112 * 3];  // RGB, HWC, [-1, 1]
    float embedding[512];
    facex_embed(fx, face, embedding);

    // Compare two faces
    float sim = facex_similarity(emb_a, emb_b);
    // sim > 0.3 → same person

    facex_free(fx);
}
```

```bash
gcc -O3 -march=native -Iinclude -o myapp myapp.c -L. -lfacex -lm -lpthread
```

### Go

```go
import "github.com/facex-engine/facex/go/facex"

ff, _ := facex.New(facex.Config{
    Exe:     "./facex-cli",
    Weights: "./edgeface_xs_fp32.bin",
})
defer ff.Close()

embedding, _ := ff.Embed(rgbImage)
sim := facex.CosSim(embA, embB)
```

### CLI (any language via stdin/stdout)

```bash
# Pipe mode: reads 112x112x3 float32 HWC, writes 512 float32
./facex-cli weights.bin --server < faces.raw > embeddings.raw
```

---

## Build

```bash
make            # builds libfacex.a + facex-cli
make example    # builds and runs example
make encrypt    # builds weight encryption tool
```

Requirements: GCC with AVX2 support. Nothing else.

### Cross-compile for Linux (from WSL)

```bash
gcc -O3 -march=x86-64-v3 -mavx2 -mfma -static \
    -DFACEX_LIB -o libfacex.a src/*.c -lm -lpthread
```

---

## API

```c
// Initialize engine. Returns NULL on error.
// license_key: NULL for plain weights, or key string for AES-256 encrypted.
FaceX* facex_init(const char* weights_path, const char* license_key);

// Compute 512-dim face embedding from 112x112 RGB image.
// rgb_hwc: float32 array [112][112][3], values in [-1, 1].
// embedding: output buffer, 512 floats (L2-normalized).
int facex_embed(FaceX* fx, const float* rgb_hwc, float embedding[512]);

// Cosine similarity between two embeddings. Range [-1, 1].
float facex_similarity(const float emb1[512], const float emb2[512]);

// Free engine resources.
void facex_free(FaceX* fx);

// Version string.
const char* facex_version(void);
```

---

## Architecture

```
Input: 112x112 RGB float32
    ↓
  Stem: Conv 3→32, stride 4
    ↓
  Stage 0: 3× ConvNeXt blocks (C=32)
    ↓
  Stage 1: 2× ConvNeXt + XCA attention (C=64)
    ↓
  Stage 2: 8× ConvNeXt + XCA attention (C=100)
    ↓
  Stage 3: 2× ConvNeXt + XCA attention (C=192)
    ↓
  Global Average Pool → LayerNorm → FC → L2 Norm
    ↓
Output: 512-dim embedding
```

**Engine internals:**

- Pure C99 + SIMD intrinsics (AVX2, FMA, AVX-512, VNNI)
- INT8 quantized GEMM with `vpmaddubsw` (AVX2) / `vpdpbusd` (VNNI)
- FP32 packed column-panel MatMul (NR=8 AVX2, NR=16 AVX-512)
- Custom thread pool with work-stealing (WaitOnAddress / futex)
- Exact GELU via polynomial `erf` approximation (A&S 7.1.26)
- Pre-packed weights at load time for cache-optimal access
- Optional AES-256-CTR weight encryption with hardware binding

---

## Weight encryption

For commercial deployment with IP protection:

```bash
# Encrypt weights (binds to target machine hardware)
./facex-encrypt encrypt weights.bin weights.enc "LICENSE-KEY"

# Load encrypted weights
FaceX* fx = facex_init("weights.enc", "LICENSE-KEY");
```

Wrong key or different machine → load fails. Original weights never
touch disk in plaintext on the target machine.

---

## Integration paths

| Language | Method | Latency overhead |
|----------|--------|:----------------:|
| **C / C++** | `libfacex.a` + `facex.h` | 0 (native) |
| **Go** | `go/facex` subprocess | ~1 ms IPC |
| **Python** | subprocess / ctypes | ~1 ms IPC |
| **Any** | `facex-cli --server` stdin/stdout | ~1 ms IPC |
| **HTTP** | wrap in your server | your choice |

---

## Limitations

- **x86-64 only.** AVX2 required, AVX-512 optional. ARM NEON port
  planned for Q3 2026.
- **Embedding only.** Face detection and alignment are separate steps.
- **Single model.** EdgeFace-XS (1.77M params). Other models need
  weight conversion.

---

## Model

Uses [EdgeFace-XS](https://arxiv.org/abs/2307.01838) by George et al.:

- 1.77M parameters (smallest in its accuracy class)
- 99.73% LFW, competitive with models 100x larger
- Originally CC BY-NC-SA 4.0 license

---

## Repo layout

```
include/
  facex.h               — public API (5 functions)
  weight_crypto.h       — encryption API
src/
  facex.c               — API implementation
  edgeface_engine.c     — forward pass (all stages + ops)
  transformer_ops.c     — SIMD kernels (LN, GELU, MatMul, Conv)
  gemm_int8_4x8c8.c    — INT8 GEMM microkernel (AVX2 + VNNI)
  threadpool.c/h        — lock-free thread pool
  weight_crypto.c       — AES-256-CTR encryption
go/facex/               — Go binding (subprocess protocol)
examples/
  example.c             — minimal usage example
docs/                   — SVG benchmarks, logo
```

---

## FAQ

**Q: Is it really faster than ONNX Runtime?**
A: Yes. Measured on the same CPU, same model, same input. FaceX median
3.0 ms vs ONNX Runtime median 3.9 ms. The gap comes from handwritten
SIMD kernels that avoid framework overhead.

**Q: What accuracy vs ArcFace-R100?**
A: EdgeFace-XS gets 99.73% LFW vs ArcFace-R100's 99.80%. The 0.07%
gap buys you 10x speed and 60x smaller model.

**Q: Can I use this commercially?**
A: The engine code is Apache 2.0 -- fully commercial. The bundled model
weights are CC BY-NC-SA 4.0 (non-commercial). For commercial use, train
your own weights or contact for licensing.

**Q: Does it do face detection?**
A: No. FaceX is the embedding step only. Pair it with any face detector
(RetinaFace, SCRFD, YuNet, etc.) for a complete pipeline.

---

## Citation

```bibtex
@software{facex2026,
  author  = {Atinov, Baurzhan},
  title   = {FaceX: Fast CPU Face Embedding Library},
  year    = {2026},
  url     = {https://github.com/facex-engine/facex}
}
```

---

## License

Code: [Apache License 2.0](LICENSE) -- free for commercial use.
Model weights: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
(follows upstream EdgeFace license). Train your own weights for
unrestricted commercial use.

For commercial licensing: [bauratynov@gmail.com](mailto:bauratynov@gmail.com)

---

<p align="center">
  Created by <strong>Baurzhan Atinov</strong> (Kazakhstan)<br>
  <a href="https://github.com/bauratynov">GitHub</a>
</p>
