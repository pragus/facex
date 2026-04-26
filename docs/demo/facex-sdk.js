/*
 * align.js — 5-point face alignment for FaceX.
 *
 * Takes 5 keypoints from detector (left_eye, right_eye, nose, left_mouth, right_mouth)
 * and produces a 112x112 aligned face using affine transformation.
 *
 * Reference template: ArcFace standard 112x112 alignment targets.
 */

// ArcFace standard reference points for 112x112 output
const REF_POINTS = [
  [38.2946, 51.6963],  // left eye
  [73.5318, 51.5014],  // right eye
  [56.0252, 71.7366],  // nose
  [41.5493, 92.3655],  // left mouth
  [70.7299, 92.2041],  // right mouth
];

/**
 * Compute similarity transform matrix from src points to dst points.
 * Uses least-squares fit for 2D similarity (rotation + scale + translation).
 * Returns [a, b, tx, ty] where:
 *   x' = a*x - b*y + tx
 *   y' = b*x + a*y + ty
 */
function getSimilarityTransform(src, dst) {
  const n = src.length;
  let sx = 0, sy = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) {
    sx += src[i][0]; sy += src[i][1];
    dx += dst[i][0]; dy += dst[i][1];
  }
  sx /= n; sy /= n; dx /= n; dy /= n;

  let num1 = 0, num2 = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const sxc = src[i][0] - sx, syc = src[i][1] - sy;
    const dxc = dst[i][0] - dx, dyc = dst[i][1] - dy;
    num1 += dxc * sxc + dyc * syc;
    num2 += dxc * syc - dyc * sxc;
    den += sxc * sxc + syc * syc;
  }

  const a = num1 / den;
  const b = num2 / den;
  const tx = dx - a * sx + b * sy;
  const ty = dy - b * sx - a * sy;

  return [a, b, tx, ty];
}

/**
 * Apply similarity transform to warp source image to 112x112 aligned face.
 *
 * @param {CanvasRenderingContext2D} srcCtx - source canvas context (video frame)
 * @param {number} srcW - source width
 * @param {number} srcH - source height
 * @param {Array<Array<number>>} kps - 5 keypoints [[x,y], ...] in source coords
 * @param {CanvasRenderingContext2D} dstCtx - destination 112x112 canvas context
 * @returns {ImageData} - 112x112 aligned face
 */
function alignFace(srcCtx, srcW, srcH, kps, dstCtx) {
  // Get transform: destination (112x112 ref) → source (video)
  // We need inverse: for each dst pixel, find src pixel
  const [a, b, tx, ty] = getSimilarityTransform(kps, REF_POINTS);

  // Inverse transform: src = inverse(M) * dst
  // M = [a, -b, tx; b, a, ty]
  // M_inv = (1/det) * [a, b, -a*tx-b*ty; -b, a, b*tx-a*ty]
  const det = a * a + b * b;
  const ai = a / det, bi = b / det;
  const txi = -(ai * tx + bi * ty);
  const tyi = (bi * tx - ai * ty);

  const srcData = srcCtx.getImageData(0, 0, srcW, srcH);
  const dstData = dstCtx.createImageData(112, 112);
  const src = srcData.data;
  const dst = dstData.data;

  for (let dy = 0; dy < 112; dy++) {
    for (let dx = 0; dx < 112; dx++) {
      // Map dst → src using inverse transform
      const sx = ai * dx - bi * dy + txi;
      const sy = bi * dx + ai * dy + tyi;

      // Bilinear interpolation
      const x0 = Math.floor(sx), y0 = Math.floor(sy);
      const x1 = x0 + 1, y1 = y0 + 1;
      const fx = sx - x0, fy = sy - y0;

      if (x0 >= 0 && x1 < srcW && y0 >= 0 && y1 < srcH) {
        const i00 = (y0 * srcW + x0) * 4;
        const i10 = (y0 * srcW + x1) * 4;
        const i01 = (y1 * srcW + x0) * 4;
        const i11 = (y1 * srcW + x1) * 4;
        const di = (dy * 112 + dx) * 4;

        for (let c = 0; c < 3; c++) {
          dst[di + c] = Math.round(
            src[i00 + c] * (1-fx) * (1-fy) +
            src[i10 + c] * fx * (1-fy) +
            src[i01 + c] * (1-fx) * fy +
            src[i11 + c] * fx * fy
          );
        }
        dst[di + 3] = 255;
      }
    }
  }

  dstCtx.putImageData(dstData, 0, 0);
  return dstData;
}
/*
 * liveness.js — Basic liveness detection using face keypoints.
 *
 * Detects:
 * 1. Face motion (frame-to-frame bbox movement)
 * 2. Blink detection (eye aspect ratio changes)
 * 3. Multiple frame consistency (not a static photo)
 *
 * Not a security-grade anti-spoofing solution — detects basic
 * photo/screen attacks but not 3D masks or deepfakes.
 */

class LivenessDetector {
  constructor(options = {}) {
    this.historySize = options.historySize || 30; // frames to track
    this.motionThreshold = options.motionThreshold || 2.0; // pixels
    this.blinkThreshold = options.blinkThreshold || 0.22; // EAR ratio

    this._history = [];
    this._blinkCount = 0;
    this._lastEAR = 1.0;
    this._wasEyeClosed = false;
  }

  /**
   * Update with new face detection.
   * @param {Object} face - {x1,y1,x2,y2,kps:[10 values]}
   * @returns {{alive: boolean, confidence: number, reason: string}}
   */
  update(face) {
    if (!face) {
      this._history = [];
      return { alive: false, confidence: 0, reason: 'No face' };
    }

    const entry = {
      cx: (face.x1 + face.x2) / 2,
      cy: (face.y1 + face.y2) / 2,
      w: face.x2 - face.x1,
      h: face.y2 - face.y1,
      ear: this._computeEAR(face.kps),
      t: Date.now()
    };

    this._history.push(entry);
    if (this._history.length > this.historySize)
      this._history.shift();

    // Need at least 10 frames
    if (this._history.length < 10) {
      return { alive: false, confidence: 0.1, reason: 'Collecting frames...' };
    }

    // Check 1: Motion — face must move slightly (not a static photo)
    const motion = this._computeMotion();
    const hasMotion = motion > this.motionThreshold;

    // Check 2: Blink detection
    this._detectBlink(entry.ear);
    const hasBlinked = this._blinkCount > 0;

    // Check 3: Size variation (breathing, micro-movements)
    const sizeVar = this._computeSizeVariation();
    const hasSizeChange = sizeVar > 0.005;

    // Compute confidence
    let confidence = 0;
    if (hasMotion) confidence += 0.35;
    if (hasBlinked) confidence += 0.40;
    if (hasSizeChange) confidence += 0.25;

    let reason;
    if (confidence >= 0.6) reason = 'Live person detected';
    else if (!hasMotion) reason = 'No motion detected — hold still and blink';
    else if (!hasBlinked) reason = 'Please blink';
    else reason = 'Analyzing...';

    return {
      alive: confidence >= 0.6,
      confidence,
      reason,
      details: { motion, blinks: this._blinkCount, sizeVar }
    };
  }

  /** Reset state */
  reset() {
    this._history = [];
    this._blinkCount = 0;
    this._lastEAR = 1.0;
    this._wasEyeClosed = false;
  }

  // ============ Internal ============

  /** Eye Aspect Ratio from 5 keypoints.
   * kps: [lex, ley, rex, rey, nx, ny, lmx, lmy, rmx, rmy]
   * EAR ≈ distance(mouth) / distance(eyes) as a proxy.
   * Real EAR needs 6 eye landmarks — we approximate from 5 points. */
  _computeEAR(kps) {
    // left eye (0,1), right eye (2,3), nose (4,5), left mouth (6,7), right mouth (8,9)
    const eyeDist = Math.sqrt((kps[2]-kps[0])**2 + (kps[3]-kps[1])**2);
    const mouthDist = Math.sqrt((kps[8]-kps[6])**2 + (kps[9]-kps[7])**2);
    const noseToEyeL = Math.sqrt((kps[4]-kps[0])**2 + (kps[5]-kps[1])**2);
    const noseToEyeR = Math.sqrt((kps[4]-kps[2])**2 + (kps[5]-kps[3])**2);

    // Use ratio of vertical to horizontal distances as EAR proxy
    if (eyeDist < 1) return 1.0;
    return (noseToEyeL + noseToEyeR) / (2 * eyeDist);
  }

  _detectBlink(ear) {
    if (ear < this.blinkThreshold && !this._wasEyeClosed) {
      this._wasEyeClosed = true;
    } else if (ear > this.blinkThreshold + 0.05 && this._wasEyeClosed) {
      this._wasEyeClosed = false;
      this._blinkCount++;
    }
    this._lastEAR = ear;
  }

  _computeMotion() {
    if (this._history.length < 2) return 0;
    let totalMotion = 0;
    for (let i = 1; i < this._history.length; i++) {
      const dx = this._history[i].cx - this._history[i-1].cx;
      const dy = this._history[i].cy - this._history[i-1].cy;
      totalMotion += Math.sqrt(dx*dx + dy*dy);
    }
    return totalMotion / (this._history.length - 1);
  }

  _computeSizeVariation() {
    if (this._history.length < 5) return 0;
    const sizes = this._history.map(h => h.w * h.h);
    const mean = sizes.reduce((a,b) => a+b) / sizes.length;
    const variance = sizes.reduce((a,s) => a + (s-mean)**2, 0) / sizes.length;
    return Math.sqrt(variance) / mean; // coefficient of variation
  }
}
/*
 * facex-sdk.js — Unified FaceX browser SDK.
 *
 * Single entry point for face detection + alignment + embedding.
 *
 * Usage:
 *   const fx = new FaceXSDK();
 *   await fx.load();
 *   const result = await fx.verify(videoElement, referenceEmbedding);
 *   // { match: true, similarity: 0.87, faces: [...], ms: 17 }
 */

class FaceXSDK {
  constructor(options = {}) {
    this.detSize = options.detSize || 160;
    this.threshold = options.threshold || 0.3;
    this.detWeightsUrl = options.detWeightsUrl || 'det_500m_int8.bin';
    this.embWeightsUrl = options.embWeightsUrl || 'edgeface_xs_fp32.bin';
    this.onProgress = options.onProgress || null;

    this._det = null;
    this._fx = null;
    this._detHandle = 0;
    this._fxHandle = 0;
    this._ready = false;

    // Reusable canvases
    this._detCanvas = document.createElement('canvas');
    this._detCtx = this._detCanvas.getContext('2d', { willReadFrequently: true });
    this._cropCanvas = document.createElement('canvas');
    this._cropCanvas.width = 112;
    this._cropCanvas.height = 112;
    this._cropCtx = this._cropCanvas.getContext('2d', { willReadFrequently: true });
    this._srcCanvas = document.createElement('canvas');
    this._srcCtx = this._srcCanvas.getContext('2d', { willReadFrequently: true });
  }

  /** Load engines and weights. Call once. */
  async load() {
    // Check SIMD support
    if (!this._checkSimd()) {
      throw new Error('WebAssembly SIMD not supported. Chrome 91+, Firefox 89+, Safari 16.4+ required.');
    }

    this._progress('Loading detection engine...');
    this._det = await FaceDetModule();

    this._progress('Loading embedding engine...');
    this._fx = await FaceXModule();

    // Load weights with caching
    this._progress('Loading detector weights (670 KB)...');
    const detW = await this._loadCached(this.detWeightsUrl, 'facex_det_weights');
    this._det.FS.writeFile('/det.bin', detW);

    this._progress('Loading embedding weights (7 MB)...');
    const fxW = await this._loadCached(this.embWeightsUrl, 'facex_emb_weights');
    this._fx.FS.writeFile('/emb.bin', fxW);

    // Init engines
    const di = this._det.cwrap('detect_init', 'number', ['string']);
    this._detHandle = di('/det.bin');
    if (!this._detHandle) throw new Error('Face detector initialization failed');

    const fi = this._fx.cwrap('facex_init', 'number', ['string', 'string']);
    this._fxHandle = fi('/emb.bin', null);
    if (!this._fxHandle) throw new Error('Face embedder initialization failed');

    this._ready = true;
    this._progress('Ready');
    return this;
  }

  /** Check if SDK is loaded and ready */
  get ready() { return this._ready; }

  /**
   * Detect faces in a video/image/canvas element.
   * @returns {Array<{x1,y1,x2,y2,score,kps}>}
   */
  detect(source) {
    if (!this._ready) throw new Error('SDK not loaded. Call load() first.');
    const { pixels, W, H } = this._prepareDetInput(source);
    return this._runDetect(pixels, W, H);
  }

  /**
   * Compute 512-dim embedding from aligned 112x112 face.
   * @param {ImageData} faceImageData - 112x112 RGBA image
   * @returns {Float32Array} 512-dim embedding
   */
  embed(faceImageData) {
    if (!this._ready) throw new Error('SDK not loaded');
    return this._runEmbed(faceImageData);
  }

  /**
   * Full pipeline: detect → align → embed from video/image source.
   * @returns {{faces, embeddings, ms}}
   */
  process(source) {
    if (!this._ready) throw new Error('SDK not loaded');
    const t0 = performance.now();

    const { pixels, W, H, scale, dx, dy } = this._prepareDetInput(source);
    const rawFaces = this._runDetect(pixels, W, H);

    // Map faces back to source coordinates
    const faces = rawFaces.map(f => ({
      x1: (f.x1 - dx) / scale,
      y1: (f.y1 - dy) / scale,
      x2: (f.x2 - dx) / scale,
      y2: (f.y2 - dy) / scale,
      score: f.score,
      kps: f.kps.map((v, i) => (v - (i % 2 === 0 ? dx : dy)) / scale)
    }));

    // Align and embed each face
    const sourceW = this._srcCanvas.width;
    const sourceH = this._srcCanvas.height;
    const embeddings = faces.map(face => {
      const kps5 = [];
      for (let i = 0; i < 5; i++)
        kps5.push([face.kps[i * 2], face.kps[i * 2 + 1]]);

      const aligned = alignFace(this._srcCtx, sourceW, sourceH, kps5, this._cropCtx);
      return this._runEmbed(aligned);
    });

    return {
      faces,
      embeddings,
      ms: performance.now() - t0
    };
  }

  /**
   * Verify: compare live source against a reference embedding.
   * @returns {{match, similarity, faces, ms}}
   */
  verify(source, refEmbedding) {
    const result = this.process(source);
    if (result.embeddings.length === 0) {
      return { match: false, similarity: 0, faces: result.faces, ms: result.ms, noFace: true };
    }
    const sim = this.cosSim(result.embeddings[0], refEmbedding);
    return {
      match: sim > this.threshold,
      similarity: sim,
      faces: result.faces,
      embedding: result.embeddings[0],
      ms: result.ms
    };
  }

  /** Cosine similarity between two embeddings */
  cosSim(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < 512; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
  }

  /** Capture reference embedding from current video frame */
  captureReference(source) {
    const result = this.process(source);
    if (result.embeddings.length === 0) return null;
    return {
      embedding: result.embeddings[0],
      face: result.faces[0],
      alignedCanvas: this._cropCanvas
    };
  }

  // ============ Internal methods ============

  _progress(msg) {
    if (this.onProgress) this.onProgress(msg);
  }

  _checkSimd() {
    try {
      return WebAssembly.validate(new Uint8Array([
        0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,10,1,8,0,
        253,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11
      ]));
    } catch(e) { return false; }
  }

  async _loadCached(url, key) {
    // Try IndexedDB cache
    try {
      const db = await new Promise((res, rej) => {
        const r = indexedDB.open('facex-cache', 1);
        r.onupgradeneeded = () => r.result.createObjectStore('weights');
        r.onsuccess = () => res(r.result);
        r.onerror = () => rej(r.error);
      });
      const cached = await new Promise(res => {
        const tx = db.transaction('weights', 'readonly');
        const req = tx.objectStore('weights').get(key);
        req.onsuccess = () => res(req.result);
        req.onerror = () => res(null);
      });
      if (cached) return new Uint8Array(cached);

      // Fetch and cache
      const buf = await fetch(url).then(r => r.arrayBuffer());
      const tx = db.transaction('weights', 'readwrite');
      tx.objectStore('weights').put(buf, key);
      return new Uint8Array(buf);
    } catch(e) {
      // Fallback: no cache
      return new Uint8Array(await fetch(url).then(r => r.arrayBuffer()));
    }
  }

  _prepareDetInput(source) {
    // Draw source to srcCanvas at full res
    let w, h;
    if (source instanceof HTMLVideoElement) {
      w = source.videoWidth; h = source.videoHeight;
    } else if (source instanceof HTMLCanvasElement) {
      w = source.width; h = source.height;
    } else if (source instanceof HTMLImageElement) {
      w = source.naturalWidth; h = source.naturalHeight;
    } else {
      throw new Error('Source must be video, canvas, or image element');
    }

    this._srcCanvas.width = w;
    this._srcCanvas.height = h;
    this._srcCtx.drawImage(source, 0, 0);

    // Letterbox to detSize
    const ds = this.detSize;
    this._detCanvas.width = ds;
    this._detCanvas.height = ds;
    const scale = Math.min(ds / w, ds / h);
    const nw = Math.round(w * scale), nh = Math.round(h * scale);
    const dx = (ds - nw) / 2, dy = (ds - nh) / 2;
    this._detCtx.fillStyle = '#000';
    this._detCtx.fillRect(0, 0, ds, ds);
    this._detCtx.drawImage(source, dx, dy, nw, nh);

    const imgData = this._detCtx.getImageData(0, 0, ds, ds);
    const nPx = ds * ds;
    const pixels = new Float32Array(nPx * 3);
    for (let i = 0; i < nPx; i++) {
      pixels[i * 3]     = imgData.data[i * 4] / 127.5 - 1.0;
      pixels[i * 3 + 1] = imgData.data[i * 4 + 1] / 127.5 - 1.0;
      pixels[i * 3 + 2] = imgData.data[i * 4 + 2] / 127.5 - 1.0;
    }
    return { pixels, W: ds, H: ds, scale, dx, dy };
  }

  _runDetect(pixels, W, H) {
    const nPx = W * H * 3;
    const inPtr = this._det._malloc(nPx * 4);
    this._det.HEAPF32.set(pixels, inPtr >> 2);

    const maxFaces = 10;
    const outPtr = this._det._malloc(15 * 4 * maxFaces);

    const df = this._det.cwrap('detect_faces', 'number',
      ['number', 'number', 'number', 'number', 'number', 'number']);
    const n = df(this._detHandle, inPtr, W, H, outPtr, maxFaces);

    const faces = [];
    for (let i = 0; i < n; i++) {
      const base = (outPtr >> 2) + i * 15;
      const f = this._det.HEAPF32;
      faces.push({
        x1: f[base], y1: f[base+1], x2: f[base+2], y2: f[base+3],
        score: f[base+4],
        kps: Array.from(f.subarray(base + 5, base + 15))
      });
    }

    this._det._free(inPtr);
    this._det._free(outPtr);
    return faces;
  }

  _runEmbed(imageData) {
    const N = 112 * 112 * 3;
    const inPtr = this._fx._malloc(N * 4);
    const outPtr = this._fx._malloc(512 * 4);

    const heap = this._fx.HEAPF32;
    const base = inPtr >> 2;
    const px = imageData.data;
    for (let i = 0; i < 112 * 112; i++) {
      heap[base + i * 3]     = px[i * 4] / 127.5 - 1.0;
      heap[base + i * 3 + 1] = px[i * 4 + 1] / 127.5 - 1.0;
      heap[base + i * 3 + 2] = px[i * 4 + 2] / 127.5 - 1.0;
    }

    const fe = this._fx.cwrap('facex_embed', 'number', ['number', 'number', 'number']);
    fe(this._fxHandle, inPtr, outPtr);

    const emb = new Float32Array(512);
    emb.set(heap.subarray(outPtr >> 2, (outPtr >> 2) + 512));

    this._fx._free(inPtr);
    this._fx._free(outPtr);
    return emb;
  }
}
