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
    this.detWeightsUrl = options.detWeightsUrl || 'yunet_fp32.bin';
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
    this._progress('Loading detector weights (208 KB)...');
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
