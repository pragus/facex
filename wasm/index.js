/**
 * facex-wasm — Face recognition in the browser.
 *
 * This is the npm package entry point. It re-exports the classes and
 * utilities defined in the individual script files.
 *
 * Usage (bundler / ESM):
 *   import { FaceXSDK } from 'facex-wasm';
 *   const fx = new FaceXSDK();
 *   await fx.load();
 *   const result = fx.process(videoElement);
 *
 * Usage (script tag):
 *   <script src="node_modules/facex-wasm/detect.js"></script>
 *   <script src="node_modules/facex-wasm/facex.js"></script>
 *   <script src="node_modules/facex-wasm/align.js"></script>
 *   <script src="node_modules/facex-wasm/facex-sdk.js"></script>
 *   // FaceXSDK is now available as a global
 */

// The SDK files define globals (FaceXSDK, LivenessDetector, etc.) when
// loaded via <script> tags.  For bundler consumers we provide this entry
// point that pulls them together.  Because the source files are plain
// browser scripts (no module syntax), we need to reference the globals.

if (typeof globalThis !== 'undefined') {
  // Make sure the exports object exists for CJS consumers
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      get FaceXSDK() { return globalThis.FaceXSDK; },
      get LivenessDetector() { return globalThis.LivenessDetector; },
      get checkWasmSimd() { return globalThis.checkWasmSimd; },
      get checkBrowserSupport() { return globalThis.checkBrowserSupport; },
      get alignFace() { return globalThis.alignFace; },
      get loadWeightsCached() { return globalThis.loadWeightsCached; },
    };
  }
}
