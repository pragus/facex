// FaceX WASM benchmark — runs in Node.js, no browser needed
const fs = require('fs');
const path = require('path');

async function main() {
  // Load the WASM module
  const FaceXModule = require('./facex.js');
  const Module = await FaceXModule();

  // Write weights to virtual filesystem
  const weightsPath = path.join(__dirname, 'edgeface_xs_fp32.bin');
  if (!fs.existsSync(weightsPath)) {
    console.error('ERROR: edgeface_xs_fp32.bin not found in wasm/ folder');
    process.exit(1);
  }
  const weightsData = fs.readFileSync(weightsPath);
  Module.FS.writeFile('/weights.bin', weightsData);

  // Init engine
  const facex_init = Module.cwrap('facex_init', 'number', ['string', 'string']);
  const facex_embed = Module.cwrap('facex_embed', 'number', ['number', 'number', 'number']);
  const facex_similarity = Module.cwrap('facex_similarity', 'number', ['number', 'number']);
  const facex_free = Module.cwrap('facex_free', null, ['number']);

  console.log('Loading weights...');
  const t0 = performance.now();
  const fx = facex_init('/weights.bin', null);
  const loadTime = performance.now() - t0;
  if (!fx) {
    console.error('ERROR: facex_init failed');
    process.exit(1);
  }
  console.log(`Engine loaded in ${loadTime.toFixed(0)}ms`);

  // Allocate input/output buffers
  const nPixels = 112 * 112 * 3;
  const inputPtr = Module._malloc(nPixels * 4);
  const embPtr = Module._malloc(512 * 4);
  const embPtr2 = Module._malloc(512 * 4);

  // Fill with dummy face data
  for (let i = 0; i < nPixels; i++) {
    Module.setValue(inputPtr + i * 4, (i % 256) / 128.0 - 1.0, 'float');
  }

  // Warmup
  console.log('Warming up...');
  for (let i = 0; i < 5; i++) {
    facex_embed(fx, inputPtr, embPtr);
  }

  // Benchmark
  const N = 50;
  console.log(`Benchmarking ${N} inferences...`);
  const times = [];
  for (let i = 0; i < N; i++) {
    // Vary input slightly each iteration
    Module.setValue(inputPtr + (i % 100) * 4, Math.random() * 2 - 1, 'float');
    const t1 = performance.now();
    facex_embed(fx, inputPtr, embPtr);
    const t2 = performance.now();
    times.push(t2 - t1);
  }

  times.sort((a, b) => a - b);
  const min = times[0];
  const median = times[Math.floor(N / 2)];
  const mean = times.reduce((a, b) => a + b, 0) / N;
  const p95 = times[Math.floor(N * 0.95)];
  const max = times[N - 1];

  console.log('\n=== FaceX WASM Benchmark ===');
  console.log(`  min:    ${min.toFixed(2)} ms`);
  console.log(`  median: ${median.toFixed(2)} ms`);
  console.log(`  mean:   ${mean.toFixed(2)} ms`);
  console.log(`  p95:    ${p95.toFixed(2)} ms`);
  console.log(`  max:    ${max.toFixed(2)} ms`);
  console.log(`  fps:    ${(1000 / median).toFixed(0)}`);

  // Verify output
  const emb = [];
  for (let i = 0; i < 5; i++) {
    emb.push(Module.getValue(embPtr + i * 4, 'float'));
  }
  console.log(`\nEmbedding[0..4]: ${emb.map(v => v.toFixed(4)).join(', ')}`);

  // Test similarity (same input = should be ~1.0)
  facex_embed(fx, inputPtr, embPtr2);
  const sim = facex_similarity(embPtr, embPtr2);
  console.log(`Self-similarity: ${sim.toFixed(6)} (should be ~1.0)`);

  // Cleanup
  Module._free(inputPtr);
  Module._free(embPtr);
  Module._free(embPtr2);
  facex_free(fx);

  console.log('\nDone.');
}

main().catch(e => { console.error(e); process.exit(1); });
