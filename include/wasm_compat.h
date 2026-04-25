/*
 * wasm_compat.h — Compatibility shims for WASM builds.
 * Provides _mm256_fmadd_ps and other FMA intrinsics
 * that Emscripten's AVX2 emulation doesn't include.
 */
#ifndef WASM_COMPAT_H
#define WASM_COMPAT_H

#ifdef __wasm_simd128__

#include <immintrin.h>

/* FMA3 emulation: a*b+c as mul+add */
#ifndef _mm256_fmadd_ps
static __inline__ __m256 __attribute__((__always_inline__))
_mm256_fmadd_ps(__m256 a, __m256 b, __m256 c) {
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
}
#endif

#ifndef _mm256_fnmadd_ps
static __inline__ __m256 __attribute__((__always_inline__))
_mm256_fnmadd_ps(__m256 a, __m256 b, __m256 c) {
    return _mm256_sub_ps(c, _mm256_mul_ps(a, b));
}
#endif

#ifndef _mm256_fmsub_ps
static __inline__ __m256 __attribute__((__always_inline__))
_mm256_fmsub_ps(__m256 a, __m256 b, __m256 c) {
    return _mm256_sub_ps(_mm256_mul_ps(a, b), c);
}
#endif

/* _mm256_castsi256_ps provided by Emscripten */

#endif /* __wasm_simd128__ */
#endif /* WASM_COMPAT_H */
