/*
 * transformer_ops.c — FP32 ops for EdgeFace-XS Transformer blocks.
 *
 * LayerNorm, GELU, Softmax, L2Normalize, MatMul (FP32).
 * These run in FP32 (not INT8) since Transformer attention
 * involves dynamic×dynamic MatMul which doesn't benefit from INT8.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#ifndef FACEX_ENABLE_GEMM_PROFILE
#define FACEX_ENABLE_GEMM_PROFILE 0
#endif

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/transformer_ops.cc"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();

namespace facex_hwy {
namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

void gelu_dispatch(float* x, int n);

static inline bool prefer_mr8(int M, int K, int N, int lanes) {
    (void)K;
    (void)N;
    (void)lanes;
    return M >= 16;
}

static inline bool prefer_mr16(int M, int K, int N, int lanes) {
    (void)K;
    (void)N;
    return lanes >= 16 && M >= 32;
}

void layer_norm_dispatch(const float* x, int N, int C,
                         const float* gamma, const float* beta,
                         float eps, float* out) {
    const hn::ScalableTag<float> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    for (int n = 0; n < N; n++) {
        const float* row = x + (size_t)n * C;
        float* orow = out + (size_t)n * C;
        int c = 0;
        auto vsum = hn::Zero(d);
        for (; c + lanes <= C; c += lanes)
            vsum = hn::Add(vsum, hn::LoadU(d, row + c));
        float sum = hn::ReduceSum(d, vsum);
        for (; c < C; c++) sum += row[c];
        float mean = sum / C;

        c = 0;
        auto vvar = hn::Zero(d);
        const auto vmean = hn::Set(d, mean);
        for (; c + lanes <= C; c += lanes) {
            const auto diff = hn::Sub(hn::LoadU(d, row + c), vmean);
            vvar = hn::MulAdd(diff, diff, vvar);
        }
        float var_sum = hn::ReduceSum(d, vvar);
        for (; c < C; c++) {
            const float diff = row[c] - mean;
            var_sum += diff * diff;
        }
        float inv_std = 1.0f / sqrtf(var_sum / C + eps);
        const auto vinv = hn::Set(d, inv_std);
        c = 0;
        for (; c + lanes <= C; c += lanes) {
            auto y = hn::Mul(hn::Sub(hn::LoadU(d, row + c), vmean), vinv);
            y = hn::MulAdd(y, hn::LoadU(d, gamma + c), hn::LoadU(d, beta + c));
            hn::StoreU(y, d, orow + c);
        }
        for (; c < C; c++) {
            float y = (row[c] - mean) * inv_std;
            orow[c] = y * gamma[c] + beta[c];
        }
    }
}

void matmul_fp32_dispatch(const float* A, const float* B, float* C,
                          int M, int K, int N) {
    const hn::ScalableTag<float> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    int m = 0;
    for (; m + 4 <= M; m += 4) {
        const float* a0 = A + (size_t)(m + 0) * K;
        const float* a1 = A + (size_t)(m + 1) * K;
        const float* a2 = A + (size_t)(m + 2) * K;
        const float* a3 = A + (size_t)(m + 3) * K;
        float* c0 = C + (size_t)(m + 0) * N;
        float* c1 = C + (size_t)(m + 1) * N;
        float* c2 = C + (size_t)(m + 2) * N;
        float* c3 = C + (size_t)(m + 3) * N;
        int n = 0;
        for (; n + lanes <= N; n += lanes) {
            auto acc0 = hn::Zero(d);
            auto acc1 = hn::Zero(d);
            auto acc2 = hn::Zero(d);
            auto acc3 = hn::Zero(d);
            for (int k = 0; k < K; k++) {
                const auto b = hn::LoadU(d, B + (size_t)k * N + n);
                acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
            }
            hn::StoreU(acc0, d, c0 + n);
            hn::StoreU(acc1, d, c1 + n);
            hn::StoreU(acc2, d, c2 + n);
            hn::StoreU(acc3, d, c3 + n);
        }
        for (; n < N; n++) {
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            for (int k = 0; k < K; k++) {
                const float bv = B[(size_t)k * N + n];
                s0 += a0[k] * bv; s1 += a1[k] * bv;
                s2 += a2[k] * bv; s3 += a3[k] * bv;
            }
            c0[n] = s0; c1[n] = s1; c2[n] = s2; c3[n] = s3;
        }
    }
    for (; m < M; m++) {
        const float* ar = A + (size_t)m * K;
        float* cr = C + (size_t)m * N;
        int n = 0;
        for (; n + lanes <= N; n += lanes) {
            auto acc = hn::Zero(d);
            for (int k = 0; k < K; k++) {
                acc = hn::MulAdd(hn::Set(d, ar[k]), hn::LoadU(d, B + (size_t)k * N + n), acc);
            }
            hn::StoreU(acc, d, cr + n);
        }
        for (; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += ar[k] * B[(size_t)k * N + n];
            cr[n] = sum;
        }
    }
}

void matmul_fp32_packed_dispatch(const float* A, const float* B_packed, float* C,
                                 int M, int K, int N) {
    const int NR = 16;
    const int n_panels = (N + NR - 1) / NR;
    const hn::CappedTag<float, NR> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    if (lanes == NR && ((N & (NR - 1)) == 0)) {
        const int n_panels_exact = N / NR;
        int m = 0;
        if (prefer_mr16(M, K, N, lanes)) {
        for (; m + 16 <= M; m += 16) {
            for (int p = 0; p < n_panels_exact; p++) {
                const int n0 = p * NR;
                const float* bp = B_packed + (size_t)p * K * NR;
                using V = decltype(hn::Zero(d));
                V acc[16];
                for (int r = 0; r < 16; r++) acc[r] = hn::Zero(d);
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR);
                    for (int r = 0; r < 16; r++) {
                        acc[r] = hn::MulAdd(hn::Set(d, A[(size_t)(m + r) * K + k]), b, acc[r]);
                    }
                }
                for (int r = 0; r < 16; r++) {
                    hn::StoreU(acc[r], d, C + (size_t)(m + r) * N + n0);
                }
            }
        }
        }
        if (prefer_mr8(M, K, N, lanes)) {
        for (; m + 8 <= M; m += 8) {
            const float* a0 = A + (size_t)(m + 0) * K;
            const float* a1 = A + (size_t)(m + 1) * K;
            const float* a2 = A + (size_t)(m + 2) * K;
            const float* a3 = A + (size_t)(m + 3) * K;
            const float* a4 = A + (size_t)(m + 4) * K;
            const float* a5 = A + (size_t)(m + 5) * K;
            const float* a6 = A + (size_t)(m + 6) * K;
            const float* a7 = A + (size_t)(m + 7) * K;
            float* c0 = C + (size_t)(m + 0) * N;
            float* c1 = C + (size_t)(m + 1) * N;
            float* c2 = C + (size_t)(m + 2) * N;
            float* c3 = C + (size_t)(m + 3) * N;
            float* c4 = C + (size_t)(m + 4) * N;
            float* c5 = C + (size_t)(m + 5) * N;
            float* c6 = C + (size_t)(m + 6) * N;
            float* c7 = C + (size_t)(m + 7) * N;
            for (int p = 0; p < n_panels_exact; p++) {
                const int n0 = p * NR;
                const float* bp = B_packed + (size_t)p * K * NR;
                auto acc0 = hn::Zero(d), acc1 = hn::Zero(d);
                auto acc2 = hn::Zero(d), acc3 = hn::Zero(d);
                auto acc4 = hn::Zero(d), acc5 = hn::Zero(d);
                auto acc6 = hn::Zero(d), acc7 = hn::Zero(d);
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                    acc4 = hn::MulAdd(hn::Set(d, a4[k]), b, acc4);
                    acc5 = hn::MulAdd(hn::Set(d, a5[k]), b, acc5);
                    acc6 = hn::MulAdd(hn::Set(d, a6[k]), b, acc6);
                    acc7 = hn::MulAdd(hn::Set(d, a7[k]), b, acc7);
                }
                hn::StoreU(acc0, d, c0 + n0);
                hn::StoreU(acc1, d, c1 + n0);
                hn::StoreU(acc2, d, c2 + n0);
                hn::StoreU(acc3, d, c3 + n0);
                hn::StoreU(acc4, d, c4 + n0);
                hn::StoreU(acc5, d, c5 + n0);
                hn::StoreU(acc6, d, c6 + n0);
                hn::StoreU(acc7, d, c7 + n0);
            }
        }
        }
        for (; m + 4 <= M; m += 4) {
            const float* a0 = A + (size_t)(m + 0) * K;
            const float* a1 = A + (size_t)(m + 1) * K;
            const float* a2 = A + (size_t)(m + 2) * K;
            const float* a3 = A + (size_t)(m + 3) * K;
            float* c0 = C + (size_t)(m + 0) * N;
            float* c1 = C + (size_t)(m + 1) * N;
            float* c2 = C + (size_t)(m + 2) * N;
            float* c3 = C + (size_t)(m + 3) * N;
            for (int p = 0; p < n_panels_exact; p++) {
                const int n0 = p * NR;
                const float* bp = B_packed + (size_t)p * K * NR;
                auto acc0 = hn::Zero(d), acc1 = hn::Zero(d);
                auto acc2 = hn::Zero(d), acc3 = hn::Zero(d);
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                }
                hn::StoreU(acc0, d, c0 + n0);
                hn::StoreU(acc1, d, c1 + n0);
                hn::StoreU(acc2, d, c2 + n0);
                hn::StoreU(acc3, d, c3 + n0);
            }
        }
        for (; m < M; m++) {
            const float* ar = A + (size_t)m * K;
            float* cr = C + (size_t)m * N;
            for (int p = 0; p < n_panels_exact; p++) {
                const int n0 = p * NR;
                const float* bp = B_packed + (size_t)p * K * NR;
                auto acc = hn::Zero(d);
                for (int k = 0; k < K; k++) {
                    acc = hn::MulAdd(hn::Set(d, ar[k]), hn::LoadU(d, bp + (size_t)k * NR), acc);
                }
                hn::StoreU(acc, d, cr + n0);
            }
        }
        return;
    }
    if (M == 1) {
        for (int p = 0; p < n_panels; p++) {
            const int n0 = p * NR;
            const int nr = (n0 + NR <= N) ? NR : (N - n0);
            const float* bp = B_packed + (size_t)p * K * NR;
            int j = 0;
            for (; j + lanes <= nr; j += lanes) {
                auto acc = hn::Zero(d);
                for (int k = 0; k < K; k++) {
                    acc = hn::MulAdd(hn::Set(d, A[k]), hn::LoadU(d, bp + (size_t)k * NR + j), acc);
                }
                hn::StoreU(acc, d, C + n0 + j);
            }
            if (j < nr) {
                auto acc = hn::Zero(d);
                for (int k = 0; k < K; k++) {
                    acc = hn::MulAdd(hn::Set(d, A[k]), hn::LoadU(d, bp + (size_t)k * NR + j), acc);
                }
                hn::StoreN(acc, d, C + n0 + j, static_cast<size_t>(nr - j));
            }
        }
        return;
    }
    int m = 0;
    if (prefer_mr16(M, K, N, lanes)) {
    for (; m + 16 <= M; m += 16) {
        const float* a0 = A + (size_t)(m + 0) * K;
        const float* a1 = A + (size_t)(m + 1) * K;
        const float* a2 = A + (size_t)(m + 2) * K;
        const float* a3 = A + (size_t)(m + 3) * K;
        const float* a4 = A + (size_t)(m + 4) * K;
        const float* a5 = A + (size_t)(m + 5) * K;
        const float* a6 = A + (size_t)(m + 6) * K;
        const float* a7 = A + (size_t)(m + 7) * K;
        const float* a8 = A + (size_t)(m + 8) * K;
        const float* a9 = A + (size_t)(m + 9) * K;
        const float* a10 = A + (size_t)(m + 10) * K;
        const float* a11 = A + (size_t)(m + 11) * K;
        const float* a12 = A + (size_t)(m + 12) * K;
        const float* a13 = A + (size_t)(m + 13) * K;
        const float* a14 = A + (size_t)(m + 14) * K;
        const float* a15 = A + (size_t)(m + 15) * K;
        float* c0 = C + (size_t)(m + 0) * N;
        float* c1 = C + (size_t)(m + 1) * N;
        float* c2 = C + (size_t)(m + 2) * N;
        float* c3 = C + (size_t)(m + 3) * N;
        float* c4 = C + (size_t)(m + 4) * N;
        float* c5 = C + (size_t)(m + 5) * N;
        float* c6 = C + (size_t)(m + 6) * N;
        float* c7 = C + (size_t)(m + 7) * N;
        float* c8 = C + (size_t)(m + 8) * N;
        float* c9 = C + (size_t)(m + 9) * N;
        float* c10 = C + (size_t)(m + 10) * N;
        float* c11 = C + (size_t)(m + 11) * N;
        float* c12 = C + (size_t)(m + 12) * N;
        float* c13 = C + (size_t)(m + 13) * N;
        float* c14 = C + (size_t)(m + 14) * N;
        float* c15 = C + (size_t)(m + 15) * N;
        for (int p = 0; p < n_panels; p++) {
            const int n0 = p * NR;
            const int nr = (n0 + NR <= N) ? NR : (N - n0);
            const float* bp = B_packed + (size_t)p * K * NR;
            int j = 0;
            for (; j + lanes <= nr; j += lanes) {
                auto acc0 = hn::Zero(d), acc1 = hn::Zero(d), acc2 = hn::Zero(d), acc3 = hn::Zero(d);
                auto acc4 = hn::Zero(d), acc5 = hn::Zero(d), acc6 = hn::Zero(d), acc7 = hn::Zero(d);
                auto acc8 = hn::Zero(d), acc9 = hn::Zero(d), acc10 = hn::Zero(d), acc11 = hn::Zero(d);
                auto acc12 = hn::Zero(d), acc13 = hn::Zero(d), acc14 = hn::Zero(d), acc15 = hn::Zero(d);
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR + j);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                    acc4 = hn::MulAdd(hn::Set(d, a4[k]), b, acc4);
                    acc5 = hn::MulAdd(hn::Set(d, a5[k]), b, acc5);
                    acc6 = hn::MulAdd(hn::Set(d, a6[k]), b, acc6);
                    acc7 = hn::MulAdd(hn::Set(d, a7[k]), b, acc7);
                    acc8 = hn::MulAdd(hn::Set(d, a8[k]), b, acc8);
                    acc9 = hn::MulAdd(hn::Set(d, a9[k]), b, acc9);
                    acc10 = hn::MulAdd(hn::Set(d, a10[k]), b, acc10);
                    acc11 = hn::MulAdd(hn::Set(d, a11[k]), b, acc11);
                    acc12 = hn::MulAdd(hn::Set(d, a12[k]), b, acc12);
                    acc13 = hn::MulAdd(hn::Set(d, a13[k]), b, acc13);
                    acc14 = hn::MulAdd(hn::Set(d, a14[k]), b, acc14);
                    acc15 = hn::MulAdd(hn::Set(d, a15[k]), b, acc15);
                }
                hn::StoreU(acc0, d, c0 + n0 + j);
                hn::StoreU(acc1, d, c1 + n0 + j);
                hn::StoreU(acc2, d, c2 + n0 + j);
                hn::StoreU(acc3, d, c3 + n0 + j);
                hn::StoreU(acc4, d, c4 + n0 + j);
                hn::StoreU(acc5, d, c5 + n0 + j);
                hn::StoreU(acc6, d, c6 + n0 + j);
                hn::StoreU(acc7, d, c7 + n0 + j);
                hn::StoreU(acc8, d, c8 + n0 + j);
                hn::StoreU(acc9, d, c9 + n0 + j);
                hn::StoreU(acc10, d, c10 + n0 + j);
                hn::StoreU(acc11, d, c11 + n0 + j);
                hn::StoreU(acc12, d, c12 + n0 + j);
                hn::StoreU(acc13, d, c13 + n0 + j);
                hn::StoreU(acc14, d, c14 + n0 + j);
                hn::StoreU(acc15, d, c15 + n0 + j);
            }
            if (j < nr) {
                const size_t count = static_cast<size_t>(nr - j);
                auto acc0 = hn::Zero(d);
                auto acc1 = acc0, acc2 = acc0, acc3 = acc0;
                auto acc4 = acc0, acc5 = acc0, acc6 = acc0, acc7 = acc0;
                auto acc8 = acc0, acc9 = acc0, acc10 = acc0, acc11 = acc0;
                auto acc12 = acc0, acc13 = acc0, acc14 = acc0, acc15 = acc0;
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR + j);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                    acc4 = hn::MulAdd(hn::Set(d, a4[k]), b, acc4);
                    acc5 = hn::MulAdd(hn::Set(d, a5[k]), b, acc5);
                    acc6 = hn::MulAdd(hn::Set(d, a6[k]), b, acc6);
                    acc7 = hn::MulAdd(hn::Set(d, a7[k]), b, acc7);
                    acc8 = hn::MulAdd(hn::Set(d, a8[k]), b, acc8);
                    acc9 = hn::MulAdd(hn::Set(d, a9[k]), b, acc9);
                    acc10 = hn::MulAdd(hn::Set(d, a10[k]), b, acc10);
                    acc11 = hn::MulAdd(hn::Set(d, a11[k]), b, acc11);
                    acc12 = hn::MulAdd(hn::Set(d, a12[k]), b, acc12);
                    acc13 = hn::MulAdd(hn::Set(d, a13[k]), b, acc13);
                    acc14 = hn::MulAdd(hn::Set(d, a14[k]), b, acc14);
                    acc15 = hn::MulAdd(hn::Set(d, a15[k]), b, acc15);
                }
                hn::StoreN(acc0, d, c0 + n0 + j, count);
                hn::StoreN(acc1, d, c1 + n0 + j, count);
                hn::StoreN(acc2, d, c2 + n0 + j, count);
                hn::StoreN(acc3, d, c3 + n0 + j, count);
                hn::StoreN(acc4, d, c4 + n0 + j, count);
                hn::StoreN(acc5, d, c5 + n0 + j, count);
                hn::StoreN(acc6, d, c6 + n0 + j, count);
                hn::StoreN(acc7, d, c7 + n0 + j, count);
                hn::StoreN(acc8, d, c8 + n0 + j, count);
                hn::StoreN(acc9, d, c9 + n0 + j, count);
                hn::StoreN(acc10, d, c10 + n0 + j, count);
                hn::StoreN(acc11, d, c11 + n0 + j, count);
                hn::StoreN(acc12, d, c12 + n0 + j, count);
                hn::StoreN(acc13, d, c13 + n0 + j, count);
                hn::StoreN(acc14, d, c14 + n0 + j, count);
                hn::StoreN(acc15, d, c15 + n0 + j, count);
            }
        }
    }
    }
    if (prefer_mr8(M, K, N, lanes)) {
    for (; m + 8 <= M; m += 8) {
        const float* a0 = A + (size_t)(m + 0) * K;
        const float* a1 = A + (size_t)(m + 1) * K;
        const float* a2 = A + (size_t)(m + 2) * K;
        const float* a3 = A + (size_t)(m + 3) * K;
        const float* a4 = A + (size_t)(m + 4) * K;
        const float* a5 = A + (size_t)(m + 5) * K;
        const float* a6 = A + (size_t)(m + 6) * K;
        const float* a7 = A + (size_t)(m + 7) * K;
        float* c0 = C + (size_t)(m + 0) * N;
        float* c1 = C + (size_t)(m + 1) * N;
        float* c2 = C + (size_t)(m + 2) * N;
        float* c3 = C + (size_t)(m + 3) * N;
        float* c4 = C + (size_t)(m + 4) * N;
        float* c5 = C + (size_t)(m + 5) * N;
        float* c6 = C + (size_t)(m + 6) * N;
        float* c7 = C + (size_t)(m + 7) * N;
        for (int p = 0; p < n_panels; p++) {
            const int n0 = p * NR;
            const int nr = (n0 + NR <= N) ? NR : (N - n0);
            const float* bp = B_packed + (size_t)p * K * NR;
            int j = 0;
            for (; j + lanes <= nr; j += lanes) {
                auto acc0 = hn::Zero(d), acc1 = hn::Zero(d);
                auto acc2 = hn::Zero(d), acc3 = hn::Zero(d);
                auto acc4 = hn::Zero(d), acc5 = hn::Zero(d);
                auto acc6 = hn::Zero(d), acc7 = hn::Zero(d);
                const float* bp0 = bp + j;
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp0);
                    bp0 += NR;
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                    acc4 = hn::MulAdd(hn::Set(d, a4[k]), b, acc4);
                    acc5 = hn::MulAdd(hn::Set(d, a5[k]), b, acc5);
                    acc6 = hn::MulAdd(hn::Set(d, a6[k]), b, acc6);
                    acc7 = hn::MulAdd(hn::Set(d, a7[k]), b, acc7);
                }
                hn::StoreU(acc0, d, c0 + n0 + j);
                hn::StoreU(acc1, d, c1 + n0 + j);
                hn::StoreU(acc2, d, c2 + n0 + j);
                hn::StoreU(acc3, d, c3 + n0 + j);
                hn::StoreU(acc4, d, c4 + n0 + j);
                hn::StoreU(acc5, d, c5 + n0 + j);
                hn::StoreU(acc6, d, c6 + n0 + j);
                hn::StoreU(acc7, d, c7 + n0 + j);
            }
            if (j < nr) {
                auto acc0 = hn::Zero(d), acc1 = hn::Zero(d);
                auto acc2 = hn::Zero(d), acc3 = hn::Zero(d);
                auto acc4 = hn::Zero(d), acc5 = hn::Zero(d);
                auto acc6 = hn::Zero(d), acc7 = hn::Zero(d);
                const float* bp0 = bp + j;
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp0);
                    bp0 += NR;
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                    acc4 = hn::MulAdd(hn::Set(d, a4[k]), b, acc4);
                    acc5 = hn::MulAdd(hn::Set(d, a5[k]), b, acc5);
                    acc6 = hn::MulAdd(hn::Set(d, a6[k]), b, acc6);
                    acc7 = hn::MulAdd(hn::Set(d, a7[k]), b, acc7);
                }
                const size_t count = static_cast<size_t>(nr - j);
                hn::StoreN(acc0, d, c0 + n0 + j, count);
                hn::StoreN(acc1, d, c1 + n0 + j, count);
                hn::StoreN(acc2, d, c2 + n0 + j, count);
                hn::StoreN(acc3, d, c3 + n0 + j, count);
                hn::StoreN(acc4, d, c4 + n0 + j, count);
                hn::StoreN(acc5, d, c5 + n0 + j, count);
                hn::StoreN(acc6, d, c6 + n0 + j, count);
                hn::StoreN(acc7, d, c7 + n0 + j, count);
            }
        }
    }
    }
    for (; m + 4 <= M; m += 4) {
        const float* a0 = A + (size_t)(m + 0) * K;
        const float* a1 = A + (size_t)(m + 1) * K;
        const float* a2 = A + (size_t)(m + 2) * K;
        const float* a3 = A + (size_t)(m + 3) * K;
        float* c0 = C + (size_t)(m + 0) * N;
        float* c1 = C + (size_t)(m + 1) * N;
        float* c2 = C + (size_t)(m + 2) * N;
        float* c3 = C + (size_t)(m + 3) * N;
        for (int p = 0; p < n_panels; p++) {
            const int n0 = p * NR;
            const int nr = (n0 + NR <= N) ? NR : (N - n0);
            const float* bp = B_packed + (size_t)p * K * NR;
            int j = 0;
            for (; j + lanes <= nr; j += lanes) {
                auto acc0 = hn::Zero(d), acc1 = hn::Zero(d);
                auto acc2 = hn::Zero(d), acc3 = hn::Zero(d);
                const float* bp0 = bp + j;
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp0);
                    bp0 += NR;
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                }
                hn::StoreU(acc0, d, c0 + n0 + j);
                hn::StoreU(acc1, d, c1 + n0 + j);
                hn::StoreU(acc2, d, c2 + n0 + j);
                hn::StoreU(acc3, d, c3 + n0 + j);
            }
            if (j < nr) {
                auto acc0 = hn::Zero(d), acc1 = hn::Zero(d);
                auto acc2 = hn::Zero(d), acc3 = hn::Zero(d);
                const float* bp0 = bp + j;
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp0);
                    bp0 += NR;
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                }
                const size_t count = static_cast<size_t>(nr - j);
                hn::StoreN(acc0, d, c0 + n0 + j, count);
                hn::StoreN(acc1, d, c1 + n0 + j, count);
                hn::StoreN(acc2, d, c2 + n0 + j, count);
                hn::StoreN(acc3, d, c3 + n0 + j, count);
            }
        }
    }
    for (; m < M; m++) {
        const float* ar = A + (size_t)m * K;
        float* cr = C + (size_t)m * N;
        for (int p = 0; p < n_panels; p++) {
            const int n0 = p * NR;
            const int nr = (n0 + NR <= N) ? NR : (N - n0);
            const float* bp = B_packed + (size_t)p * K * NR;
            int j = 0;
            for (; j + lanes <= nr; j += lanes) {
                auto acc = hn::Zero(d);
                const float* bp0 = bp + j;
                for (int k = 0; k < K; k++) {
                    acc = hn::MulAdd(hn::Set(d, ar[k]), hn::LoadU(d, bp0), acc);
                    bp0 += NR;
                }
                hn::StoreU(acc, d, cr + n0 + j);
            }
            if (j < nr) {
                auto acc = hn::Zero(d);
                const float* bp0 = bp + j;
                for (int k = 0; k < K; k++) {
                    acc = hn::MulAdd(hn::Set(d, ar[k]), hn::LoadU(d, bp0), acc);
                    bp0 += NR;
                }
                hn::StoreN(acc, d, cr + n0 + j, static_cast<size_t>(nr - j));
            }
        }
    }
}

void matmul_fp32_packed_bias_dispatch(const float* A, const float* B_packed, const float* bias,
                                      float* C,
                                      int M, int K, int N) {
    if (!bias) {
        matmul_fp32_packed_dispatch(A, B_packed, C, M, K, N);
        return;
    }
    constexpr int NR = 16;
    const int n_panels = (N + NR - 1) / NR;
    const hn::CappedTag<float, NR> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    if (lanes == NR && ((N & (NR - 1)) == 0)) {
        const int n_panels_exact = N / NR;
        int m = 0;
        if (prefer_mr16(M, K, N, lanes)) {
            for (; m + 16 <= M; m += 16) {
                const float* a0 = A + (size_t)(m + 0) * K;
                const float* a1 = A + (size_t)(m + 1) * K;
                const float* a2 = A + (size_t)(m + 2) * K;
                const float* a3 = A + (size_t)(m + 3) * K;
                const float* a4 = A + (size_t)(m + 4) * K;
                const float* a5 = A + (size_t)(m + 5) * K;
                const float* a6 = A + (size_t)(m + 6) * K;
                const float* a7 = A + (size_t)(m + 7) * K;
                const float* a8 = A + (size_t)(m + 8) * K;
                const float* a9 = A + (size_t)(m + 9) * K;
                const float* a10 = A + (size_t)(m + 10) * K;
                const float* a11 = A + (size_t)(m + 11) * K;
                const float* a12 = A + (size_t)(m + 12) * K;
                const float* a13 = A + (size_t)(m + 13) * K;
                const float* a14 = A + (size_t)(m + 14) * K;
                const float* a15 = A + (size_t)(m + 15) * K;
                float* c0 = C + (size_t)(m + 0) * N;
                float* c1 = C + (size_t)(m + 1) * N;
                float* c2 = C + (size_t)(m + 2) * N;
                float* c3 = C + (size_t)(m + 3) * N;
                float* c4 = C + (size_t)(m + 4) * N;
                float* c5 = C + (size_t)(m + 5) * N;
                float* c6 = C + (size_t)(m + 6) * N;
                float* c7 = C + (size_t)(m + 7) * N;
                float* c8 = C + (size_t)(m + 8) * N;
                float* c9 = C + (size_t)(m + 9) * N;
                float* c10 = C + (size_t)(m + 10) * N;
                float* c11 = C + (size_t)(m + 11) * N;
                float* c12 = C + (size_t)(m + 12) * N;
                float* c13 = C + (size_t)(m + 13) * N;
                float* c14 = C + (size_t)(m + 14) * N;
                float* c15 = C + (size_t)(m + 15) * N;
                int p = 0;
                for (; p + 1 < n_panels_exact; p += 2) {
                    const int n0 = p * NR;
                    const float* bp0 = B_packed + (size_t)p * K * NR;
                    const float* bp1 = bp0 + (size_t)K * NR;
                    using V = decltype(hn::Zero(d));
                    V acc0_0 = hn::LoadU(d, bias + n0);
                    V acc1_0 = acc0_0, acc2_0 = acc0_0, acc3_0 = acc0_0;
                    V acc4_0 = acc0_0, acc5_0 = acc0_0, acc6_0 = acc0_0, acc7_0 = acc0_0;
                    V acc8_0 = acc0_0, acc9_0 = acc0_0, acc10_0 = acc0_0, acc11_0 = acc0_0;
                    V acc12_0 = acc0_0, acc13_0 = acc0_0, acc14_0 = acc0_0, acc15_0 = acc0_0;
                    for (int k = 0; k < K; k++) {
                        const auto b0 = hn::LoadU(d, bp0 + (size_t)k * NR);
                        acc0_0 = hn::MulAdd(hn::Set(d, a0[k]), b0, acc0_0);
                        acc1_0 = hn::MulAdd(hn::Set(d, a1[k]), b0, acc1_0);
                        acc2_0 = hn::MulAdd(hn::Set(d, a2[k]), b0, acc2_0);
                        acc3_0 = hn::MulAdd(hn::Set(d, a3[k]), b0, acc3_0);
                        acc4_0 = hn::MulAdd(hn::Set(d, a4[k]), b0, acc4_0);
                        acc5_0 = hn::MulAdd(hn::Set(d, a5[k]), b0, acc5_0);
                        acc6_0 = hn::MulAdd(hn::Set(d, a6[k]), b0, acc6_0);
                        acc7_0 = hn::MulAdd(hn::Set(d, a7[k]), b0, acc7_0);
                        acc8_0 = hn::MulAdd(hn::Set(d, a8[k]), b0, acc8_0);
                        acc9_0 = hn::MulAdd(hn::Set(d, a9[k]), b0, acc9_0);
                        acc10_0 = hn::MulAdd(hn::Set(d, a10[k]), b0, acc10_0);
                        acc11_0 = hn::MulAdd(hn::Set(d, a11[k]), b0, acc11_0);
                        acc12_0 = hn::MulAdd(hn::Set(d, a12[k]), b0, acc12_0);
                        acc13_0 = hn::MulAdd(hn::Set(d, a13[k]), b0, acc13_0);
                        acc14_0 = hn::MulAdd(hn::Set(d, a14[k]), b0, acc14_0);
                        acc15_0 = hn::MulAdd(hn::Set(d, a15[k]), b0, acc15_0);
                    }
                    hn::StoreU(acc0_0, d, c0 + n0);
                    hn::StoreU(acc1_0, d, c1 + n0);
                    hn::StoreU(acc2_0, d, c2 + n0);
                    hn::StoreU(acc3_0, d, c3 + n0);
                    hn::StoreU(acc4_0, d, c4 + n0);
                    hn::StoreU(acc5_0, d, c5 + n0);
                    hn::StoreU(acc6_0, d, c6 + n0);
                    hn::StoreU(acc7_0, d, c7 + n0);
                    hn::StoreU(acc8_0, d, c8 + n0);
                    hn::StoreU(acc9_0, d, c9 + n0);
                    hn::StoreU(acc10_0, d, c10 + n0);
                    hn::StoreU(acc11_0, d, c11 + n0);
                    hn::StoreU(acc12_0, d, c12 + n0);
                    hn::StoreU(acc13_0, d, c13 + n0);
                    hn::StoreU(acc14_0, d, c14 + n0);
                    hn::StoreU(acc15_0, d, c15 + n0);

                    auto acc0_1 = hn::LoadU(d, bias + n0 + NR);
                    auto acc1_1 = acc0_1;
                    auto acc2_1 = acc0_1;
                    auto acc3_1 = acc0_1;
                    auto acc4_1 = acc0_1;
                    auto acc5_1 = acc0_1;
                    auto acc6_1 = acc0_1;
                    auto acc7_1 = acc0_1;
                    auto acc8_1 = acc0_1;
                    auto acc9_1 = acc0_1;
                    auto acc10_1 = acc0_1;
                    auto acc11_1 = acc0_1;
                    auto acc12_1 = acc0_1;
                    auto acc13_1 = acc0_1;
                    auto acc14_1 = acc0_1;
                    auto acc15_1 = acc0_1;
                    for (int k = 0; k < K; k++) {
                        const auto b1 = hn::LoadU(d, bp1 + (size_t)k * NR);
                        acc0_1 = hn::MulAdd(hn::Set(d, a0[k]), b1, acc0_1);
                        acc1_1 = hn::MulAdd(hn::Set(d, a1[k]), b1, acc1_1);
                        acc2_1 = hn::MulAdd(hn::Set(d, a2[k]), b1, acc2_1);
                        acc3_1 = hn::MulAdd(hn::Set(d, a3[k]), b1, acc3_1);
                        acc4_1 = hn::MulAdd(hn::Set(d, a4[k]), b1, acc4_1);
                        acc5_1 = hn::MulAdd(hn::Set(d, a5[k]), b1, acc5_1);
                        acc6_1 = hn::MulAdd(hn::Set(d, a6[k]), b1, acc6_1);
                        acc7_1 = hn::MulAdd(hn::Set(d, a7[k]), b1, acc7_1);
                        acc8_1 = hn::MulAdd(hn::Set(d, a8[k]), b1, acc8_1);
                        acc9_1 = hn::MulAdd(hn::Set(d, a9[k]), b1, acc9_1);
                        acc10_1 = hn::MulAdd(hn::Set(d, a10[k]), b1, acc10_1);
                        acc11_1 = hn::MulAdd(hn::Set(d, a11[k]), b1, acc11_1);
                        acc12_1 = hn::MulAdd(hn::Set(d, a12[k]), b1, acc12_1);
                        acc13_1 = hn::MulAdd(hn::Set(d, a13[k]), b1, acc13_1);
                        acc14_1 = hn::MulAdd(hn::Set(d, a14[k]), b1, acc14_1);
                        acc15_1 = hn::MulAdd(hn::Set(d, a15[k]), b1, acc15_1);
                    }
                    hn::StoreU(acc0_1, d, c0 + n0 + NR);
                    hn::StoreU(acc1_1, d, c1 + n0 + NR);
                    hn::StoreU(acc2_1, d, c2 + n0 + NR);
                    hn::StoreU(acc3_1, d, c3 + n0 + NR);
                    hn::StoreU(acc4_1, d, c4 + n0 + NR);
                    hn::StoreU(acc5_1, d, c5 + n0 + NR);
                    hn::StoreU(acc6_1, d, c6 + n0 + NR);
                    hn::StoreU(acc7_1, d, c7 + n0 + NR);
                    hn::StoreU(acc8_1, d, c8 + n0 + NR);
                    hn::StoreU(acc9_1, d, c9 + n0 + NR);
                    hn::StoreU(acc10_1, d, c10 + n0 + NR);
                    hn::StoreU(acc11_1, d, c11 + n0 + NR);
                    hn::StoreU(acc12_1, d, c12 + n0 + NR);
                    hn::StoreU(acc13_1, d, c13 + n0 + NR);
                    hn::StoreU(acc14_1, d, c14 + n0 + NR);
                    hn::StoreU(acc15_1, d, c15 + n0 + NR);
                }
                if (p < n_panels_exact) {
                    const int n0 = p * NR;
                    const float* bp = B_packed + (size_t)p * K * NR;
                    using V = decltype(hn::Zero(d));
                    V acc0 = hn::LoadU(d, bias + n0);
                    V acc1 = acc0, acc2 = acc0, acc3 = acc0;
                    V acc4 = acc0, acc5 = acc0, acc6 = acc0, acc7 = acc0;
                    V acc8 = acc0, acc9 = acc0, acc10 = acc0, acc11 = acc0;
                    V acc12 = acc0, acc13 = acc0, acc14 = acc0, acc15 = acc0;
                    for (int k = 0; k < K; k++) {
                        const auto b = hn::LoadU(d, bp + (size_t)k * NR);
                        acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                        acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                        acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                        acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                        acc4 = hn::MulAdd(hn::Set(d, a4[k]), b, acc4);
                        acc5 = hn::MulAdd(hn::Set(d, a5[k]), b, acc5);
                        acc6 = hn::MulAdd(hn::Set(d, a6[k]), b, acc6);
                        acc7 = hn::MulAdd(hn::Set(d, a7[k]), b, acc7);
                        acc8 = hn::MulAdd(hn::Set(d, a8[k]), b, acc8);
                        acc9 = hn::MulAdd(hn::Set(d, a9[k]), b, acc9);
                        acc10 = hn::MulAdd(hn::Set(d, a10[k]), b, acc10);
                        acc11 = hn::MulAdd(hn::Set(d, a11[k]), b, acc11);
                        acc12 = hn::MulAdd(hn::Set(d, a12[k]), b, acc12);
                        acc13 = hn::MulAdd(hn::Set(d, a13[k]), b, acc13);
                        acc14 = hn::MulAdd(hn::Set(d, a14[k]), b, acc14);
                        acc15 = hn::MulAdd(hn::Set(d, a15[k]), b, acc15);
                    }
                    hn::StoreU(acc0, d, c0 + n0);
                    hn::StoreU(acc1, d, c1 + n0);
                    hn::StoreU(acc2, d, c2 + n0);
                    hn::StoreU(acc3, d, c3 + n0);
                    hn::StoreU(acc4, d, c4 + n0);
                    hn::StoreU(acc5, d, c5 + n0);
                    hn::StoreU(acc6, d, c6 + n0);
                    hn::StoreU(acc7, d, c7 + n0);
                    hn::StoreU(acc8, d, c8 + n0);
                    hn::StoreU(acc9, d, c9 + n0);
                    hn::StoreU(acc10, d, c10 + n0);
                    hn::StoreU(acc11, d, c11 + n0);
                    hn::StoreU(acc12, d, c12 + n0);
                    hn::StoreU(acc13, d, c13 + n0);
                    hn::StoreU(acc14, d, c14 + n0);
                    hn::StoreU(acc15, d, c15 + n0);
                }
            }
        }
        if (prefer_mr8(M, K, N, lanes)) {
            for (; m + 8 <= M; m += 8) {
                const float* a0 = A + (size_t)(m + 0) * K;
                const float* a1 = A + (size_t)(m + 1) * K;
                const float* a2 = A + (size_t)(m + 2) * K;
                const float* a3 = A + (size_t)(m + 3) * K;
                const float* a4 = A + (size_t)(m + 4) * K;
                const float* a5 = A + (size_t)(m + 5) * K;
                const float* a6 = A + (size_t)(m + 6) * K;
                const float* a7 = A + (size_t)(m + 7) * K;
                float* c0 = C + (size_t)(m + 0) * N;
                float* c1 = C + (size_t)(m + 1) * N;
                float* c2 = C + (size_t)(m + 2) * N;
                float* c3 = C + (size_t)(m + 3) * N;
                float* c4 = C + (size_t)(m + 4) * N;
                float* c5 = C + (size_t)(m + 5) * N;
                float* c6 = C + (size_t)(m + 6) * N;
                float* c7 = C + (size_t)(m + 7) * N;
                int p = 0;
                for (; p + 1 < n_panels_exact; p += 2) {
                    const int n0 = p * NR;
                    const float* bp0 = B_packed + (size_t)p * K * NR;
                    const float* bp1 = bp0 + (size_t)K * NR;
                    auto acc0 = hn::LoadU(d, bias + n0);
                    auto acc1 = acc0;
                    auto acc2 = acc0;
                    auto acc3 = acc0;
                    auto acc4 = acc0;
                    auto acc5 = acc0;
                    auto acc6 = acc0;
                    auto acc7 = acc0;
                    for (int k = 0; k < K; k++) {
                        const auto b0 = hn::LoadU(d, bp0 + (size_t)k * NR);
                        acc0 = hn::MulAdd(hn::Set(d, a0[k]), b0, acc0);
                        acc1 = hn::MulAdd(hn::Set(d, a1[k]), b0, acc1);
                        acc2 = hn::MulAdd(hn::Set(d, a2[k]), b0, acc2);
                        acc3 = hn::MulAdd(hn::Set(d, a3[k]), b0, acc3);
                        acc4 = hn::MulAdd(hn::Set(d, a4[k]), b0, acc4);
                        acc5 = hn::MulAdd(hn::Set(d, a5[k]), b0, acc5);
                        acc6 = hn::MulAdd(hn::Set(d, a6[k]), b0, acc6);
                        acc7 = hn::MulAdd(hn::Set(d, a7[k]), b0, acc7);
                    }
                    hn::StoreU(acc0, d, c0 + n0);
                    hn::StoreU(acc1, d, c1 + n0);
                    hn::StoreU(acc2, d, c2 + n0);
                    hn::StoreU(acc3, d, c3 + n0);
                    hn::StoreU(acc4, d, c4 + n0);
                    hn::StoreU(acc5, d, c5 + n0);
                    hn::StoreU(acc6, d, c6 + n0);
                    hn::StoreU(acc7, d, c7 + n0);

                    auto acc0_1 = hn::LoadU(d, bias + n0 + NR);
                    auto acc1_1 = acc0_1;
                    auto acc2_1 = acc0_1;
                    auto acc3_1 = acc0_1;
                    auto acc4_1 = acc0_1;
                    auto acc5_1 = acc0_1;
                    auto acc6_1 = acc0_1;
                    auto acc7_1 = acc0_1;
                    for (int k = 0; k < K; k++) {
                        const auto b1 = hn::LoadU(d, bp1 + (size_t)k * NR);
                        acc0_1 = hn::MulAdd(hn::Set(d, a0[k]), b1, acc0_1);
                        acc1_1 = hn::MulAdd(hn::Set(d, a1[k]), b1, acc1_1);
                        acc2_1 = hn::MulAdd(hn::Set(d, a2[k]), b1, acc2_1);
                        acc3_1 = hn::MulAdd(hn::Set(d, a3[k]), b1, acc3_1);
                        acc4_1 = hn::MulAdd(hn::Set(d, a4[k]), b1, acc4_1);
                        acc5_1 = hn::MulAdd(hn::Set(d, a5[k]), b1, acc5_1);
                        acc6_1 = hn::MulAdd(hn::Set(d, a6[k]), b1, acc6_1);
                        acc7_1 = hn::MulAdd(hn::Set(d, a7[k]), b1, acc7_1);
                    }
                    hn::StoreU(acc0_1, d, c0 + n0 + NR);
                    hn::StoreU(acc1_1, d, c1 + n0 + NR);
                    hn::StoreU(acc2_1, d, c2 + n0 + NR);
                    hn::StoreU(acc3_1, d, c3 + n0 + NR);
                    hn::StoreU(acc4_1, d, c4 + n0 + NR);
                    hn::StoreU(acc5_1, d, c5 + n0 + NR);
                    hn::StoreU(acc6_1, d, c6 + n0 + NR);
                    hn::StoreU(acc7_1, d, c7 + n0 + NR);
                }
                if (p < n_panels_exact) {
                    const int n0 = p * NR;
                    const float* bp = B_packed + (size_t)p * K * NR;
                    auto acc0 = hn::LoadU(d, bias + n0);
                    auto acc1 = acc0;
                    auto acc2 = acc0;
                    auto acc3 = acc0;
                    auto acc4 = acc0;
                    auto acc5 = acc0;
                    auto acc6 = acc0;
                    auto acc7 = acc0;
                    for (int k = 0; k < K; k++) {
                        const auto b = hn::LoadU(d, bp + (size_t)k * NR);
                        acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                        acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                        acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                        acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                        acc4 = hn::MulAdd(hn::Set(d, a4[k]), b, acc4);
                        acc5 = hn::MulAdd(hn::Set(d, a5[k]), b, acc5);
                        acc6 = hn::MulAdd(hn::Set(d, a6[k]), b, acc6);
                        acc7 = hn::MulAdd(hn::Set(d, a7[k]), b, acc7);
                    }
                    hn::StoreU(acc0, d, c0 + n0);
                    hn::StoreU(acc1, d, c1 + n0);
                    hn::StoreU(acc2, d, c2 + n0);
                    hn::StoreU(acc3, d, c3 + n0);
                    hn::StoreU(acc4, d, c4 + n0);
                    hn::StoreU(acc5, d, c5 + n0);
                    hn::StoreU(acc6, d, c6 + n0);
                    hn::StoreU(acc7, d, c7 + n0);
                }
            }
        }
        for (; m + 4 <= M; m += 4) {
            const float* a0 = A + (size_t)(m + 0) * K;
            const float* a1 = A + (size_t)(m + 1) * K;
            const float* a2 = A + (size_t)(m + 2) * K;
            const float* a3 = A + (size_t)(m + 3) * K;
            float* c0 = C + (size_t)(m + 0) * N;
            float* c1 = C + (size_t)(m + 1) * N;
            float* c2 = C + (size_t)(m + 2) * N;
            float* c3 = C + (size_t)(m + 3) * N;
            for (int p = 0; p < n_panels_exact; p++) {
                const int n0 = p * NR;
                const float* bp = B_packed + (size_t)p * K * NR;
                auto acc0 = hn::LoadU(d, bias + n0);
                auto acc1 = acc0;
                auto acc2 = acc0;
                auto acc3 = acc0;
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                }
                hn::StoreU(acc0, d, c0 + n0);
                hn::StoreU(acc1, d, c1 + n0);
                hn::StoreU(acc2, d, c2 + n0);
                hn::StoreU(acc3, d, c3 + n0);
            }
        }
        for (; m < M; m++) {
            const float* ar = A + (size_t)m * K;
            float* cr = C + (size_t)m * N;
            for (int p = 0; p < n_panels_exact; p++) {
                const int n0 = p * NR;
                const float* bp = B_packed + (size_t)p * K * NR;
                auto acc = hn::LoadU(d, bias + n0);
                for (int k = 0; k < K; k++) {
                    acc = hn::MulAdd(hn::Set(d, ar[k]), hn::LoadU(d, bp + (size_t)k * NR), acc);
                }
                hn::StoreU(acc, d, cr + n0);
            }
        }
        return;
    }
    if (M == 1) {
        for (int p = 0; p < n_panels; p++) {
            const int n0 = p * NR;
            const int nr = (n0 + NR <= N) ? NR : (N - n0);
            const float* bp = B_packed + (size_t)p * K * NR;
            int j = 0;
            for (; j + lanes <= nr; j += lanes) {
                auto acc = hn::LoadU(d, bias + n0 + j);
                for (int k = 0; k < K; k++) {
                    acc = hn::MulAdd(hn::Set(d, A[k]), hn::LoadU(d, bp + (size_t)k * NR + j), acc);
                }
                hn::StoreU(acc, d, C + n0 + j);
            }
            if (j < nr) {
                const size_t count = static_cast<size_t>(nr - j);
                auto acc = hn::LoadNOr(hn::Zero(d), d, bias + n0 + j, count);
                for (int k = 0; k < K; k++) {
                    acc = hn::MulAdd(hn::Set(d, A[k]), hn::LoadU(d, bp + (size_t)k * NR + j), acc);
                }
                hn::StoreN(acc, d, C + n0 + j, count);
            }
        }
        return;
    }
    int m = 0;
    if (prefer_mr16(M, K, N, lanes)) {
    for (; m + 16 <= M; m += 16) {
        for (int p = 0; p < n_panels; p++) {
            const int n0 = p * NR;
            const int nr = (n0 + NR <= N) ? NR : (N - n0);
            const float* bp = B_packed + (size_t)p * K * NR;
            int j = 0;
            for (; j + lanes <= nr; j += lanes) {
                using V = decltype(hn::Zero(d));
                V acc[16];
                acc[0] = hn::LoadU(d, bias + n0 + j);
                for (int r = 1; r < 16; r++) acc[r] = acc[0];
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR + j);
                    for (int r = 0; r < 16; r++) {
                        acc[r] = hn::MulAdd(hn::Set(d, A[(size_t)(m + r) * K + k]), b, acc[r]);
                    }
                }
                for (int r = 0; r < 16; r++) {
                    hn::StoreU(acc[r], d, C + (size_t)(m + r) * N + n0 + j);
                }
            }
            if (j < nr) {
                const size_t count = static_cast<size_t>(nr - j);
                using V = decltype(hn::Zero(d));
                V acc[16];
                acc[0] = hn::LoadNOr(hn::Zero(d), d, bias + n0 + j, count);
                for (int r = 1; r < 16; r++) acc[r] = acc[0];
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR + j);
                    for (int r = 0; r < 16; r++) {
                        acc[r] = hn::MulAdd(hn::Set(d, A[(size_t)(m + r) * K + k]), b, acc[r]);
                    }
                }
                for (int r = 0; r < 16; r++) {
                    hn::StoreN(acc[r], d, C + (size_t)(m + r) * N + n0 + j, count);
                }
            }
        }
    }
    }
    if (prefer_mr8(M, K, N, lanes)) {
    for (; m + 8 <= M; m += 8) {
        const float* a0 = A + (size_t)(m + 0) * K;
        const float* a1 = A + (size_t)(m + 1) * K;
        const float* a2 = A + (size_t)(m + 2) * K;
        const float* a3 = A + (size_t)(m + 3) * K;
        const float* a4 = A + (size_t)(m + 4) * K;
        const float* a5 = A + (size_t)(m + 5) * K;
        const float* a6 = A + (size_t)(m + 6) * K;
        const float* a7 = A + (size_t)(m + 7) * K;
        float* c0 = C + (size_t)(m + 0) * N;
        float* c1 = C + (size_t)(m + 1) * N;
        float* c2 = C + (size_t)(m + 2) * N;
        float* c3 = C + (size_t)(m + 3) * N;
        float* c4 = C + (size_t)(m + 4) * N;
        float* c5 = C + (size_t)(m + 5) * N;
        float* c6 = C + (size_t)(m + 6) * N;
        float* c7 = C + (size_t)(m + 7) * N;
        for (int p = 0; p < n_panels; p++) {
            const int n0 = p * NR;
            const int nr = (n0 + NR <= N) ? NR : (N - n0);
            const float* bp = B_packed + (size_t)p * K * NR;
            int j = 0;
            for (; j + lanes <= nr; j += lanes) {
                auto acc0 = hn::LoadU(d, bias + n0 + j);
                auto acc1 = acc0;
                auto acc2 = acc0;
                auto acc3 = acc0;
                auto acc4 = acc0;
                auto acc5 = acc0;
                auto acc6 = acc0;
                auto acc7 = acc0;
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR + j);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                    acc4 = hn::MulAdd(hn::Set(d, a4[k]), b, acc4);
                    acc5 = hn::MulAdd(hn::Set(d, a5[k]), b, acc5);
                    acc6 = hn::MulAdd(hn::Set(d, a6[k]), b, acc6);
                    acc7 = hn::MulAdd(hn::Set(d, a7[k]), b, acc7);
                }
                hn::StoreU(acc0, d, c0 + n0 + j);
                hn::StoreU(acc1, d, c1 + n0 + j);
                hn::StoreU(acc2, d, c2 + n0 + j);
                hn::StoreU(acc3, d, c3 + n0 + j);
                hn::StoreU(acc4, d, c4 + n0 + j);
                hn::StoreU(acc5, d, c5 + n0 + j);
                hn::StoreU(acc6, d, c6 + n0 + j);
                hn::StoreU(acc7, d, c7 + n0 + j);
            }
            if (j < nr) {
                const size_t count = static_cast<size_t>(nr - j);
                auto acc0 = hn::LoadNOr(hn::Zero(d), d, bias + n0 + j, count);
                auto acc1 = acc0;
                auto acc2 = acc0;
                auto acc3 = acc0;
                auto acc4 = acc0;
                auto acc5 = acc0;
                auto acc6 = acc0;
                auto acc7 = acc0;
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR + j);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                    acc4 = hn::MulAdd(hn::Set(d, a4[k]), b, acc4);
                    acc5 = hn::MulAdd(hn::Set(d, a5[k]), b, acc5);
                    acc6 = hn::MulAdd(hn::Set(d, a6[k]), b, acc6);
                    acc7 = hn::MulAdd(hn::Set(d, a7[k]), b, acc7);
                }
                hn::StoreN(acc0, d, c0 + n0 + j, count);
                hn::StoreN(acc1, d, c1 + n0 + j, count);
                hn::StoreN(acc2, d, c2 + n0 + j, count);
                hn::StoreN(acc3, d, c3 + n0 + j, count);
                hn::StoreN(acc4, d, c4 + n0 + j, count);
                hn::StoreN(acc5, d, c5 + n0 + j, count);
                hn::StoreN(acc6, d, c6 + n0 + j, count);
                hn::StoreN(acc7, d, c7 + n0 + j, count);
            }
        }
    }
    }
    for (; m + 4 <= M; m += 4) {
        const float* a0 = A + (size_t)(m + 0) * K;
        const float* a1 = A + (size_t)(m + 1) * K;
        const float* a2 = A + (size_t)(m + 2) * K;
        const float* a3 = A + (size_t)(m + 3) * K;
        float* c0 = C + (size_t)(m + 0) * N;
        float* c1 = C + (size_t)(m + 1) * N;
        float* c2 = C + (size_t)(m + 2) * N;
        float* c3 = C + (size_t)(m + 3) * N;
        for (int p = 0; p < n_panels; p++) {
            const int n0 = p * NR;
            const int nr = (n0 + NR <= N) ? NR : (N - n0);
            const float* bp = B_packed + (size_t)p * K * NR;
            int j = 0;
            for (; j + lanes <= nr; j += lanes) {
                auto acc0 = hn::LoadU(d, bias + n0 + j);
                auto acc1 = acc0;
                auto acc2 = acc0;
                auto acc3 = acc0;
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR + j);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                }
                hn::StoreU(acc0, d, c0 + n0 + j);
                hn::StoreU(acc1, d, c1 + n0 + j);
                hn::StoreU(acc2, d, c2 + n0 + j);
                hn::StoreU(acc3, d, c3 + n0 + j);
            }
            if (j < nr) {
                const size_t count = static_cast<size_t>(nr - j);
                auto acc0 = hn::LoadNOr(hn::Zero(d), d, bias + n0 + j, count);
                auto acc1 = acc0;
                auto acc2 = acc0;
                auto acc3 = acc0;
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR + j);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                }
                hn::StoreN(acc0, d, c0 + n0 + j, count);
                hn::StoreN(acc1, d, c1 + n0 + j, count);
                hn::StoreN(acc2, d, c2 + n0 + j, count);
                hn::StoreN(acc3, d, c3 + n0 + j, count);
            }
        }
    }
    for (; m < M; m++) {
        const float* ar = A + (size_t)m * K;
        float* cr = C + (size_t)m * N;
        for (int p = 0; p < n_panels; p++) {
            const int n0 = p * NR;
            const int nr = (n0 + NR <= N) ? NR : (N - n0);
            const float* bp = B_packed + (size_t)p * K * NR;
            int j = 0;
            for (; j + lanes <= nr; j += lanes) {
                auto acc = hn::LoadU(d, bias + n0 + j);
                for (int k = 0; k < K; k++) {
                    acc = hn::MulAdd(hn::Set(d, ar[k]), hn::LoadU(d, bp + (size_t)k * NR + j), acc);
                }
                hn::StoreU(acc, d, cr + n0 + j);
            }
            if (j < nr) {
                const size_t count = static_cast<size_t>(nr - j);
                auto acc = hn::LoadNOr(hn::Zero(d), d, bias + n0 + j, count);
                for (int k = 0; k < K; k++) {
                    acc = hn::MulAdd(hn::Set(d, ar[k]), hn::LoadU(d, bp + (size_t)k * NR + j), acc);
                }
                hn::StoreN(acc, d, cr + n0 + j, count);
            }
        }
    }
}

void matmul_bias_gelu_packed_dispatch(const float* A, const float* B_packed, const float* bias,
                                      float* C,
                                      int M, int K, int N) {
    if (!bias) {
        matmul_fp32_packed_dispatch(A, B_packed, C, M, K, N);
        gelu_dispatch(C, M * N);
        return;
    }
    constexpr int NR = 16;
    const int n_panels = (N + NR - 1) / NR;
    const hn::CappedTag<float, NR> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    const auto v_c1 = hn::Set(d, 0.7978845608f);
    const auto v_c2 = hn::Set(d, 0.044715f);
    const auto v_half = hn::Set(d, 0.5f);
    const auto v_one = hn::Set(d, 1.0f);
    const auto v_neg_one = hn::Set(d, -1.0f);
    const auto v_27 = hn::Set(d, 27.0f);
    const auto v_9 = hn::Set(d, 9.0f);
    if (lanes == NR && ((N & (NR - 1)) == 0)) {
        const int n_panels_exact = N / NR;
        int m = 0;
        if (prefer_mr16(M, K, N, lanes)) {
        for (; m + 16 <= M; m += 16) {
            const float* a0 = A + (size_t)(m + 0) * K;
            const float* a1 = A + (size_t)(m + 1) * K;
            const float* a2 = A + (size_t)(m + 2) * K;
            const float* a3 = A + (size_t)(m + 3) * K;
            const float* a4 = A + (size_t)(m + 4) * K;
            const float* a5 = A + (size_t)(m + 5) * K;
            const float* a6 = A + (size_t)(m + 6) * K;
            const float* a7 = A + (size_t)(m + 7) * K;
            const float* a8 = A + (size_t)(m + 8) * K;
            const float* a9 = A + (size_t)(m + 9) * K;
            const float* a10 = A + (size_t)(m + 10) * K;
            const float* a11 = A + (size_t)(m + 11) * K;
            const float* a12 = A + (size_t)(m + 12) * K;
            const float* a13 = A + (size_t)(m + 13) * K;
            const float* a14 = A + (size_t)(m + 14) * K;
            const float* a15 = A + (size_t)(m + 15) * K;
            float* c0 = C + (size_t)(m + 0) * N;
            float* c1 = C + (size_t)(m + 1) * N;
            float* c2 = C + (size_t)(m + 2) * N;
            float* c3 = C + (size_t)(m + 3) * N;
            float* c4 = C + (size_t)(m + 4) * N;
            float* c5 = C + (size_t)(m + 5) * N;
            float* c6 = C + (size_t)(m + 6) * N;
            float* c7 = C + (size_t)(m + 7) * N;
            float* c8 = C + (size_t)(m + 8) * N;
            float* c9 = C + (size_t)(m + 9) * N;
            float* c10 = C + (size_t)(m + 10) * N;
            float* c11 = C + (size_t)(m + 11) * N;
            float* c12 = C + (size_t)(m + 12) * N;
            float* c13 = C + (size_t)(m + 13) * N;
            float* c14 = C + (size_t)(m + 14) * N;
            float* c15 = C + (size_t)(m + 15) * N;
            for (int p = 0; p < n_panels_exact; p++) {
                const int n0 = p * NR;
                const float* bp = B_packed + (size_t)p * K * NR;
                auto acc0 = hn::LoadU(d, bias + n0);
                auto acc1 = acc0, acc2 = acc0, acc3 = acc0;
                auto acc4 = acc0, acc5 = acc0, acc6 = acc0, acc7 = acc0;
                auto acc8 = acc0, acc9 = acc0, acc10 = acc0, acc11 = acc0;
                auto acc12 = acc0, acc13 = acc0, acc14 = acc0, acc15 = acc0;
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                    acc4 = hn::MulAdd(hn::Set(d, a4[k]), b, acc4);
                    acc5 = hn::MulAdd(hn::Set(d, a5[k]), b, acc5);
                    acc6 = hn::MulAdd(hn::Set(d, a6[k]), b, acc6);
                    acc7 = hn::MulAdd(hn::Set(d, a7[k]), b, acc7);
                    acc8 = hn::MulAdd(hn::Set(d, a8[k]), b, acc8);
                    acc9 = hn::MulAdd(hn::Set(d, a9[k]), b, acc9);
                    acc10 = hn::MulAdd(hn::Set(d, a10[k]), b, acc10);
                    acc11 = hn::MulAdd(hn::Set(d, a11[k]), b, acc11);
                    acc12 = hn::MulAdd(hn::Set(d, a12[k]), b, acc12);
                    acc13 = hn::MulAdd(hn::Set(d, a13[k]), b, acc13);
                    acc14 = hn::MulAdd(hn::Set(d, a14[k]), b, acc14);
                    acc15 = hn::MulAdd(hn::Set(d, a15[k]), b, acc15);
                }
#define FACEX_GELU_VEC(v) \
                do { \
                    const auto vx3 = hn::Mul(hn::Mul((v), (v)), (v)); \
                    const auto inner = hn::Mul(v_c1, hn::MulAdd(v_c2, vx3, (v))); \
                    const auto inner2 = hn::Mul(inner, inner); \
                    const auto num = hn::Mul(inner, hn::Add(v_27, inner2)); \
                    const auto den = hn::MulAdd(v_9, inner2, v_27); \
                    auto t = hn::Mul(num, hn::ApproximateReciprocal(den)); \
                    t = hn::Max(hn::Min(t, v_one), v_neg_one); \
                    (v) = hn::Mul(hn::Mul(v_half, (v)), hn::Add(v_one, t)); \
                } while (0)
                FACEX_GELU_VEC(acc0); FACEX_GELU_VEC(acc1);
                FACEX_GELU_VEC(acc2); FACEX_GELU_VEC(acc3);
                FACEX_GELU_VEC(acc4); FACEX_GELU_VEC(acc5);
                FACEX_GELU_VEC(acc6); FACEX_GELU_VEC(acc7);
                FACEX_GELU_VEC(acc8); FACEX_GELU_VEC(acc9);
                FACEX_GELU_VEC(acc10); FACEX_GELU_VEC(acc11);
                FACEX_GELU_VEC(acc12); FACEX_GELU_VEC(acc13);
                FACEX_GELU_VEC(acc14); FACEX_GELU_VEC(acc15);
#undef FACEX_GELU_VEC
                hn::StoreU(acc0, d, c0 + n0);
                hn::StoreU(acc1, d, c1 + n0);
                hn::StoreU(acc2, d, c2 + n0);
                hn::StoreU(acc3, d, c3 + n0);
                hn::StoreU(acc4, d, c4 + n0);
                hn::StoreU(acc5, d, c5 + n0);
                hn::StoreU(acc6, d, c6 + n0);
                hn::StoreU(acc7, d, c7 + n0);
                hn::StoreU(acc8, d, c8 + n0);
                hn::StoreU(acc9, d, c9 + n0);
                hn::StoreU(acc10, d, c10 + n0);
                hn::StoreU(acc11, d, c11 + n0);
                hn::StoreU(acc12, d, c12 + n0);
                hn::StoreU(acc13, d, c13 + n0);
                hn::StoreU(acc14, d, c14 + n0);
                hn::StoreU(acc15, d, c15 + n0);
            }
        }
        }
        if (prefer_mr8(M, K, N, lanes)) {
        for (; m + 8 <= M; m += 8) {
            const float* a0 = A + (size_t)(m + 0) * K;
            const float* a1 = A + (size_t)(m + 1) * K;
            const float* a2 = A + (size_t)(m + 2) * K;
            const float* a3 = A + (size_t)(m + 3) * K;
            const float* a4 = A + (size_t)(m + 4) * K;
            const float* a5 = A + (size_t)(m + 5) * K;
            const float* a6 = A + (size_t)(m + 6) * K;
            const float* a7 = A + (size_t)(m + 7) * K;
            float* c0 = C + (size_t)(m + 0) * N;
            float* c1 = C + (size_t)(m + 1) * N;
            float* c2 = C + (size_t)(m + 2) * N;
            float* c3 = C + (size_t)(m + 3) * N;
        float* c4 = C + (size_t)(m + 4) * N;
        float* c5 = C + (size_t)(m + 5) * N;
        float* c6 = C + (size_t)(m + 6) * N;
        float* c7 = C + (size_t)(m + 7) * N;
        int p = 0;
        for (; p + 1 < n_panels_exact; p += 2) {
            const int n0 = p * NR;
            const float* bp0 = B_packed + (size_t)p * K * NR;
            const float* bp1 = bp0 + (size_t)K * NR;
            auto acc0 = hn::LoadU(d, bias + n0);
            auto acc1 = acc0, acc2 = acc0, acc3 = acc0;
            auto acc4 = acc0, acc5 = acc0, acc6 = acc0, acc7 = acc0;
            auto acc8 = hn::LoadU(d, bias + n0 + NR);
            auto acc9 = acc8, acc10 = acc8, acc11 = acc8;
            auto acc12 = acc8, acc13 = acc8, acc14 = acc8, acc15 = acc8;
            for (int k = 0; k < K; k++) {
                const auto b0 = hn::LoadU(d, bp0 + (size_t)k * NR);
                const auto b1 = hn::LoadU(d, bp1 + (size_t)k * NR);
                acc0 = hn::MulAdd(hn::Set(d, a0[k]), b0, acc0);
                acc1 = hn::MulAdd(hn::Set(d, a1[k]), b0, acc1);
                acc2 = hn::MulAdd(hn::Set(d, a2[k]), b0, acc2);
                acc3 = hn::MulAdd(hn::Set(d, a3[k]), b0, acc3);
                acc4 = hn::MulAdd(hn::Set(d, a4[k]), b0, acc4);
                acc5 = hn::MulAdd(hn::Set(d, a5[k]), b0, acc5);
                acc6 = hn::MulAdd(hn::Set(d, a6[k]), b0, acc6);
                acc7 = hn::MulAdd(hn::Set(d, a7[k]), b0, acc7);
                acc8 = hn::MulAdd(hn::Set(d, a0[k]), b1, acc8);
                acc9 = hn::MulAdd(hn::Set(d, a1[k]), b1, acc9);
                acc10 = hn::MulAdd(hn::Set(d, a2[k]), b1, acc10);
                acc11 = hn::MulAdd(hn::Set(d, a3[k]), b1, acc11);
                acc12 = hn::MulAdd(hn::Set(d, a4[k]), b1, acc12);
                acc13 = hn::MulAdd(hn::Set(d, a5[k]), b1, acc13);
                acc14 = hn::MulAdd(hn::Set(d, a6[k]), b1, acc14);
                acc15 = hn::MulAdd(hn::Set(d, a7[k]), b1, acc15);
            }
#define FACEX_GELU_VEC(v) \
            do { \
                const auto vx3 = hn::Mul(hn::Mul((v), (v)), (v)); \
                const auto inner = hn::Mul(v_c1, hn::MulAdd(v_c2, vx3, (v))); \
                const auto inner2 = hn::Mul(inner, inner); \
                const auto num = hn::Mul(inner, hn::Add(v_27, inner2)); \
                const auto den = hn::MulAdd(v_9, inner2, v_27); \
                auto t = hn::Mul(num, hn::ApproximateReciprocal(den)); \
                t = hn::Max(hn::Min(t, v_one), v_neg_one); \
                (v) = hn::Mul(hn::Mul(v_half, (v)), hn::Add(v_one, t)); \
            } while (0)
            FACEX_GELU_VEC(acc0); FACEX_GELU_VEC(acc1);
            FACEX_GELU_VEC(acc2); FACEX_GELU_VEC(acc3);
            FACEX_GELU_VEC(acc4); FACEX_GELU_VEC(acc5);
            FACEX_GELU_VEC(acc6); FACEX_GELU_VEC(acc7);
            FACEX_GELU_VEC(acc8); FACEX_GELU_VEC(acc9);
            FACEX_GELU_VEC(acc10); FACEX_GELU_VEC(acc11);
            FACEX_GELU_VEC(acc12); FACEX_GELU_VEC(acc13);
            FACEX_GELU_VEC(acc14); FACEX_GELU_VEC(acc15);
#undef FACEX_GELU_VEC
            hn::StoreU(acc0, d, c0 + n0);
            hn::StoreU(acc1, d, c1 + n0);
            hn::StoreU(acc2, d, c2 + n0);
            hn::StoreU(acc3, d, c3 + n0);
            hn::StoreU(acc4, d, c4 + n0);
            hn::StoreU(acc5, d, c5 + n0);
            hn::StoreU(acc6, d, c6 + n0);
            hn::StoreU(acc7, d, c7 + n0);
            hn::StoreU(acc8, d, c0 + n0 + NR);
            hn::StoreU(acc9, d, c1 + n0 + NR);
            hn::StoreU(acc10, d, c2 + n0 + NR);
            hn::StoreU(acc11, d, c3 + n0 + NR);
            hn::StoreU(acc12, d, c4 + n0 + NR);
            hn::StoreU(acc13, d, c5 + n0 + NR);
            hn::StoreU(acc14, d, c6 + n0 + NR);
            hn::StoreU(acc15, d, c7 + n0 + NR);
        }
        for (; p < n_panels_exact; p++) {
            const int n0 = p * NR;
            const float* bp = B_packed + (size_t)p * K * NR;
            auto acc0 = hn::LoadU(d, bias + n0);
            auto acc1 = acc0, acc2 = acc0, acc3 = acc0;
            auto acc4 = acc0, acc5 = acc0, acc6 = acc0, acc7 = acc0;
            for (int k = 0; k < K; k++) {
                const auto b = hn::LoadU(d, bp + (size_t)k * NR);
                acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                acc4 = hn::MulAdd(hn::Set(d, a4[k]), b, acc4);
                acc5 = hn::MulAdd(hn::Set(d, a5[k]), b, acc5);
                acc6 = hn::MulAdd(hn::Set(d, a6[k]), b, acc6);
                acc7 = hn::MulAdd(hn::Set(d, a7[k]), b, acc7);
            }
#define FACEX_GELU_VEC(v) \
            do { \
                const auto vx3 = hn::Mul(hn::Mul((v), (v)), (v)); \
                const auto inner = hn::Mul(v_c1, hn::MulAdd(v_c2, vx3, (v))); \
                const auto inner2 = hn::Mul(inner, inner); \
                const auto num = hn::Mul(inner, hn::Add(v_27, inner2)); \
                const auto den = hn::MulAdd(v_9, inner2, v_27); \
                auto t = hn::Mul(num, hn::ApproximateReciprocal(den)); \
                t = hn::Max(hn::Min(t, v_one), v_neg_one); \
                (v) = hn::Mul(hn::Mul(v_half, (v)), hn::Add(v_one, t)); \
            } while (0)
            FACEX_GELU_VEC(acc0); FACEX_GELU_VEC(acc1);
            FACEX_GELU_VEC(acc2); FACEX_GELU_VEC(acc3);
            FACEX_GELU_VEC(acc4); FACEX_GELU_VEC(acc5);
            FACEX_GELU_VEC(acc6); FACEX_GELU_VEC(acc7);
#undef FACEX_GELU_VEC
            hn::StoreU(acc0, d, c0 + n0);
            hn::StoreU(acc1, d, c1 + n0);
            hn::StoreU(acc2, d, c2 + n0);
            hn::StoreU(acc3, d, c3 + n0);
            hn::StoreU(acc4, d, c4 + n0);
            hn::StoreU(acc5, d, c5 + n0);
            hn::StoreU(acc6, d, c6 + n0);
            hn::StoreU(acc7, d, c7 + n0);
        }
        }
        }
        for (; m + 4 <= M; m += 4) {
            const float* a0 = A + (size_t)(m + 0) * K;
            const float* a1 = A + (size_t)(m + 1) * K;
            const float* a2 = A + (size_t)(m + 2) * K;
            const float* a3 = A + (size_t)(m + 3) * K;
            float* c0 = C + (size_t)(m + 0) * N;
            float* c1 = C + (size_t)(m + 1) * N;
            float* c2 = C + (size_t)(m + 2) * N;
            float* c3 = C + (size_t)(m + 3) * N;
            for (int p = 0; p < n_panels_exact; p++) {
                const int n0 = p * NR;
                const float* bp = B_packed + (size_t)p * K * NR;
                auto acc0 = hn::LoadU(d, bias + n0);
                auto acc1 = acc0, acc2 = acc0, acc3 = acc0;
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                }
#define FACEX_GELU_VEC(v) \
                do { \
                    const auto vx3 = hn::Mul(hn::Mul((v), (v)), (v)); \
                    const auto inner = hn::Mul(v_c1, hn::MulAdd(v_c2, vx3, (v))); \
                    const auto inner2 = hn::Mul(inner, inner); \
                    const auto num = hn::Mul(inner, hn::Add(v_27, inner2)); \
                    const auto den = hn::MulAdd(v_9, inner2, v_27); \
                    auto t = hn::Mul(num, hn::ApproximateReciprocal(den)); \
                    t = hn::Max(hn::Min(t, v_one), v_neg_one); \
                    (v) = hn::Mul(hn::Mul(v_half, (v)), hn::Add(v_one, t)); \
                } while (0)
                FACEX_GELU_VEC(acc0); FACEX_GELU_VEC(acc1);
                FACEX_GELU_VEC(acc2); FACEX_GELU_VEC(acc3);
#undef FACEX_GELU_VEC
                hn::StoreU(acc0, d, c0 + n0);
                hn::StoreU(acc1, d, c1 + n0);
                hn::StoreU(acc2, d, c2 + n0);
                hn::StoreU(acc3, d, c3 + n0);
            }
        }
        for (; m < M; m++) {
            const float* ar = A + (size_t)m * K;
            float* cr = C + (size_t)m * N;
            for (int p = 0; p < n_panels_exact; p++) {
                const int n0 = p * NR;
                const float* bp = B_packed + (size_t)p * K * NR;
                auto v = hn::LoadU(d, bias + n0);
                for (int k = 0; k < K; k++) {
                    v = hn::MulAdd(hn::Set(d, ar[k]), hn::LoadU(d, bp + (size_t)k * NR), v);
                }
                const auto vx3 = hn::Mul(hn::Mul(v, v), v);
                const auto inner = hn::Mul(v_c1, hn::MulAdd(v_c2, vx3, v));
                const auto inner2 = hn::Mul(inner, inner);
                const auto num = hn::Mul(inner, hn::Add(v_27, inner2));
                const auto den = hn::MulAdd(v_9, inner2, v_27);
                auto t = hn::Mul(num, hn::ApproximateReciprocal(den));
                t = hn::Max(hn::Min(t, v_one), v_neg_one);
                hn::StoreU(hn::Mul(hn::Mul(v_half, v), hn::Add(v_one, t)), d, cr + n0);
            }
        }
        return;
    }
    int m = 0;
    if (prefer_mr8(M, K, N, lanes)) {
    for (; m + 8 <= M; m += 8) {
        const float* a0 = A + (size_t)(m + 0) * K;
        const float* a1 = A + (size_t)(m + 1) * K;
        const float* a2 = A + (size_t)(m + 2) * K;
        const float* a3 = A + (size_t)(m + 3) * K;
        const float* a4 = A + (size_t)(m + 4) * K;
        const float* a5 = A + (size_t)(m + 5) * K;
        const float* a6 = A + (size_t)(m + 6) * K;
        const float* a7 = A + (size_t)(m + 7) * K;
        float* c0 = C + (size_t)(m + 0) * N;
        float* c1 = C + (size_t)(m + 1) * N;
        float* c2 = C + (size_t)(m + 2) * N;
        float* c3 = C + (size_t)(m + 3) * N;
        float* c4 = C + (size_t)(m + 4) * N;
        float* c5 = C + (size_t)(m + 5) * N;
        float* c6 = C + (size_t)(m + 6) * N;
        float* c7 = C + (size_t)(m + 7) * N;
        for (int p = 0; p < n_panels; p++) {
            const int n0 = p * NR;
            const int nr = (n0 + NR <= N) ? NR : (N - n0);
            const float* bp = B_packed + (size_t)p * K * NR;
            int j = 0;
            for (; j + lanes <= nr; j += lanes) {
                auto acc0 = hn::LoadU(d, bias + n0 + j);
                auto acc1 = acc0;
                auto acc2 = acc0;
                auto acc3 = acc0;
                auto acc4 = acc0;
                auto acc5 = acc0;
                auto acc6 = acc0;
                auto acc7 = acc0;
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR + j);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                    acc4 = hn::MulAdd(hn::Set(d, a4[k]), b, acc4);
                    acc5 = hn::MulAdd(hn::Set(d, a5[k]), b, acc5);
                    acc6 = hn::MulAdd(hn::Set(d, a6[k]), b, acc6);
                    acc7 = hn::MulAdd(hn::Set(d, a7[k]), b, acc7);
                }
#define FACEX_GELU_VEC(v) \
                do { \
                    const auto vx3 = hn::Mul(hn::Mul((v), (v)), (v)); \
                    const auto inner = hn::Mul(v_c1, hn::MulAdd(v_c2, vx3, (v))); \
                    const auto inner2 = hn::Mul(inner, inner); \
                    const auto num = hn::Mul(inner, hn::Add(v_27, inner2)); \
                    const auto den = hn::MulAdd(v_9, inner2, v_27); \
                    auto t = hn::Mul(num, hn::ApproximateReciprocal(den)); \
                    t = hn::Max(hn::Min(t, v_one), v_neg_one); \
                    (v) = hn::Mul(hn::Mul(v_half, (v)), hn::Add(v_one, t)); \
                } while (0)
                FACEX_GELU_VEC(acc0);
                FACEX_GELU_VEC(acc1);
                FACEX_GELU_VEC(acc2);
                FACEX_GELU_VEC(acc3);
                FACEX_GELU_VEC(acc4);
                FACEX_GELU_VEC(acc5);
                FACEX_GELU_VEC(acc6);
                FACEX_GELU_VEC(acc7);
#undef FACEX_GELU_VEC
                hn::StoreU(acc0, d, c0 + n0 + j);
                hn::StoreU(acc1, d, c1 + n0 + j);
                hn::StoreU(acc2, d, c2 + n0 + j);
                hn::StoreU(acc3, d, c3 + n0 + j);
                hn::StoreU(acc4, d, c4 + n0 + j);
                hn::StoreU(acc5, d, c5 + n0 + j);
                hn::StoreU(acc6, d, c6 + n0 + j);
                hn::StoreU(acc7, d, c7 + n0 + j);
            }
            if (j < nr) {
                const size_t count = static_cast<size_t>(nr - j);
                auto acc0 = hn::LoadNOr(hn::Zero(d), d, bias + n0 + j, count);
                auto acc1 = acc0;
                auto acc2 = acc0;
                auto acc3 = acc0;
                auto acc4 = acc0;
                auto acc5 = acc0;
                auto acc6 = acc0;
                auto acc7 = acc0;
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR + j);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                    acc4 = hn::MulAdd(hn::Set(d, a4[k]), b, acc4);
                    acc5 = hn::MulAdd(hn::Set(d, a5[k]), b, acc5);
                    acc6 = hn::MulAdd(hn::Set(d, a6[k]), b, acc6);
                    acc7 = hn::MulAdd(hn::Set(d, a7[k]), b, acc7);
                }
#define FACEX_GELU_VEC(v) \
                do { \
                    const auto vx3 = hn::Mul(hn::Mul((v), (v)), (v)); \
                    const auto inner = hn::Mul(v_c1, hn::MulAdd(v_c2, vx3, (v))); \
                    const auto inner2 = hn::Mul(inner, inner); \
                    const auto num = hn::Mul(inner, hn::Add(v_27, inner2)); \
                    const auto den = hn::MulAdd(v_9, inner2, v_27); \
                    auto t = hn::Mul(num, hn::ApproximateReciprocal(den)); \
                    t = hn::Max(hn::Min(t, v_one), v_neg_one); \
                    (v) = hn::Mul(hn::Mul(v_half, (v)), hn::Add(v_one, t)); \
                } while (0)
                FACEX_GELU_VEC(acc0);
                FACEX_GELU_VEC(acc1);
                FACEX_GELU_VEC(acc2);
                FACEX_GELU_VEC(acc3);
                FACEX_GELU_VEC(acc4);
                FACEX_GELU_VEC(acc5);
                FACEX_GELU_VEC(acc6);
                FACEX_GELU_VEC(acc7);
#undef FACEX_GELU_VEC
                hn::StoreN(acc0, d, c0 + n0 + j, count);
                hn::StoreN(acc1, d, c1 + n0 + j, count);
                hn::StoreN(acc2, d, c2 + n0 + j, count);
                hn::StoreN(acc3, d, c3 + n0 + j, count);
                hn::StoreN(acc4, d, c4 + n0 + j, count);
                hn::StoreN(acc5, d, c5 + n0 + j, count);
                hn::StoreN(acc6, d, c6 + n0 + j, count);
                hn::StoreN(acc7, d, c7 + n0 + j, count);
            }
        }
    }
    }
    for (; m + 4 <= M; m += 4) {
        const float* a0 = A + (size_t)(m + 0) * K;
        const float* a1 = A + (size_t)(m + 1) * K;
        const float* a2 = A + (size_t)(m + 2) * K;
        const float* a3 = A + (size_t)(m + 3) * K;
        float* c0 = C + (size_t)(m + 0) * N;
        float* c1 = C + (size_t)(m + 1) * N;
        float* c2 = C + (size_t)(m + 2) * N;
        float* c3 = C + (size_t)(m + 3) * N;
        for (int p = 0; p < n_panels; p++) {
            const int n0 = p * NR;
            const int nr = (n0 + NR <= N) ? NR : (N - n0);
            const float* bp = B_packed + (size_t)p * K * NR;
            int j = 0;
            for (; j + lanes <= nr; j += lanes) {
                auto acc0 = hn::LoadU(d, bias + n0 + j);
                auto acc1 = acc0;
                auto acc2 = acc0;
                auto acc3 = acc0;
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR + j);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                }
#define FACEX_GELU_VEC(v) \
                do { \
                    const auto vx3 = hn::Mul(hn::Mul((v), (v)), (v)); \
                    const auto inner = hn::Mul(v_c1, hn::MulAdd(v_c2, vx3, (v))); \
                    const auto inner2 = hn::Mul(inner, inner); \
                    const auto num = hn::Mul(inner, hn::Add(v_27, inner2)); \
                    const auto den = hn::MulAdd(v_9, inner2, v_27); \
                    auto t = hn::Mul(num, hn::ApproximateReciprocal(den)); \
                    t = hn::Max(hn::Min(t, v_one), v_neg_one); \
                    (v) = hn::Mul(hn::Mul(v_half, (v)), hn::Add(v_one, t)); \
                } while (0)
                FACEX_GELU_VEC(acc0);
                FACEX_GELU_VEC(acc1);
                FACEX_GELU_VEC(acc2);
                FACEX_GELU_VEC(acc3);
#undef FACEX_GELU_VEC
                hn::StoreU(acc0, d, c0 + n0 + j);
                hn::StoreU(acc1, d, c1 + n0 + j);
                hn::StoreU(acc2, d, c2 + n0 + j);
                hn::StoreU(acc3, d, c3 + n0 + j);
            }
            if (j < nr) {
                const size_t count = static_cast<size_t>(nr - j);
                auto acc0 = hn::LoadNOr(hn::Zero(d), d, bias + n0 + j, count);
                auto acc1 = acc0;
                auto acc2 = acc0;
                auto acc3 = acc0;
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR + j);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                }
#define FACEX_GELU_VEC(v) \
                do { \
                    const auto vx3 = hn::Mul(hn::Mul((v), (v)), (v)); \
                    const auto inner = hn::Mul(v_c1, hn::MulAdd(v_c2, vx3, (v))); \
                    const auto inner2 = hn::Mul(inner, inner); \
                    const auto num = hn::Mul(inner, hn::Add(v_27, inner2)); \
                    const auto den = hn::MulAdd(v_9, inner2, v_27); \
                    auto t = hn::Mul(num, hn::ApproximateReciprocal(den)); \
                    t = hn::Max(hn::Min(t, v_one), v_neg_one); \
                    (v) = hn::Mul(hn::Mul(v_half, (v)), hn::Add(v_one, t)); \
                } while (0)
                FACEX_GELU_VEC(acc0);
                FACEX_GELU_VEC(acc1);
                FACEX_GELU_VEC(acc2);
                FACEX_GELU_VEC(acc3);
#undef FACEX_GELU_VEC
                hn::StoreN(acc0, d, c0 + n0 + j, count);
                hn::StoreN(acc1, d, c1 + n0 + j, count);
                hn::StoreN(acc2, d, c2 + n0 + j, count);
                hn::StoreN(acc3, d, c3 + n0 + j, count);
            }
        }
    }
    for (; m < M; m++) {
        const float* ar = A + (size_t)m * K;
        float* cr = C + (size_t)m * N;
        for (int p = 0; p < n_panels; p++) {
            const int n0 = p * NR;
            const int nr = (n0 + NR <= N) ? NR : (N - n0);
            const float* bp = B_packed + (size_t)p * K * NR;
            int j = 0;
            for (; j + lanes <= nr; j += lanes) {
                auto v = hn::LoadU(d, bias + n0 + j);
                for (int k = 0; k < K; k++) {
                    v = hn::MulAdd(hn::Set(d, ar[k]), hn::LoadU(d, bp + (size_t)k * NR + j), v);
                }
                const auto vx3 = hn::Mul(hn::Mul(v, v), v);
                const auto inner = hn::Mul(v_c1, hn::MulAdd(v_c2, vx3, v));
                const auto inner2 = hn::Mul(inner, inner);
                const auto num = hn::Mul(inner, hn::Add(v_27, inner2));
                const auto den = hn::MulAdd(v_9, inner2, v_27);
                auto t = hn::Mul(num, hn::ApproximateReciprocal(den));
                t = hn::Max(hn::Min(t, v_one), v_neg_one);
                hn::StoreU(hn::Mul(hn::Mul(v_half, v), hn::Add(v_one, t)), d, cr + n0 + j);
            }
            if (j < nr) {
                const size_t count = static_cast<size_t>(nr - j);
                auto v = hn::LoadNOr(hn::Zero(d), d, bias + n0 + j, count);
                for (int k = 0; k < K; k++) {
                    v = hn::MulAdd(hn::Set(d, ar[k]), hn::LoadU(d, bp + (size_t)k * NR + j), v);
                }
                const auto vx3 = hn::Mul(hn::Mul(v, v), v);
                const auto inner = hn::Mul(v_c1, hn::MulAdd(v_c2, vx3, v));
                const auto inner2 = hn::Mul(inner, inner);
                const auto num = hn::Mul(inner, hn::Add(v_27, inner2));
                const auto den = hn::MulAdd(v_9, inner2, v_27);
                auto t = hn::Mul(num, hn::ApproximateReciprocal(den));
                t = hn::Max(hn::Min(t, v_one), v_neg_one);
                hn::StoreN(hn::Mul(hn::Mul(v_half, v), hn::Add(v_one, t)), d, cr + n0 + j, count);
            }
        }
    }
}

void gelu_dispatch(float* x, int n) {
    int i = 0;
    const hn::ScalableTag<float> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    const auto v_c1 = hn::Set(d, 0.7978845608f);
    const auto v_c2 = hn::Set(d, 0.044715f);
    const auto v_half = hn::Set(d, 0.5f);
    const auto v_one = hn::Set(d, 1.0f);
    const auto v_neg_one = hn::Set(d, -1.0f);
    const auto v_27 = hn::Set(d, 27.0f);
    const auto v_9 = hn::Set(d, 9.0f);
    for (; i + lanes <= n; i += lanes) {
        const auto vx = hn::LoadU(d, x + i);
        const auto vx3 = hn::Mul(hn::Mul(vx, vx), vx);
        const auto inner = hn::Mul(v_c1, hn::MulAdd(v_c2, vx3, vx));
        const auto inner2 = hn::Mul(inner, inner);
        const auto num = hn::Mul(inner, hn::Add(v_27, inner2));
        const auto den = hn::MulAdd(v_9, inner2, v_27);
        auto t = hn::Mul(num, hn::ApproximateReciprocal(den));
        t = hn::Max(hn::Min(t, v_one), v_neg_one);
        hn::StoreU(hn::Mul(hn::Mul(v_half, vx), hn::Add(v_one, t)), d, x + i);
    }
    for (; i < n; i++) {
        float v = x[i];
        float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
        float t = inner * (27.0f + inner * inner) / (27.0f + 9.0f * inner * inner);
        if (t > 1.0f) t = 1.0f;
        if (t < -1.0f) t = -1.0f;
        x[i] = 0.5f * v * (1.0f + t);
    }
}

void l2_normalize_dispatch(float* x, int N, int C, float min_norm) {
    const hn::ScalableTag<float> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    for (int n = 0; n < N; n++) {
        float* row = x + (size_t)n * C;
        int c = 0;
        auto vacc = hn::Zero(d);
        for (; c + lanes <= C; c += lanes) {
            const auto v = hn::LoadU(d, row + c);
            vacc = hn::MulAdd(v, v, vacc);
        }
        float norm_sq = hn::ReduceSum(d, vacc);
        for (; c < C; c++) norm_sq += row[c] * row[c];
        float norm = sqrtf(norm_sq);
        if (norm < min_norm) norm = min_norm;
        const auto vinv = hn::Set(d, 1.0f / norm);
        c = 0;
        for (; c + lanes <= C; c += lanes) {
            hn::StoreU(hn::Mul(hn::LoadU(d, row + c), vinv), d, row + c);
        }
        for (; c < C; c++) row[c] *= 1.0f / norm;
    }
}

void add_fp32_dispatch(float* x, const float* y, int n) {
    const hn::ScalableTag<float> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    int i = 0;
    for (; i + lanes <= n; i += lanes) {
        hn::StoreU(hn::Add(hn::LoadU(d, x + i), hn::LoadU(d, y + i)), d, x + i);
    }
    for (; i < n; i++) x[i] += y[i];
}

void scale_fp32_dispatch(float* x, int n, float scale) {
    const hn::ScalableTag<float> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    const auto vscale = hn::Set(d, scale);
    int i = 0;
    for (; i + lanes <= n; i += lanes) {
        hn::StoreU(hn::Mul(hn::LoadU(d, x + i), vscale), d, x + i);
    }
    for (; i < n; i++) x[i] *= scale;
}

void bias_gamma_residual_fp32_dispatch(const float* src, float* dst,
                                       const float* bias, const float* gamma,
                                       const float* residual, int rows, int cols) {
    const hn::ScalableTag<float> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    if (bias) {
        for (int r = 0; r < rows; r++) {
            const float* srow = src + (size_t)r * cols;
            const float* rrow = residual + (size_t)r * cols;
            float* drow = dst + (size_t)r * cols;
            int c = 0;
            for (; c + lanes <= cols; c += lanes) {
                auto v = hn::Add(hn::LoadU(d, srow + c), hn::LoadU(d, bias + c));
                v = hn::MulAdd(v, hn::LoadU(d, gamma + c), hn::LoadU(d, rrow + c));
                hn::StoreU(v, d, drow + c);
            }
            for (; c < cols; c++) drow[c] = (srow[c] + bias[c]) * gamma[c] + rrow[c];
        }
        return;
    }
    for (int r = 0; r < rows; r++) {
        const float* srow = src + (size_t)r * cols;
        const float* rrow = residual + (size_t)r * cols;
        float* drow = dst + (size_t)r * cols;
        int c = 0;
        for (; c + lanes <= cols; c += lanes) {
            auto v = hn::LoadU(d, srow + c);
            v = hn::MulAdd(v, hn::LoadU(d, gamma + c), hn::LoadU(d, rrow + c));
            hn::StoreU(v, d, drow + c);
        }
        for (; c < cols; c++) {
            drow[c] = srow[c] * gamma[c] + rrow[c];
        }
    }
}

void matmul_residual_bias_gamma_packed_dispatch(const float* A, const float* B_packed,
                                                const float* bias, const float* gamma,
                                                const float* residual, float* C,
                                                int M, int K, int N) {
    constexpr int NR = 16;
    const int n_panels = (N + NR - 1) / NR;
    const hn::CappedTag<float, NR> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    if (lanes == NR && ((N & (NR - 1)) == 0)) {
        const int n_panels_exact = N / NR;
        int m = 0;
        if (prefer_mr16(M, K, N, lanes)) {
        for (; m + 16 <= M; m += 16) {
            const float* a0 = A + (size_t)(m + 0) * K;
            const float* a1 = A + (size_t)(m + 1) * K;
            const float* a2 = A + (size_t)(m + 2) * K;
            const float* a3 = A + (size_t)(m + 3) * K;
            const float* a4 = A + (size_t)(m + 4) * K;
            const float* a5 = A + (size_t)(m + 5) * K;
            const float* a6 = A + (size_t)(m + 6) * K;
            const float* a7 = A + (size_t)(m + 7) * K;
            const float* a8 = A + (size_t)(m + 8) * K;
            const float* a9 = A + (size_t)(m + 9) * K;
            const float* a10 = A + (size_t)(m + 10) * K;
            const float* a11 = A + (size_t)(m + 11) * K;
            const float* a12 = A + (size_t)(m + 12) * K;
            const float* a13 = A + (size_t)(m + 13) * K;
            const float* a14 = A + (size_t)(m + 14) * K;
            const float* a15 = A + (size_t)(m + 15) * K;
            float* c0 = C + (size_t)(m + 0) * N;
            float* c1 = C + (size_t)(m + 1) * N;
            float* c2 = C + (size_t)(m + 2) * N;
            float* c3 = C + (size_t)(m + 3) * N;
            float* c4 = C + (size_t)(m + 4) * N;
            float* c5 = C + (size_t)(m + 5) * N;
            float* c6 = C + (size_t)(m + 6) * N;
            float* c7 = C + (size_t)(m + 7) * N;
            float* c8 = C + (size_t)(m + 8) * N;
            float* c9 = C + (size_t)(m + 9) * N;
            float* c10 = C + (size_t)(m + 10) * N;
            float* c11 = C + (size_t)(m + 11) * N;
            float* c12 = C + (size_t)(m + 12) * N;
            float* c13 = C + (size_t)(m + 13) * N;
            float* c14 = C + (size_t)(m + 14) * N;
            float* c15 = C + (size_t)(m + 15) * N;
            const float* r0 = residual + (size_t)(m + 0) * N;
            const float* r1 = residual + (size_t)(m + 1) * N;
            const float* r2 = residual + (size_t)(m + 2) * N;
            const float* r3 = residual + (size_t)(m + 3) * N;
            const float* r4 = residual + (size_t)(m + 4) * N;
            const float* r5 = residual + (size_t)(m + 5) * N;
            const float* r6 = residual + (size_t)(m + 6) * N;
            const float* r7 = residual + (size_t)(m + 7) * N;
            const float* r8 = residual + (size_t)(m + 8) * N;
            const float* r9 = residual + (size_t)(m + 9) * N;
            const float* r10 = residual + (size_t)(m + 10) * N;
            const float* r11 = residual + (size_t)(m + 11) * N;
            const float* r12 = residual + (size_t)(m + 12) * N;
            const float* r13 = residual + (size_t)(m + 13) * N;
            const float* r14 = residual + (size_t)(m + 14) * N;
            const float* r15 = residual + (size_t)(m + 15) * N;
            int p = 0;
            for (; p + 1 < n_panels_exact; p += 2) {
                const int n0 = p * NR;
                const float* bp0 = B_packed + (size_t)p * K * NR;
                const float* bp1 = bp0 + (size_t)K * NR;
                using V = decltype(hn::Zero(d));
                V acc0_0 = hn::LoadU(d, bias + n0);
                V acc1_0 = acc0_0, acc2_0 = acc0_0, acc3_0 = acc0_0;
                V acc4_0 = acc0_0, acc5_0 = acc0_0, acc6_0 = acc0_0, acc7_0 = acc0_0;
                V acc8_0 = acc0_0, acc9_0 = acc0_0, acc10_0 = acc0_0, acc11_0 = acc0_0;
                V acc12_0 = acc0_0, acc13_0 = acc0_0, acc14_0 = acc0_0, acc15_0 = acc0_0;
                for (int k = 0; k < K; k++) {
                    const auto b0 = hn::LoadU(d, bp0 + (size_t)k * NR);
                    acc0_0 = hn::MulAdd(hn::Set(d, a0[k]), b0, acc0_0);
                    acc1_0 = hn::MulAdd(hn::Set(d, a1[k]), b0, acc1_0);
                    acc2_0 = hn::MulAdd(hn::Set(d, a2[k]), b0, acc2_0);
                    acc3_0 = hn::MulAdd(hn::Set(d, a3[k]), b0, acc3_0);
                    acc4_0 = hn::MulAdd(hn::Set(d, a4[k]), b0, acc4_0);
                    acc5_0 = hn::MulAdd(hn::Set(d, a5[k]), b0, acc5_0);
                    acc6_0 = hn::MulAdd(hn::Set(d, a6[k]), b0, acc6_0);
                    acc7_0 = hn::MulAdd(hn::Set(d, a7[k]), b0, acc7_0);
                    acc8_0 = hn::MulAdd(hn::Set(d, a8[k]), b0, acc8_0);
                    acc9_0 = hn::MulAdd(hn::Set(d, a9[k]), b0, acc9_0);
                    acc10_0 = hn::MulAdd(hn::Set(d, a10[k]), b0, acc10_0);
                    acc11_0 = hn::MulAdd(hn::Set(d, a11[k]), b0, acc11_0);
                    acc12_0 = hn::MulAdd(hn::Set(d, a12[k]), b0, acc12_0);
                    acc13_0 = hn::MulAdd(hn::Set(d, a13[k]), b0, acc13_0);
                    acc14_0 = hn::MulAdd(hn::Set(d, a14[k]), b0, acc14_0);
                    acc15_0 = hn::MulAdd(hn::Set(d, a15[k]), b0, acc15_0);
                }
                const auto vb0 = hn::LoadU(d, bias + n0);
                const auto vg0 = hn::LoadU(d, gamma + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc0_0, vb0), vg0, hn::LoadU(d, r0 + n0)), d, c0 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc1_0, vb0), vg0, hn::LoadU(d, r1 + n0)), d, c1 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc2_0, vb0), vg0, hn::LoadU(d, r2 + n0)), d, c2 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc3_0, vb0), vg0, hn::LoadU(d, r3 + n0)), d, c3 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc4_0, vb0), vg0, hn::LoadU(d, r4 + n0)), d, c4 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc5_0, vb0), vg0, hn::LoadU(d, r5 + n0)), d, c5 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc6_0, vb0), vg0, hn::LoadU(d, r6 + n0)), d, c6 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc7_0, vb0), vg0, hn::LoadU(d, r7 + n0)), d, c7 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc8_0, vb0), vg0, hn::LoadU(d, r8 + n0)), d, c8 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc9_0, vb0), vg0, hn::LoadU(d, r9 + n0)), d, c9 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc10_0, vb0), vg0, hn::LoadU(d, r10 + n0)), d, c10 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc11_0, vb0), vg0, hn::LoadU(d, r11 + n0)), d, c11 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc12_0, vb0), vg0, hn::LoadU(d, r12 + n0)), d, c12 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc13_0, vb0), vg0, hn::LoadU(d, r13 + n0)), d, c13 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc14_0, vb0), vg0, hn::LoadU(d, r14 + n0)), d, c14 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc15_0, vb0), vg0, hn::LoadU(d, r15 + n0)), d, c15 + n0);

                const auto vb1 = hn::LoadU(d, bias + n0 + NR);
                const auto vg1 = hn::LoadU(d, gamma + n0 + NR);
                V acc0_1 = hn::LoadU(d, bias + n0 + NR);
                V acc1_1 = acc0_1, acc2_1 = acc0_1, acc3_1 = acc0_1;
                V acc4_1 = acc0_1, acc5_1 = acc0_1, acc6_1 = acc0_1, acc7_1 = acc0_1;
                V acc8_1 = acc0_1, acc9_1 = acc0_1, acc10_1 = acc0_1, acc11_1 = acc0_1;
                V acc12_1 = acc0_1, acc13_1 = acc0_1, acc14_1 = acc0_1, acc15_1 = acc0_1;
                for (int k = 0; k < K; k++) {
                    const auto b1 = hn::LoadU(d, bp1 + (size_t)k * NR);
                    acc0_1 = hn::MulAdd(hn::Set(d, a0[k]), b1, acc0_1);
                    acc1_1 = hn::MulAdd(hn::Set(d, a1[k]), b1, acc1_1);
                    acc2_1 = hn::MulAdd(hn::Set(d, a2[k]), b1, acc2_1);
                    acc3_1 = hn::MulAdd(hn::Set(d, a3[k]), b1, acc3_1);
                    acc4_1 = hn::MulAdd(hn::Set(d, a4[k]), b1, acc4_1);
                    acc5_1 = hn::MulAdd(hn::Set(d, a5[k]), b1, acc5_1);
                    acc6_1 = hn::MulAdd(hn::Set(d, a6[k]), b1, acc6_1);
                    acc7_1 = hn::MulAdd(hn::Set(d, a7[k]), b1, acc7_1);
                    acc8_1 = hn::MulAdd(hn::Set(d, a8[k]), b1, acc8_1);
                    acc9_1 = hn::MulAdd(hn::Set(d, a9[k]), b1, acc9_1);
                    acc10_1 = hn::MulAdd(hn::Set(d, a10[k]), b1, acc10_1);
                    acc11_1 = hn::MulAdd(hn::Set(d, a11[k]), b1, acc11_1);
                    acc12_1 = hn::MulAdd(hn::Set(d, a12[k]), b1, acc12_1);
                    acc13_1 = hn::MulAdd(hn::Set(d, a13[k]), b1, acc13_1);
                    acc14_1 = hn::MulAdd(hn::Set(d, a14[k]), b1, acc14_1);
                    acc15_1 = hn::MulAdd(hn::Set(d, a15[k]), b1, acc15_1);
                }
                hn::StoreU(hn::MulAdd(hn::Add(acc0_1, vb1), vg1, hn::LoadU(d, r0 + n0 + NR)), d, c0 + n0 + NR);
                hn::StoreU(hn::MulAdd(hn::Add(acc1_1, vb1), vg1, hn::LoadU(d, r1 + n0 + NR)), d, c1 + n0 + NR);
                hn::StoreU(hn::MulAdd(hn::Add(acc2_1, vb1), vg1, hn::LoadU(d, r2 + n0 + NR)), d, c2 + n0 + NR);
                hn::StoreU(hn::MulAdd(hn::Add(acc3_1, vb1), vg1, hn::LoadU(d, r3 + n0 + NR)), d, c3 + n0 + NR);
                hn::StoreU(hn::MulAdd(hn::Add(acc4_1, vb1), vg1, hn::LoadU(d, r4 + n0 + NR)), d, c4 + n0 + NR);
                hn::StoreU(hn::MulAdd(hn::Add(acc5_1, vb1), vg1, hn::LoadU(d, r5 + n0 + NR)), d, c5 + n0 + NR);
                hn::StoreU(hn::MulAdd(hn::Add(acc6_1, vb1), vg1, hn::LoadU(d, r6 + n0 + NR)), d, c6 + n0 + NR);
                hn::StoreU(hn::MulAdd(hn::Add(acc7_1, vb1), vg1, hn::LoadU(d, r7 + n0 + NR)), d, c7 + n0 + NR);
                hn::StoreU(hn::MulAdd(hn::Add(acc8_1, vb1), vg1, hn::LoadU(d, r8 + n0 + NR)), d, c8 + n0 + NR);
                hn::StoreU(hn::MulAdd(hn::Add(acc9_1, vb1), vg1, hn::LoadU(d, r9 + n0 + NR)), d, c9 + n0 + NR);
                hn::StoreU(hn::MulAdd(hn::Add(acc10_1, vb1), vg1, hn::LoadU(d, r10 + n0 + NR)), d, c10 + n0 + NR);
                hn::StoreU(hn::MulAdd(hn::Add(acc11_1, vb1), vg1, hn::LoadU(d, r11 + n0 + NR)), d, c11 + n0 + NR);
                hn::StoreU(hn::MulAdd(hn::Add(acc12_1, vb1), vg1, hn::LoadU(d, r12 + n0 + NR)), d, c12 + n0 + NR);
                hn::StoreU(hn::MulAdd(hn::Add(acc13_1, vb1), vg1, hn::LoadU(d, r13 + n0 + NR)), d, c13 + n0 + NR);
                hn::StoreU(hn::MulAdd(hn::Add(acc14_1, vb1), vg1, hn::LoadU(d, r14 + n0 + NR)), d, c14 + n0 + NR);
                hn::StoreU(hn::MulAdd(hn::Add(acc15_1, vb1), vg1, hn::LoadU(d, r15 + n0 + NR)), d, c15 + n0 + NR);
            } 
            if (p < n_panels_exact) {
                const int n0 = p * NR;
                const float* bp = B_packed + (size_t)p * K * NR;
                using V = decltype(hn::Zero(d));
                V acc0 = hn::LoadU(d, bias + n0);
                V acc1 = acc0, acc2 = acc0, acc3 = acc0;
                V acc4 = acc0, acc5 = acc0, acc6 = acc0, acc7 = acc0;
                V acc8 = acc0, acc9 = acc0, acc10 = acc0, acc11 = acc0;
                V acc12 = acc0, acc13 = acc0, acc14 = acc0, acc15 = acc0;
                const auto vg = hn::LoadU(d, gamma + n0);
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                    acc4 = hn::MulAdd(hn::Set(d, a4[k]), b, acc4);
                    acc5 = hn::MulAdd(hn::Set(d, a5[k]), b, acc5);
                    acc6 = hn::MulAdd(hn::Set(d, a6[k]), b, acc6);
                    acc7 = hn::MulAdd(hn::Set(d, a7[k]), b, acc7);
                    acc8 = hn::MulAdd(hn::Set(d, a8[k]), b, acc8);
                    acc9 = hn::MulAdd(hn::Set(d, a9[k]), b, acc9);
                    acc10 = hn::MulAdd(hn::Set(d, a10[k]), b, acc10);
                    acc11 = hn::MulAdd(hn::Set(d, a11[k]), b, acc11);
                    acc12 = hn::MulAdd(hn::Set(d, a12[k]), b, acc12);
                    acc13 = hn::MulAdd(hn::Set(d, a13[k]), b, acc13);
                    acc14 = hn::MulAdd(hn::Set(d, a14[k]), b, acc14);
                    acc15 = hn::MulAdd(hn::Set(d, a15[k]), b, acc15);
                }
                const auto vb = hn::LoadU(d, bias + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc0, vb), vg, hn::LoadU(d, r0 + n0)), d, c0 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc1, vb), vg, hn::LoadU(d, r1 + n0)), d, c1 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc2, vb), vg, hn::LoadU(d, r2 + n0)), d, c2 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc3, vb), vg, hn::LoadU(d, r3 + n0)), d, c3 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc4, vb), vg, hn::LoadU(d, r4 + n0)), d, c4 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc5, vb), vg, hn::LoadU(d, r5 + n0)), d, c5 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc6, vb), vg, hn::LoadU(d, r6 + n0)), d, c6 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc7, vb), vg, hn::LoadU(d, r7 + n0)), d, c7 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc8, vb), vg, hn::LoadU(d, r8 + n0)), d, c8 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc9, vb), vg, hn::LoadU(d, r9 + n0)), d, c9 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc10, vb), vg, hn::LoadU(d, r10 + n0)), d, c10 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc11, vb), vg, hn::LoadU(d, r11 + n0)), d, c11 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc12, vb), vg, hn::LoadU(d, r12 + n0)), d, c12 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc13, vb), vg, hn::LoadU(d, r13 + n0)), d, c13 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc14, vb), vg, hn::LoadU(d, r14 + n0)), d, c14 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc15, vb), vg, hn::LoadU(d, r15 + n0)), d, c15 + n0);
            }
        }
        }
        if (prefer_mr8(M, K, N, lanes)) {
        for (; m + 8 <= M; m += 8) {
            const float* a0 = A + (size_t)(m + 0) * K;
            const float* a1 = A + (size_t)(m + 1) * K;
            const float* a2 = A + (size_t)(m + 2) * K;
            const float* a3 = A + (size_t)(m + 3) * K;
            const float* a4 = A + (size_t)(m + 4) * K;
            const float* a5 = A + (size_t)(m + 5) * K;
            const float* a6 = A + (size_t)(m + 6) * K;
            const float* a7 = A + (size_t)(m + 7) * K;
            float* c0 = C + (size_t)(m + 0) * N;
            float* c1 = C + (size_t)(m + 1) * N;
            float* c2 = C + (size_t)(m + 2) * N;
            float* c3 = C + (size_t)(m + 3) * N;
            float* c4 = C + (size_t)(m + 4) * N;
            float* c5 = C + (size_t)(m + 5) * N;
            float* c6 = C + (size_t)(m + 6) * N;
            float* c7 = C + (size_t)(m + 7) * N;
            const float* r0 = residual + (size_t)(m + 0) * N;
            const float* r1 = residual + (size_t)(m + 1) * N;
            const float* r2 = residual + (size_t)(m + 2) * N;
            const float* r3 = residual + (size_t)(m + 3) * N;
            const float* r4 = residual + (size_t)(m + 4) * N;
            const float* r5 = residual + (size_t)(m + 5) * N;
            const float* r6 = residual + (size_t)(m + 6) * N;
            const float* r7 = residual + (size_t)(m + 7) * N;
            for (int p = 0; p < n_panels_exact; p++) {
                const int n0 = p * NR;
                const float* bp = B_packed + (size_t)p * K * NR;
                auto acc0 = hn::LoadU(d, bias + n0);
                auto acc1 = acc0;
                auto acc2 = acc0;
                auto acc3 = acc0;
                auto acc4 = acc0;
                auto acc5 = acc0;
                auto acc6 = acc0;
                auto acc7 = acc0;
                const auto vg = hn::LoadU(d, gamma + n0);
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                    acc4 = hn::MulAdd(hn::Set(d, a4[k]), b, acc4);
                    acc5 = hn::MulAdd(hn::Set(d, a5[k]), b, acc5);
                    acc6 = hn::MulAdd(hn::Set(d, a6[k]), b, acc6);
                    acc7 = hn::MulAdd(hn::Set(d, a7[k]), b, acc7);
                }
                const auto vb = hn::LoadU(d, bias + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc0, vb), vg, hn::LoadU(d, r0 + n0)), d, c0 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc1, vb), vg, hn::LoadU(d, r1 + n0)), d, c1 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc2, vb), vg, hn::LoadU(d, r2 + n0)), d, c2 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc3, vb), vg, hn::LoadU(d, r3 + n0)), d, c3 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc4, vb), vg, hn::LoadU(d, r4 + n0)), d, c4 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc5, vb), vg, hn::LoadU(d, r5 + n0)), d, c5 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc6, vb), vg, hn::LoadU(d, r6 + n0)), d, c6 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc7, vb), vg, hn::LoadU(d, r7 + n0)), d, c7 + n0);
            }
        }
        }
        for (; m + 4 <= M; m += 4) {
            const float* a0 = A + (size_t)(m + 0) * K;
            const float* a1 = A + (size_t)(m + 1) * K;
            const float* a2 = A + (size_t)(m + 2) * K;
            const float* a3 = A + (size_t)(m + 3) * K;
            float* c0 = C + (size_t)(m + 0) * N;
            float* c1 = C + (size_t)(m + 1) * N;
            float* c2 = C + (size_t)(m + 2) * N;
            float* c3 = C + (size_t)(m + 3) * N;
            const float* r0 = residual + (size_t)(m + 0) * N;
            const float* r1 = residual + (size_t)(m + 1) * N;
            const float* r2 = residual + (size_t)(m + 2) * N;
            const float* r3 = residual + (size_t)(m + 3) * N;
            for (int p = 0; p < n_panels_exact; p++) {
                const int n0 = p * NR;
                const float* bp = B_packed + (size_t)p * K * NR;
                auto acc0 = hn::LoadU(d, bias + n0);
                auto acc1 = acc0;
                auto acc2 = acc0;
                auto acc3 = acc0;
                const auto vg = hn::LoadU(d, gamma + n0);
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                }
                const auto vb = hn::LoadU(d, bias + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc0, vb), vg, hn::LoadU(d, r0 + n0)), d, c0 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc1, vb), vg, hn::LoadU(d, r1 + n0)), d, c1 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc2, vb), vg, hn::LoadU(d, r2 + n0)), d, c2 + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc3, vb), vg, hn::LoadU(d, r3 + n0)), d, c3 + n0);
            }
        }
        for (; m < M; m++) {
            const float* ar = A + (size_t)m * K;
            const float* rr = residual + (size_t)m * N;
            float* cr = C + (size_t)m * N;
            for (int p = 0; p < n_panels_exact; p++) {
                const int n0 = p * NR;
                const float* bp = B_packed + (size_t)p * K * NR;
                auto acc = hn::LoadU(d, bias + n0);
                for (int k = 0; k < K; k++) {
                    acc = hn::MulAdd(hn::Set(d, ar[k]), hn::LoadU(d, bp + (size_t)k * NR), acc);
                }
                const auto vb = hn::LoadU(d, bias + n0);
                const auto vg = hn::LoadU(d, gamma + n0);
                hn::StoreU(hn::MulAdd(hn::Add(acc, vb), vg, hn::LoadU(d, rr + n0)),
                           d, cr + n0);
            }
        }
        return;
    }

    int m = 0;
    if (prefer_mr16(M, K, N, lanes)) {
    for (; m + 16 <= M; m += 16) {
        for (int p = 0; p < n_panels; p++) {
            const int n0 = p * NR;
            const int nr = (n0 + NR <= N) ? NR : (N - n0);
            const float* bp = B_packed + (size_t)p * K * NR;
            int j = 0;
            for (; j + lanes <= nr; j += lanes) {
                using V = decltype(hn::Zero(d));
                V acc[16];
                for (int r = 0; r < 16; r++) acc[r] = hn::Zero(d);
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR + j);
                    for (int r = 0; r < 16; r++) {
                        acc[r] = hn::MulAdd(hn::Set(d, A[(size_t)(m + r) * K + k]), b, acc[r]);
                    }
                }
                const auto vb = hn::LoadU(d, bias + n0 + j);
                const auto vg = hn::LoadU(d, gamma + n0 + j);
                for (int r = 0; r < 16; r++) {
                    const float* rr = residual + (size_t)(m + r) * N;
                    float* cr = C + (size_t)(m + r) * N;
                    auto v = hn::MulAdd(hn::Add(acc[r], vb), vg, hn::LoadU(d, rr + n0 + j));
                    hn::StoreU(v, d, cr + n0 + j);
                }
            }
            if (j < nr) {
                const size_t count = static_cast<size_t>(nr - j);
                using V = decltype(hn::Zero(d));
                V acc[16];
                for (int r = 0; r < 16; r++) acc[r] = hn::Zero(d);
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR + j);
                    for (int r = 0; r < 16; r++) {
                        acc[r] = hn::MulAdd(hn::Set(d, A[(size_t)(m + r) * K + k]), b, acc[r]);
                    }
                }
                const auto vb = hn::LoadNOr(hn::Zero(d), d, bias + n0 + j, count);
                const auto vg = hn::LoadNOr(hn::Zero(d), d, gamma + n0 + j, count);
                for (int r = 0; r < 16; r++) {
                    const float* rr = residual + (size_t)(m + r) * N;
                    float* cr = C + (size_t)(m + r) * N;
                    auto v = hn::MulAdd(hn::Add(acc[r], vb), vg, hn::LoadNOr(hn::Zero(d), d, rr + n0 + j, count));
                    hn::StoreN(v, d, cr + n0 + j, count);
                }
            }
        }
    }
    }
    if (prefer_mr8(M, K, N, lanes)) {
    for (; m + 8 <= M; m += 8) {
        const float* a0 = A + (size_t)(m + 0) * K;
        const float* a1 = A + (size_t)(m + 1) * K;
        const float* a2 = A + (size_t)(m + 2) * K;
        const float* a3 = A + (size_t)(m + 3) * K;
        const float* a4 = A + (size_t)(m + 4) * K;
        const float* a5 = A + (size_t)(m + 5) * K;
        const float* a6 = A + (size_t)(m + 6) * K;
        const float* a7 = A + (size_t)(m + 7) * K;
        float* c0 = C + (size_t)(m + 0) * N;
        float* c1 = C + (size_t)(m + 1) * N;
        float* c2 = C + (size_t)(m + 2) * N;
        float* c3 = C + (size_t)(m + 3) * N;
        float* c4 = C + (size_t)(m + 4) * N;
        float* c5 = C + (size_t)(m + 5) * N;
        float* c6 = C + (size_t)(m + 6) * N;
        float* c7 = C + (size_t)(m + 7) * N;
        const float* r0 = residual + (size_t)(m + 0) * N;
        const float* r1 = residual + (size_t)(m + 1) * N;
        const float* r2 = residual + (size_t)(m + 2) * N;
        const float* r3 = residual + (size_t)(m + 3) * N;
        const float* r4 = residual + (size_t)(m + 4) * N;
        const float* r5 = residual + (size_t)(m + 5) * N;
        const float* r6 = residual + (size_t)(m + 6) * N;
        const float* r7 = residual + (size_t)(m + 7) * N;
        for (int p = 0; p < n_panels; p++) {
            const int n0 = p * NR;
            const int nr = (n0 + NR <= N) ? NR : (N - n0);
            const float* bp = B_packed + (size_t)p * K * NR;
            int j = 0;
            for (; j + lanes <= nr; j += lanes) {
                auto acc0 = hn::Zero(d), acc1 = hn::Zero(d);
                auto acc2 = hn::Zero(d), acc3 = hn::Zero(d);
                auto acc4 = hn::Zero(d), acc5 = hn::Zero(d);
                auto acc6 = hn::Zero(d), acc7 = hn::Zero(d);
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR + j);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                    acc4 = hn::MulAdd(hn::Set(d, a4[k]), b, acc4);
                    acc5 = hn::MulAdd(hn::Set(d, a5[k]), b, acc5);
                    acc6 = hn::MulAdd(hn::Set(d, a6[k]), b, acc6);
                    acc7 = hn::MulAdd(hn::Set(d, a7[k]), b, acc7);
                }
                const auto vb = hn::LoadU(d, bias + n0 + j);
                const auto vg = hn::LoadU(d, gamma + n0 + j);
                hn::StoreU(hn::MulAdd(hn::Add(acc0, vb), vg, hn::LoadU(d, r0 + n0 + j)), d, c0 + n0 + j);
                hn::StoreU(hn::MulAdd(hn::Add(acc1, vb), vg, hn::LoadU(d, r1 + n0 + j)), d, c1 + n0 + j);
                hn::StoreU(hn::MulAdd(hn::Add(acc2, vb), vg, hn::LoadU(d, r2 + n0 + j)), d, c2 + n0 + j);
                hn::StoreU(hn::MulAdd(hn::Add(acc3, vb), vg, hn::LoadU(d, r3 + n0 + j)), d, c3 + n0 + j);
                hn::StoreU(hn::MulAdd(hn::Add(acc4, vb), vg, hn::LoadU(d, r4 + n0 + j)), d, c4 + n0 + j);
                hn::StoreU(hn::MulAdd(hn::Add(acc5, vb), vg, hn::LoadU(d, r5 + n0 + j)), d, c5 + n0 + j);
                hn::StoreU(hn::MulAdd(hn::Add(acc6, vb), vg, hn::LoadU(d, r6 + n0 + j)), d, c6 + n0 + j);
                hn::StoreU(hn::MulAdd(hn::Add(acc7, vb), vg, hn::LoadU(d, r7 + n0 + j)), d, c7 + n0 + j);
            }
            if (j < nr) {
                const size_t count = static_cast<size_t>(nr - j);
                auto acc0 = hn::Zero(d), acc1 = hn::Zero(d);
                auto acc2 = hn::Zero(d), acc3 = hn::Zero(d);
                auto acc4 = hn::Zero(d), acc5 = hn::Zero(d);
                auto acc6 = hn::Zero(d), acc7 = hn::Zero(d);
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR + j);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                    acc4 = hn::MulAdd(hn::Set(d, a4[k]), b, acc4);
                    acc5 = hn::MulAdd(hn::Set(d, a5[k]), b, acc5);
                    acc6 = hn::MulAdd(hn::Set(d, a6[k]), b, acc6);
                    acc7 = hn::MulAdd(hn::Set(d, a7[k]), b, acc7);
                }
                const auto vb = hn::LoadNOr(hn::Zero(d), d, bias + n0 + j, count);
                const auto vg = hn::LoadNOr(hn::Zero(d), d, gamma + n0 + j, count);
                hn::StoreN(hn::MulAdd(hn::Add(acc0, vb), vg, hn::LoadNOr(hn::Zero(d), d, r0 + n0 + j, count)), d, c0 + n0 + j, count);
                hn::StoreN(hn::MulAdd(hn::Add(acc1, vb), vg, hn::LoadNOr(hn::Zero(d), d, r1 + n0 + j, count)), d, c1 + n0 + j, count);
                hn::StoreN(hn::MulAdd(hn::Add(acc2, vb), vg, hn::LoadNOr(hn::Zero(d), d, r2 + n0 + j, count)), d, c2 + n0 + j, count);
                hn::StoreN(hn::MulAdd(hn::Add(acc3, vb), vg, hn::LoadNOr(hn::Zero(d), d, r3 + n0 + j, count)), d, c3 + n0 + j, count);
                hn::StoreN(hn::MulAdd(hn::Add(acc4, vb), vg, hn::LoadNOr(hn::Zero(d), d, r4 + n0 + j, count)), d, c4 + n0 + j, count);
                hn::StoreN(hn::MulAdd(hn::Add(acc5, vb), vg, hn::LoadNOr(hn::Zero(d), d, r5 + n0 + j, count)), d, c5 + n0 + j, count);
                hn::StoreN(hn::MulAdd(hn::Add(acc6, vb), vg, hn::LoadNOr(hn::Zero(d), d, r6 + n0 + j, count)), d, c6 + n0 + j, count);
                hn::StoreN(hn::MulAdd(hn::Add(acc7, vb), vg, hn::LoadNOr(hn::Zero(d), d, r7 + n0 + j, count)), d, c7 + n0 + j, count);
            }
        }
    }
    }
    for (; m + 4 <= M; m += 4) {
        const float* a0 = A + (size_t)(m + 0) * K;
        const float* a1 = A + (size_t)(m + 1) * K;
        const float* a2 = A + (size_t)(m + 2) * K;
        const float* a3 = A + (size_t)(m + 3) * K;
        float* c0 = C + (size_t)(m + 0) * N;
        float* c1 = C + (size_t)(m + 1) * N;
        float* c2 = C + (size_t)(m + 2) * N;
        float* c3 = C + (size_t)(m + 3) * N;
        const float* r0 = residual + (size_t)(m + 0) * N;
        const float* r1 = residual + (size_t)(m + 1) * N;
        const float* r2 = residual + (size_t)(m + 2) * N;
        const float* r3 = residual + (size_t)(m + 3) * N;
        for (int p = 0; p < n_panels; p++) {
            const int n0 = p * NR;
            const int nr = (n0 + NR <= N) ? NR : (N - n0);
            const float* bp = B_packed + (size_t)p * K * NR;
            int j = 0;
            for (; j + lanes <= nr; j += lanes) {
                auto acc0 = hn::Zero(d), acc1 = hn::Zero(d);
                auto acc2 = hn::Zero(d), acc3 = hn::Zero(d);
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR + j);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                }
                const auto vb = hn::LoadU(d, bias + n0 + j);
                const auto vg = hn::LoadU(d, gamma + n0 + j);
                hn::StoreU(hn::MulAdd(hn::Add(acc0, vb), vg, hn::LoadU(d, r0 + n0 + j)), d, c0 + n0 + j);
                hn::StoreU(hn::MulAdd(hn::Add(acc1, vb), vg, hn::LoadU(d, r1 + n0 + j)), d, c1 + n0 + j);
                hn::StoreU(hn::MulAdd(hn::Add(acc2, vb), vg, hn::LoadU(d, r2 + n0 + j)), d, c2 + n0 + j);
                hn::StoreU(hn::MulAdd(hn::Add(acc3, vb), vg, hn::LoadU(d, r3 + n0 + j)), d, c3 + n0 + j);
            }
            if (j < nr) {
                const size_t count = static_cast<size_t>(nr - j);
                auto acc0 = hn::Zero(d), acc1 = hn::Zero(d);
                auto acc2 = hn::Zero(d), acc3 = hn::Zero(d);
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR + j);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                }
                const auto vb = hn::LoadNOr(hn::Zero(d), d, bias + n0 + j, count);
                const auto vg = hn::LoadNOr(hn::Zero(d), d, gamma + n0 + j, count);
                hn::StoreN(hn::MulAdd(hn::Add(acc0, vb), vg, hn::LoadNOr(hn::Zero(d), d, r0 + n0 + j, count)), d, c0 + n0 + j, count);
                hn::StoreN(hn::MulAdd(hn::Add(acc1, vb), vg, hn::LoadNOr(hn::Zero(d), d, r1 + n0 + j, count)), d, c1 + n0 + j, count);
                hn::StoreN(hn::MulAdd(hn::Add(acc2, vb), vg, hn::LoadNOr(hn::Zero(d), d, r2 + n0 + j, count)), d, c2 + n0 + j, count);
                hn::StoreN(hn::MulAdd(hn::Add(acc3, vb), vg, hn::LoadNOr(hn::Zero(d), d, r3 + n0 + j, count)), d, c3 + n0 + j, count);
            }
        }
    }
    for (; m < M; m++) {
        const float* ar = A + (size_t)m * K;
        const float* rr = residual + (size_t)m * N;
        float* cr = C + (size_t)m * N;
        for (int p = 0; p < n_panels; p++) {
            const int n0 = p * NR;
            const int nr = (n0 + NR <= N) ? NR : (N - n0);
            const float* bp = B_packed + (size_t)p * K * NR;
            int j = 0;
            for (; j + lanes <= nr; j += lanes) {
                auto acc = hn::Zero(d);
                for (int k = 0; k < K; k++) {
                    acc = hn::MulAdd(hn::Set(d, ar[k]), hn::LoadU(d, bp + (size_t)k * NR + j), acc);
                }
                const auto v = hn::MulAdd(hn::Add(acc, hn::LoadU(d, bias + n0 + j)),
                                          hn::LoadU(d, gamma + n0 + j),
                                          hn::LoadU(d, rr + n0 + j));
                hn::StoreU(v, d, cr + n0 + j);
            }
            if (j < nr) {
                const size_t count = static_cast<size_t>(nr - j);
                auto acc = hn::Zero(d);
                for (int k = 0; k < K; k++) {
                    acc = hn::MulAdd(hn::Set(d, ar[k]), hn::LoadU(d, bp + (size_t)k * NR + j), acc);
                }
                const auto v = hn::MulAdd(hn::Add(acc, hn::LoadNOr(hn::Zero(d), d, bias + n0 + j, count)),
                                          hn::LoadNOr(hn::Zero(d), d, gamma + n0 + j, count),
                                          hn::LoadNOr(hn::Zero(d), d, rr + n0 + j, count));
                hn::StoreN(v, d, cr + n0 + j, count);
            }
        }
    }
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace facex_hwy

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace facex_hwy {
HWY_EXPORT(layer_norm_dispatch);
HWY_EXPORT(matmul_fp32_dispatch);
HWY_EXPORT(matmul_fp32_packed_dispatch);
HWY_EXPORT(matmul_fp32_packed_bias_dispatch);
HWY_EXPORT(matmul_bias_gelu_packed_dispatch);
HWY_EXPORT(matmul_residual_bias_gamma_packed_dispatch);
HWY_EXPORT(gelu_dispatch);
HWY_EXPORT(l2_normalize_dispatch);
HWY_EXPORT(add_fp32_dispatch);
HWY_EXPORT(scale_fp32_dispatch);
HWY_EXPORT(bias_gamma_residual_fp32_dispatch);
}

namespace hn = hwy::HWY_NAMESPACE;

extern "C" {

#if FACEX_ENABLE_GEMM_PROFILE
typedef struct {
    const char* kind;
    int M, K, N;
    unsigned long long calls;
    unsigned long long nanos;
} GemmProfileEntry;

static GemmProfileEntry g_gemm_profile[128];
static int g_gemm_profile_count = 0;
static int g_gemm_profile_enabled = -1;
static int g_gemm_profile_registered = 0;

static unsigned long long profile_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long long)ts.tv_sec * 1000000000ull + (unsigned long long)ts.tv_nsec;
}

static void gemm_profile_dump(void) {
    if (g_gemm_profile_enabled != 1) return;
    int order[128];
    for (int i = 0; i < g_gemm_profile_count; i++) order[i] = i;
    for (int i = 0; i < g_gemm_profile_count; i++) {
        for (int j = i + 1; j < g_gemm_profile_count; j++) {
            if (g_gemm_profile[order[j]].nanos > g_gemm_profile[order[i]].nanos) {
                int tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
        }
    }
    fprintf(stderr, "\nFaceX GEMM shape profile:\n");
    fprintf(stderr, "kind,M,K,N,calls,time_ms,GFLOP/s\n");
    for (int i = 0; i < g_gemm_profile_count; i++) {
        const GemmProfileEntry* e = &g_gemm_profile[order[i]];
        const double ms = (double)e->nanos / 1.0e6;
        const double flops = 2.0 * (double)e->M * (double)e->K * (double)e->N * (double)e->calls;
        const double gflops = e->nanos ? flops / (double)e->nanos : 0.0;
        fprintf(stderr, "%s,%d,%d,%d,%llu,%.3f,%.2f\n",
                e->kind, e->M, e->K, e->N, e->calls, ms, gflops);
    }
}

static int gemm_profile_is_enabled(void) {
    if (g_gemm_profile_enabled < 0) {
        const char* env = getenv("FACEX_PROFILE_GEMM");
        g_gemm_profile_enabled = (env && env[0] && env[0] != '0') ? 1 : 0;
        if (g_gemm_profile_enabled && !g_gemm_profile_registered) {
            atexit(gemm_profile_dump);
            g_gemm_profile_registered = 1;
        }
    }
    return g_gemm_profile_enabled;
}

static unsigned long long gemm_profile_begin(void) {
    return gemm_profile_is_enabled() ? profile_now_ns() : 0;
}

static void gemm_profile_record(const char* kind, int M, int K, int N,
                                unsigned long long start_ns) {
    if (!start_ns) return;
    const unsigned long long elapsed = profile_now_ns() - start_ns;
    for (int i = 0; i < g_gemm_profile_count; i++) {
        GemmProfileEntry* e = &g_gemm_profile[i];
        if (e->kind == kind && e->M == M && e->K == K && e->N == N) {
            e->calls++;
            e->nanos += elapsed;
            return;
        }
    }
    if (g_gemm_profile_count < (int)(sizeof(g_gemm_profile) / sizeof(g_gemm_profile[0]))) {
        GemmProfileEntry* e = &g_gemm_profile[g_gemm_profile_count++];
        e->kind = kind;
        e->M = M;
        e->K = K;
        e->N = N;
        e->calls = 1;
        e->nanos = elapsed;
    }
}
#else
static inline unsigned long long gemm_profile_begin(void) {
    return 0;
}

static inline void gemm_profile_record(const char*, int, int, int,
                                       unsigned long long) {}
#endif


/* ============ LayerNorm ============ */
/* out[i] = gamma[i] * (x[i] - mean) / sqrt(var + eps) + beta[i] */
void layer_norm_fp32(const float* x, int N, int C,
                     const float* gamma, const float* beta,
                     float eps, float* out) {
    static const auto fn = HWY_DYNAMIC_POINTER(facex_hwy::layer_norm_dispatch);
    fn(x, N, C, gamma, beta, eps, out);
    return;

    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    for (int n = 0; n < N; n++) {
        const float* row = x + (size_t)n * C;
        float* orow = out + (size_t)n * C;
        int c = 0;
        auto vsum = hn::Zero(d);
        for (; c + static_cast<int>(lanes) <= C; c += static_cast<int>(lanes))
            vsum = hn::Add(vsum, hn::LoadU(d, row + c));
        float sum = hn::ReduceSum(d, vsum);
        for (; c < C; c++) sum += row[c];
        float mean = sum / C;

        c = 0;
        auto vvar = hn::Zero(d);
        const auto vmean = hn::Set(d, mean);
        for (; c + static_cast<int>(lanes) <= C; c += static_cast<int>(lanes)) {
            const auto diff = hn::Sub(hn::LoadU(d, row + c), vmean);
            vvar = hn::MulAdd(diff, diff, vvar);
        }
        float var_sum = hn::ReduceSum(d, vvar);
        for (; c < C; c++) { float diff = row[c] - mean; var_sum += diff * diff; }
        float inv_std = 1.0f / sqrtf(var_sum / C + eps);

        c = 0;
        const auto vis = hn::Set(d, inv_std);
        for (; c + static_cast<int>(lanes) <= C; c += static_cast<int>(lanes)) {
            auto v = hn::Mul(hn::Sub(hn::LoadU(d, row + c), vmean), vis);
            if (gamma) v = hn::Mul(v, hn::LoadU(d, gamma + c));
            if (beta) v = hn::Add(v, hn::LoadU(d, beta + c));
            hn::StoreU(v, d, orow + c);
        }
        for (; c < C; c++) {
            float v = (row[c] - mean) * inv_std;
            if (gamma) v *= gamma[c];
            if (beta) v += beta[c];
            orow[c] = v;
        }
    }
}

/* ============ GELU (fast approximation) ============ */
/* gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * Using fast tanh: tanh(x) ≈ x*(27+x^2)/(27+9*x^2) for |x|<3, clamp otherwise */
static inline float fast_tanh(float x) {
    if (x > 4.0f) return 1.0f;
    if (x < -4.0f) return -1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

void gelu_fp32(float* x, int n) {
    static const auto fn = HWY_DYNAMIC_POINTER(facex_hwy::gelu_dispatch);
    fn(x, n);
}

/* ============ Softmax ============ */
/* softmax along last dim: out[i] = exp(x[i] - max) / sum(exp(x - max)) */
void softmax_fp32(float* x, int N, int C) {
    for (int n = 0; n < N; n++) {
        float* row = x + (size_t)n * C;
        float mx = row[0];
        for (int c = 1; c < C; c++) if (row[c] > mx) mx = row[c];
        float sum = 0;
        for (int c = 0; c < C; c++) {
            row[c] = expf(row[c] - mx);
            sum += row[c];
        }
        float inv = 1.0f / sum;
        for (int c = 0; c < C; c++) row[c] *= inv;
    }
}

void add_fp32(float* x, const float* y, int n) {
    static const auto fn = HWY_DYNAMIC_POINTER(facex_hwy::add_fp32_dispatch);
    fn(x, y, n);
}

void scale_fp32(float* x, int n, float scale) {
    static const auto fn = HWY_DYNAMIC_POINTER(facex_hwy::scale_fp32_dispatch);
    fn(x, n, scale);
}

void bias_gamma_residual_fp32(const float* src, float* dst,
                              const float* bias, const float* gamma,
                              const float* residual, int rows, int cols) {
    static const auto fn = HWY_DYNAMIC_POINTER(facex_hwy::bias_gamma_residual_fp32_dispatch);
    fn(src, dst, bias, gamma, residual, rows, cols);
}

/* ============ Dynamic INT8 MatMul ============ */
/* C[M,N] = A_fp32[M,K] @ W_int8_packed[K,N], with dynamic A quantization.
 * W is pre-packed in c8 format. A is quantized per-tensor at runtime. */
void matmul_dynamic_int8(const float* A_fp32, int M, int K, int N,
                         const void* W_packed, const int32_t* col_sums,
                         const float* w_scales, float* C_fp32)
{
    /* 1. Find max|A| for per-tensor quantization */
    float a_max = 0;
    int n_elem = M * K;
    for (int i = 0; i < n_elem; i++) {
        float v = A_fp32[i] < 0 ? -A_fp32[i] : A_fp32[i];
        if (v > a_max) a_max = v;
    }
    float a_scale = a_max / 127.0f;
    if (a_scale < 1e-8f) a_scale = 1e-8f;
    float a_inv = 1.0f / a_scale;

    /* 2. Quantize A to int8 — use static workspace to avoid malloc */
    int K_padded = (K + 7) & ~7;
    static int8_t* s_A_int8 = NULL;
    static int32_t* s_C_int32 = NULL;
    static size_t s_A_cap = 0, s_C_cap = 0;
    size_t a_need = (size_t)M * K_padded;
    size_t c_need = (size_t)M * N;
    if (a_need > s_A_cap) { free(s_A_int8); s_A_int8 = (int8_t*)malloc(a_need); s_A_cap = a_need; }
    if (c_need > s_C_cap) { free(s_C_int32); s_C_int32 = (int32_t*)malloc(c_need * 4); s_C_cap = c_need; }

    int8_t* A_int8 = s_A_int8;
    int i_a = 0;
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            int q = (int)(A_fp32[m*K+k] * a_inv + (A_fp32[m*K+k] >= 0 ? 0.5f : -0.5f));
            if (q > 127) q = 127; if (q < -128) q = -128;
            A_int8[m*K_padded+k] = (int8_t)q;
        }
        for (int k = K; k < K_padded; k++) A_int8[m*K_padded+k] = 0;
    }

    /* 3. INT8 GEMM */
    extern void int8_gemm_4x8c8(const int8_t*, int, int, int,
                                  const void*, int32_t*, const int32_t*);
    int32_t* C_int32 = s_C_int32;
    int8_gemm_4x8c8(A_int8, M, K_padded, N, W_packed, C_int32, col_sums);

    /* 4. Dequant: fp32 = int32 * a_scale * w_scale[n] */
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            C_fp32[m*N+n] = (float)C_int32[m*N+n] * a_scale * w_scales[n];
}

/* matmul_dynamic_int8_gelu not needed — GELU already vectorized and fast */

/* ============ MatMul FP32 ============ */
/* C[M,N] = A[M,K] @ B[K,N]
 * Portable scalar fallback used for unpacked weights. */
void matmul_fp32(const float* A, const float* B, float* C,
                 int M, int K, int N) {
    static const auto fn = HWY_DYNAMIC_POINTER(facex_hwy::matmul_fp32_dispatch);
    const unsigned long long t0 = gemm_profile_begin();
    fn(A, B, C, M, K, N);
    gemm_profile_record("plain", M, K, N, t0);
    return;
    const hn::ScalableTag<float> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    int m = 0;
    for (; m + 4 <= M; m += 4) {
        const float* a0 = A + (size_t)(m + 0) * K;
        const float* a1 = A + (size_t)(m + 1) * K;
        const float* a2 = A + (size_t)(m + 2) * K;
        const float* a3 = A + (size_t)(m + 3) * K;
        float* c0 = C + (size_t)(m + 0) * N;
        float* c1 = C + (size_t)(m + 1) * N;
        float* c2 = C + (size_t)(m + 2) * N;
        float* c3 = C + (size_t)(m + 3) * N;
        int n = 0;
        for (; n + lanes <= N; n += lanes) {
            auto acc0 = hn::Zero(d);
            auto acc1 = hn::Zero(d);
            auto acc2 = hn::Zero(d);
            auto acc3 = hn::Zero(d);
            for (int k = 0; k < K; k++) {
                const auto b = hn::LoadU(d, B + (size_t)k * N + n);
                acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
            }
            hn::StoreU(acc0, d, c0 + n);
            hn::StoreU(acc1, d, c1 + n);
            hn::StoreU(acc2, d, c2 + n);
            hn::StoreU(acc3, d, c3 + n);
        }
        for (; n < N; n++) {
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            for (int k = 0; k < K; k++) {
                const float bv = B[(size_t)k * N + n];
                s0 += a0[k] * bv;
                s1 += a1[k] * bv;
                s2 += a2[k] * bv;
                s3 += a3[k] * bv;
            }
            c0[n] = s0;
            c1[n] = s1;
            c2[n] = s2;
            c3[n] = s3;
        }
    }
    for (; m < M; m++) {
        const float* ar = A + (size_t)m * K;
        float* cr = C + (size_t)m * N;
        int n = 0;
        for (; n + lanes <= N; n += lanes) {
            auto acc = hn::Zero(d);
            for (int k = 0; k < K; k++) {
                acc = hn::MulAdd(hn::Set(d, ar[k]), hn::LoadU(d, B + (size_t)k * N + n), acc);
            }
            hn::StoreU(acc, d, cr + n);
        }
        for (; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += ar[k] * B[(size_t)k * N + n];
            cr[n] = sum;
        }
    }
}

/* ============ FP32 B-packing: [K,N] -> [ceil(N/16), K, 16] panel format ============ */
/* Pack weight matrix B[K,N] into column-panel format for cache-friendly GEMM.
 * Each panel of 16 columns is stored contiguously along K dimension. */
void pack_b_fp32(const float* B, int K, int N, float* packed) {
    int NR = 16;
    int n_panels = (N + NR - 1) / NR;
    for (int p = 0; p < n_panels; p++) {
        int n_start = p * NR;
        int nr = (n_start + NR <= N) ? NR : (N - n_start);
        float* dst = packed + (size_t)p * K * NR;
        for (int k = 0; k < K; k++) {
            const float* src = B + (size_t)k * N + n_start;
            int j = 0;
            for (; j < nr; j++) dst[k * NR + j] = src[j];
            for (; j < NR; j++) dst[k * NR + j] = 0; /* zero-pad last panel */
        }
    }
}

int packed_b_fp32_size(int K, int N) {
    return ((N + 15) / 16) * K * 16;
}

/* ============ Threaded tile worker for packed GEMM ============ */
#include "threadpool.h"
typedef struct { const float* A; const float* B; float* C; int K,N,n_panels; } PGemmCtx;

/* ============ MatMul with pre-packed B ============ */
/* B_packed layout: [ceil(N/16), K, 16]. */
void matmul_fp32_packed(const float* A, const float* B_packed, float* C,
                        int M, int K, int N) {
    static const auto fn = HWY_DYNAMIC_POINTER(facex_hwy::matmul_fp32_packed_dispatch);
    const unsigned long long t0 = gemm_profile_begin();
    fn(A, B_packed, C, M, K, N);
    gemm_profile_record("packed", M, K, N, t0);
    return;
    const int NR = 16;
    const int n_panels = (N + NR - 1) / NR;
    const hn::CappedTag<float, NR> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    int m = 0;
    for (; m + 4 <= M; m += 4) {
        const float* a0 = A + (size_t)(m + 0) * K;
        const float* a1 = A + (size_t)(m + 1) * K;
        const float* a2 = A + (size_t)(m + 2) * K;
        const float* a3 = A + (size_t)(m + 3) * K;
        float* c0 = C + (size_t)(m + 0) * N;
        float* c1 = C + (size_t)(m + 1) * N;
        float* c2 = C + (size_t)(m + 2) * N;
        float* c3 = C + (size_t)(m + 3) * N;
        for (int p = 0; p < n_panels; p++) {
            const int n0 = p * NR;
            const int nr = (n0 + NR <= N) ? NR : (N - n0);
            const float* bp = B_packed + (size_t)p * K * NR;
            int j = 0;
            for (; j + lanes <= nr; j += lanes) {
                auto acc0 = hn::Zero(d);
                auto acc1 = hn::Zero(d);
                auto acc2 = hn::Zero(d);
                auto acc3 = hn::Zero(d);
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR + j);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                }
                hn::StoreU(acc0, d, c0 + n0 + j);
                hn::StoreU(acc1, d, c1 + n0 + j);
                hn::StoreU(acc2, d, c2 + n0 + j);
                hn::StoreU(acc3, d, c3 + n0 + j);
            }
            for (; j < nr; j++) {
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                for (int k = 0; k < K; k++) {
                    const float bv = bp[(size_t)k * NR + j];
                    s0 += a0[k] * bv;
                    s1 += a1[k] * bv;
                    s2 += a2[k] * bv;
                    s3 += a3[k] * bv;
                }
                c0[n0 + j] = s0;
                c1[n0 + j] = s1;
                c2[n0 + j] = s2;
                c3[n0 + j] = s3;
            }
        }
    }
    for (; m < M; m++) {
        const float* ar = A + (size_t)m * K;
        float* cr = C + (size_t)m * N;
        for (int p = 0; p < n_panels; p++) {
            const int n0 = p * NR;
            const int nr = (n0 + NR <= N) ? NR : (N - n0);
            const float* bp = B_packed + (size_t)p * K * NR;
            int j = 0;
            for (; j + lanes <= nr; j += lanes) {
                auto acc = hn::Zero(d);
                for (int k = 0; k < K; k++) {
                    acc = hn::MulAdd(hn::Set(d, ar[k]), hn::LoadU(d, bp + (size_t)k * NR + j), acc);
                }
                hn::StoreU(acc, d, cr + n0 + j);
            }
            for (; j < nr; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) sum += ar[k] * bp[(size_t)k * NR + j];
                cr[n0 + j] = sum;
            }
        }
    }
}

/* ============ Packed MatMul + fused bias ============ */
/* Same as matmul_fp32_packed but adds bias[n] to each output element during store.
 * Eliminates separate bias addition pass. */
void matmul_fp32_packed_bias(const float* A, const float* B_packed, const float* bias,
                             float* C, int M, int K, int N) {
    static const auto fn = HWY_DYNAMIC_POINTER(facex_hwy::matmul_fp32_packed_bias_dispatch);
    const unsigned long long t0 = gemm_profile_begin();
    fn(A, B_packed, bias, C, M, K, N);
    gemm_profile_record("packed_bias", M, K, N, t0);
    return;
    const int NR = 16;
    const int n_panels = (N + NR - 1) / NR;
    const hn::CappedTag<float, NR> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    int m = 0;
    for (; m + 4 <= M; m += 4) {
        const float* a0 = A + (size_t)(m + 0) * K;
        const float* a1 = A + (size_t)(m + 1) * K;
        const float* a2 = A + (size_t)(m + 2) * K;
        const float* a3 = A + (size_t)(m + 3) * K;
        float* c0 = C + (size_t)(m + 0) * N;
        float* c1 = C + (size_t)(m + 1) * N;
        float* c2 = C + (size_t)(m + 2) * N;
        float* c3 = C + (size_t)(m + 3) * N;
        for (int p = 0; p < n_panels; p++) {
            const int n0 = p * NR;
            const int nr = (n0 + NR <= N) ? NR : (N - n0);
            const float* bp = B_packed + (size_t)p * K * NR;
            int j = 0;
            for (; j + lanes <= nr; j += lanes) {
                auto acc0 = hn::Zero(d);
                auto acc1 = hn::Zero(d);
                auto acc2 = hn::Zero(d);
                auto acc3 = hn::Zero(d);
                for (int k = 0; k < K; k++) {
                    const auto b = hn::LoadU(d, bp + (size_t)k * NR + j);
                    acc0 = hn::MulAdd(hn::Set(d, a0[k]), b, acc0);
                    acc1 = hn::MulAdd(hn::Set(d, a1[k]), b, acc1);
                    acc2 = hn::MulAdd(hn::Set(d, a2[k]), b, acc2);
                    acc3 = hn::MulAdd(hn::Set(d, a3[k]), b, acc3);
                }
                if (bias) {
                    const auto vb = hn::LoadU(d, bias + n0 + j);
                    acc0 = hn::Add(acc0, vb);
                    acc1 = hn::Add(acc1, vb);
                    acc2 = hn::Add(acc2, vb);
                    acc3 = hn::Add(acc3, vb);
                }
                hn::StoreU(acc0, d, c0 + n0 + j);
                hn::StoreU(acc1, d, c1 + n0 + j);
                hn::StoreU(acc2, d, c2 + n0 + j);
                hn::StoreU(acc3, d, c3 + n0 + j);
            }
            for (; j < nr; j++) {
                float s0 = bias ? bias[n0 + j] : 0.0f;
                float s1 = s0, s2 = s0, s3 = s0;
                for (int k = 0; k < K; k++) {
                    const float bv = bp[(size_t)k * NR + j];
                    s0 += a0[k] * bv;
                    s1 += a1[k] * bv;
                    s2 += a2[k] * bv;
                    s3 += a3[k] * bv;
                }
                c0[n0 + j] = s0;
                c1[n0 + j] = s1;
                c2[n0 + j] = s2;
                c3[n0 + j] = s3;
            }
        }
    }
    for (; m < M; m++) {
        const float* ar = A + (size_t)m * K;
        float* cr = C + (size_t)m * N;
        for (int p = 0; p < n_panels; p++) {
            const int n0 = p * NR;
            const int nr = (n0 + NR <= N) ? NR : (N - n0);
            const float* bp = B_packed + (size_t)p * K * NR;
            int j = 0;
            for (; j + lanes <= nr; j += lanes) {
                auto acc = hn::Zero(d);
                for (int k = 0; k < K; k++) {
                    acc = hn::MulAdd(hn::Set(d, ar[k]), hn::LoadU(d, bp + (size_t)k * NR + j), acc);
                }
                if (bias) acc = hn::Add(acc, hn::LoadU(d, bias + n0 + j));
                hn::StoreU(acc, d, cr + n0 + j);
            }
            for (; j < nr; j++) {
                float sum = bias ? bias[n0 + j] : 0.0f;
                for (int k = 0; k < K; k++) sum += ar[k] * bp[(size_t)k * NR + j];
                cr[n0 + j] = sum;
            }
        }
    }
}

/* ============ Fused MatMul + Bias + GELU (single memory pass) ============ */
/* Computes: C[m,n] = GELU(A[M,K] × B_packed[K,N] + bias[N])
 * Bias and GELU applied in-register during store — saves 2 memory round-trips.
 * Uses A&S 7.1.26 erf approximation, same as gelu_fp32. */

void matmul_bias_gelu_packed(const float* A, const float* B_packed, const float* bias,
                              float* C, int M, int K, int N) {
    static const auto fn = HWY_DYNAMIC_POINTER(facex_hwy::matmul_bias_gelu_packed_dispatch);
    const unsigned long long t0 = gemm_profile_begin();
    fn(A, B_packed, bias, C, M, K, N);
    gemm_profile_record("packed_bias_gelu", M, K, N, t0);
    return;
    const int NR = 16;
    const int n_panels = (N + NR - 1) / NR;
    const hn::CappedTag<float, NR> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    const auto v_c1 = hn::Set(d, 0.7978845608f);
    const auto v_c2 = hn::Set(d, 0.044715f);
    const auto v_half = hn::Set(d, 0.5f);
    const auto v_one = hn::Set(d, 1.0f);
    const auto v_neg_one = hn::Set(d, -1.0f);
    const auto v_27 = hn::Set(d, 27.0f);
    const auto v_9 = hn::Set(d, 9.0f);
    for (int m = 0; m < M; m++) {
        const float* ar = A + (size_t)m * K;
        float* cr = C + (size_t)m * N;
        for (int p = 0; p < n_panels; p++) {
            const int n0 = p * NR;
            const int nr = (n0 + NR <= N) ? NR : (N - n0);
            const float* bp = B_packed + (size_t)p * K * NR;
            int j = 0;
            for (; j + lanes <= nr; j += lanes) {
                auto v = hn::Zero(d);
                for (int k = 0; k < K; k++) {
                    v = hn::MulAdd(hn::Set(d, ar[k]), hn::LoadU(d, bp + (size_t)k * NR + j), v);
                }
                if (bias) v = hn::Add(v, hn::LoadU(d, bias + n0 + j));
                const auto vx3 = hn::Mul(hn::Mul(v, v), v);
                const auto inner = hn::Mul(v_c1, hn::MulAdd(v_c2, vx3, v));
                const auto inner2 = hn::Mul(inner, inner);
                const auto num = hn::Mul(inner, hn::Add(v_27, inner2));
                const auto den = hn::MulAdd(v_9, inner2, v_27);
                auto t = hn::Mul(num, hn::ApproximateReciprocal(den));
                t = hn::Max(hn::Min(t, v_one), v_neg_one);
                hn::StoreU(hn::Mul(hn::Mul(v_half, v), hn::Add(v_one, t)), d, cr + n0 + j);
            }
            for (; j < nr; j++) {
                float v = bias ? bias[n0 + j] : 0.0f;
                for (int k = 0; k < K; k++) v += ar[k] * bp[(size_t)k * NR + j];
                float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
                float t = inner * (27.0f + inner * inner) / (27.0f + 9.0f * inner * inner);
                if (t > 1.0f) t = 1.0f;
                if (t < -1.0f) t = -1.0f;
                cr[n0 + j] = 0.5f * v * (1.0f + t);
            }
        }
    }
}

void matmul_residual_bias_gamma_packed(const float* A, const float* B_packed,
                                       const float* bias, const float* gamma,
                                       const float* residual, float* C,
                                       int M, int K, int N) {
    static const auto fn = HWY_DYNAMIC_POINTER(facex_hwy::matmul_residual_bias_gamma_packed_dispatch);
    const unsigned long long t0 = gemm_profile_begin();
    fn(A, B_packed, bias, gamma, residual, C, M, K, N);
    gemm_profile_record("packed_residual", M, K, N, t0);
}

/* ============ MatMul + Add (bias) ============ */
void matmul_bias_fp32(const float* A, const float* B, const float* bias,
                      float* C, int M, int K, int N) {
    matmul_fp32(A, B, C, M, K, N);
    if (bias) {
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
                C[(size_t)m * N + n] += bias[n];
    }
}

/* ============ L2 Normalize ============ */
/* Normalize each row to unit L2 norm, with clipping */
void l2_normalize_fp32(float* x, int N, int C, float min_norm) {
    const hn::ScalableTag<float> d;
    const int lanes = static_cast<int>(hn::Lanes(d));
    for (int n = 0; n < N; n++) {
        float* row = x + (size_t)n * C;
        int c = 0;
        auto vacc = hn::Zero(d);
        for (; c + lanes <= C; c += lanes) {
            const auto v = hn::LoadU(d, row + c);
            vacc = hn::MulAdd(v, v, vacc);
        }
        float norm_sq = hn::ReduceSum(d, vacc);
        for (; c < C; c++) norm_sq += row[c] * row[c];
        float norm = sqrtf(norm_sq);
        if (norm < min_norm) norm = min_norm;
        float inv = 1.0f / norm;
        const auto vinv = hn::Set(d, inv);
        c = 0;
        for (; c + lanes <= C; c += lanes) hn::StoreU(hn::Mul(hn::LoadU(d, row + c), vinv), d, row + c);
        for (; c < C; c++) row[c] *= inv;
    }
}

/* ============ Adaptive Average Pool ============ */
/* Pool spatial dims H×W → 1×1, for [N, C, H, W] layout */
void adaptive_avg_pool_fp32(const float* x, int C, int H, int W, float* out) {
    int HW = H * W;
    float inv = 1.0f / HW;
    for (int c = 0; c < C; c++) {
        float sum = 0;
        for (int hw = 0; hw < HW; hw++)
            sum += x[(size_t)c * HW + hw];
        out[c] = sum * inv;
    }
}

/* ============ Depthwise Conv NxN (generalized) ============ */
/* Supports arbitrary kernel size (3,5,7,9), stride 1, pad=K/2 */
/* DW Conv NxN on HWC layout */
void depthwise_conv_nxn_hwc_fp32(
    const float* in, int H, int W, int C,
    const float* weights, /* [C, K, K] */
    const float* bias,
    int K,
    float* out)
{
    int pad = K / 2;

    /* Weights assumed pre-transposed to [K*K, C] layout */
    const float* w_t = weights;
    const hn::ScalableTag<float> d;
    const int lanes = static_cast<int>(hn::Lanes(d));

    for (int oy = 0; oy < H; oy++) {
        for (int ox = 0; ox < W; ox++) {
            float* o = out + ((size_t)oy * W + ox) * C;
            if (oy >= pad && oy + pad < H && ox >= pad && ox + pad < W) {
                int c = 0;
                for (; c + lanes <= C; c += lanes) {
                    auto acc = bias ? hn::LoadU(d, bias + c) : hn::Zero(d);
                    for (int ky = 0; ky < K; ky++) {
                        const int iy = oy - pad + ky;
                        for (int kx = 0; kx < K; kx++) {
                            const int ix = ox - pad + kx;
                            const float* inp = in + ((size_t)iy * W + ix) * C;
                            const float* wt = w_t + (size_t)(ky * K + kx) * C;
                            acc = hn::MulAdd(hn::LoadU(d, inp + c), hn::LoadU(d, wt + c), acc);
                        }
                    }
                    hn::StoreU(acc, d, o + c);
                }
                for (; c < C; c++) {
                    float sum = bias ? bias[c] : 0.0f;
                    for (int ky = 0; ky < K; ky++) {
                        const int iy = oy - pad + ky;
                        for (int kx = 0; kx < K; kx++) {
                            const int ix = ox - pad + kx;
                            const float* inp = in + ((size_t)iy * W + ix) * C;
                            const float* wt = w_t + (size_t)(ky * K + kx) * C;
                            sum += inp[c] * wt[c];
                        }
                    }
                    o[c] = sum;
                }
            } else {
                int c = 0;
                for (; c + lanes <= C; c += lanes) {
                    auto acc = bias ? hn::LoadU(d, bias + c) : hn::Zero(d);
                    for (int ky = 0; ky < K; ky++) {
                        const int iy = oy - pad + ky;
                        if (iy < 0 || iy >= H) continue;
                        for (int kx = 0; kx < K; kx++) {
                            const int ix = ox - pad + kx;
                            if (ix < 0 || ix >= W) continue;
                            const float* inp = in + ((size_t)iy * W + ix) * C;
                            const float* wt = w_t + (size_t)(ky * K + kx) * C;
                            acc = hn::MulAdd(hn::LoadU(d, inp + c), hn::LoadU(d, wt + c), acc);
                        }
                    }
                    hn::StoreU(acc, d, o + c);
                }
                for (; c < C; c++) {
                    float sum = bias ? bias[c] : 0.0f;
                    for (int ky = 0; ky < K; ky++) {
                        const int iy = oy - pad + ky;
                        if (iy < 0 || iy >= H) continue;
                        for (int kx = 0; kx < K; kx++) {
                            const int ix = ox - pad + kx;
                            if (ix < 0 || ix >= W) continue;
                            const float* inp = in + ((size_t)iy * W + ix) * C;
                            const float* wt = w_t + (size_t)(ky * K + kx) * C;
                            sum += inp[c] * wt[c];
                        }
                    }
                    o[c] = sum;
                }
            }
        }
    }
    /* w_t is pre-transposed, no free needed */
}

void depthwise_conv_nxn_fp32(
    const float* in, int H, int W, int C,
    const float* weights, /* [C, 1, K, K] */
    const float* bias,
    int K, /* kernel size */
    float* out)
{
    int pad = K / 2;
    int OH = H, OW = W;

    /* Channel-first: each channel is independent spatial conv */
    for (int c = 0; c < C; c++) {
        const float* in_c = in + (size_t)c * H * W;
        const float* w_c = weights + (size_t)c * K * K;
        float* out_c = out + (size_t)c * OH * OW;
        float b = bias ? bias[c] : 0;

        for (int oy = 0; oy < OH; oy++) {
            for (int ox = 0; ox < OW; ox++) {
                float sum = b;
                for (int ky = 0; ky < K; ky++) {
                    int iy = oy - pad + ky;
                    if (iy < 0 || iy >= H) continue;
                    for (int kx = 0; kx < K; kx++) {
                        int ix = ox - pad + kx;
                        if (ix < 0 || ix >= W) continue;
                        sum += in_c[(size_t)iy * W + ix] * w_c[(size_t)ky * K + kx];
                    }
                }
                out_c[(size_t)oy * OW + ox] = sum;
            }
        }
    }
}

}  /* extern "C" */

#endif  /* HWY_ONCE */
