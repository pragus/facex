/*
 * gemm_int8_4x8c8.cc — Portable INT8 GEMM kernel.
 *
 * Tile: MR=4, NR=8, KR=8 (c8 packing)
 * The packed layout is kept compatible with the original c8 path.
 *
 * Handles signed×signed via offset trick:
 *   A_u8 = A_s8 + 128
 *   compensation = 128 * column_sum(W) per output channel
 *
 * Weight packing layout per NR=8 group:
 *   [8 x int32 bias] [K/8 blocks of 64 bytes: 8 columns, each 8 k-values contiguous]
 *
 * B block layout (64 bytes per KR=8 block):
 *   8 columns × 8 k-values, k-values contiguous per column.
 *
 * Used for: pointwise 1x1 conv and standard 3x3 conv (via im2col)
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "hwy/highway.h"

namespace hn = hwy::HWY_NAMESPACE;

extern "C" {

/* ============ Weight packing ============ */

/*
 * Pack weights from [Cout, K] row-major (int8) to c8 format.
 * Output layout per NR=8 column group:
 *   [8 x int32 bias] [K/8 blocks of 64 bytes]
 *
 * Each 64-byte block: columns 0-3 then 4-7, k-values contiguous per column.
 * Also computes col_sums for signed offset compensation.
 */
void pack_weights_4x8c8(
    const int8_t* weights,  /* [Cout, K] row-major */
    const float* bias,      /* [Cout] or NULL */
    int K, int Cout,
    void* packed_w,         /* output: packed weights */
    int32_t* col_sums)      /* output: 128 * sum(w) per output channel [Cout] */
{
    uint8_t* out = (uint8_t*)packed_w;
    int K_padded = (K + 7) & ~7;

    for (int co = 0; co < Cout; co += 8) {
        /* Write bias (8 x int32) */
        int32_t* bias_ptr = (int32_t*)out;
        for (int j = 0; j < 8; j++) {
            if (bias && (co + j) < Cout)
                bias_ptr[j] = (int32_t)(bias[co + j]);
            else
                bias_ptr[j] = 0;
        }
        out += 8 * sizeof(int32_t);

        /* Compute column sums for compensation */
        for (int j = 0; j < 8; j++) {
            int32_t sum = 0;
            if ((co + j) < Cout) {
                for (int k = 0; k < K; k++) {
                    sum += (int32_t)weights[(co + j) * K + k];
                }
            }
            col_sums[co + j] = 128 * sum;
        }

        /* Pack weights in c8 layout: [col, 8 k-values]. */
        for (int k = 0; k < K_padded; k += 8) {
            for (int j = 0; j < 8; j++) {
                for (int kk = 0; kk < 8; kk++) {
                    if ((co + j) < Cout && (k + kk) < K) {
                        *out++ = (uint8_t)weights[(co + j) * K + (k + kk)];
                    } else {
                        *out++ = 0;
                    }
                }
            }
        }
    }
}

/* Size of packed weights buffer */
int packed_weights_size_4x8c8(int K, int Cout) {
    int K_padded = (K + 7) & ~7;
    int nr_groups = (Cout + 7) / 8;
    /* Per group: 8*int32 bias + K_padded*8 bytes weights */
    return nr_groups * (8 * sizeof(int32_t) + K_padded * 8);
}

/* ============ Activation packing: signed→unsigned ============ */

void convert_s8_to_u8(const int8_t* in, uint8_t* out, int n) {
    int i = 0;
    const hn::ScalableTag<uint8_t> d;
    const size_t lanes = hn::Lanes(d);
    const auto flip = hn::Set(d, 0x80u);
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(in);
    for (; i + static_cast<int>(lanes) <= n; i += static_cast<int>(lanes)) {
        hn::StoreU(hn::Xor(hn::LoadU(d, bytes + i), flip), d, out + i);
    }
    for (; i < n; i++) {
        out[i] = (uint8_t)((int)in[i] + 128);
    }
}

/* ============ GEMM Microkernel ============ */

/*
 * gemm_4x8c8_ukernel: MR=4 rows × NR=8 columns.
 *
 * Portable implementation over the c8 packed layout.
 */
static void gemm_4x8c8_ukernel(
    const int8_t* a, int a_stride,
    const int8_t* w,
    int32_t* c, int c_stride,
    int K, int mr)
{
    for (int r = 0; r < mr; r++) {
        const int8_t* ar = a + (size_t)r * a_stride;
        int32_t acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        for (int k = 0; k < K; k += 8) {
            const int8_t* wb = w + (size_t)(k / 8) * 64;
            for (int j = 0; j < 8; j++) {
                const int8_t* wj = wb + j * 8;
                int32_t sum = 0;
                for (int kk = 0; kk < 8; kk++) {
                    sum += ((int32_t)ar[k + kk] + 128) * (int32_t)wj[kk];
                }
                acc[j] += sum;
            }
        }

        memcpy(c + (size_t)r * c_stride, acc, sizeof(acc));
    }
}

/* ============ Full GEMM with blocking ============ */

/* Thread-local workspace for A_u8 conversion to avoid malloc per call */
static __thread uint8_t* tls_a_u8 = NULL;
static __thread size_t tls_a_u8_size = 0;

static uint8_t* get_a_u8_workspace(size_t needed) {
    if (needed > tls_a_u8_size) {
        free(tls_a_u8);
        tls_a_u8 = (uint8_t*)malloc(needed);
        tls_a_u8_size = needed;
    }
    return tls_a_u8;
}

void int8_gemm_4x8c8(
    const int8_t* A, int M, int K, int N,
    const void* B_packed,
    int32_t* C,
    const int32_t* col_sums)
{
    int K_padded = (K + 7) & ~7;

    /* With inline s8→u8 XOR in microkernel, no pre-conversion needed.
     * Just pad A to K_padded if needed. */
    const int8_t* A_eff = A;
    int A_stride = K;
    int8_t* A_pad = NULL;
    if (K != K_padded) {
        A_pad = (int8_t*)get_a_u8_workspace((size_t)M * K_padded);
        for (int m = 0; m < M; m++) {
            memcpy(A_pad + (size_t)m * K_padded, A + (size_t)m * K, K);
            memset(A_pad + (size_t)m * K_padded + K, 0, K_padded - K);
        }
        A_eff = A_pad;
        A_stride = K_padded;
    }

    #pragma omp parallel for schedule(dynamic)
    for (int m = 0; m < M; m += 4) {
        int mr = (m + 4 <= M) ? 4 : M - m;

        const uint8_t* w_ptr = (const uint8_t*)B_packed;

        for (int n = 0; n < N; n += 8) {
            const int8_t* w_data = (const int8_t*)(w_ptr + 32);

            int32_t acc[4 * 8] __attribute__((aligned(32)));
            memset(acc, 0, sizeof(acc));

            gemm_4x8c8_ukernel(
                A_eff + (size_t)m * A_stride, A_stride,
                w_data,
                acc, 8,
                K_padded, mr);

            /* Apply compensation and store to C */
            for (int i = 0; i < mr; i++) {
                for (int j = 0; j < 8 && (n + j) < N; j++) {
                    C[(size_t)(m + i) * N + (n + j)] = acc[i * 8 + j] - col_sums[n + j];
                }
            }

            /* Advance weight pointer to next NR group */
            w_ptr += 32 + (size_t)K_padded * 8; /* bias + weights */
        }
    }
    /* A_u8 is workspace-managed, not freed here */
}

/* ============ Fused GEMM + epilogue (dequant+bias+ReLU+requant → int8 output) ============ */

void int8_gemm_4x8c8_fused(
    const int8_t* A, int M, int K, int N,
    const void* B_packed,
    int8_t* out,           /* output: [M, N] int8 (fused result) */
    const int32_t* col_sums,
    const float* w_scales,   /* [N] weight scales */
    const float* bias,       /* [N] bias or NULL */
    const float* act_scale,  /* [N] output activation scale (per-tensor: all same) */
    int do_relu)
{
    int K_padded = (K + 7) & ~7;

    /* No pre-conversion: microkernel XORs s8→u8 inline */
    const int8_t* A_eff = A;
    int A_stride = K;
    int8_t* A_pad = NULL;
    if (K != K_padded) {
        A_pad = (int8_t*)get_a_u8_workspace((size_t)M * K_padded);
        for (int m = 0; m < M; m++) {
            memcpy(A_pad + (size_t)m * K_padded, A + (size_t)m * K, K);
            memset(A_pad + (size_t)m * K_padded + K, 0, K_padded - K);
        }
        A_eff = A_pad;
        A_stride = K_padded;
    }

    /* Precompute per-channel: combined_scale = 1.0 * w_scale (folded: in_scale=1.0) */
    float inv_out[512];
    for (int c = 0; c < N; c++)
        inv_out[c] = 1.0f / (act_scale[c] + 1e-9f);

    #pragma omp parallel for schedule(dynamic)
    for (int m = 0; m < M; m += 4) {
        int mr = (m + 4 <= M) ? 4 : M - m;
        const uint8_t* w_ptr = (const uint8_t*)B_packed;

        for (int n = 0; n < N; n += 8) {
            const int8_t* w_data = (const int8_t*)(w_ptr + 32);

            int32_t acc[4 * 8] __attribute__((aligned(32)));
            memset(acc, 0, sizeof(acc));

            gemm_4x8c8_ukernel(
                A_eff + (size_t)m * A_stride, A_stride,
                w_data, acc, 8, K_padded, mr);

            /* Fused: compensate + dequant + bias + relu + requant → int8 */
            for (int i = 0; i < mr; i++) {
                int8_t* orow = out + (size_t)(m + i) * N;
                for (int j = 0; j < 8 && (n + j) < N; j++) {
                    int c = n + j;
                    float fp = (float)(acc[i * 8 + j] - col_sums[c]) * w_scales[c];
                    if (bias) fp += bias[c];
                    if (do_relu && fp < 0) fp = 0;
                    int q = (int)lrintf(fp * inv_out[c]);
                    if (q > 127) q = 127;
                    if (q < -128) q = -128;
                    orow[c] = (int8_t)q;
                }
            }

            w_ptr += 32 + (size_t)K_padded * 8;
        }
    }
}

}  /* extern "C" */
