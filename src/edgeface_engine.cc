/*
 * edgeface_engine.c — Native FP32 EdgeFace-XS embedding engine.
 *
 * Loads weights from edgeface_xs_fp32.bin, runs forward pass,
 * outputs 512-dim embedding. Zero dependencies.
 *
 * Architecture: 4 stages × ConvNeXt blocks + 3 SDPA attention + head.
 * All ops in FP32 with portable MatMul.
 *
 * Build: gcc -O3 -o fastface_edgexs.exe \
 *   edgeface_engine.c kernels/transformer_ops.c kernels/gemm_int8_4x8c8.cc -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Weight decryption (optional, linked from crypto/weight_crypto.c) */
#ifndef NO_CRYPTO
extern "C" {
extern int weight_decrypt_inplace(uint8_t* raw, size_t raw_size, const char* license);
}
#endif

/* External ops */
extern "C" {
extern void layer_norm_fp32(const float*, int, int, const float*, const float*, float, float*);
extern void gelu_fp32(float*, int);
extern void softmax_fp32(float*, int, int);
extern void matmul_fp32(const float*, const float*, float*, int, int, int);
extern void l2_normalize_fp32(float*, int, int, float);
extern void add_fp32(float*, const float*, int);
extern void scale_fp32(float*, int, float);
extern void bias_gamma_residual_fp32(const float*, float*, const float*, const float*, const float*, int, int);
extern void depthwise_conv_nxn_hwc_fp32(const float*, int, int, int, const float*, const float*, int, float*);
extern void matmul_fp32_packed(const float*, const float*, float*, int, int, int);
extern void matmul_fp32_packed_bias(const float*, const float*, const float*, float*, int, int, int);
extern void matmul_bias_gelu_packed(const float*, const float*, const float*, float*, int, int, int);
extern void matmul_residual_bias_gamma_packed(const float*, const float*, const float*, const float*, const float*, float*, int, int, int);
extern void pack_weights_4x8c8(const int8_t*, const float*, int, int, void*, int32_t*);
extern int packed_weights_size_4x8c8(int, int);
extern void pack_b_fp32(const float*, int, int, float*);
extern int packed_b_fp32_size(int, int);
}

/* ============ Weight loader ============ */
/* INT8 packed MatMul weight */
typedef struct {
    void* packed;
    int32_t* col_sums;
    float* w_scales;
    int K, N;  /* original dimensions */
} PackedMM;

/* FP32 pre-packed MatMul weight (column-panel format for cache-friendly access) */
typedef struct {
    float* data;  /* packed [ceil(N/16), K, 16] */
    int K, N;
} PackedFP32;

typedef struct {
    float** tensors;   /* array of float pointers */
    int n_tensors;
    uint8_t* raw;      /* raw file data */
    size_t raw_size;
    /* INT8 pre-packed MatMul weights */
    PackedMM* mm;      /* array, indexed same as tensors */
    /* FP32 pre-packed MatMul weights */
    PackedFP32* fp;    /* array, indexed same as tensors */
} Weights;

static const char* g_license_key = NULL;

static int load_weights(const char* path, Weights* w) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    fseek(f, 0, SEEK_END);
    w->raw_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    w->raw = (uint8_t*)malloc(w->raw_size);
    if (!w->raw) {
        fclose(f);
        return -1;
    }

    size_t read_sz = fread(w->raw, 1, w->raw_size, f);
    fclose(f);
    if (read_sz != w->raw_size) {
        free(w->raw);
        w->raw = NULL;
        return -1;
    }

    /* Handle encrypted weights (EFXE magic) */
#ifndef NO_CRYPTO
    if (w->raw_size >= 20 && memcmp(w->raw, "EFXE", 4) == 0) {
        if (!g_license_key || !g_license_key[0]) {
            fprintf(stderr, "ERROR: Encrypted weights require LICENSE_KEY\n");
            fprintf(stderr, "Set LICENSE_KEY environment variable or pass --license KEY\n");
            free(w->raw);
            return -3;
        }
        int r = weight_decrypt_inplace(w->raw, w->raw_size, g_license_key);
        if (r == -2) {
            fprintf(stderr, "ERROR: Invalid license key for this machine\n");
            free(w->raw);
            return -2;
        } else if (r != 0) {
            fprintf(stderr, "ERROR: Decryption failed (%d)\n", r);
            free(w->raw);
            return r;
        }
        /* After decryption, EFXS payload starts at offset 20 */
        uint8_t* payload = w->raw + 20;
        size_t payload_size = w->raw_size - 20;

        if (memcmp(payload, "EFXS", 4) != 0) {
            free(w->raw);
            return -1;
        }

        uint32_t n = *(uint32_t*)(payload + 4);
        w->n_tensors = n;
        w->tensors = (float**)malloc(n * sizeof(float*));

        size_t off = 8;
        for (uint32_t i = 0; i < n; i++) {
            uint32_t sz = *(uint32_t*)(payload + off);
            off += 4;
            w->tensors[i] = (float*)(payload + off);
            off += sz;
        }
        return 0;
    }
#endif

    /* Unencrypted weights (EFXS magic) */
    if (memcmp(w->raw, "EFXS", 4) != 0) return -1;

    uint32_t n = *(uint32_t*)(w->raw + 4);
    w->n_tensors = n;
    w->tensors = (float**)malloc(n * sizeof(float*));

    size_t off = 8;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t sz = *(uint32_t*)(w->raw + off);
        off += 4;
        w->tensors[i] = (float*)(w->raw + off);
        off += sz;
    }

    return 0;
}

#define W(idx) (weights->tensors[idx])

/* Forward declarations */
static void matmul_w(const float* A, const float* B, float* C,
                     int M, int K, int N, const PackedMM* mm);
static void matmul_wp(const float* A, float* C,
                      int M, int K, int N, const PackedFP32* fp);
static void matmul_wpb(const float* A, float* C, const float* bias,
                       int M, int K, int N, const PackedFP32* fp);
static void matmul_wpbg(const float* A, float* C, const float* bias,
                        int M, int K, int N, const PackedFP32* fp);
static void matmul_wpbrg(const float* A, float* C, const float* bias,
                         const float* gamma, const float* residual,
                         int M, int K, int N, const PackedFP32* fp);

/* ============ XCA (Cross-Covariance Attention) Block ============ */
/* EdgeFace XCA: Split+DW conv → pos embed →
 * LN → QKV project → reshape [n_heads, head_dim, HW] →
 * L2 norm Q,K along HW → Q@K^T * temperature → softmax → @V →
 * reshape → project → gamma_xca * output + residual → LN → MLP → gamma + residual */

/* DW conv for XCA: CASCADED structure.
 * Split channels → Conv(split_0) → result_0
 * → result_0 + split_1 → Conv → result_1
 * → result_1 + split_2 → Conv → result_2
 * → Concat [result_0, result_1, ..., result_{n-1}, split_last_unchanged]
 * Input/output in NCHW. Each conv weight is [split_ch, 1, K, K] DW conv. */
static void xca_dw_conv_nchw(const float* in_nchw, int C, int H, int W,
                              int n_splits, const int* split_sizes,
                              int n_convs, const float** conv_ws, const float** conv_bs,
                              int K, float* out_nchw, float* conv_scratch) {
    int HW = H * W;
    int pad = K / 2;
    int ch_offset = 0;
    float* prev_result = NULL; /* pointer to previous conv result in out_nchw */

    for (int s = 0; s < n_splits; s++) {
        int sc = split_sizes[s];
        if (s < n_convs && conv_ws[s]) {
            const float* w = conv_ws[s]; /* [sc, 1, K, K] */
            const float* b = conv_bs[s]; /* [sc] */

            /* Prepare input: for first conv, use split directly.
             * For subsequent convs, input = prev_result + current_split */
            const float* conv_in;
            if (s == 0) {
                conv_in = in_nchw + (size_t)ch_offset * HW; /* split_0 */
            } else {
                /* Add prev_result + current_split → scratch */
                const float* split_data = in_nchw + (size_t)ch_offset * HW;
                for (int i = 0; i < sc * HW; i++)
                    conv_scratch[i] = prev_result[i] + split_data[i];
                conv_in = conv_scratch;
            }

            /* DW conv */
            float* result = out_nchw + (size_t)ch_offset * HW;
            for (int c = 0; c < sc; c++) {
                for (int oy = 0; oy < H; oy++)
                    for (int ox = 0; ox < W; ox++) {
                        float sum = b ? b[c] : 0;
                        for (int ky = 0; ky < K; ky++) {
                            int iy = oy - pad + ky;
                            if (iy < 0 || iy >= H) continue;
                            for (int kx = 0; kx < K; kx++) {
                                int ix = ox - pad + kx;
                                if (ix < 0 || ix >= W) continue;
                                sum += conv_in[c * HW + iy * W + ix] *
                                       w[(c * K + ky) * K + kx];
                            }
                        }
                        result[c * HW + oy * W + ox] = sum;
                    }
            }
            prev_result = result;
        } else {
            /* Copy unchanged */
            memcpy(out_nchw + (size_t)ch_offset * HW,
                   in_nchw + (size_t)ch_offset * HW,
                   (size_t)sc * HW * sizeof(float));
        }
        ch_offset += sc;
    }
}

static void xca_block(float* x_hwc, int H, int W, int C,
                      const float* gamma_xca, const float* gamma,
                      /* DW conv for split heads */
                      int n_dw_splits, const int* dw_split_sizes,
                      int n_dw_convs, const float** dw_ws, const float** dw_bs, int dw_K,
                      /* Position embedding: Conv1x1 on CONSTANT pos map */
                      const float* pos_const_nchw, /* [1,C,H,W] or NULL */
                      const float* pos_w, const float* pos_b, /* Conv1x1 [C,C,1,1] OIHW */
                      /* XCA attention */
                      const float* xca_ln_w, const float* xca_ln_b,
                      const float* qkv_w0, const float* qkv_w1, const float* qkv_b,
                      const float* temperature, int n_heads,
                      const float* proj_w0, const float* proj_w1, const float* proj_b,
                      /* Post-attention MLP */
                      const float* ln_w, const float* ln_b,
                      const float* mlp_w0, const float* mlp_w1, const float* mlp_b1,
                      const float* mlp_w2, const float* mlp_w3, const float* mlp_b3,
                      int rank, int hidden,
                      float* tmp,
                      const PackedFP32* fp_qkv0, const PackedFP32* fp_qkv1,
                      const PackedFP32* fp_proj0, const PackedFP32* fp_proj1,
                      const PackedFP32* fp_mlp0, const PackedFP32* fp_mlp1,
                      const PackedFP32* fp_mlp2, const PackedFP32* fp_mlp3)
{
    int HW = H * W;
    int head_dim = C / n_heads;

    /* Workspace layout: residual, nchw buf, qkv, attention scratch */
    float* residual = tmp;
    float* nchw_buf = residual + HW * C;
    float* t1 = nchw_buf + C * HW;  /* after NCHW buffer */
    float* t2 = t1 + HW * C * 3 + HW * rank; /* scratch */

    /* Save ORIGINAL input as residual for MLP at the end */
    float* residual_orig = residual;
    float* residual_dw = nchw_buf; /* reuse nchw_buf for second residual after DW section */
    memcpy(residual_orig, x_hwc, (size_t)HW * C * sizeof(float));

    /* 1. DW conv: HWC→NCHW, split/conv/concat, NCHW→HWC */
    if (n_dw_convs > 0 && dw_ws) {
        /* HWC → NCHW (use t1 as temp) */
        float* nchw_in = t1;
        for (int hw = 0; hw < HW; hw++)
            for (int c = 0; c < C; c++)
                nchw_in[c * HW + hw] = x_hwc[hw * C + c];

        float* nchw_out = nchw_in + C * HW; /* separate output */
        float* conv_scratch = nchw_out + C * HW; /* scratch for cascaded adds */
        xca_dw_conv_nchw(nchw_in, C, H, W, n_dw_splits, dw_split_sizes,
                         n_dw_convs, dw_ws, dw_bs, dw_K, nchw_out, conv_scratch);

        /* NCHW → HWC */
        for (int hw = 0; hw < HW; hw++)
            for (int c = 0; c < C; c++)
                x_hwc[hw * C + c] = nchw_out[c * HW + hw];
    }

    /* 2. Position embedding: add pre-computed constant (cached at load time) */
    if (pos_const_nchw) {
        /* pos_const_nchw now points to pre-computed [HW, C] HWC result */
        add_fp32(x_hwc, pos_const_nchw, HW * C);
    }

    /* Save x_after_dw_pos as residual for attention */
    memcpy(residual_dw, x_hwc, (size_t)HW * C * sizeof(float));

    /* 3. LayerNorm */
    layer_norm_fp32(x_hwc, HW, C, xca_ln_w, xca_ln_b, 1e-6f, x_hwc);

    /* 4. QKV projection: LoRaLin [HW,C] → [HW,3C] */
    int qkv_dim = C * 3;
    float* qkv = t1;
    float* qkv_temp = t1 + HW * qkv_dim;
    matmul_wp(x_hwc, qkv_temp, HW, C, rank, fp_qkv0);
    matmul_wpb(qkv_temp, qkv, qkv_b, HW, rank, qkv_dim, fp_qkv1);

    /* 5. Reshape QKV: [HW, 3C] → [HW, 3, n_heads, head_dim] →
     *    transpose to [3, n_heads, head_dim, HW] → split Q,K,V */
    /* Each Q,K,V: [n_heads, head_dim, HW] */
    float* Q_nhd = t2;  /* [n_heads, head_dim, HW] */
    float* K_nhd = Q_nhd + n_heads * head_dim * HW;
    float* V_nhd = K_nhd + n_heads * head_dim * HW;

    /* QKV layout from LoRaLin: [HW, 3*C] where 3*C = Q0..QC, K0..KC, V0..VC
     * But ONNX reshape is [HW, 3, n_heads, head_dim], so element at
     * qkv[hw, q, h, d] = qkv_flat[hw * 3*n_heads*head_dim + q*n_heads*head_dim + h*head_dim + d]
     * Transpose to [q, h, d, hw]: Q[h,d,hw] = qkv[hw, 0, h, d] */
    for (int hw = 0; hw < HW; hw++)
        for (int h = 0; h < n_heads; h++)
            for (int d = 0; d < head_dim; d++) {
                int qkv_idx = hw * qkv_dim + 0 * C + h * head_dim + d;
                Q_nhd[h * head_dim * HW + d * HW + hw] = qkv[qkv_idx];
                qkv_idx = hw * qkv_dim + 1 * C + h * head_dim + d;
                K_nhd[h * head_dim * HW + d * HW + hw] = qkv[qkv_idx];
                qkv_idx = hw * qkv_dim + 2 * C + h * head_dim + d;
                V_nhd[h * head_dim * HW + d * HW + hw] = qkv[qkv_idx];
            }

    /* 6. L2 normalize Q and K along HW dim */
    for (int h = 0; h < n_heads; h++)
        for (int d = 0; d < head_dim; d++) {
            float* q_row = Q_nhd + h * head_dim * HW + d * HW;
            float* k_row = K_nhd + h * head_dim * HW + d * HW;
            l2_normalize_fp32(q_row, 1, HW, 1e-12f);
            l2_normalize_fp32(k_row, 1, HW, 1e-12f);
        }

    /* 7. Per-head channel attention using matmul_fp32 for Q@K^T and attn@V */
    float* attn_buf = V_nhd + n_heads * head_dim * HW;
    /* Transpose K: [head_dim, HW] → [HW, head_dim] for matmul compatibility */
    float* K_T = attn_buf + n_heads * head_dim * head_dim; /* temp [HW, head_dim] per head */
    for (int h = 0; h < n_heads; h++) {
        float* Q_h = Q_nhd + h * head_dim * HW;  /* [head_dim, HW] */
        float* K_h = K_nhd + h * head_dim * HW;  /* [head_dim, HW] */
        float* A_h = attn_buf + h * head_dim * head_dim; /* [head_dim, head_dim] */

        /* Transpose K_h [head_dim, HW] → K_T [HW, head_dim] */
        for (int d = 0; d < head_dim; d++)
            for (int s = 0; s < HW; s++)
                K_T[s * head_dim + d] = K_h[d * HW + s];

        /* A = Q @ K^T: [head_dim, HW] @ [HW, head_dim] = [head_dim, head_dim] */
        matmul_fp32(Q_h, K_T, A_h, head_dim, HW, head_dim);

        /* Scale by temperature */
        if (temperature) {
            float temp = temperature[h];
            scale_fp32(A_h, head_dim * head_dim, temp);
        }

        /* Softmax per row */
        softmax_fp32(A_h, head_dim, head_dim);

        /* out = A @ V: [head_dim, head_dim] @ [head_dim, HW] = [head_dim, HW] */
        float* V_h = V_nhd + h * head_dim * HW;
        float* out_h = Q_nhd + h * head_dim * HW; /* reuse Q buffer */
        matmul_fp32(A_h, V_h, out_h, head_dim, head_dim, HW);
    }

    /* 8. Reshape attention output: [n_heads, head_dim, HW] → [HW, C] */
    for (int hw = 0; hw < HW; hw++)
        for (int h = 0; h < n_heads; h++)
            for (int d = 0; d < head_dim; d++)
                x_hwc[hw * C + h * head_dim + d] = Q_nhd[h * head_dim * HW + d * HW + hw];

    /* 9. Output projection: LoRaLin [HW,C] → [HW,C] */
    if (fp_proj0 && fp_proj0->data && fp_proj0->K == C && fp_proj0->N == C) {
        float* proj_temp = t1;
        matmul_wpb(x_hwc, proj_temp, proj_b, HW, C, C, fp_proj0);
        memcpy(x_hwc, proj_temp, (size_t)HW * C * sizeof(float));
    } else {
        float* proj_temp = t1;
        matmul_wp(x_hwc, proj_temp, HW, C, rank, fp_proj0);
        matmul_wpb(proj_temp, x_hwc, proj_b, HW, rank, C, fp_proj1);
    }

    /* 10. gamma_xca * output + residual_dw (attention residual = after DW+pos) */
    bias_gamma_residual_fp32(x_hwc, x_hwc, NULL, gamma_xca, residual_dw, HW, C);

    /* 11. LN + MLP + gamma + residual_orig (MLP residual = original block input) */
    layer_norm_fp32(x_hwc, HW, C, ln_w, ln_b, 1e-6f, x_hwc);

    float* mt1 = t1;
    float* mt2 = mt1 + HW * rank;
    matmul_wp(x_hwc, mt1, HW, C, rank, fp_mlp0);
    matmul_wpbg(mt1, mt2, mlp_b1, HW, rank, hidden, fp_mlp1);
    matmul_wp(mt2, mt1, HW, hidden, rank, fp_mlp2);
    matmul_wpbrg(mt1, x_hwc, mlp_b3, gamma, residual_orig, HW, rank, C, fp_mlp3);
}

/* ============ Conv2D FP32 (HWC in → HWC out) ============ */
/* Standard conv with OIHW weights, operating on HWC tensors.
 * portable: compute across output channels (Cout). */
/* conv2d_hwc with pre-transposed weights in [Cin*KK, Cout] layout.
 * At load time, call conv2d_hwc_reorder_weights() to transpose OIHW → [Cin*KK, Cout]. */
static void conv2d_hwc_reorder_weights(float* w, int Cout, int Cin, int KK) {
    /* In-place transpose from [Cout, Cin*KK] to [Cin*KK, Cout] */
    float* tmp = (float*)malloc((size_t)Cout * Cin * KK * sizeof(float));
    memcpy(tmp, w, (size_t)Cout * Cin * KK * sizeof(float));
    for (int co = 0; co < Cout; co++)
        for (int ci_ki = 0; ci_ki < Cin * KK; ci_ki++)
            w[ci_ki * Cout + co] = tmp[co * Cin * KK + ci_ki];
    free(tmp);
}

static void conv2d_hwc(const float* in_hwc, int Cin, int H, int W,
                       const float* w, /* REORDERED to [Cin*KK, Cout] */
                       const float* b, int Cout, int K, int stride,
                       float* out_hwc) {
    int Ho = (H - K) / stride + 1, Wo = (W - K) / stride + 1;
    int KK = K * K;
    for (int oy = 0; oy < Ho; oy++)
        for (int ox = 0; ox < Wo; ox++) {
            float* o = out_hwc + ((size_t)oy * Wo + ox) * Cout;
            int co = 0;
            for (; co < Cout; co++) o[co] = b ? b[co] : 0;
            for (int ky = 0; ky < K; ky++)
                for (int kx = 0; kx < K; kx++) {
                    int iy = oy * stride + ky, ix = ox * stride + kx;
                    const float* inp = in_hwc + ((size_t)iy * W + ix) * Cin;
                    int ki = ky * K + kx;
                    for (int ci = 0; ci < Cin; ci++) {
                        float in_val = inp[ci];
                        /* Reordered w: w[(ci*KK+ki)*Cout + co] — CONTIGUOUS along Cout! */
                        const float* wrow = w + (size_t)(ci * KK + ki) * Cout;
                        co = 0;
                        for (; co < Cout; co++) o[co] += in_val * wrow[co];
                    }
                }
        }
}

static void conv2d_hwc_packed(const float* in_hwc, int Cin, int H, int W,
                              const float* bias, int Cout, int K, int stride,
                              const PackedFP32* fp, float* im2col,
                              float* out_hwc) {
    int Ho = (H - K) / stride + 1, Wo = (W - K) / stride + 1;
    int row_k = Cin * K * K;
    for (int oy = 0; oy < Ho; oy++)
        for (int ox = 0; ox < Wo; ox++) {
            float* row = im2col + ((size_t)oy * Wo + ox) * row_k;
            int p = 0;
            for (int ci = 0; ci < Cin; ci++)
                for (int ky = 0; ky < K; ky++)
                    for (int kx = 0; kx < K; kx++)
                        row[p++] = in_hwc[((size_t)(oy * stride + ky) * W + ox * stride + kx) * Cin + ci];
        }
    matmul_fp32_packed_bias(im2col, fp->data, bias, out_hwc, Ho * Wo, row_k, Cout);
}

/* ============ Conv2D FP32 (NCHW) ============ */
static void conv2d(const float* in, int Cin, int H, int W,
                   const float* w, const float* b, int Cout, int K, int stride,
                   float* out) {
    int Ho = (H - K) / stride + 1, Wo = (W - K) / stride + 1;
    for (int co = 0; co < Cout; co++) {
        float bias = b ? b[co] : 0;
        for (int oy = 0; oy < Ho; oy++)
            for (int ox = 0; ox < Wo; ox++) {
                float sum = bias;
                for (int ci = 0; ci < Cin; ci++)
                    for (int ky = 0; ky < K; ky++)
                        for (int kx = 0; kx < K; kx++)
                            sum += in[((size_t)ci*H + oy*stride+ky)*W + ox*stride+kx] *
                                   w[(((size_t)co*Cin+ci)*K+ky)*K+kx];
                out[((size_t)co*Ho+oy)*Wo+ox] = sum;
            }
    }
}

/* ============ MatMul with optional INT8 acceleration ============ */
static void matmul_auto(const float* A, int M, int K, int N,
                        const float* B_fp32, const PackedMM* mm)
{
    /* Allocate output — caller must know where to put result */
    /* Actually this doesn't work — we need output pointer. Rethink. */
}

static void matmul_w(const float* A, const float* B, float* C,
                     int M, int K, int N, const PackedMM* mm)
{
    (void)mm; /* INT8 disabled for now */
    /* Check for FP32 packed weight — stored in the global weights struct */
    /* We pass B as the ORIGINAL weight pointer; the caller checks fp[] */
    matmul_fp32(A, B, C, M, K, N);
}

/* Packed FP32 matmul — uses pre-packed weights for cache-friendly access.
 * Optional bias: added to each output row during store (avoids separate pass). */

extern void matmul_specialized(const float*, const float*, float*, int, int, int);
extern void matmul_jit_or_packed(const float*, const float*, float*, int, int, int);

static Weights* _g_weights = NULL; /* set in forward pass for INT8 lookup */

/* Zero-overhead matmul: inline, direct call, no dispatch */
static inline void matmul_wp(const float* A, float* C,
                             int M, int K, int N, const PackedFP32* fp)
{
    matmul_fp32_packed(A, fp->data, C, M, fp->K, fp->N);
}
static void matmul_wpb(const float* A, float* C, const float* bias,
                       int M, int K, int N, const PackedFP32* fp)
{
    if (fp && fp->data)
        matmul_fp32_packed_bias(A, fp->data, bias, C, M, K, N);
}

static void matmul_wpbg(const float* A, float* C, const float* bias,
                        int M, int K, int N, const PackedFP32* fp)
{
    if (fp && fp->data)
        matmul_bias_gelu_packed(A, fp->data, bias, C, M, K, N);
}

static void matmul_wpbrg(const float* A, float* C, const float* bias,
                         const float* gamma, const float* residual,
                         int M, int K, int N, const PackedFP32* fp)
{
    if (fp && fp->data)
        matmul_residual_bias_gamma_packed(A, fp->data, bias, gamma, residual, C, M, K, N);
}

template <int Channels>
static inline void hwc_mean(const float* in, int HW, float* out) {
    for (int c = 0; c < Channels; c++) {
        float sum = 0.0f;
        for (int hw = 0; hw < HW; hw++) sum += in[(size_t)hw * Channels + c];
        out[c] = sum / HW;
    }
}

static void pack_transposed_fc_weight(const float* wt_nk, int K, int N,
                                      PackedFP32* fp) {
    float* wt_kn = (float*)malloc((size_t)K * N * sizeof(float));
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++)
            wt_kn[(size_t)k * N + n] = wt_nk[(size_t)n * K + k];

    int sz = packed_b_fp32_size(K, N);
    float* packed;
#ifdef _WIN32
    packed = (float*)_aligned_malloc((size_t)sz * sizeof(float), 64);
#else
    if (posix_memalign((void**)&packed, 64, (size_t)sz * sizeof(float)) != 0) packed = NULL;
#endif
    if (!packed) packed = (float*)malloc((size_t)sz * sizeof(float));
    pack_b_fp32(wt_kn, K, N, packed);
    fp->data = packed;
    fp->K = K;
    fp->N = N;
    free(wt_kn);
}

static void pack_folded_projection_weight(const float* w0, const float* w1,
                                          int C, int rank, PackedFP32* fp) {
    float* folded = (float*)calloc((size_t)C * C, sizeof(float));
    for (int k = 0; k < C; k++)
        for (int r = 0; r < rank; r++) {
            const float a = w0[(size_t)k * rank + r];
            const float* w1_row = w1 + (size_t)r * C;
            float* out = folded + (size_t)k * C;
            for (int n = 0; n < C; n++) out[n] += a * w1_row[n];
        }

    int sz = packed_b_fp32_size(C, C);
    float* packed;
#ifdef _WIN32
    packed = (float*)_aligned_malloc((size_t)sz * sizeof(float), 64);
#else
    if (posix_memalign((void**)&packed, 64, (size_t)sz * sizeof(float)) != 0) packed = NULL;
#endif
    if (!packed) packed = (float*)malloc((size_t)sz * sizeof(float));
    pack_b_fp32(folded, C, C, packed);
    fp->data = packed;
    fp->K = C;
    fp->N = C;
    free(folded);
}

/* ============ Threaded MLP worker (file scope for proper compilation) ============ */
#include "threadpool.h"
typedef struct {
    float *x, *t1, *t2, *residual;
    const float *gamma, *mlp_b1, *mlp_b3;
    const PackedFP32 *fp0, *fp1, *fp2, *fp3;
    int C, rank, hidden;
} MlpCtx;

static void _mlp_rows(void* ctx_, int start, int end) {
    MlpCtx* g = (MlpCtx*)ctx_;
    int rows = end - start;
    if (rows <= 0) return;
    float* x_s = g->x + start * g->C;
    float* t1_s = g->t1 + start * g->rank;
    float* t2_s = g->t2 + start * g->hidden;
    float* res_s = g->residual + start * g->C;
    matmul_fp32_packed(x_s, g->fp0->data, t1_s, rows, g->fp0->K, g->fp0->N);
    matmul_bias_gelu_packed(t1_s, g->fp1->data, g->mlp_b1, t2_s, rows, g->fp1->K, g->fp1->N);
    matmul_fp32_packed(t2_s, g->fp2->data, t1_s, rows, g->fp2->K, g->fp2->N);
    matmul_fp32_packed(t1_s, g->fp3->data, x_s, rows, g->fp3->K, g->fp3->N);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < g->C; c++)
            x_s[r*g->C+c] = (x_s[r*g->C+c] + g->mlp_b3[c]) * g->gamma[c] + res_s[r*g->C+c];
}

/* ============ ConvNeXt Block (HWC in/out, NCHW for DW) ============ */
/* x[HW,C] → LN → DW Conv → gamma*x → LN → MLP(4 MatMul) → +residual */
static void convnext_block(float* x_hwc, int H, int W, int C,
                           const float* gamma,
                           const float* dw_w, const float* dw_b, int K,
                           const float* ln_w, const float* ln_b,
                           const float* mlp_w0, const float* mlp_w1,
                           const float* mlp_b1,
                           const float* mlp_w2, const float* mlp_w3,
                           const float* mlp_b3,
                           int rank, int hidden,
                           float* tmp,
                           const PackedFP32* fp0, const PackedFP32* fp1,
                           const PackedFP32* fp2, const PackedFP32* fp3)
{
    int HW = H * W;
    float* residual = tmp;
    float* dw_out = residual + HW * C;
    float* t1 = dw_out + HW * C;
    float* t2 = t1 + HW * rank;

    memcpy(residual, x_hwc, (size_t)HW * C * sizeof(float));

    /* DW Conv: output to dw_out, then use dw_out for LN+MLP (avoids memcpy!) */
    depthwise_conv_nxn_hwc_fp32(x_hwc, H, W, C, dw_w, dw_b, K, dw_out);
    /* No memcpy — use dw_out directly for subsequent ops */

    /* LayerNorm on dw_out (in-place) */
    layer_norm_fp32(dw_out, HW, C, ln_w, ln_b, 1e-6f, dw_out);

    {
        /* Single-threaded MLP operating on dw_out */
        matmul_wp(dw_out, t1, HW, C, rank, fp0);
        matmul_wpbg(t1, t2, mlp_b1, HW, rank, hidden, fp1);
        matmul_wp(t2, t1, HW, hidden, rank, fp2);
        matmul_wpbrg(t1, x_hwc, mlp_b3, gamma, residual, HW, rank, C, fp3);
    }
}

/* ============ Forward Pass ============ */
static void edgeface_forward(const float* input_chw, /* [3, 112, 112] */
                             Weights* weights,
                             float* embedding) /* [512] */
{
    _g_weights = weights; /* enable INT8 lookup in matmul_wp */
    /* Static workspace — avoids malloc overhead on every call */
    static float work[2 * 1024 * 1024]; /* 8MB workspace */
    static float x_buf[28 * 28 * 192]; /* largest: 784 positions × 192 channels */

    float* x = x_buf;

    /* === STEM: Conv 3→32 4×4 s4 → output directly in HWC === */
    {
        float* stem_cols = work; /* [784, 3*4*4] */
        for (int oy = 0; oy < 28; oy++)
            for (int ox = 0; ox < 28; ox++) {
                float* row = stem_cols + ((size_t)oy * 28 + ox) * 48;
                int k = 0;
                for (int ci = 0; ci < 3; ci++)
                    for (int ky = 0; ky < 4; ky++) {
                        const float* src = input_chw + (size_t)ci * 112 * 112 + (oy * 4 + ky) * 112 + ox * 4;
                        row[k++] = src[0];
                        row[k++] = src[1];
                        row[k++] = src[2];
                        row[k++] = src[3];
                    }
            }
        matmul_fp32_packed_bias(stem_cols, weights->fp[0].data, W(1), x, 784, 48, 32);
    }

    layer_norm_fp32(x, 784, 32, W(2), W(3), 1e-6f, x);

    /* === STAGE 0: 3 ConvNeXt blocks (C=32, K=3, rank=19, hidden=128) === */
    /* Block 0: gamma=17, dw_w=4, dw_b=5, ln_w=6, ln_b=7,
     *          mlp_w0=8[32,19], mlp_w1=9[19,128], mlp_b1=10,
     *          mlp_w2=14[128,19], mlp_w3=15[19,32], mlp_b3=16 */
    /* Test: INT8 for first block only, FP32 for rest */
    #define FP(i) (&weights->fp[i])
    #define I8(i) (&weights->mm[i])  /* INT8 packed */

    /* TEST: first block INT8 to verify correctness */
    convnext_block(x, 28,28,32, W(17), W(4),W(5),3, W(6),W(7),
                   W(8),W(9),W(10), W(14),W(15),W(16), 19,128, work,
                   FP(8),FP(9),FP(14),FP(15));
    /* Note: INT8 test disabled — need to change convnext_block to accept PackedMM* */

    convnext_block(x, 28,28,32, W(28), W(18),W(19),3, W(20),W(21),
                   W(22),W(23),W(24), W(25),W(26),W(27), 19,128, work,
                   FP(22),FP(23),FP(25),FP(26));

    convnext_block(x, 28,28,32, W(39), W(29),W(30),3, W(31),W(32),
                   W(33),W(34),W(35), W(36),W(37),W(38), 19,128, work,
                   FP(33),FP(34),FP(36),FP(37));
    /* === DOWNSAMPLE 0→1: LN(40,41) + Conv 32→64 2×2 s2 (42,43) — all HWC === */
    layer_norm_fp32(x, 784, 32, W(40), W(41), 1e-6f, x);
    {
        conv2d_hwc_packed(x, 32, 28, 28, W(43), 64, 2, 2, FP(42), work, x);
    }

    /* === STAGE 1: 2 ConvNeXt blocks (C=64, K=5, rank=38, hidden=256) === */
    /* x already has enough space for all stages */

    convnext_block(x, 14,14,64, W(54), W(44),W(45),5, W(46),W(47),
                   W(48),W(49),W(50), W(51),W(52),W(53), 38,256, work,
                   FP(48),FP(49),FP(51),FP(52));
    convnext_block(x, 14,14,64, W(65), W(55),W(56),5, W(57),W(58),
                   W(59),W(60),W(61), W(62),W(63),W(64), 38,256, work,
                   FP(59),FP(60),FP(62),FP(63));

    /* Stage 1 XCA attention (tensors 66-100) */
    /* DW: split [32,32], conv first 32 only. Pos: Conv1x1(const[69],W70,W71). */
    {
        int s1_dw_splits[] = {32, 32};
        const float* s1_dw_ws[] = {W(66)};
        const float* s1_dw_bs[] = {W(67)};
        xca_block(x, 14,14,64,
                  W(90)/*gamma_xca*/, W(100)/*gamma*/,
                  2, s1_dw_splits, 1, s1_dw_ws, s1_dw_bs, 3, /* DW conv */
                  W(69)/*pos_const_nchw*/, W(70), W(71), /* pos embed */
                  W(72),W(73), /* xca LN */
                  W(74),W(75),W(76), /* QKV LoRaLin + bias (192 = 3×64) */
                  W(85), 4, /* temperature, n_heads */
                  W(87),W(88),W(89), /* proj LoRaLin + bias */
                  W(92),W(93), /* post-attn LN */
                  W(94),W(95),W(96), W(97),W(98),W(99), /* MLP */
                  38, 256, work,
                  FP(74),FP(75),FP(87),FP(88),
                  FP(94),FP(95),FP(97),FP(98));
    }

    /* === DOWNSAMPLE 1→2: LN(101,102) + Conv 64→100 2×2 s2 (103,104) — HWC === */
    layer_norm_fp32(x, 196, 64, W(101), W(102), 1e-6f, x);
    {
        conv2d_hwc_packed(x, 64, 14, 14, W(104), 100, 2, 2, FP(103), work, x);
    }

    /* === STAGE 2: 8 ConvNeXt blocks === */
    for (int b = 0; b < 8; b++) {
        int base = 105 + b * 11;
        convnext_block(x, 7,7,100, W(base+10), W(base),W(base+1),7,
                       W(base+2),W(base+3),
                       W(base+4),W(base+5),W(base+6), W(base+7),W(base+8),W(base+9),
                       60,400, work,
                       FP(base+4),FP(base+5),FP(base+7),FP(base+8));
    }

    /* Stage 2 XCA attention (tensors 193-220) */
    /* DW: split [34,34,32], conv first 2 splits. No pos embed. */
    {
        int s2_dw_splits[] = {34, 34, 32};
        const float* s2_dw_ws[] = {W(193), W(195)};
        const float* s2_dw_bs[] = {W(194), W(196)};
        xca_block(x, 7,7,100,
                  W(210), W(220),
                  3, s2_dw_splits, 2, s2_dw_ws, s2_dw_bs, 3, /* DW conv */
                  NULL, NULL, NULL, /* No pos embed */
                  W(198),W(199),
                  W(200),W(201),W(202),
                  W(205), 4,
                  W(207),W(208),W(209),
                  W(212),W(213),
                  W(214),W(215),W(216), W(217),W(218),W(219),
                  60, 400, work,
                  FP(200),FP(201),FP(207),FP(208),
                  FP(214),FP(215),FP(217),FP(218));
    }

    /* === DOWNSAMPLE 2→3: LN(221,222) + Conv 100→192 2×2 s2 (223,224) — HWC === */
    layer_norm_fp32(x, 49, 100, W(221), W(222), 1e-6f, x);
    {
        int Ho = (7-2)/2+1; /* =3 */
        conv2d_hwc_packed(x, 100, 7, 7, W(224), 192, 2, 2, FP(223), work, x);
    }
    int H3 = 3, W3 = 3;

    /* === STAGE 3: 2 ConvNeXt blocks (C=192, K=9, rank=115, hidden=768) === */
    /* Block 0: tensors 225-235 */
    convnext_block(x, H3,W3,192, W(235), W(225),W(226),9, W(227),W(228),
                   W(229),W(230),W(231), W(232),W(233),W(234), 115,768, work,
                   FP(229),FP(230),FP(232),FP(233));
    convnext_block(x, H3,W3,192, W(246), W(236),W(237),9, W(238),W(239),
                   W(240),W(241),W(242), W(243),W(244),W(245), 115,768, work,
                   FP(240),FP(241),FP(243),FP(244));

    /* Stage 3 XCA attention (tensors 247-276) */
    /* DW: split [48,48,48,48], conv first 3 splits. No pos embed. */
    {
        int s3_dw_splits[] = {48, 48, 48, 48};
        const float* s3_dw_ws[] = {W(247), W(249), W(251)};
        const float* s3_dw_bs[] = {W(248), W(250), W(252)};
        xca_block(x, H3,W3,192,
                  W(266), W(276),
                  4, s3_dw_splits, 3, s3_dw_ws, s3_dw_bs, 3, /* DW conv */
                  NULL, NULL, NULL, /* No pos embed */
                  W(254),W(255),
                  W(256),W(257),W(258),
                  W(261), 4,
                  W(263),W(264),W(265),
                  W(268),W(269),
                  W(270),W(271),W(272), W(273),W(274),W(275),
                  115, 768, work,
                  FP(256),FP(257),FP(263),FP(264),
                  FP(270),FP(271),FP(273),FP(274));
    }

    /* === HEAD: ReduceMean + LN + FC(192→115) + FC(115→512) === */
    /* Global average pool: x[H3*W3, 192] → mean across positions → [192] */
    float pooled[192];
    int HW3 = H3 * W3;
    hwc_mean<192>(x, HW3, pooled);

    /* LN(280,281) */
    layer_norm_fp32(pooled, 1, 192, W(280), W(281), 1e-6f, pooled);

    /* FC weights are stored as [N,K]; prepacked copies use [K,N]. */
    float fc1[115];
    {
        matmul_wp(pooled, fc1, 1, 192, 115, FP(283));
    }

    /* FC: 115→512 + bias */
    {
        matmul_wpb(fc1, embedding, W(285), 1, 115, 512, FP(284));
    }

    l2_normalize_fp32(embedding, 1, 512, 1e-12f);

    /* x and work are static — no free needed */
}

/* ============ Engine Init (used by both main and library API) ============ */
static int engine_init(const char* weights_path, Weights* weights) {
    /* Init thread pool for parallel GEMM */
    {
        extern void tp_init(int);
        tp_init(4);
    }

    if (load_weights(weights_path, weights) != 0) {
        fprintf(stderr, "Failed to load %s\n", weights_path);
        return -1;
    }
    /* Allocate packed MatMul weight arrays */
    weights->mm = (PackedMM*)calloc(weights->n_tensors, sizeof(PackedMM));
    weights->fp = (PackedFP32*)calloc(weights->n_tensors, sizeof(PackedFP32));


    /* Pre-pack MatMul weights to INT8 c8 format */
    {
        /* MatMul weight indices (2D tensors used in MatMul) */
        int mm_idx[] = {8,9,14,15,22,23,25,26,33,34,36,37,48,49,51,52,59,60,62,63,
                        74,75,87,88,94,95,97,98,109,110,112,113,120,121,123,124,
                        131,132,134,135,142,143,145,146,153,154,156,157,164,165,
                        167,168,175,176,178,179,186,187,189,190,200,201,207,208,
                        214,215,217,218,229,230,232,233,240,241,243,244,256,257,
                        263,264,270,271,273,274,283,284,-1};

        /* Weight sizes from names.txt (K×N pairs) */
        /* Each weight W[K,N] is used as: A[M,K] @ W[K,N] → C[M,N]
         * c8 GEMM expects B as [N, K] (Cout=N, K=K)
         * So we need to transpose W to [N, K] before packing */
        int mm_shapes[][2] = {
            {32,19},{19,128},{128,19},{19,32}, /* stage 0 block 0 */
            {32,19},{19,128},{128,19},{19,32}, /* stage 0 block 1 */
            {32,19},{19,128},{128,19},{19,32}, /* stage 0 block 2 */
            {64,38},{38,256},{256,38},{38,64}, /* stage 1 block 0 */
            {64,38},{38,256},{256,38},{38,64}, /* stage 1 block 1 */
            {64,38},{38,192}, /* stage 1 attn QKV proj */
            {64,38},{38,64},  /* stage 1 attn out proj */
            {64,38},{38,256},{256,38},{38,64}, /* stage 1 attn MLP */
            /* stage 2: 8 blocks × 4 weights */
            {100,60},{60,400},{400,60},{60,100},
            {100,60},{60,400},{400,60},{60,100},
            {100,60},{60,400},{400,60},{60,100},
            {100,60},{60,400},{400,60},{60,100},
            {100,60},{60,400},{400,60},{60,100},
            {100,60},{60,400},{400,60},{60,100},
            {100,60},{60,400},{400,60},{60,100},
            {100,60},{60,400},{400,60},{60,100},
            /* stage 2 attn */
            {100,60},{60,300},
            {100,60},{60,100},
            {100,60},{60,400},{400,60},{60,100},
            /* stage 3: 2 blocks */
            {192,115},{115,768},{768,115},{115,192},
            {192,115},{115,768},{768,115},{115,192},
            /* stage 3 attn */
            {192,115},{115,576},
            {192,115},{115,192},
            {192,115},{115,768},{768,115},{115,192},
            /* head */
            {115,192},{512,115},
        };

        int packed_count = 0;
        for (int i = 0; mm_idx[i] >= 0; i++) {
            int idx = mm_idx[i];
            int K = mm_shapes[i][0], N = mm_shapes[i][1];

            if (N < 8) continue; /* Skip tiny N (c8 GEMM needs N≥8) */

            float* w_fp32 = weights->tensors[idx]; /* [K, N] row-major */

            /* Quantize per-output-channel: transpose to [N, K], quantize per N */
            int8_t* w_int8 = (int8_t*)malloc((size_t)N * K);
            float* w_scales = (float*)malloc(N * sizeof(float));

            for (int n = 0; n < N; n++) {
                float mx = 0;
                for (int k = 0; k < K; k++) {
                    float v = w_fp32[k * N + n];
                    if (v < 0) v = -v;
                    if (v > mx) mx = v;
                }
                w_scales[n] = (mx > 1e-8f) ? mx / 127.0f : 1e-8f;
                for (int k = 0; k < K; k++) {
                    float v = w_fp32[k * N + n] / w_scales[n];
                    int q = (int)(v + (v >= 0 ? 0.5f : -0.5f));
                    if (q > 127) q = 127; if (q < -128) q = -128;
                    w_int8[n * K + k] = (int8_t)q; /* [N, K] layout for c8 pack */
                }
            }

            int pw_size = packed_weights_size_4x8c8(K, N);
            void* packed = malloc(pw_size);
            int32_t* col_sums = (int32_t*)calloc(((N+7)&~7), sizeof(int32_t));
            pack_weights_4x8c8(w_int8, NULL, K, N, packed, col_sums);

            weights->mm[idx].packed = packed;
            weights->mm[idx].col_sums = col_sums;
            weights->mm[idx].w_scales = w_scales;
            weights->mm[idx].K = K;
            weights->mm[idx].N = N;
            packed_count++;

            free(w_int8);
        }
    }

    /* Pre-pack FP32 MatMul weights into column-panel format [ceil(N/16), K, 16] */
    {
        /* Same indices and shapes as INT8 packing */
        int fp_idx[] = {8,9,14,15,22,23,25,26,33,34,36,37,48,49,51,52,59,60,62,63,
                        74,75,87,88,94,95,97,98, /* stage 1 XCA */
                        109,110,112,113,120,121,123,124,131,132,134,135,142,143,145,146,
                        153,154,156,157,164,165,167,168,175,176,178,179,186,187,189,190,
                        200,201,207,208,214,215,217,218, /* stage 2 XCA */
                        229,230,232,233,240,241,243,244,
                        256,257,263,264,270,271,273,274, /* stage 3 XCA */
                        -1};
        int fp_shapes[][2] = {
            {32,19},{19,128},{128,19},{19,32},
            {32,19},{19,128},{128,19},{19,32},
            {32,19},{19,128},{128,19},{19,32},
            {64,38},{38,256},{256,38},{38,64},
            {64,38},{38,256},{256,38},{38,64},
            {64,38},{38,192}, {64,38},{38,64}, {64,38},{38,256},{256,38},{38,64}, /* s1 XCA */
            {100,60},{60,400},{400,60},{60,100},
            {100,60},{60,400},{400,60},{60,100},
            {100,60},{60,400},{400,60},{60,100},
            {100,60},{60,400},{400,60},{60,100},
            {100,60},{60,400},{400,60},{60,100},
            {100,60},{60,400},{400,60},{60,100},
            {100,60},{60,400},{400,60},{60,100},
            {100,60},{60,400},{400,60},{60,100},
            {100,60},{60,300}, {100,60},{60,100}, {100,60},{60,400},{400,60},{60,100}, /* s2 XCA */
            {192,115},{115,768},{768,115},{115,192},
            {192,115},{115,768},{768,115},{115,192},
            {192,115},{115,576}, {192,115},{115,192}, {192,115},{115,768},{768,115},{115,192}, /* s3 XCA */
        };
        for (int i = 0; fp_idx[i] >= 0; i++) {
            int idx = fp_idx[i];
            int K_ = fp_shapes[i][0], N_ = fp_shapes[i][1];
            int sz = packed_b_fp32_size(K_, N_);
            float* packed;
#ifdef _WIN32
#ifdef _WIN32
            packed = (float*)_aligned_malloc((size_t)sz * sizeof(float), 64);
#else
            if (posix_memalign((void**)&packed, 64, (size_t)sz * sizeof(float)) != 0) packed = NULL;
#endif
#else
            packed = (float*)aligned_alloc(32, (size_t)sz * sizeof(float));
#endif
            if (!packed) packed = (float*)malloc((size_t)sz * sizeof(float));
            pack_b_fp32(weights->tensors[idx], K_, N_, packed);
            weights->fp[idx].data = packed;
            weights->fp[idx].K = K_;
            weights->fp[idx].N = N_;
        }
    }

    /* Pre-transpose DW conv weights: [C, K*K] → [K*K, C] for HWC loads */
    /* DW weight tensor indices from names.txt */
    int dw_indices[] = {4,18,29, 44,55, 66, 105,116,127,138,149,160,171,182, 193,195,
                        225,236, 247,249,251, -1};
    int dw_K[] = {3,3,3, 5,5, 3, 7,7,7,7,7,7,7,7, 3,3, 9,9, 3,3,3};
    for (int di = 0; dw_indices[di] >= 0; di++) {
        int idx = dw_indices[di];
        int K = dw_K[di];
        int KK = K * K;
        /* Find C from weight shape: weight is [C, 1, K, K] = C*KK floats */
        /* We need to know C — get from the next tensor (bias) which has C floats */
        /* Actually just compute: total_floats / KK */
        /* But we don't store sizes... just hardcode known C values */
    }
    /* Simpler: transpose in-place in the weight buffer */
    /* DW weights are [C, K*K] stored contiguously. Transpose to [K*K, C] */
    {
        /* Stage 0: C=32, K=3, indices 4,18,29 */
        int stage0_dw[] = {4, 18, 29};
        for (int j = 0; j < 3; j++) {
            int idx = stage0_dw[j]; int C = 32, KK = 9;
            float* tmp = (float*)malloc(C * KK * sizeof(float));
            memcpy(tmp, weights->tensors[idx], C * KK * sizeof(float));
            for (int c = 0; c < C; c++)
                for (int ki = 0; ki < KK; ki++)
                    weights->tensors[idx][ki * C + c] = tmp[c * KK + ki];
            free(tmp);
        }
        /* Stage 1: C=64, K=5, indices 44,55 */
        int stage1_dw[] = {44, 55};
        for (int j = 0; j < 2; j++) {
            int idx = stage1_dw[j]; int C = 64, KK = 25;
            float* tmp = (float*)malloc(C * KK * sizeof(float));
            memcpy(tmp, weights->tensors[idx], C * KK * sizeof(float));
            for (int c = 0; c < C; c++)
                for (int ki = 0; ki < KK; ki++)
                    weights->tensors[idx][ki * C + c] = tmp[c * KK + ki];
            free(tmp);
        }
        /* Stage 2: C=100, K=7, indices 105..182 (every 11) */
        for (int b = 0; b < 8; b++) {
            int idx = 105 + b * 11; int C = 100, KK = 49;
            float* tmp = (float*)malloc(C * KK * sizeof(float));
            memcpy(tmp, weights->tensors[idx], C * KK * sizeof(float));
            for (int c = 0; c < C; c++)
                for (int ki = 0; ki < KK; ki++)
                    weights->tensors[idx][ki * C + c] = tmp[c * KK + ki];
            free(tmp);
        }
        /* Stage 3: C=192, K=9, indices 225, 236 */
        int stage3_dw[] = {225, 236};
        for (int j = 0; j < 2; j++) {
            int idx = stage3_dw[j]; int C = 192, KK = 81;
            float* tmp = (float*)malloc(C * KK * sizeof(float));
            memcpy(tmp, weights->tensors[idx], C * KK * sizeof(float));
            for (int c = 0; c < C; c++)
                for (int ki = 0; ki < KK; ki++)
                    weights->tensors[idx][ki * C + c] = tmp[c * KK + ki];
            free(tmp);
        }
    }

    /* Reorder stem/downsample conv weights: OIHW [Cout,Cin,K,K] → [Cin*KK, Cout] */
    conv2d_hwc_reorder_weights(weights->tensors[0], 32, 3, 16);    /* stem: 3→32 4×4 */
    conv2d_hwc_reorder_weights(weights->tensors[42], 64, 32, 4);   /* ds 0→1: 32→64 2×2 */
    conv2d_hwc_reorder_weights(weights->tensors[103], 100, 64, 4); /* ds 1→2: 64→100 2×2 */
    conv2d_hwc_reorder_weights(weights->tensors[223], 192, 100, 4);/* ds 2→3: 100→192 2×2 */

    {
        const int conv_idx[] = {0, 42, 103, 223};
        const int conv_k[] = {48, 128, 256, 400};
        const int conv_n[] = {32, 64, 100, 192};
        for (int i = 0; i < 4; i++) {
            int idx = conv_idx[i];
            int K_ = conv_k[i];
            int N_ = conv_n[i];
            int sz = packed_b_fp32_size(K_, N_);
            float* packed;
#ifdef _WIN32
            packed = (float*)_aligned_malloc((size_t)sz * sizeof(float), 64);
#else
            if (posix_memalign((void**)&packed, 64, (size_t)sz * sizeof(float)) != 0) packed = NULL;
#endif
            if (!packed) packed = (float*)malloc((size_t)sz * sizeof(float));
            pack_b_fp32(weights->tensors[idx], K_, N_, packed);
            weights->fp[idx].data = packed;
            weights->fp[idx].K = K_;
            weights->fp[idx].N = N_;
        }
    }

    pack_transposed_fc_weight(weights->tensors[283], 192, 115, &weights->fp[283]);
    pack_transposed_fc_weight(weights->tensors[284], 115, 512, &weights->fp[284]);

    pack_folded_projection_weight(weights->tensors[87], weights->tensors[88], 64, 38, &weights->fp[87]);
    pack_folded_projection_weight(weights->tensors[207], weights->tensors[208], 100, 60, &weights->fp[207]);
    pack_folded_projection_weight(weights->tensors[263], weights->tensors[264], 192, 115, &weights->fp[263]);

    /* Register JIT kernels for small-K matmuls (optional — link jit_gemm.c) */
#ifdef USE_JIT
    {
        extern void jit_register(int, int);
        int jit_sizes[][2] = {
            {32,19},{19,128},{128,19},{19,32},
            {64,38},{38,256},{256,38},{38,64},
            {38,192},{60,400},{60,100},{60,300},
            {-1,-1}
        };
        for (int i = 0; jit_sizes[i][0] > 0; i++)
            jit_register(jit_sizes[i][0], jit_sizes[i][1]);
    }
#endif

    /* Pre-compute stage 1 position embedding: Conv1x1(const[69], W70, W71) → [196, 64] HWC */
    /* This is constant — same result every forward pass. Cache it in tensor 69's slot. */
    {
        int HW = 196, C = 64;
        float* pos_cached = (float*)malloc((size_t)HW * C * sizeof(float));
        const float* pos_const = weights->tensors[69]; /* [1, 64, 14, 14] NCHW */
        const float* pos_w = weights->tensors[70];      /* [64, 64, 1, 1] = [Cout, Cin] */
        const float* pos_b = weights->tensors[71];      /* [64] */
        for (int hw = 0; hw < HW; hw++)
            for (int co = 0; co < C; co++) {
                float sum = pos_b[co];
                for (int ci = 0; ci < C; ci++)
                    sum += pos_const[ci * HW + hw] * pos_w[co * C + ci];
                pos_cached[hw * C + co] = sum;
            }
        weights->tensors[69] = pos_cached; /* Replace with pre-computed HWC result */
    }

    return 0;
}

/* ============ Main ============ */
#ifndef FACEX_LIB
int main(int argc, char** argv) {
    const char* weights_path = "data/edgeface_xs_fp32.bin";
    if (argc > 1) weights_path = argv[1];

    /* Parse --license KEY from args or LICENSE_KEY env */
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--license") == 0) {
            g_license_key = argv[i + 1];
            break;
        }
    }
    if (!g_license_key) g_license_key = getenv("LICENSE_KEY");

    Weights weights;
    if (engine_init(weights_path, &weights) != 0) {
        fprintf(stderr, "Failed to load %s\n", weights_path);
        return 1;
    }

    /* Load input: if --stdin, read 112*112*3 floats (CHW, [-1,1] normalized) */
    float input[3 * 112 * 112];
    if (argc > 2 && strcmp(argv[2], "--stdin") == 0) {
        #ifdef _WIN32
        _setmode(_fileno(stdin), 0x8000); /* _O_BINARY */
        #endif
        size_t rd = fread(input, sizeof(float), 3*112*112, stdin);
        if (rd != 3*112*112) {
            fprintf(stderr, "Expected %d floats, got %zu\n", 3*112*112, rd);
            return 1;
        }
    } else {
        for (int i = 0; i < 3*112*112; i++)
            input[i] = (float)(i % 256) / 128.0f - 1.0f;
    }

    float embedding[512];

    /* Mode selection */
    const char* mode = (argc > 2) ? argv[2] : "";
    int is_server = (strcmp(mode, "--server") == 0);
    int is_batch = (strcmp(mode, "--batch") == 0);

    if (is_server || is_batch) {
        #ifdef _WIN32
        _setmode(_fileno(stdin), 0x8000);
        _setmode(_fileno(stdout), 0x8000);
        #endif
        float hwc_buf[3 * 112 * 112];
        /* --server: reads HWC float32, converts to CHW (Go server protocol)
         * --batch:  reads CHW float32 directly (Python test protocol) */
        while (1) {
            float* read_buf = is_server ? hwc_buf : input;
            if (fread(read_buf, sizeof(float), 3*112*112, stdin) != 3*112*112)
                break;
            if (is_server) {
                /* HWC → CHW conversion */
                for (int h = 0; h < 112; h++)
                    for (int w = 0; w < 112; w++)
                        for (int c = 0; c < 3; c++)
                            input[c * 112 * 112 + h * 112 + w] = hwc_buf[(h * 112 + w) * 3 + c];
            }
            edgeface_forward(input, &weights, embedding);
            fwrite(embedding, sizeof(float), 512, stdout);
            fflush(stdout);
        }
    } else {
        /* Single image: warmup + benchmark */
        edgeface_forward(input, &weights, embedding);

        struct timespec t0, t1;
        int ITERS = 200;
        const char* iters_env = getenv("FACEX_BENCH_ITERS");
        if (iters_env) {
            int parsed = atoi(iters_env);
            if (parsed > 0) ITERS = parsed;
        }
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < ITERS; i++)
            edgeface_forward(input, &weights, embedding);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = ((t1.tv_sec-t0.tv_sec)*1e3 + (t1.tv_nsec-t0.tv_nsec)/1e6) / ITERS;

        fprintf(stderr, "EdgeFace-XS forward: %.2f ms\n", ms);
        fprintf(stderr, "Embedding[0..4]: %.4f %.4f %.4f %.4f %.4f\n",
                embedding[0], embedding[1], embedding[2], embedding[3], embedding[4]);
        #ifdef _WIN32
        _setmode(_fileno(stdout), 0x8000);
        #endif
        fwrite(embedding, sizeof(float), 512, stdout);
        fflush(stdout);
    }

    free(weights.tensors);
    free(weights.raw);
    return 0;
}
#endif /* FACEX_LIB */
