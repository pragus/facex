/*
 * align.c — Affine face alignment using 5 keypoints.
 *
 * Transforms detected face region to 112×112 aligned crop
 * matching ArcFace/EdgeFace training alignment.
 *
 * Reference landmarks for 112×112 (ArcFace standard):
 *   left_eye:    (38.2946, 51.6963)
 *   right_eye:   (73.5318, 51.5014)
 *   nose:        (56.0252, 71.7366)
 *   left_mouth:  (41.5493, 92.3655)
 *   right_mouth: (70.7299, 92.2041)
 */

#include <math.h>
#include <string.h>
#include <stdint.h>

/* Reference 112×112 landmarks (ArcFace order) */
static const float REF_KPS[10] = {
    38.2946f, 51.6963f,  /* left eye */
    73.5318f, 51.5014f,  /* right eye */
    56.0252f, 71.7366f,  /* nose */
    41.5493f, 92.3655f,  /* left mouth */
    70.7299f, 92.2041f   /* right mouth */
};

/*
 * Compute similarity transform (rotation + uniform scale + translation)
 * from src keypoints to dst keypoints using least-squares.
 *
 * Transform: [a -b tx] [x]   [x']
 *            [b  a ty] [y] = [y']
 *
 * Returns: M[6] = {a, -b, tx, b, a, ty}
 */
static void estimate_similarity_transform(const float src[10], const float dst[10], float M[6]) {
    /* Solve for [a, b, tx, ty] using normal equations:
     * For each point pair (sx,sy) -> (dx,dy):
     *   dx = a*sx - b*sy + tx
     *   dy = b*sx + a*sy + ty
     */
    float sx_sum = 0, sy_sum = 0, dx_sum = 0, dy_sum = 0;
    float sxsx = 0, sysy = 0, sxsy = 0;
    float sxdx = 0, sydx = 0, sxdy = 0, sydy = 0;
    int n = 5;

    for (int i = 0; i < n; i++) {
        float sx = src[i*2], sy = src[i*2+1];
        float dx = dst[i*2], dy = dst[i*2+1];
        sx_sum += sx; sy_sum += sy;
        dx_sum += dx; dy_sum += dy;
        sxsx += sx*sx; sysy += sy*sy; sxsy += sx*sy;
        sxdx += sx*dx; sydx += sy*dx;
        sxdy += sx*dy; sydy += sy*dy;
    }

    /* Normal equations for similarity transform:
     * [sum(sx²+sy²)  0             sum(sx) -sum(sy)] [a ]   [sum(sx*dx + sy*dy)]
     * [0             sum(sx²+sy²)  sum(sy)  sum(sx)] [b ]   [sum(sx*dy - sy*dx)]
     * [sum(sx)       sum(sy)       n        0      ] [tx] = [sum(dx)           ]
     * [-sum(sy)      sum(sx)       0        n      ] [ty]   [sum(dy)           ]
     *
     * Simplified: solve 2×2 for (a,b), then get (tx,ty)
     */
    float ss = sxsx + sysy;
    float rhs_a = sxdx + sydy;
    float rhs_b = sxdy - sydx;

    /* [ss    0   sx_sum -sy_sum] [a]   [rhs_a ]
     * [0     ss  sy_sum  sx_sum] [b] = [rhs_b ]
     * [sx    sy  n       0     ] [tx]  [dx_sum]
     * [-sy   sx  0       n     ] [ty]  [dy_sum] */

    /* From rows 3,4: tx = (dx_sum - a*sx_sum + b*sy_sum) / n
     *                ty = (dy_sum - b*sx_sum - a*sy_sum) / n
     * Substitute into rows 1,2:
     * a*(ss - (sx_sum²+sy_sum²)/n) + 0*b = rhs_a - (sx_sum*dx_sum+sy_sum*dy_sum)/n
     * 0*a + b*(ss - (sx_sum²+sy_sum²)/n) = rhs_b - (sy_sum*dx_sum-sx_sum*dy_sum)/n ... actually not quite.
     * Easier: just solve directly. */

    float A00 = ss, A02 = sx_sum, A03 = -sy_sum;
    float A12 = sy_sum, A13 = sx_sum;
    float b0 = rhs_a, b1 = rhs_b;
    float b2 = dx_sum, b3 = dy_sum;

    /* Eliminate tx,ty from first two equations using last two */
    /* a*ss + tx*sx - ty*sy = rhs_a  ... (1)
     * b*ss + tx*sy + ty*sx = rhs_b  ... (2)
     * a*sx + b*sy + tx*n = dx       ... (3) → tx = (dx - a*sx - b*sy)/n
     * -a*sy + b*sx + ty*n = dy      ... (4) → ty = (dy + a*sy - b*sx)/n  */
    /* Hmm, let me just use the simple form. */

    float sx_m = sx_sum / n, sy_m = sy_sum / n;
    float dx_m = dx_sum / n, dy_m = dy_sum / n;

    /* Center the points */
    float num_a = 0, num_b = 0, denom = 0;
    for (int i = 0; i < n; i++) {
        float sx = src[i*2] - sx_m, sy = src[i*2+1] - sy_m;
        float dx = dst[i*2] - dx_m, dy = dst[i*2+1] - dy_m;
        num_a += sx*dx + sy*dy;
        num_b += sx*dy - sy*dx;
        denom += sx*sx + sy*sy;
    }

    float a = num_a / denom;
    float b = num_b / denom;
    float tx = dx_m - a*sx_m + b*sy_m;
    float ty = dy_m - b*sx_m - a*sy_m;

    M[0] = a;  M[1] = -b; M[2] = tx;
    M[3] = b;  M[4] = a;  M[5] = ty;
}

/*
 * Apply affine transform to warp src image to 112×112 aligned face.
 *
 * src_rgb:  source image, uint8 HWC [0,255]
 * src_w, src_h: source dimensions
 * kps:      5 detected keypoints [x0,y0,x1,y1,...,x4,y4]
 * out_f32:  output 112×112×3 float32 HWC [-1, 1]
 */
void align_face(const uint8_t* src_rgb, int src_w, int src_h,
                const float kps[10], float* out_f32) {
    float M[6];
    /* We want: dst_point = M * src_point
     * But for warping we need inverse: for each dst pixel, find src pixel.
     * So compute M: detected_kps → reference_kps, then invert. */
    estimate_similarity_transform(kps, REF_KPS, M);

    /* Invert 2×2 part: [a -b]^-1 = 1/(a²+b²) * [a  b]
     *                  [b  a]                     [-b a] */
    float a = M[0], b = M[3], tx = M[2], ty = M[5];
    float det = a*a + b*b;
    float ia = a / det, ib = -b / det;
    float itx = -(ia*tx + (-ib)*ty);
    float ity = -(ib*tx + ia*ty);
    /* Inverse: for dst pixel (dx,dy) → src pixel (sx,sy):
     * sx = ia*dx + (-ib)*dy + itx
     * sy = ib*dx + ia*dy + ity */

    for (int dy = 0; dy < 112; dy++) {
        for (int dx = 0; dx < 112; dx++) {
            float sx = ia*dx - ib*dy + itx;  /* wait, ib = -b/det, so -ib = b/det */
            float sy_f = (b/det)*dx + ia*dy + ity;
            /* Actually let me redo properly */
            /* Forward: [a -b tx] maps src→dst. Inverse maps dst→src.
             * inv = [a/(a²+b²)   b/(a²+b²)   -(a*tx+b*ty)/(a²+b²) ]
             *       [-b/(a²+b²)  a/(a²+b²)   (b*tx-a*ty)/(a²+b²)  ] */
            float inv_a = a/det, inv_b = b/det;
            sx = inv_a * dx + inv_b * dy + (-(a*tx + b*ty))/det;  /* Hmm getting messy */
            /* Start over with clean inverse */
            sx = (a*(dx - tx) + b*(dy - ty)) / det;
            sy_f = (-b*(dx - tx) + a*(dy - ty)) / det;

            /* Bilinear interpolation */
            int ix = (int)floorf(sx), iy = (int)floorf(sy_f);
            float fx = sx - ix, fy = sy_f - iy;

            for (int c = 0; c < 3; c++) {
                float v = 0;
                if (ix >= 0 && ix < src_w-1 && iy >= 0 && iy < src_h-1) {
                    float v00 = src_rgb[(iy*src_w + ix)*3 + c];
                    float v10 = src_rgb[(iy*src_w + ix+1)*3 + c];
                    float v01 = src_rgb[((iy+1)*src_w + ix)*3 + c];
                    float v11 = src_rgb[((iy+1)*src_w + ix+1)*3 + c];
                    v = v00*(1-fx)*(1-fy) + v10*fx*(1-fy) + v01*(1-fx)*fy + v11*fx*fy;
                } else if (ix >= 0 && ix < src_w && iy >= 0 && iy < src_h) {
                    v = src_rgb[(iy*src_w + ix)*3 + c];
                }
                out_f32[(dy*112 + dx)*3 + c] = v / 127.5f - 1.0f;  /* normalize to [-1, 1] */
            }
        }
    }
}
