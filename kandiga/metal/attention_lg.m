/*
 * attention_lg.m — Fused GatedDeltaNet attention on CPU (NEON).
 *
 * One C call replaces ~15 MLX Metal dispatches per layer.
 * All computation on CPU using NEON vectorized 4-bit dequant matvec.
 * Eliminates mx.eval sync overhead entirely for attention.
 *
 * 122B config: hidden=3072, key_dim=2048 (16h×128d), value_dim=8192 (64h×128d)
 *              conv_dim=12288, conv_kernel=4
 *
 * Build:
 *   clang -shared -o libkandiga_attn_lg.dylib attention_lg.m \
 *         -framework Accelerate -O2 -march=native
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

/* bf16 helpers */
static inline float bf16_to_f32(uint16_t bf) {
    uint32_t bits = (uint32_t)bf << 16; float f; memcpy(&f, &bits, 4); return f;
}
static inline uint16_t f32_to_bf16(float f) {
    uint32_t bits; memcpy(&bits, &f, 4); return (uint16_t)(bits >> 16);
}

/* ----------------------------------------------------------------------- */
/* NEON 4-bit dequant matvec (same kernel as expert MLP)                   */
/* ----------------------------------------------------------------------- */
static void matvec4bit(
    const uint32_t* w, const uint16_t* s, const uint16_t* b,
    const float* x, float* out, int od, int id
) {
    int ppr = id/8, ng = id/64, ppg = 8;
    for (int r = 0; r < od; r++) {
        const uint32_t* wr = w + r*ppr;
        const uint16_t* sr = s + r*ng;
        const uint16_t* br = b + r*ng;
#ifdef __ARM_NEON__
        float32x4_t acc = vdupq_n_f32(0);
        for (int g = 0; g < ng; g++) {
            float sc = bf16_to_f32(sr[g]), bi = bf16_to_f32(br[g]);
            float32x4_t vs = vdupq_n_f32(sc), vb = vdupq_n_f32(bi);
            for (int p = 0; p < ppg; p += 4) {
                const uint32_t* wp = wr + g*ppg + p;
                int xb = g*64 + p*8;
                for (int pp = 0; pp < 4; pp++) {
                    uint32_t pk = wp[pp];
                    float32x4_t lo = {(float)(pk&0xF),(float)((pk>>4)&0xF),(float)((pk>>8)&0xF),(float)((pk>>12)&0xF)};
                    float32x4_t hi = {(float)((pk>>16)&0xF),(float)((pk>>20)&0xF),(float)((pk>>24)&0xF),(float)((pk>>28)&0xF)};
                    acc = vfmaq_f32(acc, vfmaq_f32(vb, lo, vs), vld1q_f32(x + xb + pp*8));
                    acc = vfmaq_f32(acc, vfmaq_f32(vb, hi, vs), vld1q_f32(x + xb + pp*8 + 4));
                }
            }
        }
        float32x2_t sp = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        out[r] = vget_lane_f32(vpadd_f32(sp, sp), 0);
#else
        float a = 0;
        for (int g = 0; g < ng; g++) {
            float sc = bf16_to_f32(sr[g]), bi = bf16_to_f32(br[g]);
            for (int p = 0; p < ppg; p++) {
                uint32_t pk = wr[g*ppg+p]; int xb = g*64+p*8;
                for (int n = 0; n < 8; n++)
                    a += ((float)((pk>>(n*4))&0xF)*sc+bi) * x[xb+n];
            }
        }
        out[r] = a;
#endif
    }
}

/* ----------------------------------------------------------------------- */
/* Helper ops                                                               */
/* ----------------------------------------------------------------------- */
static void silu_inplace(float* x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

static void sigmoid_inplace(float* x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = 1.0f / (1.0f + expf(-x[i]));
}

static void rmsnorm(float* x, int dim) {
    float ss = 0;
    for (int i = 0; i < dim; i++) ss += x[i]*x[i];
    float s = 1.0f / sqrtf(ss / dim + 1e-6f);
    for (int i = 0; i < dim; i++) x[i] *= s;
}

/* ----------------------------------------------------------------------- */
/* Per-layer weight pointers (extracted from MLX model at init)             */
/* ----------------------------------------------------------------------- */
typedef struct {
    /* QKV projection: hidden -> conv_dim */
    uint32_t* qkv_w; uint16_t* qkv_s; uint16_t* qkv_b;
    /* Z gate: hidden -> value_dim */
    uint32_t* z_w; uint16_t* z_s; uint16_t* z_b;
    /* Beta: hidden -> num_v_heads */
    uint32_t* beta_w; uint16_t* beta_s; uint16_t* beta_b;
    /* Alpha: hidden -> num_v_heads */
    uint32_t* alpha_w; uint16_t* alpha_s; uint16_t* alpha_b;
    /* Output: value_dim -> hidden */
    uint32_t* out_w; uint16_t* out_s; uint16_t* out_b;
    /* Conv1d depthwise weights: [conv_dim, kernel] */
    float* conv_w;
    /* Delta-net params */
    float* A_log;    /* [num_v_heads] */
    float* dt_bias;  /* [num_v_heads] */
    /* Gated norm weight */
    float* norm_w;   /* [head_v_dim] */
} LayerWeights;

typedef struct {
    LayerWeights* layers;
    int num_layers;
    int hidden;      /* 3072 */
    int key_dim;     /* 2048 */
    int value_dim;   /* 8192 */
    int conv_dim;    /* 12288 */
    int num_k_heads; /* 16 */
    int num_v_heads; /* 64 */
    int head_k_dim;  /* 128 */
    int head_v_dim;  /* 128 */
    int conv_kernel;  /* 4 */

    /* Per-layer state */
    float* conv_states;   /* [num_layers][conv_kernel-1][conv_dim] */
    float* delta_states;  /* [num_layers][num_v_heads][head_v_dim][head_k_dim] */

    /* Scratch buffers (pre-allocated) */
    float* qkv_buf;   /* [conv_dim] */
    float* z_buf;      /* [value_dim] */
    float* beta_buf;   /* [num_v_heads] */
    float* alpha_buf;  /* [num_v_heads] */
    float* out_buf;    /* [value_dim] */
    float* conv_tmp;   /* [conv_dim] for conv input */
} FusedAttn;

/* ----------------------------------------------------------------------- */
/* Init / Destroy                                                           */
/* ----------------------------------------------------------------------- */
void* kandiga_fused_attn_init(int num_layers, int hidden,
    int key_dim, int value_dim, int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim, int conv_kernel)
{
    FusedAttn* e = (FusedAttn*)calloc(1, sizeof(FusedAttn));
    if (!e) return NULL;

    e->num_layers = num_layers;
    e->hidden = hidden;
    e->key_dim = key_dim;
    e->value_dim = value_dim;
    e->conv_dim = key_dim * 2 + value_dim;
    e->num_k_heads = num_k_heads;
    e->num_v_heads = num_v_heads;
    e->head_k_dim = head_k_dim;
    e->head_v_dim = head_v_dim;
    e->conv_kernel = conv_kernel;

    e->layers = (LayerWeights*)calloc(num_layers, sizeof(LayerWeights));

    /* State buffers */
    int cs = (conv_kernel - 1) * e->conv_dim;
    int ds = num_v_heads * head_v_dim * head_k_dim;
    e->conv_states = (float*)calloc(num_layers * cs, sizeof(float));
    e->delta_states = (float*)calloc(num_layers * ds, sizeof(float));

    /* Scratch */
    e->qkv_buf = (float*)calloc(e->conv_dim, sizeof(float));
    e->z_buf = (float*)calloc(value_dim, sizeof(float));
    e->beta_buf = (float*)calloc(num_v_heads, sizeof(float));
    e->alpha_buf = (float*)calloc(num_v_heads, sizeof(float));
    e->out_buf = (float*)calloc(value_dim, sizeof(float));
    e->conv_tmp = (float*)calloc(conv_kernel * e->conv_dim, sizeof(float));

    fprintf(stderr, "[kandiga-attn] Fused: %d layers, h=%d, kd=%d, vd=%d\n",
            num_layers, hidden, key_dim, value_dim);
    return e;
}

void kandiga_fused_attn_destroy(void* ptr) {
    if (!ptr) return;
    FusedAttn* e = (FusedAttn*)ptr;
    /* Free extracted weight copies */
    for (int i = 0; i < e->num_layers; i++) {
        free(e->layers[i].qkv_w); free(e->layers[i].qkv_s); free(e->layers[i].qkv_b);
        free(e->layers[i].z_w); free(e->layers[i].z_s); free(e->layers[i].z_b);
        free(e->layers[i].beta_w); free(e->layers[i].beta_s); free(e->layers[i].beta_b);
        free(e->layers[i].alpha_w); free(e->layers[i].alpha_s); free(e->layers[i].alpha_b);
        free(e->layers[i].out_w); free(e->layers[i].out_s); free(e->layers[i].out_b);
        free(e->layers[i].conv_w); free(e->layers[i].A_log); free(e->layers[i].dt_bias);
        free(e->layers[i].norm_w);
    }
    free(e->layers);
    free(e->conv_states); free(e->delta_states);
    free(e->qkv_buf); free(e->z_buf); free(e->beta_buf);
    free(e->alpha_buf); free(e->out_buf); free(e->conv_tmp);
    free(e);
    fprintf(stderr, "[kandiga-attn] Destroyed\n");
}

/* ----------------------------------------------------------------------- */
/* Set layer weights (called from Python during init)                       */
/* ----------------------------------------------------------------------- */
void kandiga_fused_attn_set_weights(void* ptr, int layer_idx,
    void* qkv_w, void* qkv_s, void* qkv_b, int qkv_w_bytes, int qkv_s_bytes,
    void* z_w, void* z_s, void* z_b, int z_w_bytes, int z_s_bytes,
    void* beta_w, void* beta_s, void* beta_b, int beta_w_bytes, int beta_s_bytes,
    void* alpha_w, void* alpha_s, void* alpha_b, int alpha_w_bytes, int alpha_s_bytes,
    void* out_w, void* out_s, void* out_b, int out_w_bytes, int out_s_bytes,
    void* conv_w, int conv_w_bytes,
    void* A_log, int A_log_bytes,
    void* dt_bias, int dt_bias_bytes,
    void* norm_w, int norm_w_bytes)
{
    FusedAttn* e = (FusedAttn*)ptr;
    LayerWeights* lw = &e->layers[layer_idx];

    /* Copy weights to our own memory (decoupled from MLX) */
    #define COPY(dst, src, bytes) do { dst = malloc(bytes); memcpy(dst, src, bytes); } while(0)
    COPY(lw->qkv_w, qkv_w, qkv_w_bytes); COPY(lw->qkv_s, qkv_s, qkv_s_bytes); COPY(lw->qkv_b, qkv_b, qkv_s_bytes);
    COPY(lw->z_w, z_w, z_w_bytes); COPY(lw->z_s, z_s, z_s_bytes); COPY(lw->z_b, z_b, z_s_bytes);
    COPY(lw->beta_w, beta_w, beta_w_bytes); COPY(lw->beta_s, beta_s, beta_s_bytes); COPY(lw->beta_b, beta_b, beta_s_bytes);
    COPY(lw->alpha_w, alpha_w, alpha_w_bytes); COPY(lw->alpha_s, alpha_s, alpha_s_bytes); COPY(lw->alpha_b, alpha_b, alpha_s_bytes);
    COPY(lw->out_w, out_w, out_w_bytes); COPY(lw->out_s, out_s, out_s_bytes); COPY(lw->out_b, out_b, out_s_bytes);
    COPY(lw->conv_w, conv_w, conv_w_bytes);
    COPY(lw->A_log, A_log, A_log_bytes);
    COPY(lw->dt_bias, dt_bias, dt_bias_bytes);
    COPY(lw->norm_w, norm_w, norm_w_bytes);
    #undef COPY
}

/* ----------------------------------------------------------------------- */
/* Reset state (call between conversations)                                 */
/* ----------------------------------------------------------------------- */
void kandiga_fused_attn_reset(void* ptr) {
    FusedAttn* e = (FusedAttn*)ptr;
    int cs = (e->conv_kernel - 1) * e->conv_dim;
    int ds = e->num_v_heads * e->head_v_dim * e->head_k_dim;
    memset(e->conv_states, 0, e->num_layers * cs * sizeof(float));
    memset(e->delta_states, 0, e->num_layers * ds * sizeof(float));
}

/* ----------------------------------------------------------------------- */
/* Fused GatedDeltaNet decode (single token, S=1)                          */
/*                                                                          */
/* One C call = one complete attention layer. Zero Metal dispatch.          */
/* ----------------------------------------------------------------------- */
int kandiga_fused_attn_decode(void* ptr, int li, const float* x, float* output) {
    FusedAttn* e = (FusedAttn*)ptr;
    if (!e || li < 0 || li >= e->num_layers) return -1;
    LayerWeights* lw = &e->layers[li];
    if (!lw->qkv_w) return -1;  /* weights not set */

    int H = e->hidden;
    int KD = e->key_dim;
    int VD = e->value_dim;
    int CD = e->conv_dim;
    int NK = e->num_k_heads;
    int NV = e->num_v_heads;
    int DK = e->head_k_dim;
    int DV = e->head_v_dim;
    int CK = e->conv_kernel;

    /* 1. Projections (4-bit dequant matvec on NEON) */
    matvec4bit(lw->qkv_w, lw->qkv_s, lw->qkv_b, x, e->qkv_buf, CD, H);
    matvec4bit(lw->z_w, lw->z_s, lw->z_b, x, e->z_buf, VD, H);
    matvec4bit(lw->beta_w, lw->beta_s, lw->beta_b, x, e->beta_buf, NV, H);
    matvec4bit(lw->alpha_w, lw->alpha_s, lw->alpha_b, x, e->alpha_buf, NV, H);

    /* 2. Conv1d: shift conv_state, append new, depthwise conv + SiLU */
    int cs_size = (CK - 1) * CD;
    float* cs = e->conv_states + li * cs_size;

    /* Build conv input: [conv_state(CK-1 rows), new_qkv(1 row)] */
    memcpy(e->conv_tmp, cs, (CK-1) * CD * sizeof(float));
    memcpy(e->conv_tmp + (CK-1) * CD, e->qkv_buf, CD * sizeof(float));

    /* Update conv_state: shift left by 1, append new */
    memmove(cs, cs + CD, (CK-2) * CD * sizeof(float));
    memcpy(cs + (CK-2) * CD, e->qkv_buf, CD * sizeof(float));

    /* Depthwise conv1d: for each channel, dot product with kernel weights */
    for (int c = 0; c < CD; c++) {
        float sum = 0;
        for (int k = 0; k < CK; k++) {
            sum += e->conv_tmp[k * CD + c] * lw->conv_w[c * CK + k];
        }
        e->qkv_buf[c] = sum;
    }
    silu_inplace(e->qkv_buf, CD);

    /* 3. Split into Q, K, V */
    float* Q = e->qkv_buf;           /* [NK * DK = KD] */
    float* K = e->qkv_buf + KD;      /* [NK * DK = KD] */
    float* V = e->qkv_buf + 2*KD;    /* [NV * DV = VD] */

    /* 4. RMSNorm Q and K (no weight, per head) */
    float inv_scale_sq = 1.0f / (DK * DK);  /* (head_dim^-0.5)^2 */
    float inv_scale = 1.0f / sqrtf((float)DK);
    for (int h = 0; h < NK; h++) {
        rmsnorm(Q + h*DK, DK);
        for (int d = 0; d < DK; d++) Q[h*DK+d] *= inv_scale_sq;
        rmsnorm(K + h*DK, DK);
        for (int d = 0; d < DK; d++) K[h*DK+d] *= inv_scale;
    }

    /* 5. Gated delta state update */
    sigmoid_inplace(e->beta_buf, NV);  /* beta = sigmoid(beta_proj) */

    /* g = exp(A_log * sigmoid(alpha) + dt_bias) per v-head */
    float g_buf[64]; /* max 64 v-heads */
    for (int h = 0; h < NV; h++) {
        float sig_a = 1.0f / (1.0f + expf(-e->alpha_buf[h]));
        g_buf[h] = expf(lw->A_log[h] * sig_a + lw->dt_bias[h]);
    }

    /* Delta state update: for each v-head group */
    /* K has NK heads, V has NV heads. NV/NK = heads_per_group */
    int ds_offset = li * NV * DV * DK;
    float* state = e->delta_states + ds_offset;
    int hpg = NV / NK;  /* heads per KV group (64/16 = 4) */

    for (int kh = 0; kh < NK; kh++) {
        float* k_vec = K + kh * DK;  /* [DK] */
        for (int vg = 0; vg < hpg; vg++) {
            int vh = kh * hpg + vg;
            float* v_vec = V + vh * DV;  /* [DV] */
            float* S = state + vh * DV * DK;  /* [DV, DK] */
            float beta = e->beta_buf[vh];
            float g = g_buf[vh];

            /* S = g * S + beta * (v ⊗ k) — outer product update */
            for (int dv = 0; dv < DV; dv++) {
                float bv = beta * v_vec[dv];
                for (int dk = 0; dk < DK; dk += 4) {
                    S[dv*DK+dk]   = g * S[dv*DK+dk]   + bv * k_vec[dk];
                    S[dv*DK+dk+1] = g * S[dv*DK+dk+1] + bv * k_vec[dk+1];
                    S[dv*DK+dk+2] = g * S[dv*DK+dk+2] + bv * k_vec[dk+2];
                    S[dv*DK+dk+3] = g * S[dv*DK+dk+3] + bv * k_vec[dk+3];
                }
            }

            /* out[vh] = S^T @ q = sum over dk: S[dv,dk] * q[dk] */
            float* q_vec = Q + kh * DK;
            float* o_vec = e->out_buf + vh * DV;
            for (int dv = 0; dv < DV; dv++) {
                float sum = 0;
                for (int dk = 0; dk < DK; dk += 4) {
                    sum += S[dv*DK+dk]   * q_vec[dk];
                    sum += S[dv*DK+dk+1] * q_vec[dk+1];
                    sum += S[dv*DK+dk+2] * q_vec[dk+2];
                    sum += S[dv*DK+dk+3] * q_vec[dk+3];
                }
                o_vec[dv] = sum;
            }
        }
    }

    /* 6. Gated norm: out = rmsnorm(out, norm_w) * silu(z) */
    /* Per-head RMSNorm with weight, then multiply by SiLU(z) */
    for (int vh = 0; vh < NV; vh++) {
        float* o = e->out_buf + vh * DV;
        float* z = e->z_buf + vh * DV;
        /* RMSNorm with weight */
        float ss = 0;
        for (int d = 0; d < DV; d++) ss += o[d]*o[d];
        float s = 1.0f / sqrtf(ss / DV + 1e-6f);
        for (int d = 0; d < DV; d++) {
            float normed = o[d] * s * lw->norm_w[d];
            float gate = z[d] / (1.0f + expf(-z[d]));
            o[d] = normed * gate;
        }
    }

    /* 7. Output projection: value_dim -> hidden */
    matvec4bit(lw->out_w, lw->out_s, lw->out_b, e->out_buf, output, H, VD);

    return 0;
}

/* bf16 input/output wrapper */
int kandiga_fused_attn_decode_bf16(void* ptr, int li,
    const void* x_bf16, void* out_bf16)
{
    FusedAttn* e = (FusedAttn*)ptr;
    int H = e->hidden;

    /* Convert bf16 input to f32 */
    const uint16_t* xb = (const uint16_t*)x_bf16;
    float x_f32[8192]; /* max hidden */
    for (int i = 0; i < H; i++) x_f32[i] = bf16_to_f32(xb[i]);

    float out_f32[8192];
    int ret = kandiga_fused_attn_decode(ptr, li, x_f32, out_f32);

    /* Convert f32 output to bf16 */
    uint16_t* ob = (uint16_t*)out_bf16;
    for (int i = 0; i < H; i++) ob[i] = f32_to_bf16(out_f32[i]);

    return ret;
}
