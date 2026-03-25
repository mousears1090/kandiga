/*
 * metal_attn_lg.m — Metal GPU fused attention for large MoE models.
 *
 * Single command buffer per layer: all attention ops fused into one
 * GPU submission. Eliminates per-op Metal dispatch overhead.
 *
 * Uses parameterized kernels from attention.metal.
 *
 * Build:
 *   1. Compile attention.metal -> attention.metallib
 *      xcrun -sdk macosx metal -O2 -o attention.air attention.metal
 *      xcrun -sdk macosx metallib -o attention.metallib attention.air
 *   2. Compile this host:
 *      clang -shared -o libkandiga_metal_attn_lg.dylib metal_attn_lg.m \
 *            -framework Foundation -framework Metal -framework MetalPerformanceShaders \
 *            -fobjc-arc -O2 -march=native
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>

/* ----------------------------------------------------------------------- */
/* Engine state                                                             */
/* ----------------------------------------------------------------------- */
typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;

    /* Pipeline states for each kernel */
    id<MTLComputePipelineState> matvec_4bit_ps;
    id<MTLComputePipelineState> rmsnorm_ps;
    id<MTLComputePipelineState> depthwise_conv1d_silu_ps;
    id<MTLComputePipelineState> gated_delta_state_ps;
    id<MTLComputePipelineState> rmsnorm_gated_ps;
    id<MTLComputePipelineState> residual_add_ps;

    /* Model dimensions */
    int hidden;         /* 3072 */
    int key_dim;        /* 2048 total */
    int value_dim;      /* 8192 total */
    int conv_dim;       /* 12288 */
    int num_k_heads;    /* 16 */
    int num_v_heads;    /* 64 */
    int head_k_dim;     /* 128 */
    int head_v_dim;     /* 128 */
    int conv_kernel;    /* 4 */
    int num_layers;

    /* Per-layer GPU buffers for attention weights */
    id<MTLBuffer> __strong * qkv_w_bufs;    /* [num_layers] */
    id<MTLBuffer> __strong * qkv_s_bufs;
    id<MTLBuffer> __strong * qkv_b_bufs;
    id<MTLBuffer> __strong * z_w_bufs;
    id<MTLBuffer> __strong * z_s_bufs;
    id<MTLBuffer> __strong * z_b_bufs;
    id<MTLBuffer> __strong * beta_w_bufs;
    id<MTLBuffer> __strong * beta_s_bufs;
    id<MTLBuffer> __strong * beta_b_bufs;
    id<MTLBuffer> __strong * alpha_w_bufs;
    id<MTLBuffer> __strong * alpha_s_bufs;
    id<MTLBuffer> __strong * alpha_b_bufs;
    id<MTLBuffer> __strong * out_w_bufs;
    id<MTLBuffer> __strong * out_s_bufs;
    id<MTLBuffer> __strong * out_b_bufs;
    id<MTLBuffer> __strong * conv_w_bufs;
    id<MTLBuffer> __strong * A_log_bufs;
    id<MTLBuffer> __strong * dt_bias_bufs;
    id<MTLBuffer> __strong * norm_w_bufs;

    /* Scratch GPU buffers (shared across layers, reused per token) */
    id<MTLBuffer> qkv_buf;     /* half[conv_dim] */
    id<MTLBuffer> z_buf;        /* half[value_dim] */
    id<MTLBuffer> beta_buf;     /* half[num_v_heads] */
    id<MTLBuffer> alpha_buf;    /* half[num_v_heads] */
    id<MTLBuffer> attn_out_buf; /* half[value_dim] */
    id<MTLBuffer> final_out_buf;/* half[hidden] */

    /* Conv state + delta state */
    id<MTLBuffer> __strong * conv_state_bufs;  /* [num_layers] half[conv_kernel-1, conv_dim] */
    id<MTLBuffer> __strong * delta_state_bufs; /* [num_layers] float[num_v, val_dim, key_dim] */

    /* Param buffer */
    id<MTLBuffer> params_buf;

} MetalAttnEngine;

/* ----------------------------------------------------------------------- */
/* Init                                                                     */
/* ----------------------------------------------------------------------- */
void* kandiga_metal_attn_init(int num_layers, int hidden,
    int key_dim, int value_dim, int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim, int conv_kernel)
{
    MetalAttnEngine* e = (MetalAttnEngine*)calloc(1, sizeof(MetalAttnEngine));
    if (!e) return NULL;

    e->device = MTLCreateSystemDefaultDevice();
    if (!e->device) {
        fprintf(stderr, "[kandiga-metal] ERROR: No Metal device\n");
        free(e); return NULL;
    }
    e->queue = [e->device newCommandQueue];

    /* Load Metal library from .metallib */
    NSString* metalDir = [NSString stringWithFormat:@"%s",
        [[[NSBundle mainBundle] bundlePath] UTF8String]];
    /* Try relative to dylib location */
    NSString* libPath = nil;
    Dl_info info;
    if (dladdr((void*)kandiga_metal_attn_init, &info)) {
        NSString* dir = [[NSString stringWithUTF8String:info.dli_fname]
                          stringByDeletingLastPathComponent];
        libPath = [dir stringByAppendingPathComponent:@"attention.metallib"];
    }

    NSError* err = nil;
    if (libPath && [[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
        NSURL* url = [NSURL fileURLWithPath:libPath];
        e->library = [e->device newLibraryWithURL:url error:&err];
    }
    if (!e->library) {
        fprintf(stderr, "[kandiga-metal] ERROR: Cannot load attention.metallib: %s\n",
                err ? [[err localizedDescription] UTF8String] : "not found");
        free(e); return NULL;
    }

    /* Create pipeline states */
    #define MAKE_PS(name, var) do { \
        id<MTLFunction> fn = [e->library newFunctionWithName:@name]; \
        if (fn) { var = [e->device newComputePipelineStateWithFunction:fn error:&err]; \
            if (!var) fprintf(stderr, "[kandiga-metal] WARNING: %s pipeline failed\n", name); } \
        else fprintf(stderr, "[kandiga-metal] WARNING: %s not found\n", name); \
    } while(0)

    MAKE_PS("matvec_4bit", e->matvec_4bit_ps);
    MAKE_PS("rmsnorm_forward", e->rmsnorm_ps);
    MAKE_PS("depthwise_conv1d_silu", e->depthwise_conv1d_silu_ps);
    MAKE_PS("gated_delta_state_update", e->gated_delta_state_ps);
    MAKE_PS("rmsnorm_gated_kernel", e->rmsnorm_gated_ps);
    MAKE_PS("residual_add", e->residual_add_ps);
    #undef MAKE_PS

    /* Store dimensions */
    e->hidden = hidden;
    e->key_dim = key_dim;
    e->value_dim = value_dim;
    e->conv_dim = key_dim * 2 + value_dim;
    e->num_k_heads = num_k_heads;
    e->num_v_heads = num_v_heads;
    e->head_k_dim = head_k_dim;
    e->head_v_dim = head_v_dim;
    e->conv_kernel = conv_kernel;
    e->num_layers = num_layers;

    /* Allocate per-layer buffer arrays */
    #define ALLOC_BUFS(name) e->name = (id<MTLBuffer> __strong *)calloc(num_layers, sizeof(id<MTLBuffer>))
    ALLOC_BUFS(qkv_w_bufs); ALLOC_BUFS(qkv_s_bufs); ALLOC_BUFS(qkv_b_bufs);
    ALLOC_BUFS(z_w_bufs); ALLOC_BUFS(z_s_bufs); ALLOC_BUFS(z_b_bufs);
    ALLOC_BUFS(beta_w_bufs); ALLOC_BUFS(beta_s_bufs); ALLOC_BUFS(beta_b_bufs);
    ALLOC_BUFS(alpha_w_bufs); ALLOC_BUFS(alpha_s_bufs); ALLOC_BUFS(alpha_b_bufs);
    ALLOC_BUFS(out_w_bufs); ALLOC_BUFS(out_s_bufs); ALLOC_BUFS(out_b_bufs);
    ALLOC_BUFS(conv_w_bufs); ALLOC_BUFS(A_log_bufs); ALLOC_BUFS(dt_bias_bufs);
    ALLOC_BUFS(norm_w_bufs);
    ALLOC_BUFS(conv_state_bufs); ALLOC_BUFS(delta_state_bufs);
    #undef ALLOC_BUFS

    /* Allocate scratch buffers */
    e->qkv_buf = [e->device newBufferWithLength:e->conv_dim * 2 options:MTLResourceStorageModeShared];
    e->z_buf = [e->device newBufferWithLength:value_dim * 2 options:MTLResourceStorageModeShared];
    e->beta_buf = [e->device newBufferWithLength:num_v_heads * 2 options:MTLResourceStorageModeShared];
    e->alpha_buf = [e->device newBufferWithLength:num_v_heads * 2 options:MTLResourceStorageModeShared];
    e->attn_out_buf = [e->device newBufferWithLength:value_dim * 2 options:MTLResourceStorageModeShared];
    e->final_out_buf = [e->device newBufferWithLength:hidden * 2 options:MTLResourceStorageModeShared];

    /* Per-layer conv state + delta state */
    int conv_state_bytes = (conv_kernel - 1) * e->conv_dim * 2;  /* half */
    int delta_state_bytes = num_v_heads * head_v_dim * head_k_dim * 4;  /* float32 */
    for (int i = 0; i < num_layers; i++) {
        e->conv_state_bufs[i] = [e->device newBufferWithLength:conv_state_bytes
                                  options:MTLResourceStorageModeShared];
        e->delta_state_bufs[i] = [e->device newBufferWithLength:delta_state_bytes
                                   options:MTLResourceStorageModeShared];
        memset([e->conv_state_bufs[i] contents], 0, conv_state_bytes);
        memset([e->delta_state_bufs[i] contents], 0, delta_state_bytes);
    }

    fprintf(stderr, "[kandiga-metal] GPU attention: %d layers, h=%d, kd=%d, vd=%d\n",
            num_layers, hidden, key_dim, value_dim);
    return e;
}

void kandiga_metal_attn_destroy(void* ptr) {
    if (!ptr) return;
    MetalAttnEngine* e = (MetalAttnEngine*)ptr;
    /* ARC handles Metal object release */
    free(e->qkv_w_bufs); free(e->qkv_s_bufs); free(e->qkv_b_bufs);
    free(e->z_w_bufs); free(e->z_s_bufs); free(e->z_b_bufs);
    free(e->beta_w_bufs); free(e->beta_s_bufs); free(e->beta_b_bufs);
    free(e->alpha_w_bufs); free(e->alpha_s_bufs); free(e->alpha_b_bufs);
    free(e->out_w_bufs); free(e->out_s_bufs); free(e->out_b_bufs);
    free(e->conv_w_bufs); free(e->A_log_bufs); free(e->dt_bias_bufs);
    free(e->norm_w_bufs);
    free(e->conv_state_bufs); free(e->delta_state_bufs);
    free(e);
    fprintf(stderr, "[kandiga-metal] Destroyed\n");
}

/* ----------------------------------------------------------------------- */
/* Set layer weights — copies data to GPU Metal buffers                    */
/* ----------------------------------------------------------------------- */
void kandiga_metal_attn_set_weights(void* ptr, int layer_idx,
    void* qkv_w, int qkv_w_bytes, void* qkv_s, int qkv_s_bytes, void* qkv_b,
    void* z_w, int z_w_bytes, void* z_s, int z_s_bytes, void* z_b,
    void* beta_w, int beta_w_bytes, void* beta_s, int beta_s_bytes, void* beta_b,
    void* alpha_w, int alpha_w_bytes, void* alpha_s, int alpha_s_bytes, void* alpha_b,
    void* out_w, int out_w_bytes, void* out_s, int out_s_bytes, void* out_b,
    void* conv_w, int conv_w_bytes,
    void* A_log, int A_log_bytes,
    void* dt_bias, int dt_bias_bytes,
    void* norm_w, int norm_w_bytes)
{
    MetalAttnEngine* e = (MetalAttnEngine*)ptr;

    #define COPY_BUF(arr, src, bytes) do { \
        arr[layer_idx] = [e->device newBufferWithBytes:src length:bytes \
                          options:MTLResourceStorageModeShared]; \
    } while(0)

    COPY_BUF(e->qkv_w_bufs, qkv_w, qkv_w_bytes);
    COPY_BUF(e->qkv_s_bufs, qkv_s, qkv_s_bytes);
    COPY_BUF(e->qkv_b_bufs, qkv_b, qkv_s_bytes);
    COPY_BUF(e->z_w_bufs, z_w, z_w_bytes);
    COPY_BUF(e->z_s_bufs, z_s, z_s_bytes);
    COPY_BUF(e->z_b_bufs, z_b, z_s_bytes);
    COPY_BUF(e->beta_w_bufs, beta_w, beta_w_bytes);
    COPY_BUF(e->beta_s_bufs, beta_s, beta_s_bytes);
    COPY_BUF(e->beta_b_bufs, beta_b, beta_s_bytes);
    COPY_BUF(e->alpha_w_bufs, alpha_w, alpha_w_bytes);
    COPY_BUF(e->alpha_s_bufs, alpha_s, alpha_s_bytes);
    COPY_BUF(e->alpha_b_bufs, alpha_b, alpha_s_bytes);
    COPY_BUF(e->out_w_bufs, out_w, out_w_bytes);
    COPY_BUF(e->out_s_bufs, out_s, out_s_bytes);
    COPY_BUF(e->out_b_bufs, out_b, out_s_bytes);
    COPY_BUF(e->conv_w_bufs, conv_w, conv_w_bytes);
    COPY_BUF(e->A_log_bufs, A_log, A_log_bytes);
    COPY_BUF(e->dt_bias_bufs, dt_bias, dt_bias_bytes);
    COPY_BUF(e->norm_w_bufs, norm_w, norm_w_bytes);
    #undef COPY_BUF
}

/* ----------------------------------------------------------------------- */
/* Fused decode — ONE command buffer for entire attention layer             */
/* ----------------------------------------------------------------------- */
int kandiga_metal_attn_decode(void* ptr, int layer_idx,
    const void* x_half, void* out_half)
{
    MetalAttnEngine* e = (MetalAttnEngine*)ptr;
    if (!e || layer_idx < 0 || layer_idx >= e->num_layers) return -1;
    if (!e->matvec_4bit_ps || !e->gated_delta_state_ps) return -1;

    /* Create input buffer wrapping caller's data (no copy on unified memory) */
    id<MTLBuffer> input_buf = [e->device newBufferWithBytesNoCopy:(void*)x_half
                                length:e->hidden * 2
                                options:MTLResourceStorageModeShared
                                deallocator:nil];

    id<MTLCommandBuffer> cb = [e->queue commandBuffer];

    /* --- 1. QKV projection: hidden -> conv_dim --- */
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:e->matvec_4bit_ps];
        [enc setBuffer:e->qkv_w_bufs[layer_idx] offset:0 atIndex:0];
        [enc setBuffer:e->qkv_s_bufs[layer_idx] offset:0 atIndex:1];
        [enc setBuffer:e->qkv_b_bufs[layer_idx] offset:0 atIndex:2];
        [enc setBuffer:input_buf offset:0 atIndex:3];
        [enc setBuffer:e->qkv_buf offset:0 atIndex:4];
        int32_t dims[2] = { (int32_t)e->conv_dim, (int32_t)e->hidden };
        [enc setBytes:&dims[0] length:4 atIndex:5];
        [enc setBytes:&dims[1] length:4 atIndex:6];
        int32_t gs = 64;
        [enc setBytes:&gs length:4 atIndex:7];
        MTLSize grid = MTLSizeMake(e->conv_dim, 1, 1);
        MTLSize tg = MTLSizeMake(MIN(256, e->conv_dim), 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }

    /* --- 2. Z projection: hidden -> value_dim --- */
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:e->matvec_4bit_ps];
        [enc setBuffer:e->z_w_bufs[layer_idx] offset:0 atIndex:0];
        [enc setBuffer:e->z_s_bufs[layer_idx] offset:0 atIndex:1];
        [enc setBuffer:e->z_b_bufs[layer_idx] offset:0 atIndex:2];
        [enc setBuffer:input_buf offset:0 atIndex:3];
        [enc setBuffer:e->z_buf offset:0 atIndex:4];
        int32_t dims[2] = { (int32_t)e->value_dim, (int32_t)e->hidden };
        [enc setBytes:&dims[0] length:4 atIndex:5];
        [enc setBytes:&dims[1] length:4 atIndex:6];
        int32_t gs = 64;
        [enc setBytes:&gs length:4 atIndex:7];
        MTLSize grid = MTLSizeMake(e->value_dim, 1, 1);
        MTLSize tg = MTLSizeMake(MIN(256, e->value_dim), 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }

    /* --- 3. Beta + Alpha projections (small) --- */
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:e->matvec_4bit_ps];
        [enc setBuffer:e->beta_w_bufs[layer_idx] offset:0 atIndex:0];
        [enc setBuffer:e->beta_s_bufs[layer_idx] offset:0 atIndex:1];
        [enc setBuffer:e->beta_b_bufs[layer_idx] offset:0 atIndex:2];
        [enc setBuffer:input_buf offset:0 atIndex:3];
        [enc setBuffer:e->beta_buf offset:0 atIndex:4];
        int32_t dims[2] = { (int32_t)e->num_v_heads, (int32_t)e->hidden };
        [enc setBytes:&dims[0] length:4 atIndex:5];
        [enc setBytes:&dims[1] length:4 atIndex:6];
        int32_t gs = 64;
        [enc setBytes:&gs length:4 atIndex:7];
        MTLSize grid = MTLSizeMake(e->num_v_heads, 1, 1);
        MTLSize tg = MTLSizeMake(MIN(64, e->num_v_heads), 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:e->matvec_4bit_ps];
        [enc setBuffer:e->alpha_w_bufs[layer_idx] offset:0 atIndex:0];
        [enc setBuffer:e->alpha_s_bufs[layer_idx] offset:0 atIndex:1];
        [enc setBuffer:e->alpha_b_bufs[layer_idx] offset:0 atIndex:2];
        [enc setBuffer:input_buf offset:0 atIndex:3];
        [enc setBuffer:e->alpha_buf offset:0 atIndex:4];
        int32_t dims[2] = { (int32_t)e->num_v_heads, (int32_t)e->hidden };
        [enc setBytes:&dims[0] length:4 atIndex:5];
        [enc setBytes:&dims[1] length:4 atIndex:6];
        int32_t gs = 64;
        [enc setBytes:&gs length:4 atIndex:7];
        MTLSize grid = MTLSizeMake(e->num_v_heads, 1, 1);
        MTLSize tg = MTLSizeMake(MIN(64, e->num_v_heads), 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }

    /* --- 4. Gated delta state update --- */
    /* Note: conv1d + split + rmsnorm happen here too but for now we skip conv
       and go straight to the state update as a first cut */
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:e->gated_delta_state_ps];
        /* Q = qkv_buf[0..key_dim], K = qkv_buf[key_dim..2*key_dim], V = qkv_buf[2*key_dim..] */
        [enc setBuffer:e->qkv_buf offset:0 atIndex:0];                    /* q */
        [enc setBuffer:e->qkv_buf offset:e->key_dim*2 atIndex:1];         /* k */
        [enc setBuffer:e->qkv_buf offset:e->key_dim*4 atIndex:2];         /* v */
        [enc setBuffer:e->alpha_buf offset:0 atIndex:3];                   /* a */
        [enc setBuffer:e->beta_buf offset:0 atIndex:4];                    /* b */
        [enc setBuffer:e->A_log_bufs[layer_idx] offset:0 atIndex:5];
        [enc setBuffer:e->dt_bias_bufs[layer_idx] offset:0 atIndex:6];
        [enc setBuffer:e->delta_state_bufs[layer_idx] offset:0 atIndex:7];
        [enc setBuffer:e->attn_out_buf offset:0 atIndex:8];
        int32_t nk = e->num_k_heads, nv = e->num_v_heads;
        int32_t kd = e->head_k_dim, vd = e->head_v_dim;
        [enc setBytes:&nk length:4 atIndex:9];
        [enc setBytes:&nv length:4 atIndex:10];
        [enc setBytes:&kd length:4 atIndex:11];
        [enc setBytes:&vd length:4 atIndex:12];
        MTLSize grid = MTLSizeMake(nv, vd, 1);
        MTLSize tg = MTLSizeMake(1, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }

    /* --- 5. Gated norm + output projection --- */
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:e->rmsnorm_gated_ps];
        [enc setBuffer:e->attn_out_buf offset:0 atIndex:0];  /* hidden */
        [enc setBuffer:e->z_buf offset:0 atIndex:1];          /* gate */
        [enc setBuffer:e->norm_w_bufs[layer_idx] offset:0 atIndex:2];
        [enc setBuffer:e->attn_out_buf offset:0 atIndex:3];  /* output (in-place ok) */
        int32_t nh = e->num_v_heads, hd = e->head_v_dim;
        float eps = 1e-6f;
        [enc setBytes:&nh length:4 atIndex:4];
        [enc setBytes:&hd length:4 atIndex:5];
        [enc setBytes:&eps length:4 atIndex:6];
        MTLSize grid = MTLSizeMake(nh, 1, 1);
        MTLSize tg = MTLSizeMake(MIN(128, hd), 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }

    /* --- 6. Output projection: value_dim -> hidden --- */
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:e->matvec_4bit_ps];
        [enc setBuffer:e->out_w_bufs[layer_idx] offset:0 atIndex:0];
        [enc setBuffer:e->out_s_bufs[layer_idx] offset:0 atIndex:1];
        [enc setBuffer:e->out_b_bufs[layer_idx] offset:0 atIndex:2];
        [enc setBuffer:e->attn_out_buf offset:0 atIndex:3];
        [enc setBuffer:e->final_out_buf offset:0 atIndex:4];
        int32_t dims[2] = { (int32_t)e->hidden, (int32_t)e->value_dim };
        [enc setBytes:&dims[0] length:4 atIndex:5];
        [enc setBytes:&dims[1] length:4 atIndex:6];
        int32_t gs = 64;
        [enc setBytes:&gs length:4 atIndex:7];
        MTLSize grid = MTLSizeMake(e->hidden, 1, 1);
        MTLSize tg = MTLSizeMake(MIN(256, e->hidden), 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }

    /* Submit and wait */
    [cb commit];
    [cb waitUntilCompleted];

    /* Copy output */
    memcpy(out_half, [e->final_out_buf contents], e->hidden * 2);

    return 0;
}
