// attention.metal — Metal compute kernels for the complete transformer forward pass
//
// Kernels for Qwen3.5-35B-A3B single-token decode:
//   1. embed_lookup      — Token embedding table lookup
//   2. rmsnorm_forward   — RMSNorm (reusable for all norm layers)
//   3. qkv_proj_4bit     — 4-bit quantized matrix-vector multiply (generic)
//   4. qkv_proj_8bit     — 8-bit quantized matrix-vector multiply (for router/gate)
//   5. rope_kernel       — Rotary positional embeddings (partial, 64/256 dims)
//   6. gqa_attention     — Full GQA self-attention with KV cache (10 layers)
//   7. gated_delta_proj  — GatedDeltaNet linear projections + conv1d (30 layers)
//   8. gated_delta_state — GatedDeltaNet state update + output
//   9. rmsnorm_gated     — RMSNormGated (gated norm for linear attention)
//  10. sigmoid_gate_mul  — sigmoid(gate) * value (for attention output gating)
//  11. lm_head_4bit      — LM head matmul (2048 -> vocab_size, 4-bit)
//  12. residual_add      — Element-wise residual addition
//  13. vec_add_half      — Element-wise vector addition (half precision)
//
// All quantized weights use 4-bit packed uint32 with bfloat16 scales/biases,
// group_size=64 (matching existing expert_mlp.metal conventions).

#include <metal_stdlib>
using namespace metal;

// bfloat16 support — use Metal's built-in if available, otherwise typedef






// ---------------------------------------------------------------------------
// Forward pass parameter buffer
// ---------------------------------------------------------------------------
struct ForwardParams {
    int32_t hidden_size;      // 2048
    int32_t vocab_size;       // 248320
    int32_t num_layers;       // 40
    int32_t group_size;       // 64
    float   rms_norm_eps;     // 1e-6
    int32_t position;         // current token position (for RoPE)
    int32_t kv_len;           // current KV cache length (before this token)
    // Full attention params
    int32_t num_q_heads;      // 16
    int32_t num_kv_heads;     // 2
    int32_t head_dim;         // 256
    int32_t rope_dim;         // 64 (partial_rotary_factor * head_dim)
    float   rope_theta;       // 10000000.0
    // Linear attention params
    int32_t lin_num_k_heads;  // 16
    int32_t lin_num_v_heads;  // 32
    int32_t lin_key_dim;      // 128 (per head)
    int32_t lin_val_dim;      // 128 (per head)
    int32_t conv_kernel_size; // 4
    // Generic matmul params (reusable)
    int32_t matmul_out_dim;   // varies per projection
    int32_t matmul_in_dim;    // varies per projection
};

// ---------------------------------------------------------------------------
// Dequantized dot product — 4-bit packed weights x half input (shared mem)
// Identical to expert_mlp.metal's dequant_dot_half
// ---------------------------------------------------------------------------
inline float dequant_dot_4bit(
    device const uint32_t* weight_row,
    threadgroup const half* x_shared,
    device const bfloat*   scales_row,
    device const bfloat*   biases_row,
    uint in_dim,
    uint group_size
) {
    float acc = 0.0f;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;

    for (uint g = 0; g < num_groups; g++) {
        float scale = float(scales_row[g]);
        float bias  = float(biases_row[g]);

        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t packed = weight_row[g * packed_per_group + p];
            uint x_base = g * group_size + p * 8;

            float x0 = float(x_shared[x_base + 0]);
            float x1 = float(x_shared[x_base + 1]);
            float x2 = float(x_shared[x_base + 2]);
            float x3 = float(x_shared[x_base + 3]);
            float x4 = float(x_shared[x_base + 4]);
            float x5 = float(x_shared[x_base + 5]);
            float x6 = float(x_shared[x_base + 6]);
            float x7 = float(x_shared[x_base + 7]);

            acc = fma(fma(float((packed      ) & 0xF), scale, bias), x0, acc);
            acc = fma(fma(float((packed >>  4) & 0xF), scale, bias), x1, acc);
            acc = fma(fma(float((packed >>  8) & 0xF), scale, bias), x2, acc);
            acc = fma(fma(float((packed >> 12) & 0xF), scale, bias), x3, acc);
            acc = fma(fma(float((packed >> 16) & 0xF), scale, bias), x4, acc);
            acc = fma(fma(float((packed >> 20) & 0xF), scale, bias), x5, acc);
            acc = fma(fma(float((packed >> 24) & 0xF), scale, bias), x6, acc);
            acc = fma(fma(float((packed >> 28) & 0xF), scale, bias), x7, acc);
        }
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Dequantized dot product — 4-bit, device memory input (no shared)
// ---------------------------------------------------------------------------
inline float dequant_dot_4bit_device(
    device const uint32_t* weight_row,
    device const half*     x,
    device const bfloat*   scales_row,
    device const bfloat*   biases_row,
    uint in_dim,
    uint group_size
) {
    float acc = 0.0f;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;

    for (uint g = 0; g < num_groups; g++) {
        float scale = float(scales_row[g]);
        float bias  = float(biases_row[g]);

        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t packed = weight_row[g * packed_per_group + p];
            uint x_base = g * group_size + p * 8;

            acc = fma(fma(float((packed      ) & 0xF), scale, bias), float(x[x_base + 0]), acc);
            acc = fma(fma(float((packed >>  4) & 0xF), scale, bias), float(x[x_base + 1]), acc);
            acc = fma(fma(float((packed >>  8) & 0xF), scale, bias), float(x[x_base + 2]), acc);
            acc = fma(fma(float((packed >> 12) & 0xF), scale, bias), float(x[x_base + 3]), acc);
            acc = fma(fma(float((packed >> 16) & 0xF), scale, bias), float(x[x_base + 4]), acc);
            acc = fma(fma(float((packed >> 20) & 0xF), scale, bias), float(x[x_base + 5]), acc);
            acc = fma(fma(float((packed >> 24) & 0xF), scale, bias), float(x[x_base + 6]), acc);
            acc = fma(fma(float((packed >> 28) & 0xF), scale, bias), float(x[x_base + 7]), acc);
        }
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Dequantized dot product — 8-bit packed weights x half input
// 8-bit: each uint32 contains 4 values (8 bits each)
// Used for router gate and shared_expert_gate (quant_predicate says 8-bit)
// ---------------------------------------------------------------------------
inline float dequant_dot_8bit(
    device const uint32_t* weight_row,
    threadgroup const half* x_shared,
    device const bfloat*   scales_row,
    device const bfloat*   biases_row,
    uint in_dim,
    uint group_size
) {
    float acc = 0.0f;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 4;  // 4 values per uint32 for 8-bit

    for (uint g = 0; g < num_groups; g++) {
        float scale = float(scales_row[g]);
        float bias  = float(biases_row[g]);

        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t packed = weight_row[g * packed_per_group + p];
            uint x_base = g * group_size + p * 4;

            float x0 = float(x_shared[x_base + 0]);
            float x1 = float(x_shared[x_base + 1]);
            float x2 = float(x_shared[x_base + 2]);
            float x3 = float(x_shared[x_base + 3]);

            acc = fma(fma(float((packed      ) & 0xFF), scale, bias), x0, acc);
            acc = fma(fma(float((packed >>  8) & 0xFF), scale, bias), x1, acc);
            acc = fma(fma(float((packed >> 16) & 0xFF), scale, bias), x2, acc);
            acc = fma(fma(float((packed >> 24) & 0xFF), scale, bias), x3, acc);
        }
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Kernel 1: embed_lookup
//
// Look up token embedding from 4-bit quantized embedding table.
// Embedding table: (vocab_size, hidden_size/8) uint32 packed
// Output: half[hidden_size]
//
// Grid: (hidden_size, 1, 1)
// Threadgroup: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void embed_lookup(
    device const uint32_t* embed_weight [[buffer(0)]],  // packed 4-bit
    device const bfloat*   embed_scales [[buffer(1)]],
    device const bfloat*   embed_biases [[buffer(2)]],
    device       half*     output       [[buffer(3)]],  // half[hidden_size]
    device const int32_t&  token_id     [[buffer(4)]],
    device const ForwardParams& params  [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint hidden_size = uint(params.hidden_size);
    uint group_size = uint(params.group_size);

    if (gid >= hidden_size) return;

    int tid = token_id;
    // For embedding, each row is hidden_size values packed as 4-bit
    // Row tid: weight starts at tid * (hidden_size/8) uint32s
    // scales at tid * (hidden_size/group_size)
    uint packed_per_row = hidden_size / 8;
    uint groups_per_row = hidden_size / group_size;

    // Find which group and which position within group
    uint group_idx = gid / group_size;
    uint pos_in_group = gid % group_size;
    uint packed_idx = pos_in_group / 8;
    uint nibble_idx = pos_in_group % 8;

    float scale = float(embed_scales[uint(tid) * groups_per_row + group_idx]);
    float bias  = float(embed_biases[uint(tid) * groups_per_row + group_idx]);

    uint32_t packed = embed_weight[uint(tid) * packed_per_row + group_idx * (group_size / 8) + packed_idx];
    float nibble = float((packed >> (nibble_idx * 4)) & 0xF);
    float val = fma(nibble, scale, bias);
    output[gid] = half(val);
}

// ---------------------------------------------------------------------------
// Kernel 2: rmsnorm_forward
//
// RMSNorm: output[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
// Weight is bfloat16.
//
// Grid: (1, 1, 1) — single threadgroup
// Threadgroup: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void rmsnorm_forward(
    device const half*     x       [[buffer(0)]],
    device const bfloat*   weight  [[buffer(1)]],
    device       half*     output  [[buffer(2)]],
    device const int32_t&  dim     [[buffer(3)]],
    device const float&    eps     [[buffer(4)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]]
) {
    uint n = uint(dim);

    threadgroup float partial_sums[256];
    float local_sum = 0.0f;
    for (uint i = tid; i < n; i += tgs) {
        float v = float(x[i]);
        local_sum += v * v;
    }
    partial_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float shared_scale;
    if (tid == 0) {
        float mean_sq = partial_sums[0] / float(n);
        shared_scale = 1.0f / sqrt(mean_sq + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = shared_scale;
    for (uint i = tid; i < n; i += tgs) {
        float v = float(x[i]) * scale * float(weight[i]);
        output[i] = half(v);
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: matvec_4bit — SIMD group reduction with shared memory
//
// 4-bit quantized matrix-vector multiply.
// Each threadgroup: 256 threads = 8 SIMD groups, processing 8 output rows.
// Phase 1: All 256 threads cooperatively load x into shared memory.
// Phase 2: Each SIMD group (32 lanes) computes one row's dot product,
//          splitting across quantization groups and reducing via simd_sum().
//
// For small projections (out_dim < 8), some SIMD groups are idle.
// For the common case (out_dim >> 8), this amortizes x loading well.
//
// Dispatch: threadgroups=(ceil(out_dim/8), 1, 1), threadsPerThreadgroup=(256, 1, 1)
// ---------------------------------------------------------------------------
kernel void matvec_4bit(
    device const half*     x_input  [[buffer(0)]],  // half[in_dim]
    device const uint32_t* weight   [[buffer(1)]],  // (out_dim, in_dim/8) uint32
    device const bfloat*   scales   [[buffer(2)]],  // (out_dim, in_dim/gs) bf16
    device const bfloat*   biases   [[buffer(3)]],  // (out_dim, in_dim/gs) bf16
    device       half*     output   [[buffer(4)]],  // half[out_dim]
    device const int32_t&  out_dim  [[buffer(5)]],
    device const int32_t&  in_dim   [[buffer(6)]],
    device const int32_t&  grp_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    uint out_d = uint(out_dim);
    uint in_d  = uint(in_dim);
    uint gs    = uint(grp_size);

    // Load input into shared memory — all 256 threads cooperate
    threadgroup half x_shared[8192];
    for (uint i = tid; i < in_d; i += 256) {
        x_shared[i] = x_input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each SIMD group handles one output row
    uint row = tgid * 8 + sg_idx;
    if (row >= out_d) return;

    uint w_stride = in_d / 8;
    uint s_stride = in_d / gs;
    uint ppg = gs / 8;  // packed uint32 per group

    device const uint32_t* w_row = weight + row * w_stride;
    device const bfloat*   s_row = scales + row * s_stride;
    device const bfloat*   b_row = biases + row * s_stride;

    // 32 SIMD lanes split across quantization groups
    uint num_groups = in_d / gs;
    float acc = 0.0f;

    for (uint g = lane; g < num_groups; g += 32) {
        float sc = float(s_row[g]);
        float bi = float(b_row[g]);
        device const uint32_t* gbase = w_row + g * ppg;
        uint xb = g * gs;

        // Vectorized: process 4 packed uint32 (32 elements) per iteration
        // For gs=64, ppg=8, so we do 2 iterations of 4
        uint p = 0;
        for (; p + 3 < ppg; p += 4) {
            uint4 pk4 = *((device const uint4*)(gbase + p));
            uint xo = xb + p * 8;
            uint32_t pk;

            pk = pk4.x;
            acc = fma(fma(float((pk      ) & 0xF), sc, bi), float(x_shared[xo+ 0]), acc);
            acc = fma(fma(float((pk >>  4) & 0xF), sc, bi), float(x_shared[xo+ 1]), acc);
            acc = fma(fma(float((pk >>  8) & 0xF), sc, bi), float(x_shared[xo+ 2]), acc);
            acc = fma(fma(float((pk >> 12) & 0xF), sc, bi), float(x_shared[xo+ 3]), acc);
            acc = fma(fma(float((pk >> 16) & 0xF), sc, bi), float(x_shared[xo+ 4]), acc);
            acc = fma(fma(float((pk >> 20) & 0xF), sc, bi), float(x_shared[xo+ 5]), acc);
            acc = fma(fma(float((pk >> 24) & 0xF), sc, bi), float(x_shared[xo+ 6]), acc);
            acc = fma(fma(float((pk >> 28) & 0xF), sc, bi), float(x_shared[xo+ 7]), acc);
            pk = pk4.y;
            acc = fma(fma(float((pk      ) & 0xF), sc, bi), float(x_shared[xo+ 8]), acc);
            acc = fma(fma(float((pk >>  4) & 0xF), sc, bi), float(x_shared[xo+ 9]), acc);
            acc = fma(fma(float((pk >>  8) & 0xF), sc, bi), float(x_shared[xo+10]), acc);
            acc = fma(fma(float((pk >> 12) & 0xF), sc, bi), float(x_shared[xo+11]), acc);
            acc = fma(fma(float((pk >> 16) & 0xF), sc, bi), float(x_shared[xo+12]), acc);
            acc = fma(fma(float((pk >> 20) & 0xF), sc, bi), float(x_shared[xo+13]), acc);
            acc = fma(fma(float((pk >> 24) & 0xF), sc, bi), float(x_shared[xo+14]), acc);
            acc = fma(fma(float((pk >> 28) & 0xF), sc, bi), float(x_shared[xo+15]), acc);
            pk = pk4.z;
            acc = fma(fma(float((pk      ) & 0xF), sc, bi), float(x_shared[xo+16]), acc);
            acc = fma(fma(float((pk >>  4) & 0xF), sc, bi), float(x_shared[xo+17]), acc);
            acc = fma(fma(float((pk >>  8) & 0xF), sc, bi), float(x_shared[xo+18]), acc);
            acc = fma(fma(float((pk >> 12) & 0xF), sc, bi), float(x_shared[xo+19]), acc);
            acc = fma(fma(float((pk >> 16) & 0xF), sc, bi), float(x_shared[xo+20]), acc);
            acc = fma(fma(float((pk >> 20) & 0xF), sc, bi), float(x_shared[xo+21]), acc);
            acc = fma(fma(float((pk >> 24) & 0xF), sc, bi), float(x_shared[xo+22]), acc);
            acc = fma(fma(float((pk >> 28) & 0xF), sc, bi), float(x_shared[xo+23]), acc);
            pk = pk4.w;
            acc = fma(fma(float((pk      ) & 0xF), sc, bi), float(x_shared[xo+24]), acc);
            acc = fma(fma(float((pk >>  4) & 0xF), sc, bi), float(x_shared[xo+25]), acc);
            acc = fma(fma(float((pk >>  8) & 0xF), sc, bi), float(x_shared[xo+26]), acc);
            acc = fma(fma(float((pk >> 12) & 0xF), sc, bi), float(x_shared[xo+27]), acc);
            acc = fma(fma(float((pk >> 16) & 0xF), sc, bi), float(x_shared[xo+28]), acc);
            acc = fma(fma(float((pk >> 20) & 0xF), sc, bi), float(x_shared[xo+29]), acc);
            acc = fma(fma(float((pk >> 24) & 0xF), sc, bi), float(x_shared[xo+30]), acc);
            acc = fma(fma(float((pk >> 28) & 0xF), sc, bi), float(x_shared[xo+31]), acc);
        }
        // Scalar remainder
        for (; p < ppg; p++) {
            uint32_t packed = gbase[p];
            uint xo = xb + p * 8;
            acc = fma(fma(float((packed      ) & 0xF), sc, bi), float(x_shared[xo+0]), acc);
            acc = fma(fma(float((packed >>  4) & 0xF), sc, bi), float(x_shared[xo+1]), acc);
            acc = fma(fma(float((packed >>  8) & 0xF), sc, bi), float(x_shared[xo+2]), acc);
            acc = fma(fma(float((packed >> 12) & 0xF), sc, bi), float(x_shared[xo+3]), acc);
            acc = fma(fma(float((packed >> 16) & 0xF), sc, bi), float(x_shared[xo+4]), acc);
            acc = fma(fma(float((packed >> 20) & 0xF), sc, bi), float(x_shared[xo+5]), acc);
            acc = fma(fma(float((packed >> 24) & 0xF), sc, bi), float(x_shared[xo+6]), acc);
            acc = fma(fma(float((packed >> 28) & 0xF), sc, bi), float(x_shared[xo+7]), acc);
        }
    }

    // SIMD group reduction — all 32 lanes sum in one hardware cycle
    float total = simd_sum(acc);
    if (lane == 0) {
        output[row] = half(total);
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: matvec_4bit_f32out — SIMD group reduction with shared memory
//
// Same as matvec_4bit but outputs float32 (for logits / router).
// 256 threads = 8 SIMD groups per threadgroup, each handling one row.
//
// Dispatch: threadgroups=(ceil(out_dim/8), 1, 1), threadsPerThreadgroup=(256, 1, 1)
// ---------------------------------------------------------------------------
kernel void matvec_4bit_f32out(
    device const half*     x_input  [[buffer(0)]],
    device const uint32_t* weight   [[buffer(1)]],
    device const bfloat*   scales   [[buffer(2)]],
    device const bfloat*   biases   [[buffer(3)]],
    device       float*    output   [[buffer(4)]],
    device const int32_t&  out_dim  [[buffer(5)]],
    device const int32_t&  in_dim   [[buffer(6)]],
    device const int32_t&  grp_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    uint out_d = uint(out_dim);
    uint in_d  = uint(in_dim);
    uint gs    = uint(grp_size);

    threadgroup half x_shared[8192];
    for (uint i = tid; i < in_d; i += 256) {
        x_shared[i] = x_input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = tgid * 8 + sg_idx;
    if (row >= out_d) return;

    uint w_stride = in_d / 8;
    uint s_stride = in_d / gs;
    uint ppg = gs / 8;

    device const uint32_t* w_row = weight + row * w_stride;
    device const bfloat*   s_row = scales + row * s_stride;
    device const bfloat*   b_row = biases + row * s_stride;

    uint num_groups = in_d / gs;
    float acc = 0.0f;

    for (uint g = lane; g < num_groups; g += 32) {
        float sc = float(s_row[g]);
        float bi = float(b_row[g]);
        device const uint32_t* gbase = w_row + g * ppg;
        uint xb = g * gs;

        uint p = 0;
        for (; p + 3 < ppg; p += 4) {
            uint4 pk4 = *((device const uint4*)(gbase + p));
            uint xo = xb + p * 8;
            uint32_t pk;

            pk = pk4.x;
            acc = fma(fma(float((pk      ) & 0xF), sc, bi), float(x_shared[xo+ 0]), acc);
            acc = fma(fma(float((pk >>  4) & 0xF), sc, bi), float(x_shared[xo+ 1]), acc);
            acc = fma(fma(float((pk >>  8) & 0xF), sc, bi), float(x_shared[xo+ 2]), acc);
            acc = fma(fma(float((pk >> 12) & 0xF), sc, bi), float(x_shared[xo+ 3]), acc);
            acc = fma(fma(float((pk >> 16) & 0xF), sc, bi), float(x_shared[xo+ 4]), acc);
            acc = fma(fma(float((pk >> 20) & 0xF), sc, bi), float(x_shared[xo+ 5]), acc);
            acc = fma(fma(float((pk >> 24) & 0xF), sc, bi), float(x_shared[xo+ 6]), acc);
            acc = fma(fma(float((pk >> 28) & 0xF), sc, bi), float(x_shared[xo+ 7]), acc);
            pk = pk4.y;
            acc = fma(fma(float((pk      ) & 0xF), sc, bi), float(x_shared[xo+ 8]), acc);
            acc = fma(fma(float((pk >>  4) & 0xF), sc, bi), float(x_shared[xo+ 9]), acc);
            acc = fma(fma(float((pk >>  8) & 0xF), sc, bi), float(x_shared[xo+10]), acc);
            acc = fma(fma(float((pk >> 12) & 0xF), sc, bi), float(x_shared[xo+11]), acc);
            acc = fma(fma(float((pk >> 16) & 0xF), sc, bi), float(x_shared[xo+12]), acc);
            acc = fma(fma(float((pk >> 20) & 0xF), sc, bi), float(x_shared[xo+13]), acc);
            acc = fma(fma(float((pk >> 24) & 0xF), sc, bi), float(x_shared[xo+14]), acc);
            acc = fma(fma(float((pk >> 28) & 0xF), sc, bi), float(x_shared[xo+15]), acc);
            pk = pk4.z;
            acc = fma(fma(float((pk      ) & 0xF), sc, bi), float(x_shared[xo+16]), acc);
            acc = fma(fma(float((pk >>  4) & 0xF), sc, bi), float(x_shared[xo+17]), acc);
            acc = fma(fma(float((pk >>  8) & 0xF), sc, bi), float(x_shared[xo+18]), acc);
            acc = fma(fma(float((pk >> 12) & 0xF), sc, bi), float(x_shared[xo+19]), acc);
            acc = fma(fma(float((pk >> 16) & 0xF), sc, bi), float(x_shared[xo+20]), acc);
            acc = fma(fma(float((pk >> 20) & 0xF), sc, bi), float(x_shared[xo+21]), acc);
            acc = fma(fma(float((pk >> 24) & 0xF), sc, bi), float(x_shared[xo+22]), acc);
            acc = fma(fma(float((pk >> 28) & 0xF), sc, bi), float(x_shared[xo+23]), acc);
            pk = pk4.w;
            acc = fma(fma(float((pk      ) & 0xF), sc, bi), float(x_shared[xo+24]), acc);
            acc = fma(fma(float((pk >>  4) & 0xF), sc, bi), float(x_shared[xo+25]), acc);
            acc = fma(fma(float((pk >>  8) & 0xF), sc, bi), float(x_shared[xo+26]), acc);
            acc = fma(fma(float((pk >> 12) & 0xF), sc, bi), float(x_shared[xo+27]), acc);
            acc = fma(fma(float((pk >> 16) & 0xF), sc, bi), float(x_shared[xo+28]), acc);
            acc = fma(fma(float((pk >> 20) & 0xF), sc, bi), float(x_shared[xo+29]), acc);
            acc = fma(fma(float((pk >> 24) & 0xF), sc, bi), float(x_shared[xo+30]), acc);
            acc = fma(fma(float((pk >> 28) & 0xF), sc, bi), float(x_shared[xo+31]), acc);
        }
        for (; p < ppg; p++) {
            uint32_t packed = gbase[p];
            uint xo = xb + p * 8;
            acc = fma(fma(float((packed      ) & 0xF), sc, bi), float(x_shared[xo+0]), acc);
            acc = fma(fma(float((packed >>  4) & 0xF), sc, bi), float(x_shared[xo+1]), acc);
            acc = fma(fma(float((packed >>  8) & 0xF), sc, bi), float(x_shared[xo+2]), acc);
            acc = fma(fma(float((packed >> 12) & 0xF), sc, bi), float(x_shared[xo+3]), acc);
            acc = fma(fma(float((packed >> 16) & 0xF), sc, bi), float(x_shared[xo+4]), acc);
            acc = fma(fma(float((packed >> 20) & 0xF), sc, bi), float(x_shared[xo+5]), acc);
            acc = fma(fma(float((packed >> 24) & 0xF), sc, bi), float(x_shared[xo+6]), acc);
            acc = fma(fma(float((packed >> 28) & 0xF), sc, bi), float(x_shared[xo+7]), acc);
        }
    }

    float total = simd_sum(acc);
    if (lane == 0) {
        output[row] = total;
    }
}

// ---------------------------------------------------------------------------
// Kernel 3b: rmsnorm_matvec_4bit — SIMD group reduction with shared memory
//
// Fused RMSNorm + matvec_4bit: eliminates one dispatch + barrier per layer.
// Phase 1: All 256 threads cooperatively compute RMSNorm using SIMD + shared
//          mem cross-group reduction, apply norm to x_shared.
//          First threadgroup writes normed_out for subsequent projections.
// Phase 2: Each SIMD group (32 lanes) handles one row, reads normed x_shared.
//
// Dispatch: threadgroups=(ceil(out_dim/8), 1, 1), threadsPerThreadgroup=(256, 1, 1)
// ---------------------------------------------------------------------------
kernel void rmsnorm_matvec_4bit(
    device const half*     x_input    [[buffer(0)]],  // half[in_dim] (pre-norm)
    device const bfloat*   norm_wt    [[buffer(1)]],  // bfloat16[in_dim] RMSNorm weight
    device const uint32_t* weight     [[buffer(2)]],  // (out_dim, in_dim/8) uint32
    device const bfloat*   scales     [[buffer(3)]],  // (out_dim, in_dim/gs) bf16
    device const bfloat*   biases     [[buffer(4)]],  // (out_dim, in_dim/gs) bf16
    device       half*     output     [[buffer(5)]],  // half[out_dim]
    device       half*     normed_out [[buffer(6)]],  // half[in_dim] — normed x for other projections
    device const int32_t&  out_dim    [[buffer(7)]],
    device const int32_t&  in_dim     [[buffer(8)]],
    device const int32_t&  grp_size   [[buffer(9)]],
    device const float&    eps        [[buffer(10)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    uint out_d = uint(out_dim);
    uint in_d  = uint(in_dim);
    uint gs    = uint(grp_size);

    // Phase 1: RMSNorm — all 256 threads cooperate
    threadgroup half x_shared[8192];

    // Load x and compute partial sum-of-squares
    float local_sq = 0.0f;
    for (uint i = tid; i < in_d; i += 256) {
        float v = float(x_input[i]);
        local_sq += v * v;
        x_shared[i] = x_input[i];
    }

    // SIMD reduction within each of 8 SIMD groups, then cross-group via shared
    float sg_sum = simd_sum(local_sq);
    threadgroup float sg_partial[8];
    if (lane == 0) {
        sg_partial[sg_idx] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float norm_scale_shared;
    if (tid == 0) {
        float total_sq = 0.0f;
        for (uint s = 0; s < 8; s++) total_sq += sg_partial[s];
        norm_scale_shared = 1.0f / sqrt(total_sq / float(in_d) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float norm_sc = norm_scale_shared;

    // Apply norm to shared memory + write normed output (first threadgroup only)
    for (uint i = tid; i < in_d; i += 256) {
        float normed = float(x_shared[i]) * norm_sc * float(norm_wt[i]);
        x_shared[i] = half(normed);
    }
    if (tgid == 0) {
        for (uint i = tid; i < in_d; i += 256) {
            normed_out[i] = x_shared[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: SIMD dot product — each SIMD group handles one row
    uint row = tgid * 8 + sg_idx;
    if (row >= out_d) return;

    uint w_stride = in_d / 8;
    uint s_stride = in_d / gs;
    uint ppg = gs / 8;

    device const uint32_t* w_row = weight + row * w_stride;
    device const bfloat*   s_row = scales + row * s_stride;
    device const bfloat*   b_row = biases + row * s_stride;

    // 32 SIMD lanes split across quantization groups
    uint num_groups = in_d / gs;
    float acc = 0.0f;

    for (uint g = lane; g < num_groups; g += 32) {
        float sc = float(s_row[g]);
        float bi = float(b_row[g]);
        device const uint32_t* gbase = w_row + g * ppg;
        uint xb = g * gs;

        uint p = 0;
        for (; p + 3 < ppg; p += 4) {
            uint4 pk4 = *((device const uint4*)(gbase + p));
            uint xo = xb + p * 8;
            uint32_t pk;

            pk = pk4.x;
            acc = fma(fma(float((pk      ) & 0xF), sc, bi), float(x_shared[xo+ 0]), acc);
            acc = fma(fma(float((pk >>  4) & 0xF), sc, bi), float(x_shared[xo+ 1]), acc);
            acc = fma(fma(float((pk >>  8) & 0xF), sc, bi), float(x_shared[xo+ 2]), acc);
            acc = fma(fma(float((pk >> 12) & 0xF), sc, bi), float(x_shared[xo+ 3]), acc);
            acc = fma(fma(float((pk >> 16) & 0xF), sc, bi), float(x_shared[xo+ 4]), acc);
            acc = fma(fma(float((pk >> 20) & 0xF), sc, bi), float(x_shared[xo+ 5]), acc);
            acc = fma(fma(float((pk >> 24) & 0xF), sc, bi), float(x_shared[xo+ 6]), acc);
            acc = fma(fma(float((pk >> 28) & 0xF), sc, bi), float(x_shared[xo+ 7]), acc);
            pk = pk4.y;
            acc = fma(fma(float((pk      ) & 0xF), sc, bi), float(x_shared[xo+ 8]), acc);
            acc = fma(fma(float((pk >>  4) & 0xF), sc, bi), float(x_shared[xo+ 9]), acc);
            acc = fma(fma(float((pk >>  8) & 0xF), sc, bi), float(x_shared[xo+10]), acc);
            acc = fma(fma(float((pk >> 12) & 0xF), sc, bi), float(x_shared[xo+11]), acc);
            acc = fma(fma(float((pk >> 16) & 0xF), sc, bi), float(x_shared[xo+12]), acc);
            acc = fma(fma(float((pk >> 20) & 0xF), sc, bi), float(x_shared[xo+13]), acc);
            acc = fma(fma(float((pk >> 24) & 0xF), sc, bi), float(x_shared[xo+14]), acc);
            acc = fma(fma(float((pk >> 28) & 0xF), sc, bi), float(x_shared[xo+15]), acc);
            pk = pk4.z;
            acc = fma(fma(float((pk      ) & 0xF), sc, bi), float(x_shared[xo+16]), acc);
            acc = fma(fma(float((pk >>  4) & 0xF), sc, bi), float(x_shared[xo+17]), acc);
            acc = fma(fma(float((pk >>  8) & 0xF), sc, bi), float(x_shared[xo+18]), acc);
            acc = fma(fma(float((pk >> 12) & 0xF), sc, bi), float(x_shared[xo+19]), acc);
            acc = fma(fma(float((pk >> 16) & 0xF), sc, bi), float(x_shared[xo+20]), acc);
            acc = fma(fma(float((pk >> 20) & 0xF), sc, bi), float(x_shared[xo+21]), acc);
            acc = fma(fma(float((pk >> 24) & 0xF), sc, bi), float(x_shared[xo+22]), acc);
            acc = fma(fma(float((pk >> 28) & 0xF), sc, bi), float(x_shared[xo+23]), acc);
            pk = pk4.w;
            acc = fma(fma(float((pk      ) & 0xF), sc, bi), float(x_shared[xo+24]), acc);
            acc = fma(fma(float((pk >>  4) & 0xF), sc, bi), float(x_shared[xo+25]), acc);
            acc = fma(fma(float((pk >>  8) & 0xF), sc, bi), float(x_shared[xo+26]), acc);
            acc = fma(fma(float((pk >> 12) & 0xF), sc, bi), float(x_shared[xo+27]), acc);
            acc = fma(fma(float((pk >> 16) & 0xF), sc, bi), float(x_shared[xo+28]), acc);
            acc = fma(fma(float((pk >> 20) & 0xF), sc, bi), float(x_shared[xo+29]), acc);
            acc = fma(fma(float((pk >> 24) & 0xF), sc, bi), float(x_shared[xo+30]), acc);
            acc = fma(fma(float((pk >> 28) & 0xF), sc, bi), float(x_shared[xo+31]), acc);
        }
        for (; p < ppg; p++) {
            uint32_t packed = gbase[p];
            uint xo = xb + p * 8;
            acc = fma(fma(float((packed      ) & 0xF), sc, bi), float(x_shared[xo+0]), acc);
            acc = fma(fma(float((packed >>  4) & 0xF), sc, bi), float(x_shared[xo+1]), acc);
            acc = fma(fma(float((packed >>  8) & 0xF), sc, bi), float(x_shared[xo+2]), acc);
            acc = fma(fma(float((packed >> 12) & 0xF), sc, bi), float(x_shared[xo+3]), acc);
            acc = fma(fma(float((packed >> 16) & 0xF), sc, bi), float(x_shared[xo+4]), acc);
            acc = fma(fma(float((packed >> 20) & 0xF), sc, bi), float(x_shared[xo+5]), acc);
            acc = fma(fma(float((packed >> 24) & 0xF), sc, bi), float(x_shared[xo+6]), acc);
            acc = fma(fma(float((packed >> 28) & 0xF), sc, bi), float(x_shared[xo+7]), acc);
        }
    }

    float total = simd_sum(acc);
    if (lane == 0) {
        output[row] = half(total);
    }
}

// ---------------------------------------------------------------------------
// Kernel 5: rope_kernel
//
// Apply RoPE to Q and K vectors (partial rotation, 64/256 dims).
// Only the first rope_dim elements of each head are rotated.
// The rest are left unchanged.
//
// For Qwen3.5: partial_rotary_factor=0.25, head_dim=256, so rope_dim=64
//
// Uses non-interleaved (MLX default, traditional=false) layout:
//   pair i: element[i] pairs with element[i + rope_dim/2]
//   freq_i = 1 / theta^(2i / rope_dim)
//
// Q: half[num_q_heads * head_dim] = half[4096]
// K: half[num_kv_heads * head_dim] = half[512]
// position: int (for computing angles)
//
// Grid: (num_heads * rope_dim/2, 1, 1)
// Threadgroup: (64, 1, 1)
// ---------------------------------------------------------------------------
kernel void rope_kernel(
    device       half*     data      [[buffer(0)]],  // Q or K: half[n_heads * head_dim]
    device const int32_t&  n_heads   [[buffer(1)]],
    device const int32_t&  head_dim  [[buffer(2)]],
    device const int32_t&  rope_dim  [[buffer(3)]],  // 64
    device const int32_t&  position  [[buffer(4)]],
    device const float&    theta     [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint nh = uint(n_heads);
    uint hd = uint(head_dim);
    uint rd = uint(rope_dim);
    int pos = position;

    uint half_rd = rd / 2;
    uint total_pairs = nh * half_rd;
    if (gid >= total_pairs) return;

    uint h = gid / half_rd;
    uint pair_idx = gid % half_rd;

    // Frequency for this pair
    float freq = 1.0f / pow(theta, float(2 * pair_idx) / float(rd));
    float angle = float(pos) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    // Non-interleaved layout: pair element i with element i + half_rd
    uint head_base = h * hd;
    uint idx0 = head_base + pair_idx;
    uint idx1 = head_base + pair_idx + half_rd;
    float x0 = float(data[idx0]);
    float x1 = float(data[idx1]);

    // Rotate
    data[idx0] = half(x0 * cos_a - x1 * sin_a);
    data[idx1] = half(x0 * sin_a + x1 * cos_a);
}

// ---------------------------------------------------------------------------
// Kernel 6: gqa_attention_decode
//
// Full GQA self-attention for single-token decode.
// Q: (16 heads, 256) — already projected and RoPE'd
// KV cache: (2 KV heads, max_len, 256) for both K and V
// GQA: each KV head serves 8 Q heads
//
// For each Q head:
//   score[t] = Q_h . K_kv[t] / sqrt(head_dim) for t in 0..kv_len
//   weights = softmax(scores)
//   out_h = sum(weights[t] * V_kv[t])
//
// Grid: (num_q_heads, 1, 1) = (16, 1, 1)
// Threadgroup: (256, 1, 1) — parallel over head_dim for dot products
// ---------------------------------------------------------------------------
kernel void gqa_attention_decode(
    device const half*     query      [[buffer(0)]],   // half[num_q * head_dim]
    device const half*     k_cache    [[buffer(1)]],   // half[num_kv * max_len * head_dim]
    device const half*     v_cache    [[buffer(2)]],   // half[num_kv * max_len * head_dim]
    device       half*     output     [[buffer(3)]],   // half[num_q * head_dim]
    device const int32_t&  num_q      [[buffer(4)]],   // 16
    device const int32_t&  num_kv     [[buffer(5)]],   // 2
    device const int32_t&  head_dim_p [[buffer(6)]],   // 256
    device const int32_t&  kv_len     [[buffer(7)]],   // current length
    device const int32_t&  max_len    [[buffer(8)]],   // max cache length
    uint h_idx [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tgs   [[threads_per_threadgroup]]
) {
    uint nq = uint(num_q);
    uint nkv = uint(num_kv);
    uint hd = uint(head_dim_p);
    uint kvl = uint(kv_len);
    uint ml = uint(max_len);

    if (h_idx >= nq) return;

    // GQA mapping: which KV head this Q head uses
    uint kv_idx = h_idx / (nq / nkv);

    device const half* q_head = query + h_idx * hd;
    device const half* k_head = k_cache + kv_idx * ml * hd;
    device const half* v_head = v_cache + kv_idx * ml * hd;
    device half* out_head = output + h_idx * hd;

    float inv_sqrt = 1.0f / sqrt(float(hd));

    // Thread-cooperative: each thread handles a subset of time steps
    // for computing scores, then we need per-thread softmax

    // Since kv_len can be large, we need a parallel approach:
    // 1. Each thread computes scores for a subset of time steps
    // 2. We need online softmax (numerically stable)

    // For simplicity and correctness, use a sequential approach per head
    // (each threadgroup handles one Q head)

    // Step 1: Compute all attention scores
    // scores[t] = sum_d(Q[d] * K[t][d]) * inv_sqrt
    // Use shared memory for the Q vector
    threadgroup float q_shared[256];
    for (uint d = tid; d < hd; d += tgs) {
        q_shared[d] = float(q_head[d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Online softmax + weighted V accumulation
    // Each thread accumulates partial results, then we reduce
    // Using online softmax: track running max and sum
    float max_score = -1e30f;
    float sum_exp = 0.0f;
    float acc[256]; // accumulator for weighted V (max head_dim)

    // Initialize accumulator
    for (uint d = 0; d < hd; d++) {
        acc[d] = 0.0f;
    }

    // Process time steps sequentially (single thread per head)
    // This is fine for decode (kv_len is moderate, and we have 16 heads)
    if (tid == 0) {
        for (uint t = 0; t < kvl; t++) {
            // Compute score
            float score = 0.0f;
            for (uint d = 0; d < hd; d++) {
                score += q_shared[d] * float(k_head[t * hd + d]);
            }
            score *= inv_sqrt;

            // Online softmax update
            if (score > max_score) {
                float correction = exp(max_score - score);
                sum_exp = sum_exp * correction + 1.0f;
                for (uint d = 0; d < hd; d++) {
                    acc[d] = acc[d] * correction + float(v_head[t * hd + d]);
                }
                max_score = score;
            } else {
                float w = exp(score - max_score);
                sum_exp += w;
                for (uint d = 0; d < hd; d++) {
                    acc[d] += w * float(v_head[t * hd + d]);
                }
            }
        }

        // Normalize
        float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
        for (uint d = 0; d < hd; d++) {
            out_head[d] = half(acc[d] * inv_sum);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 7: kv_cache_append
//
// Append new K and V vectors to the KV cache at the given position.
// K: half[num_kv * head_dim], V: half[num_kv * head_dim]
// Cache layout: (num_kv_heads, max_len, head_dim)
//
// Grid: (num_kv * head_dim, 1, 1)
// Threadgroup: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void kv_cache_append(
    device const half*     new_k     [[buffer(0)]],   // half[num_kv * head_dim]
    device const half*     new_v     [[buffer(1)]],   // half[num_kv * head_dim]
    device       half*     k_cache   [[buffer(2)]],   // half[num_kv * max_len * head_dim]
    device       half*     v_cache   [[buffer(3)]],   // half[num_kv * max_len * head_dim]
    device const int32_t&  num_kv    [[buffer(4)]],
    device const int32_t&  head_dim  [[buffer(5)]],
    device const int32_t&  position  [[buffer(6)]],
    device const int32_t&  max_len   [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint nkv = uint(num_kv);
    uint hd = uint(head_dim);
    uint pos = uint(position);
    uint ml = uint(max_len);

    uint total = nkv * hd;
    if (gid >= total) return;

    uint h = gid / hd;
    uint d = gid % hd;

    uint cache_idx = h * ml * hd + pos * hd + d;
    k_cache[cache_idx] = new_k[gid];
    v_cache[cache_idx] = new_v[gid];
}

// ---------------------------------------------------------------------------
// Kernel 8: rmsnorm_per_head
//
// RMSNorm applied per-head (for q_norm and k_norm in full attention).
// Input: half[num_heads * head_dim], each head normalized independently.
// Weight: bfloat16[head_dim] (shared across heads)
//
// Grid: (num_heads, 1, 1)
// Threadgroup: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void rmsnorm_per_head(
    device       half*     data    [[buffer(0)]],  // in-place: half[n_heads * head_dim]
    device const bfloat*   weight  [[buffer(1)]],  // bf16[head_dim] (or NULL)
    device const int32_t&  n_heads [[buffer(2)]],
    device const int32_t&  hd      [[buffer(3)]],
    device const float&    eps     [[buffer(4)]],
    uint h_idx [[thread_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tgs   [[threads_per_threadgroup]]
) {
    uint nh = uint(n_heads);
    uint head_dim = uint(hd);
    if (h_idx >= nh) return;

    device half* head = data + h_idx * head_dim;

    threadgroup float partial_sums[256];
    float local_sum = 0.0f;
    for (uint i = tid; i < head_dim; i += tgs) {
        float v = float(head[i]);
        local_sum += v * v;
    }
    partial_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float shared_scale;
    if (tid == 0) {
        float mean_sq = partial_sums[0] / float(head_dim);
        shared_scale = 1.0f / sqrt(mean_sq + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = shared_scale;
    for (uint i = tid; i < head_dim; i += tgs) {
        float v = float(head[i]) * scale;
        // Apply weight if provided (weight pointer check done on host)
        v *= float(weight[i]);
        head[i] = half(v);
    }
}

// ---------------------------------------------------------------------------
// Kernel 9: rmsnorm_per_head_no_weight
//
// RMSNorm per-head without weight (for Q/K norm in GatedDeltaNet).
// ---------------------------------------------------------------------------
kernel void rmsnorm_per_head_no_weight(
    device       half*     data    [[buffer(0)]],
    device const int32_t&  n_heads [[buffer(1)]],
    device const int32_t&  hd      [[buffer(2)]],
    device const float&    eps     [[buffer(3)]],
    uint h_idx [[thread_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tgs   [[threads_per_threadgroup]]
) {
    uint nh = uint(n_heads);
    uint head_dim = uint(hd);
    if (h_idx >= nh) return;

    device half* head = data + h_idx * head_dim;

    threadgroup float partial_sums[256];
    float local_sum = 0.0f;
    for (uint i = tid; i < head_dim; i += tgs) {
        float v = float(head[i]);
        local_sum += v * v;
    }
    partial_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float shared_scale;
    if (tid == 0) {
        float mean_sq = partial_sums[0] / float(head_dim);
        shared_scale = 1.0f / sqrt(mean_sq + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = shared_scale;
    for (uint i = tid; i < head_dim; i += tgs) {
        head[i] = half(float(head[i]) * scale);
    }
}

// ---------------------------------------------------------------------------
// Kernel 10: sigmoid_gate_mul
//
// output = o_proj(attn_out * sigmoid(gate))
// Since o_proj is a separate matvec, this kernel just does:
//   output[i] = input[i] * sigmoid(gate[i])
//
// Grid: (dim, 1, 1)
// Threadgroup: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void sigmoid_gate_mul(
    device const half*     input   [[buffer(0)]],  // half[dim]
    device const half*     gate    [[buffer(1)]],  // half[dim]
    device       half*     output  [[buffer(2)]],  // half[dim]
    device const int32_t&  dim     [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(dim)) return;
    float v = float(input[gid]);
    float g = float(gate[gid]);
    float sig_g = 1.0f / (1.0f + exp(-g));
    output[gid] = half(v * sig_g);
}

// ---------------------------------------------------------------------------
// Kernel 11: residual_add
//
// output = a + b (element-wise, half precision)
//
// Grid: (dim, 1, 1)
// Threadgroup: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void residual_add(
    device const half*     a       [[buffer(0)]],
    device const half*     b       [[buffer(1)]],
    device       half*     output  [[buffer(2)]],
    device const int32_t&  dim     [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(dim)) return;
    output[gid] = half(float(a[gid]) + float(b[gid]));
}

// ---------------------------------------------------------------------------
// Kernel 12: depthwise_conv1d_silu
//
// Depthwise 1D convolution + SiLU activation for GatedDeltaNet.
// For single-token decode with causal conv: the "input" is the conv buffer
// (kernel_size values per channel) and we compute one output per channel.
//
// conv_buf: half[conv_dim * kernel_size] — ring buffer of recent inputs
// weights:  half[conv_dim * kernel_size] — conv1d weights (depthwise)
// output:   half[conv_dim] — silu(conv_output)
//
// Conv buffer layout: [channel][kernel_pos]
// The newest value is at position (write_pos - 1) % kernel_size.
//
// Grid: (conv_dim, 1, 1)
// Threadgroup: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void depthwise_conv1d_silu(
    device const half*     conv_buf [[buffer(0)]],  // half[conv_dim * kernel_size]
    device const bfloat*   weights  [[buffer(1)]],  // bfloat16[conv_dim * kernel_size]
    device       half*     output   [[buffer(2)]],  // half[conv_dim]
    device const int32_t&  conv_dim [[buffer(3)]],
    device const int32_t&  ksize    [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint cd = uint(conv_dim);
    uint ks = uint(ksize);
    if (gid >= cd) return;

    // Dot product over kernel window
    float acc = 0.0f;
    for (uint k = 0; k < ks; k++) {
        acc += float(conv_buf[gid * ks + k]) * float(weights[gid * ks + k]);
    }

    // SiLU activation
    float result = acc / (1.0f + exp(-acc));
    output[gid] = half(result);
}

// ---------------------------------------------------------------------------
// Kernel 13: gated_delta_state_update
//
// GatedDeltaNet recurrent state update for single-token decode.
//
// For each value head hv (0..31):
//   kv_head = hv / (num_v_heads / num_k_heads)  // maps V heads to K heads
//   g = exp(-exp(A_log[hv]) * softplus(a[hv] + dt_bias[hv]))  // decay
//   beta_val = sigmoid(b[hv])
//
//   // State: [Dv, Dk] per head
//   state[hv] *= g           // decay all elements
//   kv_mem = state[hv] @ k[kv_head]   // [Dv] = [Dv, Dk] @ [Dk]
//   delta = (v[hv] - kv_mem) * beta_val  // [Dv]
//   state[hv] += outer(delta, k[kv_head])  // rank-1 update
//   output[hv] = state[hv] @ q[kv_head]   // [Dv] = [Dv, Dk] @ [Dk]
//
// State: float[num_v_heads * val_dim * key_dim] = float[32 * 128 * 128]
// Q: half[num_k_heads * key_dim] = half[16 * 128]
// K: half[num_k_heads * key_dim] = half[16 * 128]
// V: half[num_v_heads * val_dim] = half[32 * 128]
// a: half[num_v_heads] = half[32]
// b: half[num_v_heads] = half[32]
// A_log: float[num_v_heads] = float[32]
// dt_bias: float[num_v_heads] = float[32]
// output: half[num_v_heads * val_dim] = half[32 * 128]
//
// Grid: (num_v_heads, val_dim, 1)
// Threadgroup: (1, 1, 1) — each thread handles one (hv, dv) pair
// ---------------------------------------------------------------------------
kernel void gated_delta_state_update(
    device const half*     q_vec     [[buffer(0)]],   // half[num_k * key_dim]
    device const half*     k_vec     [[buffer(1)]],   // half[num_k * key_dim]
    device const half*     v_vec     [[buffer(2)]],   // half[num_v * val_dim]
    device const half*     a_vec     [[buffer(3)]],   // half[num_v]
    device const half*     b_vec     [[buffer(4)]],   // half[num_v]
    device const float*    A_log     [[buffer(5)]],   // float[num_v] (not cast, stays float32)
    device const float*    dt_bias   [[buffer(6)]],   // float[num_v] (converted to float32 in Python)
    device       float*    state     [[buffer(7)]],   // float[num_v * val_dim * key_dim]
    device       half*     output    [[buffer(8)]],   // half[num_v * val_dim]
    device const int32_t&  num_k     [[buffer(9)]],
    device const int32_t&  num_v     [[buffer(10)]],
    device const int32_t&  key_dim   [[buffer(11)]],
    device const int32_t&  val_dim   [[buffer(12)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint hv = gid.x;
    uint dv = gid.y;
    uint nk = uint(num_k);
    uint nv = uint(num_v);
    uint kd = uint(key_dim);
    uint vd = uint(val_dim);

    if (hv >= nv || dv >= vd) return;

    // Map V head to K head
    uint hk = hv / (nv / nk);

    // Compute decay g = exp(-exp(A_log) * softplus(a + dt_bias))
    float a_val = float(a_vec[hv]);
    float dt_b = float(dt_bias[hv]);
    float A_l = A_log[hv];
    float softplus_val = log(1.0f + exp(a_val + dt_b));
    float g = exp(-exp(A_l) * softplus_val);

    // Compute beta = sigmoid(b)
    float b_val = float(b_vec[hv]);
    float beta = 1.0f / (1.0f + exp(-b_val));

    // State pointer for this head and dv row
    // State layout: [num_v, val_dim, key_dim]
    device float* s_row = state + (hv * vd + dv) * kd;

    // K pointer for this K head
    device const half* k_head = k_vec + hk * kd;
    // Q pointer for this K head
    device const half* q_head = q_vec + hk * kd;

    // V value for this (hv, dv)
    float v_val = float(v_vec[hv * vd + dv]);

    // Step 1: Decay state and compute kv_mem = dot(state_row, k)
    float kv_mem = 0.0f;
    for (uint dk = 0; dk < kd; dk++) {
        s_row[dk] *= g;
        kv_mem += s_row[dk] * float(k_head[dk]);
    }

    // Step 2: delta = (v - kv_mem) * beta
    float delta = (v_val - kv_mem) * beta;

    // Step 3: Rank-1 update: state += delta * k^T, and compute output = dot(state, q)
    float out_val = 0.0f;
    for (uint dk = 0; dk < kd; dk++) {
        s_row[dk] += delta * float(k_head[dk]);
        out_val += s_row[dk] * float(q_head[dk]);
    }

    output[hv * vd + dv] = half(out_val);
}

// ---------------------------------------------------------------------------
// Kernel 14: rmsnorm_gated_kernel
//
// RMSNormGated: x = rms_norm(hidden_states, weight) then silu(gate) * x
// This is the norm used at the output of GatedDeltaNet linear attention.
//
// hidden_states: half[num_v_heads * val_dim] = half[4096]
// gate: half[num_v_heads * val_dim] = half[4096]
// weight: bfloat16[val_dim] = bfloat16[128] (shared across heads)
// output: half[num_v_heads * val_dim] = half[4096]
//
// We apply RMSNorm per-head, then silu(gate) * normalized
// Actually, RMSNormGated does rms_norm on the whole head and applies gate.
// The implementation: x = rms_norm(x, weight, eps) then silu(gate)*x
//
// Grid: (num_v_heads, 1, 1) — one threadgroup per head
// Threadgroup: (128, 1, 1)
// ---------------------------------------------------------------------------
kernel void rmsnorm_gated_kernel(
    device const half*     hidden    [[buffer(0)]],  // half[num_heads * head_dim]
    device const half*     gate      [[buffer(1)]],  // half[num_heads * head_dim]
    device const bfloat*   weight    [[buffer(2)]],  // bfloat16[head_dim]
    device       half*     output    [[buffer(3)]],  // half[num_heads * head_dim]
    device const int32_t&  n_heads   [[buffer(4)]],
    device const int32_t&  head_dim  [[buffer(5)]],
    device const float&    eps       [[buffer(6)]],
    uint h_idx [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tgs   [[threads_per_threadgroup]]
) {
    uint nh = uint(n_heads);
    uint hd = uint(head_dim);
    if (h_idx >= nh) return;

    device const half* h_head = hidden + h_idx * hd;
    device const half* g_head = gate + h_idx * hd;
    device half* o_head = output + h_idx * hd;

    // RMSNorm
    threadgroup float partial_sums[128];
    float local_sum = 0.0f;
    for (uint i = tid; i < hd; i += tgs) {
        float v = float(h_head[i]);
        local_sum += v * v;
    }
    partial_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float shared_scale;
    if (tid == 0) {
        float mean_sq = partial_sums[0] / float(hd);
        shared_scale = 1.0f / sqrt(mean_sq + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = shared_scale;

    // Apply: silu(gate) * rms_norm(x) * weight
    // The _precise_swiglu does: silu(gate.float32) * x.float32
    for (uint i = tid; i < hd; i += tgs) {
        float normed = float(h_head[i]) * scale * float(weight[i]);
        float g = float(g_head[i]);
        float silu_g = g / (1.0f + exp(-g));
        o_head[i] = half(silu_g * normed);
    }
}

// ---------------------------------------------------------------------------
// Kernel 15: scale_vector
//
// Multiply vector by scalar: output[i] = input[i] * scale
// Used for Q/K scaling in GatedDeltaNet: q *= inv_scale^2, k *= inv_scale
//
// Grid: (dim, 1, 1)
// Threadgroup: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void scale_vector(
    device       half*     data    [[buffer(0)]],
    device const float&    scale   [[buffer(1)]],
    device const int32_t&  dim     [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(dim)) return;
    data[gid] = half(float(data[gid]) * scale);
}

// ---------------------------------------------------------------------------
// Kernel 16: conv1d_buffer_update
//
// Update the conv1d ring buffer: shift left and append new values.
// Buffer layout: [conv_dim, kernel_size]
// Shifts: buf[ch][0..ks-2] = buf[ch][1..ks-1], buf[ch][ks-1] = new[ch]
//
// Grid: (conv_dim, 1, 1)
// Threadgroup: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void conv1d_buffer_update(
    device       half*     conv_buf [[buffer(0)]],  // half[conv_dim * kernel_size]
    device const half*     new_val  [[buffer(1)]],  // half[conv_dim]
    device const int32_t&  conv_dim [[buffer(2)]],
    device const int32_t&  ksize    [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint cd = uint(conv_dim);
    uint ks = uint(ksize);
    if (gid >= cd) return;

    // Shift left
    for (uint k = 0; k < ks - 1; k++) {
        conv_buf[gid * ks + k] = conv_buf[gid * ks + k + 1];
    }
    // Append new value
    conv_buf[gid * ks + (ks - 1)] = new_val[gid];
}

// ---------------------------------------------------------------------------
// Kernel 17: split_qkv
//
// Split a contiguous QKV buffer into separate Q, K, V buffers.
// Used for GatedDeltaNet linear attention where in_proj_qkv produces
// a single [q_dim + k_dim + v_dim] output that needs splitting.
//
// qkv: half[q_dim + k_dim + v_dim]
// q: half[q_dim], k: half[k_dim], v: half[v_dim]
//
// Grid: (q_dim + k_dim + v_dim, 1, 1)
// Threadgroup: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void split_qkv(
    device const half*     qkv    [[buffer(0)]],
    device       half*     q      [[buffer(1)]],
    device       half*     k      [[buffer(2)]],
    device       half*     v      [[buffer(3)]],
    device const int32_t&  q_dim  [[buffer(4)]],
    device const int32_t&  k_dim  [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint qd = uint(q_dim);
    uint kd = uint(k_dim);
    uint total = qd + kd;  // v starts after q+k

    if (gid < qd) {
        q[gid] = qkv[gid];
    } else if (gid < total) {
        k[gid - qd] = qkv[gid];
    } else {
        v[gid - total] = qkv[gid];
    }
}

// ---------------------------------------------------------------------------
// Kernel 18: split_q_gate
//
// Split interleaved Q+gate projection into separate Q and gate buffers.
// Full attention q_proj output is [num_heads * head_dim * 2] with layout:
//   [head0_q(head_dim), head0_gate(head_dim), head1_q(head_dim), head1_gate(head_dim), ...]
//
// proj: half[num_heads * head_dim * 2]
// q: half[num_heads * head_dim]
// gate: half[num_heads * head_dim]
//
// Grid: (num_heads * head_dim, 1, 1)
// Threadgroup: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void split_q_gate(
    device const half*     proj      [[buffer(0)]],
    device       half*     q         [[buffer(1)]],
    device       half*     gate      [[buffer(2)]],
    device const int32_t&  num_heads [[buffer(3)]],
    device const int32_t&  head_dim  [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint nh = uint(num_heads);
    uint hd = uint(head_dim);
    uint total = nh * hd;
    if (gid >= total) return;

    uint h = gid / hd;
    uint d = gid % hd;

    // Source layout: [head][q(head_dim), gate(head_dim)]
    uint src_base = h * (hd * 2);
    q[gid]    = proj[src_base + d];
    gate[gid] = proj[src_base + hd + d];
}
