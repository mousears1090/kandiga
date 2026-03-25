// moe_block.metal — Metal compute kernels for the complete MoE block
//
// Handles everything EXCEPT the per-expert MLP (which uses expert_mlp.metal):
//   1. RMSNorm (post-attention layernorm)
//   2. Router gate matmul (4-bit quantized, 2048 -> 256 logits)
//   3. Shared expert MLP (gate_proj + up_proj + SwiGLU + down_proj, all 4-bit)
//   4. Shared expert gate (4-bit quantized linear, 2048 -> 1 + sigmoid)
//   5. Blend + residual (weighted expert sum + shared + residual)
//
// All quantized weights use 4-bit packed uint32 with bfloat16 scales/biases,
// group_size=64 (matching the expert MLP kernels in expert_mlp.metal).

#include <metal_stdlib>
using namespace metal;







// ---------------------------------------------------------------------------
// Shared dequantized dot product — 4-bit packed weights x half input
// Identical logic to expert_mlp.metal's dequant_dot_half
// ---------------------------------------------------------------------------
inline float dequant_dot_shared(
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

// Variant for float input (used by shared expert down_proj)
inline float dequant_dot_float_shared(
    device const uint32_t* weight_row,
    device const float*    x_row,
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

            float x0 = x_row[x_base + 0];
            float x1 = x_row[x_base + 1];
            float x2 = x_row[x_base + 2];
            float x3 = x_row[x_base + 3];
            float x4 = x_row[x_base + 4];
            float x5 = x_row[x_base + 5];
            float x6 = x_row[x_base + 6];
            float x7 = x_row[x_base + 7];

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
// MoE block parameter buffer — passed from C host code
// ---------------------------------------------------------------------------
struct MoeBlockParams {
    int32_t hidden_size;      // 2048
    int32_t expert_dim;       // 512
    int32_t num_experts;      // 256 (total router outputs)
    int32_t num_experts_per_tok; // 8 (top-K)
    int32_t group_size;       // 64
    float   rms_norm_eps;     // 1e-6
};

// ---------------------------------------------------------------------------
// Kernel 1: rmsnorm_kernel
//
// RMSNorm: output[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
//
// Uses threadgroup shared memory for parallel reduction of sum-of-squares.
// Weight is bfloat16 (model norm weights stored as bf16 in quantized model).
//
// Grid: (1, 1, 1) — single threadgroup
// Threadgroup: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void rmsnorm_kernel(
    device const half*     x       [[buffer(0)]],  // input float16[hidden_size]
    device const bfloat*   weight  [[buffer(1)]],  // norm weight bfloat16[hidden_size]
    device       half*     output  [[buffer(2)]],  // output float16[hidden_size]
    device const MoeBlockParams& params [[buffer(3)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]]
) {
    uint hidden_size = uint(params.hidden_size);
    float eps = params.rms_norm_eps;

    // Phase 1: Compute partial sum of squares
    threadgroup float partial_sums[256];
    float local_sum = 0.0f;
    for (uint i = tid; i < hidden_size; i += tgs) {
        float v = float(x[i]);
        local_sum += v * v;
    }
    partial_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Parallel reduction
    for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 3: Compute scale factor
    threadgroup float shared_scale;
    if (tid == 0) {
        float mean_sq = partial_sums[0] / float(hidden_size);
        shared_scale = 1.0f / sqrt(mean_sq + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = shared_scale;

    // Phase 4: Apply normalization and weight
    for (uint i = tid; i < hidden_size; i += tgs) {
        float v = float(x[i]) * scale * float(weight[i]);
        output[i] = half(v);
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: router_matmul_kernel
//
// Dequantized matrix-vector multiply for the router gate.
// All weights are 4-bit quantized with group_size=64.
// Router weight shape: (256, 256) uint32 = 256 rows x (2048/8) packed cols
// Router scales/biases: (256, 32) bfloat16
//
// Input: normed_x float16[2048] (loaded into shared memory)
// Output: float32[256] raw logits
//
// Grid: (256, 1, 1) — one thread per output row
// Threadgroup: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void router_matmul_kernel(
    device const half*     normed_x   [[buffer(0)]],  // float16[2048]
    device const uint32_t* router_w   [[buffer(1)]],  // (256, 256) uint32
    device const bfloat*   router_s   [[buffer(2)]],  // (256, 32)  bf16
    device const bfloat*   router_b   [[buffer(3)]],  // (256, 32)  bf16
    device       float*    logits     [[buffer(4)]],  // float32[256]
    device const MoeBlockParams& params [[buffer(5)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]]
) {
    uint hidden_size = uint(params.hidden_size);
    uint num_experts = uint(params.num_experts);
    uint group_size = uint(params.group_size);

    // Load input into shared memory
    threadgroup half x_shared[2048];
    for (uint i = tid; i < hidden_size; i += tgs) {
        x_shared[i] = normed_x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread computes one output row
    uint row = tid;
    if (row >= num_experts) return;

    uint weight_row_stride = hidden_size / 8;   // 256
    uint sb_row_stride = hidden_size / group_size; // 32

    device const uint32_t* w_row = router_w + row * weight_row_stride;
    device const bfloat*   s_row = router_s + row * sb_row_stride;
    device const bfloat*   b_row = router_b + row * sb_row_stride;

    logits[row] = dequant_dot_shared(w_row, x_shared, s_row, b_row, hidden_size, group_size);
}

// ---------------------------------------------------------------------------
// Kernel 3: shared_expert_up_gate_kernel
//
// Shared expert MLP: gate_proj + up_proj + SwiGLU
// gate_proj: (512, 256) uint32 = 512 x 2048, 4-bit quantized
// up_proj:   (512, 256) uint32 = 512 x 2048, 4-bit quantized
// Output: float[512] = silu(gate) * up
//
// Grid: (512, 1, 1)
// Threadgroup: (64, 1, 1)
// ---------------------------------------------------------------------------
kernel void shared_expert_up_gate_kernel(
    device const half*     normed_x    [[buffer(0)]],  // float16[2048]
    device const uint32_t* gate_w      [[buffer(1)]],  // (512, 256) uint32
    device const bfloat*   gate_s      [[buffer(2)]],  // (512, 32) bf16
    device const bfloat*   gate_b      [[buffer(3)]],  // (512, 32) bf16
    device const uint32_t* up_w        [[buffer(4)]],  // (512, 256) uint32
    device const bfloat*   up_s        [[buffer(5)]],  // (512, 32) bf16
    device const bfloat*   up_b        [[buffer(6)]],  // (512, 32) bf16
    device       float*    activated   [[buffer(7)]],  // float[512]
    device const MoeBlockParams& params [[buffer(8)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint gid  [[thread_position_in_grid]],
    uint tgs  [[threads_per_threadgroup]]
) {
    uint hidden_size = uint(params.hidden_size);
    uint expert_dim = uint(params.expert_dim);
    uint group_size = uint(params.group_size);

    // Load input into shared memory
    threadgroup half x_shared[2048];
    for (uint i = tid; i < hidden_size; i += tgs) {
        x_shared[i] = normed_x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = gid;
    if (row >= expert_dim) return;

    uint weight_row_stride = hidden_size / 8;   // 256
    uint sb_row_stride = hidden_size / group_size; // 32

    // gate_proj
    float gate_val = dequant_dot_shared(
        gate_w + row * weight_row_stride, x_shared,
        gate_s + row * sb_row_stride, gate_b + row * sb_row_stride,
        hidden_size, group_size
    );

    // up_proj
    float up_val = dequant_dot_shared(
        up_w + row * weight_row_stride, x_shared,
        up_s + row * sb_row_stride, up_b + row * sb_row_stride,
        hidden_size, group_size
    );

    // SwiGLU: silu(gate) * up
    float silu_gate = gate_val / (1.0f + exp(-gate_val));
    activated[row] = silu_gate * up_val;
}

// ---------------------------------------------------------------------------
// Kernel 4: shared_expert_down_kernel
//
// Shared expert down_proj: (2048, 64) uint32 = 2048 x 512, 4-bit quantized
// Input: float[512] (activated from SwiGLU)
// Output: float16[2048]
//
// Grid: (2048, 1, 1)
// Threadgroup: (64, 1, 1)
// ---------------------------------------------------------------------------
kernel void shared_expert_down_kernel(
    device const float*    activated   [[buffer(0)]],  // float[512]
    device const uint32_t* down_w      [[buffer(1)]],  // (2048, 64) uint32
    device const bfloat*   down_s      [[buffer(2)]],  // (2048, 8) bf16
    device const bfloat*   down_b      [[buffer(3)]],  // (2048, 8) bf16
    device       half*     output      [[buffer(4)]],  // float16[2048]
    device const MoeBlockParams& params [[buffer(5)]],
    uint gid  [[thread_position_in_grid]]
) {
    uint hidden_size = uint(params.hidden_size);
    uint expert_dim = uint(params.expert_dim);
    uint group_size = uint(params.group_size);

    uint row = gid;
    if (row >= hidden_size) return;

    uint weight_row_stride = expert_dim / 8;    // 64
    uint sb_row_stride = expert_dim / group_size; // 8

    float val = dequant_dot_float_shared(
        down_w + row * weight_row_stride, activated,
        down_s + row * sb_row_stride, down_b + row * sb_row_stride,
        expert_dim, group_size
    );
    output[row] = half(val);
}

// ---------------------------------------------------------------------------
// Kernel 5: shared_gate_kernel
//
// Shared expert gate: linear(2048 -> 1, 4-bit quantized) -> sigmoid
// Weight: (1, 256) uint32 = 1 x 2048
// Scales: (1, 32) bf16
// Output: float[1] = sigmoid(dot(normed_x, gate_weight))
//
// Grid: (1, 1, 1)
// Threadgroup: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void shared_gate_kernel(
    device const half*     normed_x    [[buffer(0)]],  // float16[2048]
    device const uint32_t* gate_w      [[buffer(1)]],  // (1, 256) uint32
    device const bfloat*   gate_s      [[buffer(2)]],  // (1, 32) bf16
    device const bfloat*   gate_b      [[buffer(3)]],  // (1, 32) bf16
    device       float*    gate_val    [[buffer(4)]],  // float[1]
    device const MoeBlockParams& params [[buffer(5)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]]
) {
    uint hidden_size = uint(params.hidden_size);
    uint group_size = uint(params.group_size);

    // Load input into shared memory
    threadgroup half x_shared[2048];
    for (uint i = tid; i < hidden_size; i += tgs) {
        x_shared[i] = normed_x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel dot product with reduction
    uint weight_row_stride = hidden_size / 8;
    uint sb_row_stride = hidden_size / group_size;

    // Each thread computes partial sum
    threadgroup float partial_sums[256];
    float local_sum = 0.0f;

    uint num_groups = hidden_size / group_size;
    uint packed_per_group = group_size / 8;
    uint total_packed = num_groups * packed_per_group;  // 256

    // Divide packed uint32 elements across threads
    for (uint idx = tid; idx < total_packed; idx += tgs) {
        uint g = idx / packed_per_group;
        uint p = idx % packed_per_group;

        float scale = float(gate_s[g]);
        float bias  = float(gate_b[g]);
        uint32_t packed = gate_w[idx];
        uint x_base = g * group_size + p * 8;

        local_sum += fma(float((packed      ) & 0xF), scale, bias) * float(x_shared[x_base + 0]);
        local_sum += fma(float((packed >>  4) & 0xF), scale, bias) * float(x_shared[x_base + 1]);
        local_sum += fma(float((packed >>  8) & 0xF), scale, bias) * float(x_shared[x_base + 2]);
        local_sum += fma(float((packed >> 12) & 0xF), scale, bias) * float(x_shared[x_base + 3]);
        local_sum += fma(float((packed >> 16) & 0xF), scale, bias) * float(x_shared[x_base + 4]);
        local_sum += fma(float((packed >> 20) & 0xF), scale, bias) * float(x_shared[x_base + 5]);
        local_sum += fma(float((packed >> 24) & 0xF), scale, bias) * float(x_shared[x_base + 6]);
        local_sum += fma(float((packed >> 28) & 0xF), scale, bias) * float(x_shared[x_base + 7]);
    }
    partial_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float logit = partial_sums[0];
        gate_val[0] = 1.0f / (1.0f + exp(-logit));  // sigmoid
    }
}

// ---------------------------------------------------------------------------
// Kernel 6c: router_softmax_topk
//
// GPU-side softmax + top-K selection on router logits.
// Eliminates CPU softmax/top-K between CB1 wait and expert pread,
// reducing readback from 1024 bytes (256 floats) to 32 bytes (8 int32).
//
// Input:  float32[num_experts] raw router logits (from router_matmul_kernel)
// Output: int32[top_k] expert indices, float32[top_k] normalized scores
//
// Grid: (1, 1, 1) — single threadgroup
// Threadgroup: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void router_softmax_topk(
    device const float*    logits       [[buffer(0)]],  // float32[num_experts]
    device       int32_t*  out_indices  [[buffer(1)]],  // int32[top_k]
    device       float*    out_scores   [[buffer(2)]],  // float32[top_k]
    device const MoeBlockParams& params [[buffer(3)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]]
) {
    uint num_experts = uint(params.num_experts);   // 256
    uint top_k = uint(params.num_experts_per_tok); // 8

    // --- Phase 1: Load logits into shared memory and find max ---
    threadgroup float shared_logits[256];
    threadgroup float partial_max[256];

    // Each thread loads one element (256 threads for 256 experts)
    float my_val = -INFINITY;
    if (tid < num_experts) {
        my_val = logits[tid];
        shared_logits[tid] = my_val;
    }
    partial_max[tid] = my_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction for max
    for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_max[tid] = max(partial_max[tid], partial_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = partial_max[0];

    // --- Phase 2: Compute exp(logit - max) and partial sum ---
    threadgroup float partial_sum[256];
    float my_exp = 0.0f;
    if (tid < num_experts) {
        my_exp = exp(shared_logits[tid] - max_val);
        shared_logits[tid] = my_exp;  // reuse for softmax probs
    }
    partial_sum[tid] = my_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction for sum
    for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Phase 3: Normalize to get softmax probabilities ---
    float inv_sum = 1.0f / partial_sum[0];
    if (tid < num_experts) {
        shared_logits[tid] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Phase 4: Top-K selection (single-threaded, K=8 over 256 values) ---
    // This is a tiny O(K*N) scan — faster on GPU than a CPU roundtrip.
    if (tid == 0) {
        threadgroup bool taken[256];
        for (uint i = 0; i < num_experts; i++) taken[i] = false;

        float score_sum = 0.0f;
        for (uint k = 0; k < top_k; k++) {
            int best = -1;
            float best_val = -1.0f;
            for (uint i = 0; i < num_experts; i++) {
                if (!taken[i] && shared_logits[i] > best_val) {
                    best = int(i);
                    best_val = shared_logits[i];
                }
            }
            out_indices[k] = int32_t(best);
            out_scores[k] = best_val;
            score_sum += best_val;
            taken[best] = true;
        }

        // Normalize top-K scores
        float inv_score_sum = 1.0f / score_sum;
        for (uint k = 0; k < top_k; k++) {
            out_scores[k] *= inv_score_sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 6a: blend_kernel (NO residual — for SparseMoeBlock replacement)
//
// output = sum(expert_out[k] * score[k]) + gate_val * shared_out
//
// Grid: (hidden_size, 1, 1) = (2048, 1, 1)
// Threadgroup: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void blend_kernel(
    device const float*    expert_output [[buffer(0)]],  // float[K * hidden_size]
    device const float*    scores        [[buffer(1)]],  // float[K]
    device const half*     shared_output [[buffer(2)]],  // float16[hidden_size]
    device const float*    gate_val      [[buffer(3)]],  // float[1]
    device       half*     output        [[buffer(4)]],  // float16[hidden_size]
    device const MoeBlockParams& params  [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint hidden_size = uint(params.hidden_size);
    uint K = uint(params.num_experts_per_tok);

    if (gid >= hidden_size) return;

    // Weighted sum of expert outputs
    float expert_sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        expert_sum += expert_output[k * hidden_size + gid] * scores[k];
    }

    // Add shared expert contribution (gated)
    float shared_val = float(shared_output[gid]) * gate_val[0];

    output[gid] = half(expert_sum + shared_val);
}

// ---------------------------------------------------------------------------
// Kernel 6b: blend_residual_kernel
//
// Combines all MoE block outputs:
//   output = x_residual + sum(expert_out[k] * score[k]) + gate_val * shared_out
//
// expert_output: float[K * hidden_size] — per-expert outputs from expert_down_proj
// scores: float[K] — normalized router scores
// shared_output: float16[hidden_size] — shared expert output
// gate_val: float[1] — sigmoid(shared_expert_gate(normed_x))
// x_residual: float16[hidden_size] — pre-MoE hidden state (h, before layernorm)
//
// Grid: (hidden_size, 1, 1) = (2048, 1, 1)
// Threadgroup: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void blend_residual_kernel(
    device const half*     x_residual    [[buffer(0)]],  // float16[hidden_size]
    device const float*    expert_output [[buffer(1)]],  // float[K * hidden_size]
    device const float*    scores        [[buffer(2)]],  // float[K]
    device const half*     shared_output [[buffer(3)]],  // float16[hidden_size]
    device const float*    gate_val      [[buffer(4)]],  // float[1]
    device       half*     output        [[buffer(5)]],  // float16[hidden_size]
    device const MoeBlockParams& params  [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint hidden_size = uint(params.hidden_size);
    uint K = uint(params.num_experts_per_tok);

    if (gid >= hidden_size) return;

    // 1. Weighted sum of expert outputs
    float expert_sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        expert_sum += expert_output[k * hidden_size + gid] * scores[k];
    }

    // 2. Add shared expert contribution (gated)
    float shared_val = float(shared_output[gid]) * gate_val[0];

    // 3. Add residual
    float result = float(x_residual[gid]) + expert_sum + shared_val;

    output[gid] = half(result);
}
