/*
 * bakan_forward.h — Public C API for the complete MoE block engine.
 *
 * Replaces the entire SparseMoeBlock (router + expert MLP + shared expert +
 * blending) with a single C/Metal call per layer, including the post-attention
 * layernorm and residual connection.
 *
 * Usage:
 *   void* engine = bakan_forward_init("/path/to/packed", 40);
 *   // Set per-layer weights (called from Python after loading MLX model):
 *   bakan_forward_set_weight(engine, layer, WT_POST_ATTN_NORM, data, size, rows, cols, 2);
 *   bakan_forward_set_weight(engine, layer, WT_ROUTER_WEIGHT, data, size, rows, cols, 1);
 *   // ... set all 16 weight types for each layer ...
 *
 *   // Run complete MoE block (norm + route + experts + shared + blend + residual):
 *   bakan_forward_moe_block(engine, layer_idx, h_f16, output_f16);
 *
 *   bakan_forward_destroy(engine);
 */

#ifndef BAKAN_FORWARD_H
#define BAKAN_FORWARD_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Weight type identifiers for bakan_forward_set_weight().
 * Each MoE layer requires all WT_COUNT weight buffers to be set.
 */
enum {
    WT_POST_ATTN_NORM       = 0,   /* post_attention_layernorm.weight bf16[2048] */
    WT_ROUTER_WEIGHT        = 1,   /* mlp.gate.weight          uint32(256, 256) */
    WT_ROUTER_SCALES        = 2,   /* mlp.gate.scales          bf16(256, 32) */
    WT_ROUTER_BIASES        = 3,   /* mlp.gate.biases          bf16(256, 32) */
    WT_SHARED_GATE_W        = 4,   /* mlp.shared_expert_gate.weight uint32(1, 256) */
    WT_SHARED_GATE_S        = 5,   /* mlp.shared_expert_gate.scales bf16(1, 32) */
    WT_SHARED_GATE_B        = 6,   /* mlp.shared_expert_gate.biases bf16(1, 32) */
    WT_SHARED_UP_W          = 7,   /* mlp.shared_expert.up_proj.weight uint32(512, 256) */
    WT_SHARED_UP_S          = 8,   /* mlp.shared_expert.up_proj.scales bf16(512, 32) */
    WT_SHARED_UP_B          = 9,   /* mlp.shared_expert.up_proj.biases bf16(512, 32) */
    WT_SHARED_GATE_PROJ_W   = 10,  /* mlp.shared_expert.gate_proj.weight uint32(512, 256) */
    WT_SHARED_GATE_PROJ_S   = 11,  /* mlp.shared_expert.gate_proj.scales bf16(512, 32) */
    WT_SHARED_GATE_PROJ_B   = 12,  /* mlp.shared_expert.gate_proj.biases bf16(512, 32) */
    WT_SHARED_DOWN_W        = 13,  /* mlp.shared_expert.down_proj.weight uint32(2048, 64) */
    WT_SHARED_DOWN_S        = 14,  /* mlp.shared_expert.down_proj.scales bf16(2048, 8) */
    WT_SHARED_DOWN_B        = 15,  /* mlp.shared_expert.down_proj.biases bf16(2048, 8) */
    WT_COUNT                = 16
};

/*
 * Initialize the forward engine.
 *
 * Opens all layer binary files from packed_dir, creates Metal device/queue,
 * compiles all shader pipelines, allocates scratch buffers.
 *
 * Returns an opaque engine pointer, or NULL on failure.
 */
void* bakan_forward_init(const char* packed_expert_dir, int num_layers);

/*
 * Set a weight buffer for a specific layer and weight type.
 *
 * Called from Python after loading the MLX model. Copies the weight data
 * into a Metal shared buffer for GPU access.
 *
 * Parameters:
 *   engine:      Opaque engine pointer
 *   layer_idx:   Layer index (0..num_layers-1)
 *   weight_type: One of WT_* enum values
 *   data:        Pointer to raw weight data
 *   size:        Size in bytes
 *   rows:        Number of rows (for validation)
 *   cols:        Number of columns (for validation)
 *   dtype:       0=float16, 1=uint32(packed 4-bit), 2=bfloat16(as uint16)
 *
 * Returns 0 on success, -1 on error.
 */
int bakan_forward_set_weight(void* engine, int layer_idx, int weight_type,
                              const void* data, size_t size,
                              int rows, int cols, int dtype);

/*
 * Run the complete MoE block for one layer.
 *
 * This function replaces the ENTIRE decoder layer MLP path:
 *   out = h + self.mlp(self.post_attention_layernorm(h))
 *
 * Internally does:
 *   1. post_attention_layernorm(h) -> normed
 *   2. Router: gate(normed) -> softmax -> top-k -> indices/scores
 *   3. pread expert weights from packed binary files
 *   4. Expert MLP (Metal: up+gate+SwiGLU, down_proj)
 *   5. Weighted sum of expert outputs
 *   6. Shared expert MLP (Metal: gate_proj+up_proj+SwiGLU, down_proj)
 *   7. Shared expert gate: sigmoid(gate_linear(normed))
 *   8. Blend + residual: output = h + sum(expert[k]*score[k]) + gate*shared
 *
 * Parameters:
 *   engine:     Opaque engine pointer
 *   layer_idx:  Layer index (0..num_layers-1)
 *   h_f16:      Input vector, float16[hidden_size] (post-attention residual h)
 *   output_f16: Output buffer, float16[hidden_size]
 *
 * Returns 0 on success, -1 on error.
 */
int bakan_forward_moe_block(void* engine, int layer_idx,
                             const void* h_f16, void* output_f16);

/*
 * Run the MoE block for already-normed input (no norm, no residual).
 *
 * This replaces SparseMoeBlock.__call__() — receives input that has
 * already been through post_attention_layernorm, and returns the MoE
 * output WITHOUT adding the residual (the decoder layer handles that).
 *
 * Internally does:
 *   1. Router: gate(x) -> softmax -> top-k -> indices/scores
 *   2. pread expert weights from packed binary files
 *   3. Expert MLP (Metal: up+gate+SwiGLU, down_proj)
 *   4. Weighted sum of expert outputs
 *   5. Shared expert MLP (Metal: gate_proj+up_proj+SwiGLU, down_proj)
 *   6. Shared expert gate: sigmoid(gate_linear(x))
 *   7. output = sum(expert[k]*score[k]) + gate*shared
 *
 * Note: Does NOT skip WT_POST_ATTN_NORM (that weight is ignored in this path).
 *
 * Parameters:
 *   engine:     Opaque engine pointer
 *   layer_idx:  Layer index (0..num_layers-1)
 *   x_f16:      Input vector, float16[hidden_size] (already normed)
 *   output_f16: Output buffer, float16[hidden_size]
 *
 * Returns 0 on success, -1 on error.
 */
int bakan_forward_moe_block_normed(void* engine, int layer_idx,
                                    const void* x_f16, void* output_f16);

/*
 * Destroy the engine and release all resources.
 */
void bakan_forward_destroy(void* engine);

/*
 * Get the number of layers the engine was initialized with.
 */
int bakan_forward_num_layers(void* engine);

#ifdef __cplusplus
}
#endif

#endif /* BAKAN_FORWARD_H */
