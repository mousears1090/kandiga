/*
 * full_forward.h — Public C API for the complete transformer forward pass.
 *
 * Replaces the entire MLX model decode path with C/Metal. No Python in
 * the hot loop — embedding, attention (both full and GatedDeltaNet linear),
 * MoE blocks, final norm, and LM head all execute in C with Metal dispatch.
 *
 * Usage:
 *   void* engine = bakan_full_init("/path/to/packed", 40);
 *   // Load all weights from Python:
 *   bakan_full_set_weight(engine, "layers.0.input_layernorm.weight",
 *                         data, size, ndim, shape, dtype);
 *   // ... set all weights ...
 *   int status = bakan_full_build(engine);  // finalize weight setup
 *
 *   // Forward pass:
 *   float logits[248320];
 *   int vocab;
 *   bakan_full_forward(engine, token_id, position, logits, &vocab);
 *
 *   bakan_full_reset_cache(engine);  // new conversation
 *   bakan_full_destroy(engine);
 */

#ifndef BAKAN_FULL_FORWARD_H
#define BAKAN_FULL_FORWARD_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize the full forward engine.
 *
 * Opens all layer binary files from packed_expert_dir, creates Metal
 * device/queue, compiles all shader pipelines, allocates all buffers.
 *
 * Returns an opaque engine pointer, or NULL on failure.
 */
void* bakan_full_init(const char* packed_expert_dir, int num_layers);

/*
 * Set a weight tensor by name.
 *
 * Called from Python for every parameter in the model. The name follows
 * the MLX parameter naming convention (e.g. "layers.0.input_layernorm.weight").
 *
 * Parameters:
 *   engine: Opaque engine pointer
 *   name:   Parameter name (null-terminated string)
 *   data:   Raw weight data pointer
 *   size:   Size in bytes
 *   ndim:   Number of dimensions
 *   shape:  Array of dimension sizes (length ndim)
 *   dtype:  0=float16, 1=bfloat16, 2=float32, 3=uint32(packed 4-bit quantized)
 *
 * Returns 0 on success, -1 on error, 1 if name is unrecognized (skipped).
 */
int bakan_full_set_weight(void* engine, const char* name, const void* data,
                          size_t size, int ndim, const int64_t* shape, int dtype);

/*
 * Finalize weight setup. Must be called after all set_weight calls
 * and before the first forward pass.
 *
 * Validates that all required weights are loaded and sets up any
 * derived data structures.
 *
 * Returns 0 on success, -1 on error.
 */
int bakan_full_build(void* engine);

/*
 * Run one complete forward pass for a single decode token.
 *
 * Processes the token through all 40 layers (embedding -> attention ->
 * MoE -> final norm -> LM head) and returns logits.
 *
 * Parameters:
 *   engine:        Opaque engine pointer
 *   token_id:      Input token ID
 *   position:      Position in sequence (for RoPE, starting from 0)
 *   logits_out:    Output buffer for logits, float[vocab_size]
 *   vocab_size_out: Output: vocabulary size (248320)
 *
 * Returns 0 on success, -1 on error.
 */
int bakan_full_forward(void* engine, int token_id, int position,
                       float* logits_out, int* vocab_size_out);

/*
 * Batch prefill: process all prompt tokens through the model in one call.
 *
 * Instead of calling bakan_full_forward() N times from Python, this
 * processes all N prompt tokens in a layer-major loop within a single
 * C function call. Tokens are processed sequentially within each layer
 * (attention state is recurrent), then advances to the next layer.
 *
 * Key optimizations:
 *   - Skips LM head (248320-dim matmul) for all non-last tokens
 *   - Eliminates Python overhead (ctypes calls, logits alloc, numpy)
 *   - Bulk CPU embedding for all tokens
 *   - Layer-major ordering (foundation for future batched matmul)
 *
 * Parameters:
 *   engine:        Opaque engine pointer
 *   token_ids:     Array of input token IDs (length num_tokens)
 *   num_tokens:    Number of prompt tokens to process (max 1024)
 *   logits_out:    Output buffer for logits of the LAST token, float[vocab_size]
 *   vocab_size_out: Output: vocabulary size (248320)
 *
 * Returns 0 on success, -1 on error.
 */
int bakan_full_prefill(void* engine, const int32_t* token_ids, int num_tokens,
                       float* logits_out, int* vocab_size_out);

/*
 * Clear all caches (KV cache, linear attention state, conv1d state).
 * Call between conversations.
 */
void bakan_full_reset_cache(void* engine);

/*
 * Destroy the engine and release all resources.
 */
void bakan_full_destroy(void* engine);

/*
 * Get model info.
 */
int bakan_full_num_layers(void* engine);
int bakan_full_vocab_size(void* engine);

/*
 * Enable/disable debug printing in forward pass.
 * When enabled, prints hidden state values after embedding, each layer, etc.
 */
void bakan_full_set_debug(void* engine, int enable);

/*
 * Set KV cache for a full-attention layer (after MLX prefill).
 *
 * Copies pre-computed key/value cache data into the Metal engine's
 * KV cache buffers. Used by the hybrid engine to transfer MLX prefill
 * results to the Metal decode engine.
 *
 * Parameters:
 *   engine:         Opaque engine pointer
 *   attn_layer_idx: Index into full-attention layers (0-9)
 *   keys_data:      Raw key data, half[num_kv_heads * seq_len * head_dim]
 *                   Layout: [head][seq][dim], contiguous
 *   values_data:    Raw value data, same layout as keys
 *   seq_len:        Number of tokens in cache (positions 0..seq_len-1)
 *   num_kv_heads:   Number of KV heads (2 for this model)
 *   head_dim:       Head dimension (256 for this model)
 *
 * Returns 0 on success, -1 on error.
 */
int bakan_full_set_kv_cache(void* engine, int attn_layer_idx,
                             const void* keys_data, const void* values_data,
                             int seq_len, int num_kv_heads, int head_dim);

/*
 * Set linear attention state for a GatedDeltaNet layer (after MLX prefill).
 *
 * Copies the delta-net recurrent state matrix into the Metal engine's
 * linear state buffer.
 *
 * Parameters:
 *   engine:         Opaque engine pointer
 *   lin_layer_idx:  Index into linear-attention layers (0-29)
 *   state_data:     Raw state data, float32[num_v_heads * val_dim * key_dim]
 *                   Layout: [hv][dv][dk], contiguous
 *   state_bytes:    Size in bytes of state_data
 *
 * Returns 0 on success, -1 on error.
 */
int bakan_full_set_linear_state(void* engine, int lin_layer_idx,
                                 const void* state_data, int state_bytes);

/*
 * Set conv1d buffer for a GatedDeltaNet layer (after MLX prefill).
 *
 * Copies the conv1d sliding window state into the Metal engine's
 * conv buffer. The caller must transpose from MLX layout [time][channel]
 * to Metal layout [channel][kernel_size] before calling.
 *
 * Parameters:
 *   engine:         Opaque engine pointer
 *   lin_layer_idx:  Index into linear-attention layers (0-29)
 *   conv_data:      Raw conv buffer data, half[conv_dim * kernel_size]
 *                   Layout: [channel][kernel_pos], contiguous
 *   conv_bytes:     Size in bytes of conv_data
 *
 * Returns 0 on success, -1 on error.
 */
int bakan_full_set_conv_state(void* engine, int lin_layer_idx,
                               const void* conv_data, int conv_bytes);

/*
 * Set the current sequence position (after MLX prefill).
 *
 * The Metal engine uses position to know where in the KV cache to
 * write the next token and for RoPE computation. After prefill of
 * N tokens, set position to N so the first decode token uses
 * position=N.
 *
 * Note: position is set automatically during forward/prefill calls.
 * This function is only needed for the hybrid engine path.
 *
 * Parameters:
 *   engine:    Opaque engine pointer
 *   position:  Next token position (= number of prefill tokens)
 *
 * Returns 0 on success, -1 on error.
 */
int bakan_full_set_position(void* engine, int position);

#ifdef __cplusplus
}
#endif

#endif /* BAKAN_FULL_FORWARD_H */
