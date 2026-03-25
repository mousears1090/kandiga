/*
 * kandiga_cpu_expert.h -- Public C API for CPU-side expert MLP computation.
 *
 * Computes MoE expert MLP entirely on CPU using vectorized 4-bit dequant
 * matrix-vector multiply. Designed to run in PARALLEL with GPU attention
 * on Apple Silicon's unified memory architecture.
 *
 * Architecture (KTransformers-inspired):
 *   CPU expert computation has ZERO dispatch overhead vs Metal's ~0.5ms
 *   per dispatch. Even though CPU FLOPs are slower, total latency is lower
 *   because there's no GPU sync cost.
 *
 * On Apple Silicon unified memory, CPU and GPU share the same physical
 * DRAM. CPU reads expert weights via pread (OS page cache), dequantizes
 * 4-bit weights inline, and computes gate+up+SwiGLU+down MLP.
 *
 * Usage:
 *   void* engine = bakan_cpu_expert_init("/path/to/packed", 40);
 *   int32_t experts[] = {3, 17, 42, 100, 128, 200, 211, 255};
 *   float output[8 * 2048];
 *   bakan_cpu_expert_mlp(engine, layer_idx, x_f32, experts, 8, output);
 *   bakan_cpu_expert_destroy(engine);
 */

#ifndef KANDIGA_CPU_EXPERT_H
#define KANDIGA_CPU_EXPERT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize the CPU expert engine.
 * Opens all layer files from packed_dir for pread access.
 * Returns an opaque engine pointer, or NULL on failure.
 */
void* bakan_cpu_expert_init(const char* packed_dir, int num_layers);

/*
 * Compute expert MLP for one layer on CPU (float32 I/O).
 *
 * Reads expert weights via pread, dequantizes 4-bit packed weights,
 * computes gate+up+SwiGLU+down MLP. Multiple experts computed in
 * parallel using GCD (Grand Central Dispatch).
 *
 * Returns 0 on success, -1 on error.
 */
int bakan_cpu_expert_mlp(
    void* engine,
    int layer_idx,
    const float* x_f32,
    const int32_t* expert_indices,
    int num_experts,
    float* output_f32
);

/*
 * Compute expert MLP with float16 input/output.
 * Same as above but accepts/produces float16. Internally uses float32.
 * Returns 0 on success, -1 on error.
 */
int bakan_cpu_expert_mlp_f16(
    void* engine,
    int layer_idx,
    const void* x_f16,
    const int32_t* expert_indices,
    int num_experts,
    void* output_f16
);

/*
 * Compute expert MLP with bfloat16 input, float16 output.
 * Accepts MLX's native bfloat16 directly.
 * Returns 0 on success, -1 on error.
 */
int bakan_cpu_expert_mlp_bf16(
    void* engine,
    int layer_idx,
    const void* x_bf16,
    const int32_t* expert_indices,
    int num_experts,
    void* output_f16
);

/*
 * Destroy the engine and release all resources.
 */
void bakan_cpu_expert_destroy(void* engine);

/*
 * Get the number of layers the engine was initialized with.
 */
int bakan_cpu_expert_num_layers(void* engine);

#ifdef __cplusplus
}
#endif

#endif /* KANDIGA_CPU_EXPERT_H */
