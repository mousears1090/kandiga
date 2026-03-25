"""Custom Metal kernels for Kandiga using MLX's metal_kernel API.

These kernels operate directly on MLX arrays — no buffer copying needed.
They fuse multiple operations into single GPU dispatches.
"""

import mlx.core as mx

# ---------------------------------------------------------------------------
# Fused gated delta state update + output computation
# Replaces: sigmoid(b) + compute_g + state_update + query_output
# One kernel instead of 4+ separate MLX ops
# ---------------------------------------------------------------------------

_FUSED_DELTA_HEADER = """
inline float _sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
"""

_FUSED_DELTA_SOURCE = """
    // Thread handles one (vh, dv) pair
    uint idx = thread_position_in_grid.x;
    uint nv = num_v_heads[0];
    uint kd = key_dim[0];
    uint vd = val_dim[0];
    uint nk = num_k_heads[0];

    uint vh = idx / vd;
    uint dv = idx % vd;
    if (vh >= nv) return;

    // Map V head to K head
    uint hk = vh / (nv / nk);

    // Compute g = exp(A_log * sigmoid(alpha) + dt_bias)
    float a_val = float(alpha[vh]);
    float sig_a = _sigmoid(a_val);
    float g = exp(A_log[vh] * sig_a + dt_bias[vh]);

    // Beta = sigmoid(b)
    float beta = _sigmoid(float(b_vec[vh]));

    // State pointer
    uint s_base = (vh * vd + dv) * kd;

    // K and Q pointers for this K head
    uint k_base = hk * kd;
    uint q_base = hk * kd;

    // V value
    float v_val = float(v_vec[vh * vd + dv]);

    // Decay state and compute outer product update
    float out_val = 0.0f;
    for (uint dk = 0; dk < kd; dk++) {
        float s = float(state[s_base + dk]);
        float k = float(k_vec[k_base + dk]);
        float q = float(q_vec[q_base + dk]);

        // Decay
        s *= g;
        // Delta update: s += beta * v * k
        s += beta * v_val * k;
        // Write back state
        state[s_base + dk] = s;
        // Accumulate output: out += s * q
        out_val += s * q;
    }

    output[vh * vd + dv] = float(out_val);
"""

fused_delta_update = mx.fast.metal_kernel(
    name="fused_delta_update",
    input_names=["q_vec", "k_vec", "v_vec", "alpha", "b_vec",
                 "A_log", "dt_bias", "num_k_heads", "num_v_heads",
                 "key_dim", "val_dim"],
    output_names=["state", "output"],
    header=_FUSED_DELTA_HEADER,
    source=_FUSED_DELTA_SOURCE,
)


def fused_gated_delta(q, k, v, alpha, b, A_log, dt_bias, state,
                       num_k_heads, num_v_heads, head_k_dim, head_v_dim):
    """Fused gated delta state update.

    Replaces: sigmoid + compute_g + state decay + outer product + query output
    All in one Metal dispatch instead of 5+ separate ops.

    Returns: (output, updated_state)
    """
    nv = num_v_heads
    vd = head_v_dim
    total_threads = nv * vd

    nk_arr = mx.array([num_k_heads], dtype=mx.int32)
    nv_arr = mx.array([num_v_heads], dtype=mx.int32)
    kd_arr = mx.array([head_k_dim], dtype=mx.int32)
    vd_arr = mx.array([head_v_dim], dtype=mx.int32)

    output, state_out = fused_delta_update(
        inputs=[q.reshape(-1), k.reshape(-1), v.reshape(-1),
                alpha.reshape(-1), b.reshape(-1),
                A_log, dt_bias, nk_arr, nv_arr, kd_arr, vd_arr],
        output_shapes=[(state.shape,), (nv * vd,)],
        output_dtypes=[state.dtype, mx.float32],
        grid=(total_threads, 1, 1),
        threadgroup=(min(256, total_threads), 1, 1),
        init_value=0.0,  # state is passed as input AND output
    )

    return output.reshape(1, nv, vd), state_out
