"""Tests for TurboQuant KV cache compression."""

import numpy as np
import pytest


def test_imports():
    from kandiga.kv_compress import TurboQuantCache, install_kv_compression


def test_rotation_matrix_orthogonal():
    """Rotation matrix should be orthogonal (R @ R^T = I)."""
    from kandiga.kv_compress import TurboQuantCache
    import mlx.core as mx

    cache = TurboQuantCache(head_dim=128, num_heads=8)
    R = np.array(cache.rotation)
    identity = R @ R.T
    np.testing.assert_allclose(identity, np.eye(128), atol=1e-5)


def test_quantize_dequantize_roundtrip():
    """Compressed values should be close to originals."""
    from kandiga.kv_compress import TurboQuantCache
    import mlx.core as mx

    cache = TurboQuantCache(head_dim=64, num_heads=4, seed=123)

    # Random KV data simulating attention outputs
    np.random.seed(42)
    keys = mx.array(np.random.randn(1, 4, 16, 64).astype(np.float32))
    values = mx.array(np.random.randn(1, 4, 16, 64).astype(np.float32))

    cache.update(keys, values)

    # Reconstruct
    k_recon = cache.keys
    v_recon = cache.values

    assert k_recon.shape == keys.shape
    assert v_recon.shape == values.shape

    # Check reconstruction quality — should be reasonable for 3-bit
    k_err = np.mean(np.abs(np.array(k_recon.astype(mx.float32)) - np.array(keys)))
    v_err = np.mean(np.abs(np.array(v_recon.astype(mx.float32)) - np.array(values)))

    # 3-bit quantization error should be bounded
    assert k_err < 0.5, f"Key reconstruction error too high: {k_err}"
    assert v_err < 0.5, f"Value reconstruction error too high: {v_err}"


def test_sequential_updates():
    """Cache should handle multiple sequential updates."""
    from kandiga.kv_compress import TurboQuantCache
    import mlx.core as mx

    cache = TurboQuantCache(head_dim=64, num_heads=2, seed=0)

    for t in range(5):
        k = mx.array(np.random.randn(1, 2, 1, 64).astype(np.float32))
        v = mx.array(np.random.randn(1, 2, 1, 64).astype(np.float32))
        cache.update(k, v)

    assert cache.seq_len == 5
    assert cache.keys.shape == (1, 2, 5, 64)
    assert cache.values.shape == (1, 2, 5, 64)


def test_memory_savings():
    """Compressed cache should use less memory."""
    from kandiga.kv_compress import TurboQuantCache
    import mlx.core as mx

    cache = TurboQuantCache(head_dim=128, num_heads=8)

    k = mx.array(np.random.randn(1, 8, 100, 128).astype(np.float32))
    v = mx.array(np.random.randn(1, 8, 100, 128).astype(np.float32))
    cache.update(k, v)

    compressed = cache.memory_bytes
    uncompressed = 1 * 8 * 100 * 128 * 2 * 2  # float16, keys+values

    assert compressed < uncompressed, f"Compressed {compressed} >= uncompressed {uncompressed}"


def test_empty_cache():
    """Empty cache should return None."""
    from kandiga.kv_compress import TurboQuantCache

    cache = TurboQuantCache(head_dim=128, num_heads=8)
    assert cache.keys is None
    assert cache.values is None
    assert cache.seq_len == 0
    assert cache.memory_bytes == 0


def test_cosine_similarity_preserved():
    """Attention scores depend on cosine similarity — it should be preserved."""
    from kandiga.kv_compress import TurboQuantCache
    import mlx.core as mx

    cache = TurboQuantCache(head_dim=128, num_heads=1, seed=77)

    np.random.seed(99)
    keys = mx.array(np.random.randn(1, 1, 32, 128).astype(np.float32))
    cache.update(keys, keys)  # use same for simplicity

    k_orig = np.array(keys).reshape(32, 128)
    k_recon = np.array(cache.keys.astype(mx.float32)).reshape(32, 128)

    # Compute cosine similarity between original and reconstructed
    for i in range(32):
        cos_sim = np.dot(k_orig[i], k_recon[i]) / (
            np.linalg.norm(k_orig[i]) * np.linalg.norm(k_recon[i]) + 1e-8
        )
        assert cos_sim > 0.85, f"Token {i} cosine similarity too low: {cos_sim:.3f}"
