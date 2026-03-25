"""Fused GatedDeltaNet attention on CPU (NEON).

Replaces MLX's per-op attention path with a single C function call
per layer. Eliminates Metal dispatch overhead for attention.

Only used for large models (122B/397B) where attention is the bottleneck.
35B models continue to use MLX's attention path.
"""

from __future__ import annotations

import ctypes
import os
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class FusedAttnLib:
    """Ctypes wrapper for libkandiga_attn_lg.dylib."""

    def __init__(self):
        metal_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metal")
        path = os.path.join(metal_dir, "libkandiga_attn_lg.dylib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fused attention library not found at {path}")

        self._lib = ctypes.CDLL(path)

        # init
        self._lib.kandiga_fused_attn_init.argtypes = [
            ctypes.c_int] * 9
        self._lib.kandiga_fused_attn_init.restype = ctypes.c_void_p

        # set_weights
        self._lib.kandiga_fused_attn_set_weights.argtypes = [
            ctypes.c_void_p, ctypes.c_int,  # ptr, layer_idx
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,  # qkv
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,  # z
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,  # beta
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,  # alpha
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,  # out
            ctypes.c_void_p, ctypes.c_int,  # conv_w
            ctypes.c_void_p, ctypes.c_int,  # A_log
            ctypes.c_void_p, ctypes.c_int,  # dt_bias
            ctypes.c_void_p, ctypes.c_int,  # norm_w
        ]
        self._lib.kandiga_fused_attn_set_weights.restype = None

        # decode
        self._lib.kandiga_fused_attn_decode.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self._lib.kandiga_fused_attn_decode.restype = ctypes.c_int

        # decode_bf16
        self._lib.kandiga_fused_attn_decode_bf16.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self._lib.kandiga_fused_attn_decode_bf16.restype = ctypes.c_int

        # reset
        self._lib.kandiga_fused_attn_reset.argtypes = [ctypes.c_void_p]
        self._lib.kandiga_fused_attn_reset.restype = None

        # destroy
        self._lib.kandiga_fused_attn_destroy.argtypes = [ctypes.c_void_p]
        self._lib.kandiga_fused_attn_destroy.restype = None


def _extract_qlinear(mod):
    """Extract quantized linear weight/scales/biases as contiguous numpy arrays."""
    mx.eval(mod.weight, mod.scales)
    w = np.ascontiguousarray(np.array(mod.weight, copy=False))
    s = np.ascontiguousarray(np.array(mod.scales.view(mx.uint16), copy=False))
    b_arr = getattr(mod, 'biases', None)
    if b_arr is None:
        b_arr = getattr(mod, 'bias', None)
    if b_arr is not None:
        mx.eval(b_arr)
        b = np.ascontiguousarray(np.array(b_arr.view(mx.uint16), copy=False))
    else:
        b = np.zeros_like(s)
    return w, s, b


def _ptr_and_size(arr):
    """Get ctypes pointer and byte size from numpy array."""
    return arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes


def install_fused_attention(model, lib: FusedAttnLib):
    """Extract attention weights and install fused C attention on all layers.

    Returns the engine handle and number of layers patched.
    """
    lm = model.language_model if hasattr(model, 'language_model') else model
    args = lm.args
    layers = lm.model.layers

    # Get dimensions (attributes vary by model version)
    nk = getattr(args, 'linear_num_key_heads', 16)
    nv = getattr(args, 'linear_num_value_heads', 64)
    dk = getattr(args, 'linear_key_head_dim', 128)
    dv = getattr(args, 'linear_value_head_dim', 128)
    ck = getattr(args, 'linear_conv_kernel_dim', 4)
    key_dim = nk * dk
    value_dim = nv * dv

    # Init engine
    eng = lib._lib.kandiga_fused_attn_init(
        len(layers), args.hidden_size,
        key_dim, value_dim,
        nk, nv, dk, dv, ck,
    )
    if not eng:
        raise RuntimeError("Failed to init fused attention engine")

    # Extract and set weights for each layer
    from mlx.utils import tree_flatten
    for i, layer in enumerate(layers):
        # Access attention module via children dict (more robust than attribute)
        children = layer.children()
        attn_key = None
        for k in ('linear_attn', 'self_attn', 'attention'):
            if k in children:
                attn_key = k
                break
        if attn_key is None:
            raise AttributeError(f"Layer {i} has no attention module. Children: {list(children.keys())}")
        attn = children[attn_key]

        # Only fuse GatedDeltaNet layers (skip full attention layers)
        if not hasattr(attn, 'in_proj_qkv'):
            continue

        mx.eval(*[v for _, v in tree_flatten(attn.parameters())])

        qkv_w, qkv_s, qkv_b = _extract_qlinear(attn.in_proj_qkv)
        z_w, z_s, z_b = _extract_qlinear(attn.in_proj_z)
        beta_w, beta_s, beta_b = _extract_qlinear(attn.in_proj_b)
        alpha_w, alpha_s, alpha_b = _extract_qlinear(attn.in_proj_a)
        out_w, out_s, out_b = _extract_qlinear(attn.out_proj)

        conv_w = np.array(attn.conv1d.weight.astype(mx.float32), copy=False)
        A_log = np.array(attn.A_log.astype(mx.float32), copy=False)
        dt_bias = np.array(attn.dt_bias.astype(mx.float32), copy=False)
        norm_w = np.array(attn.norm.weight.astype(mx.float32), copy=False)

        def _p(arr):
            return arr.ctypes.data_as(ctypes.c_void_p)

        lib._lib.kandiga_fused_attn_set_weights(
            eng, i,
            _p(qkv_w), _p(qkv_s), _p(qkv_b), qkv_w.nbytes, qkv_s.nbytes,
            _p(z_w), _p(z_s), _p(z_b), z_w.nbytes, z_s.nbytes,
            _p(beta_w), _p(beta_s), _p(beta_b), beta_w.nbytes, beta_s.nbytes,
            _p(alpha_w), _p(alpha_s), _p(alpha_b), alpha_w.nbytes, alpha_s.nbytes,
            _p(out_w), _p(out_s), _p(out_b), out_w.nbytes, out_s.nbytes,
            _p(conv_w), conv_w.nbytes,
            _p(A_log), A_log.nbytes,
            _p(dt_bias), dt_bias.nbytes,
            _p(norm_w), norm_w.nbytes,
        )

    return eng, len(layers)


class _FusedGatedDeltaNet(nn.Module):
    """Drop-in replacement for GatedDeltaNet that uses fused C attention."""

    def __init__(self, original, layer_idx: int, lib: FusedAttnLib, eng, hidden: int):
        super().__init__()
        self.original = original
        self._layer_idx = layer_idx
        self._lib = lib
        self._eng = eng
        self._hidden = hidden

    def __call__(self, inputs, mask=None, cache=None):
        B, S, _ = inputs.shape

        # For prefill (S > 1), use original MLX path
        if S > 1:
            return self.original(inputs, mask=mask, cache=cache)

        # Decode (S=1): use fused C attention
        x = inputs.reshape(-1, self._hidden)
        mx.eval(x)

        is_bf16 = x.dtype == mx.bfloat16
        x_np = np.array(x.view(mx.uint16) if is_bf16 else x.astype(mx.float32), copy=False)
        out_np = np.empty_like(x_np)

        if is_bf16:
            ret = self._lib._lib.kandiga_fused_attn_decode_bf16(
                self._eng, self._layer_idx,
                x_np.ctypes.data_as(ctypes.c_void_p),
                out_np.ctypes.data_as(ctypes.c_void_p),
            )
        else:
            ret = self._lib._lib.kandiga_fused_attn_decode(
                self._eng, self._layer_idx,
                x_np.ctypes.data_as(ctypes.c_void_p),
                out_np.ctypes.data_as(ctypes.c_void_p),
            )

        if ret != 0:
            # Fallback to MLX
            return self.original(inputs, mask=mask, cache=cache)

        if is_bf16:
            result = mx.array(out_np).view(mx.bfloat16)
        else:
            result = mx.array(out_np)

        return result.reshape(B, S, self._hidden)
