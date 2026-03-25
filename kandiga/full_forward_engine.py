"""Full C/Metal forward pass engine for large models (122B/397B).

Zero Python in the decode loop. All attention + MoE computed in C with
Metal GPU dispatch. Eliminates mx.eval sync overhead entirely.

Only used for models where is_large=True. 35B continues using MLX path.
"""

from __future__ import annotations

import ctypes
import os
import time
from typing import Optional

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten


class FullForwardEngine:
    """Wraps libkandiga_full_lg.dylib for complete C/Metal inference."""

    def __init__(self, packed_dir: str, num_layers: int):
        metal_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metal")
        dylib_path = os.path.join(metal_dir, "libkandiga_full_lg.dylib")

        if not os.path.exists(dylib_path):
            raise FileNotFoundError(f"Full forward library not found at {dylib_path}")

        self._lib = ctypes.CDLL(dylib_path)

        # bakan_full_init(packed_dir, num_layers) -> engine
        self._lib.bakan_full_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self._lib.bakan_full_init.restype = ctypes.c_void_p

        # bakan_full_set_weight(engine, name, data, size, ndim, shape, dtype)
        self._lib.bakan_full_set_weight.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64), ctypes.c_int,
        ]
        self._lib.bakan_full_set_weight.restype = ctypes.c_int

        # bakan_full_build(engine)
        self._lib.bakan_full_build.argtypes = [ctypes.c_void_p]
        self._lib.bakan_full_build.restype = ctypes.c_int

        # bakan_full_forward(engine, token_id, position, logits_out, vocab_size_out)
        self._lib.bakan_full_forward.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.bakan_full_forward.restype = ctypes.c_int

        # bakan_full_reset_cache(engine)
        self._lib.bakan_full_reset_cache.argtypes = [ctypes.c_void_p]
        self._lib.bakan_full_reset_cache.restype = None

        # bakan_full_destroy(engine)
        self._lib.bakan_full_destroy.argtypes = [ctypes.c_void_p]
        self._lib.bakan_full_destroy.restype = None

        self._engine = self._lib.bakan_full_init(packed_dir.encode(), num_layers)
        if not self._engine:
            raise RuntimeError("Failed to init full forward engine")

    def _set_one_weight(self, c_name: str, param):
        """Set a single weight in the C engine."""
        mx.eval(param)
        if param.dtype == mx.bfloat16:
            raw = np.array(param.view(mx.uint16), copy=False)
            dtype_code = 1
        elif param.dtype == mx.float16:
            raw = np.array(param, copy=False)
            dtype_code = 1
        elif param.dtype == mx.uint32:
            raw = np.array(param, copy=False)
            dtype_code = 3
        elif param.dtype == mx.float32:
            raw = np.array(param, copy=False)
            dtype_code = 0
        else:
            raw = np.array(param.astype(mx.float16), copy=False)
            dtype_code = 1
        raw = np.ascontiguousarray(raw)
        shape = (ctypes.c_int64 * len(raw.shape))(*raw.shape)
        return self._lib.bakan_full_set_weight(
            self._engine, c_name.encode(),
            raw.ctypes.data_as(ctypes.c_void_p),
            raw.nbytes, len(raw.shape), shape, dtype_code,
        )

    def set_weights(self, model):
        """Extract all weights from MLX model and feed to C engine.

        Fuses separate attention projections into combined_proj format
        expected by the C forward pass.
        """
        t0 = time.time()
        flat = tree_flatten(model.parameters())
        params = {name: param for name, param in flat}

        count = 0
        skipped = 0

        # Group params by layer for fusion
        for name, param in flat:
            if "switch_mlp" in name:
                skipped += 1
                continue

            c_name = name
            for prefix in ("language_model.model.", "language_model."):
                if c_name.startswith(prefix):
                    c_name = c_name[len(prefix):]
                    break

            # Skip individual attention projections (we'll fuse them below)
            if any(x in c_name for x in ("in_proj_qkv.", "in_proj_z.", "in_proj_b.", "in_proj_a.",
                                           "q_proj.", "k_proj.", "v_proj.")):
                # These get fused into combined_proj
                continue

            if self._set_one_weight(c_name, param) == 0:
                count += 1

        # Fuse attention projections per layer
        lm_prefix = "language_model.model."
        for layer_idx in range(48):
            # Check if linear attention or full attention
            lin_key = f"{lm_prefix}layers.{layer_idx}.linear_attn.in_proj_qkv.weight"
            self_key = f"{lm_prefix}layers.{layer_idx}.self_attn.q_proj.weight"

            if lin_key in params:
                # Linear attention: fuse QKV + Z + B + A into combined_proj
                attn_prefix = f"{lm_prefix}layers.{layer_idx}.linear_attn"
                for comp in ("weight", "scales", "biases"):
                    parts = []
                    for proj in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
                        key = f"{attn_prefix}.{proj}.{comp}"
                        if key in params:
                            mx.eval(params[key])
                            parts.append(params[key])
                    if parts:
                        fused = mx.concatenate(parts, axis=0)
                        c_name = f"layers.{layer_idx}.linear_attn.combined_proj.{comp}"
                        if self._set_one_weight(c_name, fused) == 0:
                            count += 1

            elif self_key in params:
                # Full attention: fuse Q+gate + K + V into combined_proj
                attn_prefix = f"{lm_prefix}layers.{layer_idx}.self_attn"
                for comp in ("weight", "scales", "biases"):
                    parts = []
                    for proj in ("q_proj", "k_proj", "v_proj"):
                        key = f"{attn_prefix}.{proj}.{comp}"
                        if key in params:
                            mx.eval(params[key])
                            parts.append(params[key])
                    if parts:
                        fused = mx.concatenate(parts, axis=0)
                        c_name = f"layers.{layer_idx}.self_attn.combined_proj.{comp}"
                        if self._set_one_weight(c_name, fused) == 0:
                            count += 1

        elapsed = time.time() - t0
        print(f"[kandiga-full] Set {count} weights ({skipped} expert skipped) in {elapsed:.1f}s")

        # Build (finalize weight setup)
        ret = self._lib.bakan_full_build(self._engine)
        if ret != 0:
            raise RuntimeError("Failed to build full forward engine")
        print("[kandiga-full] Engine built and ready")

    def forward(self, token_id: int, position: int) -> np.ndarray:
        """Run one decode step. Returns logits as float32 numpy array."""
        vocab_size = ctypes.c_int(0)
        logits = np.empty(248320, dtype=np.float32)  # max vocab

        ret = self._lib.bakan_full_forward(
            self._engine, token_id, position,
            logits.ctypes.data_as(ctypes.c_void_p),
            ctypes.byref(vocab_size),
        )
        if ret != 0:
            raise RuntimeError(f"Forward pass failed at position {position}")

        return logits[:vocab_size.value]

    def reset_cache(self):
        """Reset KV cache and linear attention state for new conversation."""
        self._lib.bakan_full_reset_cache(self._engine)

    def destroy(self):
        if self._engine:
            self._lib.bakan_full_destroy(self._engine)
            self._engine = None

    def __del__(self):
        self.destroy()


def generate_with_full_forward(engine: FullForwardEngine, tokenizer,
                                prompt: str, max_tokens: int = 256,
                                temp: float = 0.0):
    """Generate tokens using the full C/Metal forward pass.

    Yields tokens one at a time (streaming).
    """
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
        )

    engine.reset_cache()

    # Prefill: process prompt tokens
    for i, token_id in enumerate(input_ids):
        logits = engine.forward(token_id, i)

    # Decode: generate new tokens
    position = len(input_ids)
    eos_ids = set()
    if hasattr(tokenizer, 'eos_token_id'):
        eid = tokenizer.eos_token_id
        if isinstance(eid, (list, tuple)):
            eos_ids = set(eid)
        elif eid is not None:
            eos_ids = {eid}

    for _ in range(max_tokens):
        if temp == 0:
            next_token = int(np.argmax(logits))
        else:
            probs = np.exp(logits / temp)
            probs /= probs.sum()
            next_token = int(np.random.choice(len(probs), p=probs))

        if next_token in eos_ids:
            break

        text = tokenizer.decode([next_token])
        yield text

        logits = engine.forward(next_token, position)
        position += 1
