"""Pack per-expert safetensors into single raw binary files per layer.

Reads the split per-expert safetensors files and packs them into a compact
binary format that can be read with zero parsing overhead using pread().

Binary format per file:
  Header (4096 bytes):
    magic:       4 bytes  "BKEX"
    version:     uint32   1
    num_experts: uint32   256
    expert_size: uint64   1769472 (bytes per expert block)
    num_tensors: uint32   9
    tensor descriptors...
    padding to 4096 bytes

  Expert data (256 x 1769472 bytes):
    expert_000: [gate.weight][gate.scales][gate.biases]
                [up.weight][up.scales][up.biases]
                [down.weight][down.scales][down.biases]
    ...
    expert_255: same layout
"""

from __future__ import annotations

import os
import struct
import time

import mlx.core as mx
import numpy as np

NUM_EXPERTS = 256
NUM_MOE_LAYERS = 40
HEADER_SIZE = 4096
EXPERT_SIZE = 1_769_472  # 1728KB exactly

# Tensor order must match the C library's byte offsets
TENSOR_ORDER = [
    ("gate_proj.weight", (512, 256), "uint32"),
    ("gate_proj.scales", (512, 32), "bfloat16"),
    ("gate_proj.biases", (512, 32), "bfloat16"),
    ("up_proj.weight", (512, 256), "uint32"),
    ("up_proj.scales", (512, 32), "bfloat16"),
    ("up_proj.biases", (512, 32), "bfloat16"),
    ("down_proj.weight", (2048, 64), "uint32"),
    ("down_proj.scales", (2048, 8), "bfloat16"),
    ("down_proj.biases", (2048, 8), "bfloat16"),
]


def _tensor_nbytes(shape: tuple, dtype_str: str) -> int:
    """Calculate raw byte size for a tensor."""
    itemsize = 4 if dtype_str == "uint32" else 2
    return shape[0] * shape[1] * itemsize


def _build_header() -> bytes:
    """Build the 4096-byte binary header."""
    buf = bytearray(HEADER_SIZE)

    buf[0:4] = b"BKEX"
    struct.pack_into("<I", buf, 4, 1)  # version
    struct.pack_into("<I", buf, 8, NUM_EXPERTS)
    struct.pack_into("<Q", buf, 12, EXPERT_SIZE)
    struct.pack_into("<I", buf, 20, len(TENSOR_ORDER))

    offset_in_expert = 0
    pos = 24
    for name, shape, dtype_str in TENSOR_ORDER:
        nbytes = _tensor_nbytes(shape, dtype_str)
        dtype_code = 0 if dtype_str == "uint32" else 1

        name_bytes = name.encode("ascii")
        struct.pack_into("<B", buf, pos, len(name_bytes))
        pos += 1
        buf[pos: pos + len(name_bytes)] = name_bytes
        pos += 24
        struct.pack_into("<I", buf, pos, offset_in_expert)
        pos += 4
        struct.pack_into("<I", buf, pos, nbytes)
        pos += 4
        struct.pack_into("<I", buf, pos, shape[0])
        pos += 4
        struct.pack_into("<I", buf, pos, shape[1])
        pos += 4
        struct.pack_into("<B", buf, pos, dtype_code)
        pos += 1

        offset_in_expert += nbytes

    assert offset_in_expert == EXPERT_SIZE, (
        f"Tensor sizes don't sum to EXPERT_SIZE: {offset_in_expert} != {EXPERT_SIZE}"
    )
    return bytes(buf)


def _expert_to_bytes(tensors: dict[str, mx.array]) -> bytes:
    """Convert an expert's tensor dict to raw bytes in canonical order."""
    parts = []
    for name, shape, dtype_str in TENSOR_ORDER:
        tensor = tensors[name]
        mx.eval(tensor)

        if dtype_str == "uint32":
            np_arr = np.array(tensor, copy=False)
            raw = np_arr.tobytes()
        else:
            u16 = tensor.view(mx.uint16)
            mx.eval(u16)
            np_arr = np.array(u16, copy=False)
            raw = np_arr.tobytes()

        expected = _tensor_nbytes(shape, dtype_str)
        assert len(raw) == expected, f"{name}: got {len(raw)} bytes, expected {expected}"
        parts.append(raw)

    data = b"".join(parts)
    assert len(data) == EXPERT_SIZE, f"Expert data {len(data)} != {EXPERT_SIZE}"
    return data


def _pack_layer(layer_idx: int, input_dir: str, output_dir: str) -> None:
    """Pack all 256 experts for one layer into a single binary file."""
    layer_dir = os.path.join(input_dir, f"layer_{layer_idx:02d}")
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

    header = _build_header()

    with open(out_path, "wb") as f:
        f.write(header)
        for expert_idx in range(NUM_EXPERTS):
            st_path = os.path.join(layer_dir, f"expert_{expert_idx:03d}.safetensors")
            tensors = mx.load(st_path)
            mx.eval(*tensors.values())
            raw = _expert_to_bytes(tensors)
            f.write(raw)
            del tensors

    # Verify file size
    expected_size = HEADER_SIZE + NUM_EXPERTS * EXPERT_SIZE
    actual_size = os.path.getsize(out_path)
    assert actual_size == expected_size, (
        f"File size mismatch: {actual_size} != {expected_size}"
    )


def pack_experts(
    input_dir: str,
    output_dir: str,
    num_layers: int = NUM_MOE_LAYERS,
) -> None:
    """Pack all layers from split expert files into binary format."""
    os.makedirs(output_dir, exist_ok=True)

    total_start = time.time()
    for layer_idx in range(num_layers):
        layer_start = time.time()
        print(f"  Packing layer {layer_idx:2d}/{num_layers - 1}...", end=" ", flush=True)
        _pack_layer(layer_idx, input_dir, output_dir)
        elapsed = time.time() - layer_start
        total_elapsed = time.time() - total_start
        eta = (total_elapsed / (layer_idx + 1)) * (num_layers - layer_idx - 1)
        print(f"done ({elapsed:.1f}s, ETA {eta:.0f}s)")

    total_elapsed = time.time() - total_start
    print(f"  {num_layers} layer files packed in {total_elapsed:.1f}s")
