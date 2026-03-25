"""Tests for the expert packing module."""

from __future__ import annotations

import struct


def test_header_format():
    """Header should be exactly 4096 bytes with correct magic."""
    from kandiga._pack_experts import _build_header, HEADER_SIZE

    header = _build_header()
    assert len(header) == HEADER_SIZE
    assert header[:4] == b"BKEX"

    # Parse version
    version = struct.unpack_from("<I", header, 4)[0]
    assert version == 1

    # Parse num_experts
    num_experts = struct.unpack_from("<I", header, 8)[0]
    assert num_experts == 256

    # Parse expert_size
    expert_size = struct.unpack_from("<Q", header, 12)[0]
    assert expert_size == 1_769_472

    # Parse num_tensors
    num_tensors = struct.unpack_from("<I", header, 20)[0]
    assert num_tensors == 9


def test_tensor_nbytes():
    """Tensor byte sizes should be computed correctly."""
    from kandiga._pack_experts import _tensor_nbytes

    # uint32: 4 bytes per element
    assert _tensor_nbytes((512, 256), "uint32") == 512 * 256 * 4

    # bfloat16: 2 bytes per element
    assert _tensor_nbytes((512, 32), "bfloat16") == 512 * 32 * 2


def test_expert_size_consistent():
    """All tensor sizes should sum to EXPERT_SIZE."""
    from kandiga._pack_experts import TENSOR_ORDER, EXPERT_SIZE, _tensor_nbytes

    total = sum(_tensor_nbytes(shape, dtype) for _, shape, dtype in TENSOR_ORDER)
    assert total == EXPERT_SIZE
