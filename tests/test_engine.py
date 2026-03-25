"""Tests for the Kandiga engine module."""

from __future__ import annotations

import os
import pytest


def test_engine_importable():
    """Engine module should import without errors."""
    from kandiga.engine import KandigaEngine


def test_engine_defaults():
    """Engine should have sensible defaults."""
    from kandiga.engine import KandigaEngine

    engine = KandigaEngine()
    assert engine.model_path == KandigaEngine.DEFAULT_MODEL
    assert engine.fast_mode is False
    assert engine.is_ready is False


def test_engine_fast_mode():
    """Engine should accept fast_mode parameter."""
    from kandiga.engine import KandigaEngine

    engine = KandigaEngine(fast_mode=True)
    assert engine.fast_mode is True


def test_engine_custom_model():
    """Engine should accept custom model path."""
    from kandiga.engine import KandigaEngine

    engine = KandigaEngine(model_path="custom/model-path")
    assert engine.model_path == "custom/model-path"


def test_engine_stats_before_load():
    """Stats should work even before loading."""
    from kandiga.engine import KandigaEngine

    engine = KandigaEngine()
    stats = engine.stats
    assert "rss_mb" in stats
    assert "model" in stats
    assert "mode" in stats


def test_engine_cache_dir():
    """Cache dir should be derived from model name."""
    from kandiga.engine import KandigaEngine

    engine = KandigaEngine(model_path="mlx-community/Qwen3.5-35B-A3B-4bit")
    cache_dir = engine._model_cache_dir()
    assert cache_dir.endswith("Qwen3.5-35B-A3B-4bit")
    assert ".kandiga/experts" in cache_dir


def test_strip_thinking():
    """Thinking tag stripping should work correctly."""
    from kandiga.engine import _strip_thinking

    # Full think block
    assert _strip_thinking("<think>internal</think>Hello") == "Hello"

    # Only closing tag (opening was in prompt)
    assert _strip_thinking("thinking...</think>Answer") == "Answer"

    # No tags
    assert _strip_thinking("Just an answer") == "Just an answer"

    # Empty with thinking enabled (model exhausted tokens)
    assert _strip_thinking("still thinking...", thinking_enabled=True) == ""


def test_find_layers_error():
    """_find_layers should raise on unknown model structure."""
    from kandiga.engine import _find_layers
    import mlx.nn as nn

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()

    with pytest.raises(AttributeError):
        _find_layers(FakeModel())


def test_mem_stats():
    """mem_stats should return rss and gpu memory."""
    from kandiga.engine import mem_stats

    stats = mem_stats()
    assert "rss_mb" in stats
    assert "gpu_active_mb" in stats
    assert stats["rss_mb"] > 0
