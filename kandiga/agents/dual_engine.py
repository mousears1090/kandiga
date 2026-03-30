"""Dual-model engine — 4B fast worker + 35B brain.

The 4B model handles mechanical work at 34 tok/s:
  - Tool call JSON generation
  - Query classification
  - Content generation for file writes

The 35B MoE handles reasoning via SEM at 3-6 tok/s:
  - Planning multi-step tasks
  - Final synthesis with tool results
  - Verification of complex outputs

Result: tool queries drop from ~76s (3x 35B calls) to ~15s (2x 4B + 1x 35B).
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from kandiga.engine import KandigaEngine


class DualEngine:
    """Manages 4B + 35B models simultaneously.

    Both share the same GPU. The 4B is a standard dense model loaded
    via mlx_lm. The 35B uses Kandiga's SEM engine (experts on SSD).
    Total RAM: ~4GB (1.5GB shared 35B + 2.5GB dense 4B).
    """

    FAST_MODEL = "mlx-community/Qwen3.5-4B-4bit"
    BRAIN_MODEL = "mlx-community/Qwen3.5-35B-A3B-4bit"

    def __init__(self, fast_mode: bool = False, log_memory: bool = False):
        self._brain = KandigaEngine(
            model_path=self.BRAIN_MODEL,
            fast_mode=fast_mode,
            log_memory=log_memory,
        )
        self._fast_model = None
        self._fast_tokenizer = None
        self._ready = False
        self._log_memory = log_memory

    def load(self):
        """Load both models."""
        if self._ready:
            return

        # Load 35B brain via SEM
        self._brain.load()

        # Load 4B fast model via standard mlx_lm
        from mlx_lm import load as mlx_load
        self._fast_model, self._fast_tokenizer = mlx_load(self.FAST_MODEL)

        self._ready = True

    @property
    def brain(self) -> KandigaEngine:
        return self._brain

    @property
    def tokenizer(self):
        return self._brain._tokenizer

    def generate_fast(
        self,
        system: str,
        user: str,
        max_tokens: int = 1200,
        temp: float = 0.0,
    ) -> str:
        """Generate with the 4B model (fast, ~34 tok/s).

        Use for: tool call JSON, classification, content generation.
        """
        if not self._ready:
            self.load()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            formatted = self._fast_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            formatted = self._fast_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        from mlx_lm import generate as mlx_generate
        from mlx_lm.generate import make_sampler

        sampler = make_sampler(temp=temp)
        response = mlx_generate(
            self._fast_model, self._fast_tokenizer,
            prompt=formatted, max_tokens=max_tokens,
            sampler=sampler, verbose=False,
        )
        return _strip_thinking(response)

    def generate_brain(
        self,
        system: str,
        user: str,
        max_tokens: int = 2048,
        temp: float = 0.0,
    ) -> str:
        """Generate with the 35B model (smart, ~3-6 tok/s).

        Use for: planning, synthesis, complex reasoning.
        """
        if not self._ready:
            self.load()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            formatted = self._brain._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            formatted = self._brain._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        return self._brain.generate(
            formatted, max_tokens=max_tokens, temp=temp, stream=False,
        )

    def generate_brain_stream(
        self,
        system: str,
        user: str,
        max_tokens: int = 2048,
        temp: float = 0.0,
    ):
        """Stream from the 35B model. Yields tokens."""
        if not self._ready:
            self.load()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            formatted = self._brain._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            formatted = self._brain._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        for token in self._brain.generate(
            formatted, max_tokens=max_tokens, temp=temp, stream=True,
        ):
            yield token

    # --- Session (persistent KV on the 35B) ---

    def start_session(self):
        self._brain.start_session()

    def end_session(self):
        self._brain.end_session()

    def save_session(self, path: str):
        self._brain.save_session(path)

    def load_session(self, path: str):
        self._brain.load_session(path)

    def session_generate(self, user_message: str, max_tokens: int = 2048, temp: float = 0.0):
        """Generate within persistent session (constant-time follow-ups)."""
        return self._brain.session_generate(user_message, max_tokens=max_tokens, temp=temp)

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def stats(self) -> Dict[str, Any]:
        s = self._brain.stats
        s["dual_mode"] = True
        s["fast_model"] = self.FAST_MODEL
        s["brain_model"] = self.BRAIN_MODEL
        return s

    def __del__(self):
        if hasattr(self, '_brain'):
            del self._brain


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    return text
