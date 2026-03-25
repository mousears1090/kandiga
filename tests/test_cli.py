"""Tests for the Kandiga CLI."""

from __future__ import annotations

import subprocess
import sys


def test_cli_help():
    """kandiga --help should print usage and exit 0."""
    result = subprocess.run(
        [sys.executable, "-m", "kandiga.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Kandiga" in result.stdout


def test_cli_version_importable():
    """kandiga.__version__ should be importable."""
    from kandiga import __version__
    assert __version__ == "0.1.0"


def test_cli_setup_help():
    """kandiga setup --help should work."""
    result = subprocess.run(
        [sys.executable, "-m", "kandiga.cli", "setup", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "model" in result.stdout.lower()


def test_cli_chat_help():
    """kandiga chat --help should work."""
    result = subprocess.run(
        [sys.executable, "-m", "kandiga.cli", "chat", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "fast" in result.stdout.lower()


def test_cli_serve_help():
    """kandiga serve --help should work."""
    result = subprocess.run(
        [sys.executable, "-m", "kandiga.cli", "serve", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "port" in result.stdout.lower()


def test_cli_bench_help():
    """kandiga bench --help should work."""
    result = subprocess.run(
        [sys.executable, "-m", "kandiga.cli", "bench", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
