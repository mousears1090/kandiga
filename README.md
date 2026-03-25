# Kandiga

Run 35B AI models in 1.5GB of RAM. Any Mac.

Kandiga is an open-source MoE inference engine that uses **Selective Expert Materialization** to run models that would normally require 20GB+ of memory in under 2GB on any Apple Silicon Mac.

## How it works

Large MoE (Mixture of Experts) models like Qwen3.5-35B-A3B have 256 experts per layer, but only activate 8 per token. Kandiga exploits this sparsity:

1. **Shared layers** (attention, norms, embeddings) load to GPU memory (~1.5GB)
2. **Expert MLP weights** stay on disk in packed binary files (~17GB SSD)
3. **Per token**: the router selects 8 experts, which are read from SSD via `pread`
4. **CPU computes** expert MLP with NEON-vectorized 4-bit dequant + GCD parallelism
5. **GPU computes** attention simultaneously via MLX (unified memory, zero copy)

This is the [KTransformers](https://github.com/kvcache-ai/ktransformers) architecture adapted for Apple Silicon's unified memory.

## Install

```bash
pip install kandiga
```

Requirements: macOS with Apple Silicon (M1/M2/M3/M4), Python 3.10+

## Quick start

```bash
# One-time setup: download model + prepare expert files (~20 min)
kandiga setup

# Interactive chat
kandiga chat

# Fast mode (K=4 experts instead of 8, ~2x speed, slightly less quality)
kandiga chat --fast

# One-shot prompt
kandiga "What is the capital of France?"

# Start an OpenAI-compatible API server
kandiga serve

# Run benchmarks
kandiga bench
```

## Benchmarks

Measured on M4 Mac Mini (16GB), Qwen3.5-35B-A3B-4bit:

| Mode | Experts | Speed | RAM | Quality |
|------|---------|-------|-----|---------|
| Quality (K=8) | 8/256 per layer | ~3.5 tok/s | 1.5GB | Full |
| Fast (K=4) | 4/256 per layer | ~6.5 tok/s | 1.5GB | Near-equal |

For comparison, loading the full model requires 20.4GB of RAM and MLX alone achieves ~25 tok/s when it fits in memory. Kandiga trades speed for accessibility: if your Mac has 8-16GB of RAM, you can now run a 35B model that previously required 24GB+.

## KV Cache Compression (TurboQuant)

Every token generated grows a memory buffer called the **KV cache**. On a 16GB Mac, this normally limits conversations to ~4K tokens before RAM runs out.

Kandiga implements **TurboQuant** (based on [Google Research](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)) — a compression algorithm that shrinks the KV cache from 16-bit to 3-bit per element with only 4% quality loss:

1. **PolarQuant** — randomly rotates vectors to spread information, then quantizes to 3 bits
2. **QJL** — 1-bit error correction using the Johnson-Lindenstrauss transform

| Context Length | Standard (float16) | Kandiga (3-bit) | Compression |
|---------------|-------------------|-----------------|-------------|
| 4K tokens | 16.8 MB | 4.5 MB | 3.8x |
| 8K tokens | 33.6 MB | 8.9 MB | 3.8x |
| 16K tokens | 67.1 MB | 17.8 MB | 3.8x |
| 32K tokens | 134 MB | 35.3 MB | 3.8x |

- **96% cosine similarity** — attention scores stay accurate
- **Zero speed overhead** — rotation cost is negligible vs expert I/O
- **No configuration needed** — compression is automatic

## Architecture

```
User prompt
    |
    v
[Tokenizer + Chat Template]
    |
    v
[MLX Forward Pass]
    |
    +---> GPU: Attention + Norms + Router + Shared Expert + Blending
    |
    +---> CPU: Routed Expert MLP (NEON 4-bit dequant + GCD parallel)
    |         |
    |         +-- pread expert weights from SSD (OS page cache)
    |         +-- gate_proj matvec (512x2048)
    |         +-- up_proj matvec (512x2048)
    |         +-- SwiGLU activation
    |         +-- down_proj matvec (2048x512)
    |
    v
[Token Output]
```

Both CPU and GPU operate on the same physical DRAM (Apple Silicon unified memory), so there is zero data transfer overhead between them.

## API Server

Kandiga includes an OpenAI-compatible HTTP API:

```bash
kandiga serve --port 8340
```

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8340/v1", api_key="unused")
response = client.chat.completions.create(
    model="mlx-community/Qwen3.5-35B-A3B-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

## Project structure

```
kandiga/
  __init__.py          # Package version
  cli.py               # CLI entry point (argparse)
  engine.py            # Core inference engine (SEM)
  chat.py              # Interactive chat (Rich terminal UI)
  serve.py             # OpenAI-compatible HTTP API (FastAPI)
  bench.py             # Benchmarking suite
  setup.py             # Model download + expert splitting + packing
  _split_experts.py    # Split stacked weights into per-expert files
  _pack_experts.py     # Pack per-expert files into binary format
  _build.py            # Compile CPU expert dylib from source
  metal/
    kandiga_cpu_expert.h   # C API header
    kandiga_cpu_expert.m   # NEON + GCD implementation
    Makefile               # Build the dylib
  tools/
    __init__.py            # Future: web search, file access
scripts/
  install.sh           # Quick install script
tests/
  ...
```

## Development

```bash
# Clone
git clone https://github.com/kantheon/kandiga.git
cd kandiga

# Install in development mode
pip install -e ".[serve]"

# Build the CPU expert library
cd kandiga/metal && make && cd ../..

# Run tests
pytest tests/ -v
```

## License

MIT
