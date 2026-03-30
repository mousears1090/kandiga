"""Dual-model chat: 4B responds fast, 35B verifies after.
Type anything — medical notes, invoices, questions. Type 'quit' to exit."""
import time
import sys
from kandiga.engine import KandigaEngine
import mlx.core as mx

mx.set_cache_limit(256 * 1024 * 1024)

print("\033[36m╔══════════════════════════════════════════════════╗\033[0m")
print("\033[36m║           KANDIGA DUAL-MODEL CHAT                ║\033[0m")
print("\033[36m║   4B writes fast → 35B verifies after            ║\033[0m")
print("\033[36m╚══════════════════════════════════════════════════╝\033[0m")
print()

print("Loading 4B writer...", end=" ", flush=True)
writer = KandigaEngine(model_path="mlx-community/Qwen3.5-4B-4bit", fast_mode=True)
writer.load()
writer.start_session()
print("ready.")

print("Loading 35B verifier...", end=" ", flush=True)
verifier = KandigaEngine(fast_mode=True)
verifier.load()
verifier.start_session()
for _ in verifier.session_generate(
    "You are an assistant. When verifying, check facts and say VERIFIED or state errors. "
    "When asked questions, answer briefly and accurately.", max_tokens=5):
    pass
print("ready.")
print()
print("Type your message. 4B responds, then 35B verifies.")
print("Type \033[33mquit\033[0m to exit.\n")

while True:
    try:
        user_input = input("\033[36mYou:\033[0m ")
    except (EOFError, KeyboardInterrupt):
        break

    if user_input.strip().lower() in ("quit", "exit", "q"):
        break
    if not user_input.strip():
        continue

    # 4B responds
    t0 = time.time()
    first_token_time = None
    response = ""
    token_count = 0

    print("\033[37m 4B:\033[0m ", end="", flush=True)
    for token in writer.session_generate(user_input, max_tokens=300):
        if first_token_time is None:
            first_token_time = time.time()
        response += token
        token_count += 1
        print(token, end="", flush=True)

    total = time.time() - t0
    ttft = (first_token_time - t0) if first_token_time else total
    decode_time = (time.time() - first_token_time) if first_token_time else 0
    toks = token_count / decode_time if decode_time > 0 else 0
    print(f"\n\033[90m  [{ttft:.1f}s TTFT | {token_count} tok | {toks:.0f} tok/s]\033[0m")

    # 35B verifies (sequential, not background — MLX isn't thread-safe)
    mx.clear_cache()
    print("\033[90m  35B checking...\033[0m", end=" ", flush=True)
    t0v = time.time()

    # Feed context to 35B
    for _ in verifier.session_generate(
        f"User said: {user_input[:200]}\nAI responded: {response[:200]}", max_tokens=3):
        pass

    # Verify
    verify = ""
    for token in verifier.session_generate(
        "Is the AI response accurate? VERIFIED or state errors briefly.", max_tokens=40):
        verify += token

    vtime = time.time() - t0v
    verify_clean = verify.strip()[:80]

    if "VERIFIED" in verify_clean.upper():
        print(f"\033[32m✓ {verify_clean} ({vtime:.1f}s)\033[0m")
    else:
        print(f"\033[33m⚠ {verify_clean} ({vtime:.1f}s)\033[0m")
    print()

print("\n\033[36mSession ended.\033[0m")
writer.end_session()
verifier.end_session()
