A 122 billion parameter AI model that remembers your entire conversation, running locally on a MacBook in 4GB of RAM. No cloud. No API. No data ever leaving the device.

Not a hypothetical. Shipping now.

Every local LLM has the same fundamental limitation — they re-read your entire conversation on every single message. Turn 30 re-processes turns 1 through 29 from scratch. The longer you talk, the slower it gets. Until it's unusable.

That problem is now solved.

Persistent KV cache. The model reads a conversation once. Every follow-up only processes the new message.

Turn 2: 3 seconds.
Turn 30: 3 seconds.
Turn 200: 3 seconds.

Close the laptop Friday. Open it Monday. The conversation picks up exactly where it left off. No re-processing. Instant.

The benchmarks:
- 35B model: 6.3 tokens/sec, 2GB RAM
- 122B model: running in 4GB RAM, 5/5 accuracy on multi-turn recall
- 397B model: 8GB RAM
- 3 second follow-ups regardless of conversation length
- Session save and resume in 0.2 seconds
- Custom Metal GPU kernels for 3-5x faster prompt processing

But here's where it gets real.

Clinical documentation is one of the biggest time burdens in healthcare. Nurses and providers spend hours after every shift charting — time that could be spent on patient care. A frontier-class model running locally on a hospital workstation changes that equation entirely. Assessments, SOAP notes, clinical summaries — generated and organized without any patient data ever leaving the device.

No cloud endpoints. No third-party data processing agreements. No attack surface. The data lives and dies on the machine it was created on.

When a 122B model runs locally with persistent memory, it becomes a clinical tool that maintains full context across an entire shift — without sending a single byte off the device.

Open source. MIT licensed. Apple Silicon.

pip install kandiga
https://github.com/kantheon/kandiga
