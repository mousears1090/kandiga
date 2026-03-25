export default function Architecture() {
  return (
    <section id="architecture">
      <h2 className="text-center text-2xl md:text-3xl font-semibold text-[var(--text-bright)] mb-4 tracking-tight">
        How it works
      </h2>
      <p className="text-center text-[var(--text-muted)] mb-12 max-w-xl mx-auto">
        Selective Expert Materialization splits the model into shared layers (always in memory) and expert weights (loaded on demand from disk).
      </p>

      {/* Architecture diagram */}
      <div className="terminal">
        <div className="terminal-bar">
          <div className="terminal-dot bg-[#ff5f57]" />
          <div className="terminal-dot bg-[#febc2e]" />
          <div className="terminal-dot bg-[#28c840]" />
          <span className="ml-3 text-xs text-[var(--text-muted)]">architecture</span>
        </div>
        <div className="terminal-body text-[10px] sm:text-xs md:text-sm leading-relaxed overflow-x-auto flex justify-center">
          <pre className="text-[var(--text-muted)] inline-block text-left">{`  Token Input
      │
      ▼
 ┌──────────────────────────────────────┐
 │  GPU (MLX)                           │
 │  ┌──────────────────────────────┐    │
 │  │  Attention + Norms + Routing │    │
 │  │  (1.5GB in memory, always)   │    │
 │  └──────────────┬───────────────┘    │
 │                 │                     │
 │          Router selects 8/256        │
 │          expert indices              │
 └─────────────────┼────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    │              ▼              │
 ┌──┴─────┐  ┌──────────┐  ┌────┴───┐
 │Expert  │  │ Expert   │  │Expert  │
 │#42     │  │ #128     │  │#201    │
 │1.7MB   │  │ 1.7MB    │  │1.7MB   │
 └──┬─────┘  └────┬─────┘  └────┬───┘
    │              │             │
    └──────────────┼─────────────┘
                   │
 ┌─────────────────┼────────────────────┐
 │  CPU (NEON)     ▼                    │
 │  ┌──────────────────────────────┐    │
 │  │  4-bit dequant + SwiGLU     │    │
 │  │  GCD across P-cores         │    │
 │  └──────────────────────────────┘    │
 └─────────────────┼────────────────────┘
                   │
                   ▼
              Next Layer (×40)
                   │
                   ▼
              Output Token`}</pre>
        </div>
      </div>

      {/* Key insight */}
      <div className="mt-8 grid grid-cols-3 gap-3 sm:gap-4 text-center">
        {[
          { label: 'In memory', value: '1.5GB', desc: 'Shared attention layers' },
          { label: 'Per token from disk', value: '14MB', desc: '8 experts × 40 layers' },
          { label: 'On disk (never in RAM)', value: '18.9GB', desc: '10,240 expert files' },
        ].map((item, i) => (
          <div key={i} className="feature-card text-center">
            <div className="text-lg sm:text-xl font-bold text-[var(--cyan)] mb-1">{item.value}</div>
            <div className="text-xs sm:text-sm text-[var(--text-bright)] mb-1">{item.label}</div>
            <div className="text-[10px] sm:text-xs text-[var(--text-muted)]">{item.desc}</div>
          </div>
        ))}
      </div>
    </section>
  )
}
