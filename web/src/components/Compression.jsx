const contexts = [
  { tokens: '4K', standard: '16.8 MB', kandiga: '4.5 MB', ratio: '3.8x' },
  { tokens: '8K', standard: '33.6 MB', kandiga: '8.9 MB', ratio: '3.8x' },
  { tokens: '16K', standard: '67.1 MB', kandiga: '17.8 MB', ratio: '3.8x', highlight: true },
  { tokens: '32K', standard: '134 MB', kandiga: '35.3 MB', ratio: '3.8x' },
]

function ContextCard({ row }) {
  return (
    <div className={`rounded-lg p-3 border ${row.highlight
      ? 'bg-[var(--card)] border-[var(--cyan)]/30'
      : 'bg-[var(--surface)] border-[var(--card-border)]'}`}>
      <div className="text-lg font-bold text-[var(--text-bright)] mb-2">{row.tokens} <span className="text-xs font-normal text-[var(--text-muted)]">tokens</span></div>
      <div className="space-y-1.5 text-[13px]">
        <div className="flex justify-between">
          <span className="text-[11px] text-[var(--text-muted)]">Standard</span>
          <span className="text-[var(--text)]">{row.standard}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-[11px] text-[var(--cyan)] font-medium">Kandiga</span>
          <span className="text-[var(--cyan)] font-medium">{row.kandiga}</span>
        </div>
      </div>
    </div>
  )
}

export default function Compression() {
  return (
    <section>
      <h2 className="text-center text-2xl md:text-3xl font-semibold text-[var(--text-bright)] mb-4 tracking-tight">
        Longer conversations
      </h2>
      <p className="text-center text-[var(--text-muted)] mb-4 max-w-2xl mx-auto">
        Every token you generate grows a memory buffer called the <span className="text-[var(--text)]">KV cache</span>.
        On a 16GB Mac, this normally limits you to ~4K tokens before you run out of RAM.
      </p>
      <p className="text-center text-[var(--text-muted)] mb-10 max-w-2xl mx-auto">
        Kandiga implements <span className="text-[var(--text)]">TurboQuant</span> — a compression algorithm from Google Research
        that rotates and quantizes the cache to <span className="text-[var(--cyan)]">3 bits</span> with
        only 4% quality loss. Same answers, 3.8x less memory.
      </p>

      {/* Mobile: cards */}
      <div className="grid grid-cols-2 gap-3 md:hidden">
        {contexts.map((row, i) => (
          <ContextCard key={i} row={row} />
        ))}
      </div>

      {/* Desktop: table */}
      <div className="hidden md:block feature-card max-w-2xl mx-auto">
        <h3 className="text-lg font-semibold text-[var(--text-bright)] mb-5">KV cache memory by context length</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--card-border)]">
              <th className="pb-3 text-left text-[var(--text-muted)] font-medium">Context</th>
              <th className="pb-3 text-left text-[var(--text-muted)] font-medium">Standard (float16)</th>
              <th className="pb-3 text-left text-[var(--cyan)] font-medium">Kandiga (3-bit)</th>
              <th className="pb-3 text-left text-[var(--text-muted)] font-medium">Savings</th>
            </tr>
          </thead>
          <tbody>
            {contexts.map((row, i) => (
              <tr key={i} className="border-b border-[var(--card-border)]/30 last:border-0">
                <td className="py-3 text-[var(--text-bright)] font-medium">{row.tokens} tokens</td>
                <td className="py-3 text-[var(--text)]">{row.standard}</td>
                <td className="py-3 text-[var(--cyan)] font-medium">{row.kandiga}</td>
                <td className="py-3 text-[var(--text-muted)]">{row.ratio}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Explainer */}
      <div className="mt-8 grid sm:grid-cols-3 gap-4 text-center max-w-2xl mx-auto">
        {[
          { value: '96%', label: 'Cosine similarity', desc: 'Attention scores preserved' },
          { value: '3 bits', label: 'Per element', desc: 'Down from 16 bits (float16)' },
          { value: '0%', label: 'Speed overhead', desc: 'Rotation is negligible vs I/O' },
        ].map((item, i) => (
          <div key={i} className="feature-card text-center">
            <div className="text-lg font-bold text-[var(--cyan)] mb-1">{item.value}</div>
            <div className="text-xs sm:text-sm text-[var(--text-bright)] mb-1">{item.label}</div>
            <div className="text-[10px] sm:text-xs text-[var(--text-muted)]">{item.desc}</div>
          </div>
        ))}
      </div>
    </section>
  )
}
