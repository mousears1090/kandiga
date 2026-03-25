export default function Footer() {
  return (
    <footer className="py-16 px-6 text-center border-t border-[var(--card-border)]">
      <div className="max-w-4xl mx-auto">
        {/* Supported models */}
        <div className="mb-12">
          <h3 className="text-sm font-medium text-[var(--text-muted)] uppercase tracking-widest mb-4">
            Supported Models
          </h3>
          <div className="flex flex-wrap justify-center gap-3">
            {[
              { name: 'Qwen3.5-35B-A3B', status: 'default' },
              { name: 'Qwen3.5-397B-A17B', status: 'coming' },
              { name: 'Any MoE model', status: 'planned' },
            ].map((m, i) => (
              <span key={i} className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-[var(--card-border)] bg-[var(--card)] text-xs">
                <span className={`w-1.5 h-1.5 rounded-full ${
                  m.status === 'default' ? 'bg-[var(--green)]' :
                  m.status === 'coming' ? 'bg-yellow-500' : 'bg-[var(--text-muted)]'
                }`} />
                <span className="text-[var(--text)]">{m.name}</span>
              </span>
            ))}
          </div>
        </div>

        <div className="section-divider mb-8" />

        <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-[var(--text-muted)]">
          <div className="flex items-center gap-2">
            <img src="/logo.png" alt="Kandiga" className="w-5 h-5 rounded" />
            <span>Built by <a href="https://kantheon.com" target="_blank" rel="noopener noreferrer" className="text-[var(--text)] hover:text-[var(--cyan)] transition-colors no-underline">Kantheon</a></span>
            <span>·</span>
            <span>MIT License</span>
            <span>·</span>
            <span>2026</span>
          </div>

          <div className="flex items-center gap-6">
            <a href="https://github.com/kantheon/kandiga" target="_blank" rel="noopener noreferrer"
               className="hover:text-[var(--text)] transition-colors no-underline">
              GitHub
            </a>
            <a href="https://pypi.org/project/kandiga" target="_blank" rel="noopener noreferrer"
               className="hover:text-[var(--text)] transition-colors no-underline">
              PyPI
            </a>
            <a href="https://kandiga.com" target="_blank" rel="noopener noreferrer"
               className="hover:text-[var(--text)] transition-colors no-underline">
              kandiga.com
            </a>
          </div>
        </div>
      </div>
    </footer>
  )
}
