import { useState } from 'react'

function CodeBlock({ children }) {
  const [copied, setCopied] = useState(false)
  const text = typeof children === 'string' ? children : children.join('\n')

  return (
    <div className="bg-[#0a0a12] border border-[var(--card-border)] rounded-lg px-4 py-3 mt-3 font-mono text-xs sm:text-sm text-[var(--text)]">
      <div className="flex items-start justify-between gap-3">
        <pre className="whitespace-pre-wrap break-all min-w-0">{children}</pre>
        <button
          className="shrink-0 bg-[var(--card)] border border-[var(--card-border)] text-[var(--text-muted)] px-2 py-0.5 rounded text-xs cursor-pointer hover:text-[var(--text)] hover:border-[var(--cyan)] transition-all"
          onClick={() => { navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 2000) }}
        >
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
    </div>
  )
}

const steps = [
  {
    num: 1,
    title: 'Install',
    code: 'pip install kandiga',
    note: 'Requires Python 3.10+ and Apple Silicon Mac.',
  },
  {
    num: 2,
    title: 'Setup',
    code: 'kandiga setup',
    note: 'Downloads model from HuggingFace (~20GB), splits expert weights, compiles Metal library. One-time, ~25 minutes.',
  },
  {
    num: 3,
    title: 'Chat',
    code: 'kandiga chat',
    note: 'Full interactive CLI with streaming, markdown rendering, and memory stats.',
  },
  {
    num: 4,
    title: 'Serve (optional)',
    code: 'kandiga serve',
    note: 'OpenAI-compatible API at localhost:8340. Works with Cursor, Continue.dev, or any OpenAI SDK client.',
  },
]

export default function GetStarted() {
  return (
    <section id="install" className="max-w-2xl mx-auto mb-20">
      <h2 className="text-center text-2xl md:text-3xl font-semibold text-[var(--text-bright)] mb-12 tracking-tight">
        Get started in 60 seconds
      </h2>

      <div className="space-y-8">
        {steps.map((step) => (
          <div key={step.num} className="flex gap-3 sm:gap-5">
            <div className="flex-shrink-0 w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-[var(--cyan-dim)] border border-[rgba(34,211,238,0.2)] text-[var(--cyan)] flex items-center justify-center text-xs sm:text-sm font-semibold">
              {step.num}
            </div>
            <div className="flex-1 min-w-0 -mt-0.5">
              <h3 className="text-[var(--text-bright)] font-medium text-base sm:text-lg mb-1">{step.title}</h3>
              <CodeBlock>{step.code}</CodeBlock>
              <p className="text-xs text-[var(--text-muted)] mt-2">{step.note}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Fast mode callout */}
      <div className="mt-14 feature-card">
        <div className="flex items-start gap-3 sm:gap-4">
          <span className="text-xl shrink-0 mt-0.5">⚡</span>
          <div className="flex-1 min-w-0">
            <h4 className="text-[var(--text-bright)] font-medium mb-2">Want more speed?</h4>
            <p className="text-sm text-[var(--text-muted)] mb-4 leading-relaxed">
              Add <code className="text-[var(--cyan)] bg-[var(--cyan-dim)] px-1.5 py-0.5 rounded text-xs">--fast</code> to
              any command. Uses 4 experts instead of 8 — 1.7× faster with near-identical quality.
            </p>
            <CodeBlock>kandiga chat --fast  # 6.5 tok/s</CodeBlock>
          </div>
        </div>
      </div>
    </section>
  )
}
