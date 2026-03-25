import { useEffect, useState, useRef } from 'react'

const DEMO_LINES = [
  { type: 'prompt', text: 'kandiga chat --fast' },
  { type: 'blank' },
  { type: 'header', text: '  Kandiga — 35B AI in 1.5GB RAM · ⚡ Fast mode (K=4)' },
  { type: 'dim', text: '  Type your message. /quit to exit.' },
  { type: 'blank' },
  { type: 'user', text: '› What would happen if the moon disappeared?' },
  { type: 'blank' },
  { type: 'response', text: '  If the Moon were to suddenly disappear, the consequences' },
  { type: 'response', text: '  would be immediate, dramatic, and long-lasting, affecting' },
  { type: 'response', text: '  Earth\'s rotation, tides, climate, and evolution of life.' },
  { type: 'response', text: '' },
  { type: 'response', text: '  **Immediate Effects:**' },
  { type: 'response', text: '  • Tides weaken dramatically (Sun-only: ~1/3 strength)' },
  { type: 'response', text: '  • Night sky becomes significantly darker' },
  { type: 'response', text: '  • Earth\'s axial tilt destabilizes over millennia' },
  { type: 'blank' },
  { type: 'stats', text: '  144 tokens · 22.0s · 6.5 tok/s' },
]

export default function Terminal() {
  const [visibleLines, setVisibleLines] = useState(0)
  const ref = useRef(null)
  const started = useRef(false)

  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting && !started.current) {
        started.current = true
        let i = 0
        const timer = setInterval(() => {
          i++
          setVisibleLines(i)
          if (i >= DEMO_LINES.length) clearInterval(timer)
        }, 120)
      }
    }, { threshold: 0.3 })

    if (ref.current) observer.observe(ref.current)
    return () => observer.disconnect()
  }, [])

  const lineColor = {
    prompt: 'text-[var(--text-bright)]',
    header: 'text-[var(--cyan)]',
    dim: 'text-[var(--text-muted)]',
    user: 'text-[var(--text-bright)]',
    response: 'text-[var(--text)]',
    stats: 'text-[var(--text-muted)] text-xs',
  }

  return (
    <section ref={ref}>
      <h2 className="text-center text-xl font-semibold text-[var(--text-bright)] mb-8">
        See it in action
      </h2>

      <div className="bg-[#08080f] border border-[var(--card-border)] rounded-xl overflow-hidden shadow-2xl shadow-black/40">
        {/* Title bar */}
        <div className="flex items-center gap-2 px-4 py-3 bg-[#0c0c14] border-b border-[var(--card-border)]">
          <span className="w-3 h-3 rounded-full bg-[#ff5f57]" />
          <span className="w-3 h-3 rounded-full bg-[#febc2e]" />
          <span className="w-3 h-3 rounded-full bg-[#28c840]" />
          <span className="ml-2 text-xs text-[var(--text-muted)] font-mono">kandiga</span>
        </div>

        {/* Body */}
        <div className="p-4 sm:p-5 md:p-6 font-mono text-[11px] sm:text-[13px] leading-[1.8] min-h-[280px] sm:min-h-[340px]">
          {DEMO_LINES.slice(0, visibleLines).map((line, i) => {
            if (line.type === 'blank') return <div key={i} className="h-3" />
            return (
              <div key={i} className={lineColor[line.type] || 'text-[var(--text)]'}>
                {line.type === 'prompt' && <span className="text-[var(--cyan)] mr-2">$</span>}
                {line.type === 'user' && <span className="text-[var(--cyan)] font-bold mr-1">›</span>}
                {line.type === 'user' ? line.text.slice(1) : line.text}
              </div>
            )
          })}
          {visibleLines >= DEMO_LINES.length && <span className="cursor" />}
        </div>
      </div>
    </section>
  )
}
