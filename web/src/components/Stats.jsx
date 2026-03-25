import { useEffect, useRef, useState } from 'react'

function AnimatedNumber({ value, suffix = '', duration = 1500 }) {
  const [display, setDisplay] = useState(0)
  const ref = useRef(null)
  const started = useRef(false)

  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting && !started.current) {
        started.current = true
        const start = performance.now()
        const animate = (now) => {
          const progress = Math.min((now - start) / duration, 1)
          const eased = 1 - Math.pow(1 - progress, 3)
          setDisplay(eased * value)
          if (progress < 1) requestAnimationFrame(animate)
        }
        requestAnimationFrame(animate)
      }
    }, { threshold: 0.5 })

    if (ref.current) observer.observe(ref.current)
    return () => observer.disconnect()
  }, [value, duration])

  const formatted = Number.isInteger(value) ? Math.round(display) : display.toFixed(1)
  return <span ref={ref}>{formatted}{suffix}</span>
}

const stats = [
  { value: 20.4, suffix: 'GB', label: 'Model on disk', sub: '35 billion parameters' },
  { value: 1.5, suffix: 'GB', label: 'Memory used', sub: 'Less than a 4B dense model' },
  { value: 6.5, suffix: ' t/s', label: 'Fast mode', sub: 'K=4 on Apple M4' },
  { value: 3.8, suffix: 'x', label: 'Cache compression', sub: '16K+ context on 16GB Mac' },
]

const rows = [
  { label: 'Parameters', dense: '4 billion', moe: '35 billion', kandiga: '35 billion' },
  { label: 'RAM Usage', dense: '2.9 GB', moe: '~20 GB', kandiga: '1.5 GB', highlight: true },
  { label: 'Min. Mac', dense: '8 GB (any M1)', moe: '24 GB (M2 Pro+)', kandiga: '8 GB (any M1)' },
  { label: 'Speed', dense: '28 tok/s', moe: '~15 tok/s', kandiga: '3.4–6.5 tok/s' },
  { label: 'Quality', dense: 'Good', moe: 'Excellent', kandiga: 'Excellent' },
  { label: 'Expert Loading', dense: 'All weights', moe: 'All 256 in RAM', kandiga: '8 of 256 from disk' },
]

const models = [
  {
    name: '4B Dense',
    sub: 'Small model, fits anywhere',
    specs: [
      { label: 'RAM', value: '2.9 GB' },
      { label: 'Speed', value: '28 tok/s' },
      { label: 'Quality', value: 'Good' },
      { label: 'Min. Mac', value: '8 GB' },
    ],
  },
  {
    name: '35B MoE',
    sub: 'Standard loading',
    specs: [
      { label: 'RAM', value: '~20 GB', bad: true },
      { label: 'Speed', value: '~15 tok/s' },
      { label: 'Quality', value: 'Excellent' },
      { label: 'Min. Mac', value: '24 GB', bad: true },
    ],
  },
  {
    name: '35B MoE + Kandiga',
    sub: 'Selective Expert Materialization',
    featured: true,
    specs: [
      { label: 'RAM', value: '1.5 GB', good: true },
      { label: 'Speed', value: '3.4–6.5 tok/s' },
      { label: 'Quality', value: 'Excellent' },
      { label: 'Min. Mac', value: '8 GB', good: true },
    ],
  },
]

export default function Stats() {
  return (
    <section className="py-4">
      {/* Stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-16">
        {stats.map((s, i) => (
          <div key={i} className="feature-card text-center">
            <div className="text-2xl md:text-3xl font-bold text-[var(--text-bright)] mb-1 tracking-tight">
              <AnimatedNumber value={s.value} suffix={s.suffix} />
            </div>
            <div className="text-sm text-[var(--text)]">{s.label}</div>
            <div className="text-xs text-[var(--text-muted)]">{s.sub}</div>
          </div>
        ))}
      </div>

      <h3 className="text-lg font-semibold text-[var(--text-bright)] mb-5">How it compares</h3>

      {/* Mobile: 3 model cards stacked */}
      <div className="flex flex-col gap-4 md:hidden">
        {models.map((m, i) => (
          <div key={i} className={`rounded-xl p-4 border ${m.featured
            ? 'bg-[var(--card)] border-[var(--cyan)]/30'
            : 'bg-[var(--surface)] border-[var(--card-border)]'}`}>
            <div className="flex items-center justify-between mb-3">
              <div>
                <div className={`font-semibold text-sm ${m.featured ? 'text-[var(--cyan)]' : 'text-[var(--text-bright)]'}`}>{m.name}</div>
                <div className="text-[11px] text-[var(--text-muted)]">{m.sub}</div>
              </div>
              {m.featured && (
                <span className="text-[10px] font-medium text-[var(--cyan)] bg-[var(--cyan-dim)] px-2 py-0.5 rounded-full">Best</span>
              )}
            </div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-2">
              {m.specs.map((s, j) => (
                <div key={j} className="flex justify-between items-baseline">
                  <span className="text-[11px] text-[var(--text-muted)]">{s.label}</span>
                  <span className={`text-[13px] font-medium ${
                    s.good ? 'text-[var(--cyan)]' : s.bad ? 'text-red-400/80' : 'text-[var(--text)]'
                  }`}>{s.value}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Desktop: table */}
      <div className="hidden md:block feature-card overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--card-border)]">
              <th className="pb-3 text-left text-[var(--text-muted)] font-medium w-1/4"></th>
              <th className="pb-3 text-left text-[var(--text-muted)] font-medium w-1/4">4B Dense</th>
              <th className="pb-3 text-left text-[var(--text-muted)] font-medium w-1/4">35B MoE (standard)</th>
              <th className="pb-3 text-left text-[var(--cyan)] font-medium w-1/4">35B MoE (Kandiga)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className="border-b border-[var(--card-border)]/30 last:border-0">
                <td className="py-3 text-[var(--text-muted)]">{row.label}</td>
                <td className="py-3 text-[var(--text)]">{row.dense}</td>
                <td className="py-3 text-[var(--text)]">{row.moe}</td>
                <td className={`py-3 font-medium ${row.highlight ? 'text-[var(--cyan)]' : 'text-[var(--text-bright)]'}`}>{row.kandiga}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
