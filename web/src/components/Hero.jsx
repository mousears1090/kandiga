import { useEffect, useState, useRef } from 'react'

const ASCII_ART = `██╗  ██╗ █████╗ ███╗   ██╗██████╗ ██╗ ██████╗  █████╗
██║ ██╔╝██╔══██╗████╗  ██║██╔══██╗██║██╔════╝ ██╔══██╗
█████╔╝ ███████║██╔██╗ ██║██║  ██║██║██║  ███╗███████║
██╔═██╗ ██╔══██║██║╚██╗██║██║  ██║██║██║   ██║██╔══██║
██║  ██╗██║  ██║██║ ╚████║██████╔╝██║╚██████╔╝██║  ██║
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝ ╚═════╝ ╚═╝  ╚═╝`

function TypedAscii() {
  const [displayed, setDisplayed] = useState('')
  const ref = useRef(null)
  const started = useRef(false)

  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting && !started.current) {
        started.current = true
        let i = 0
        const chars = ASCII_ART
        const step = () => {
          // Type multiple chars per frame for speed
          const chunk = Math.min(i + 8, chars.length)
          setDisplayed(chars.slice(0, chunk))
          i = chunk
          if (i < chars.length) requestAnimationFrame(step)
        }
        requestAnimationFrame(step)
      }
    }, { threshold: 0.3 })

    if (ref.current) observer.observe(ref.current)
    return () => observer.disconnect()
  }, [])

  return (
    <div ref={ref} className="relative inline-block">
      <pre className="font-mono text-[var(--cyan)] text-[7px] sm:text-[9px] md:text-[11px] lg:text-[13px] leading-[1.15] select-none glow-cyan whitespace-pre">
        {displayed}
        <span className="inline-block w-[2px] sm:w-[3px] h-[8px] sm:h-[12px] bg-[var(--cyan)] ml-[1px] animate-pulse align-text-bottom" />
      </pre>
    </div>
  )
}

export default function Hero() {
  const [visible, setVisible] = useState(false)
  useEffect(() => { setTimeout(() => setVisible(true), 100) }, [])

  const installCmd = 'pip install kandiga && kandiga setup && kandiga chat'

  return (
    <section className="pt-28 sm:pt-36 md:pt-40 pb-12 sm:pb-16 md:pb-24 text-center">
      {/* Badge */}
      <div className={`inline-flex items-center gap-2 mb-8 px-4 py-1.5 rounded-full border border-[var(--card-border)] bg-[var(--card)] text-xs tracking-wide transition-all duration-700 ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
        <span className="w-1.5 h-1.5 rounded-full bg-[var(--green)]" />
        <span className="text-[var(--text-muted)]">Open Source</span>
        <span className="text-[var(--text-muted)]">·</span>
        <span className="text-[var(--text-muted)]">MIT License</span>
      </div>

      {/* ASCII Art Brand */}
      <div className={`mb-6 transition-all duration-700 delay-200 ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'}`}>
        <TypedAscii />
      </div>

      {/* Headline */}
      <div className={`transition-all duration-700 delay-300 ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'}`}>
        <p className="text-xl sm:text-2xl md:text-3xl font-medium text-[var(--text-bright)] mb-4 tracking-tight">
          35B intelligence. <span className="text-[var(--cyan)]">1.5GB memory.</span>
        </p>
        <p className="text-base md:text-lg text-[var(--text-muted)] max-w-lg mx-auto leading-relaxed">
          Run massive MoE models on any Mac.
          No cloud. No API keys. No compromises.
        </p>
      </div>

      {/* Install command */}
      <div className={`mt-12 max-w-xl mx-auto transition-all duration-700 delay-500 ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'}`}>
        <div className="bg-[var(--card)] border border-[var(--card-border)] rounded-lg px-4 sm:px-5 py-3 sm:py-4 text-left">
          <div className="flex items-start sm:items-center justify-between gap-3">
            <div className="flex items-start sm:items-center gap-2 sm:gap-3 font-mono text-xs sm:text-sm min-w-0">
              <span className="text-[var(--cyan)] select-none shrink-0 mt-0.5 sm:mt-0">$</span>
              <span className="text-[var(--text)] break-all sm:whitespace-nowrap">{installCmd}</span>
            </div>
            <button
              onClick={() => { navigator.clipboard.writeText(installCmd) }}
              className="shrink-0 bg-[var(--surface)] border border-[var(--card-border)] text-[var(--text-muted)] px-2.5 py-1 rounded text-xs cursor-pointer hover:text-[var(--text)] hover:border-[var(--cyan)] transition-all"
            >
              Copy
            </button>
          </div>
        </div>
        <p className="text-xs text-[var(--text-muted)] mt-4">
          Works on any Apple Silicon Mac · M1 through M4 · No prerequisites
        </p>
      </div>
    </section>
  )
}
