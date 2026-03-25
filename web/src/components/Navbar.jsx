import { useState } from 'react'

export default function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 backdrop-blur-xl bg-[var(--void)] border-b border-[var(--card-border)]" style={{ paddingTop: 'env(safe-area-inset-top, 0px)' }}>
      <div className="max-w-[1080px] mx-auto px-6 md:px-10 h-14 flex items-center justify-between">
        <a href="#" className="flex items-center gap-2 no-underline">
          <img src="/logo.png" alt="Kandiga" className="w-7 h-7 rounded" />
          <span className="font-bold text-[var(--cyan)] text-[15px] font-mono uppercase tracking-wider">kandiga</span>
        </a>

        <div className="hidden md:flex items-center gap-7">
          {[
            ['Features', '#features'],
            ['Architecture', '#architecture'],
            ['Install', '#install'],
          ].map(([label, href]) => (
            <a key={label} href={href}
               className="text-[13px] text-[var(--text-muted)] hover:text-[var(--text)] transition-colors no-underline">
              {label}
            </a>
          ))}
          <a href="https://github.com/kantheon/kandiga"
             target="_blank" rel="noopener noreferrer"
             className="text-[13px] text-[var(--text-muted)] hover:text-[var(--text)] transition-colors flex items-center gap-1.5 no-underline">
            <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
            </svg>
            GitHub
          </a>
        </div>

        <button className="md:hidden text-[var(--text)]" onClick={() => setMobileOpen(!mobileOpen)}>
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            {mobileOpen
              ? <path strokeLinecap="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12"/>
              : <path strokeLinecap="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16"/>}
          </svg>
        </button>
      </div>

      {mobileOpen && (
        <div className="md:hidden border-t border-[var(--card-border)] bg-[var(--void)]/95 backdrop-blur-xl px-6 py-3">
          {[['Features','#features'],['Architecture','#architecture'],['Install','#install']].map(([l,h]) => (
            <a key={l} href={h} onClick={() => setMobileOpen(false)}
               className="block py-2 text-sm text-[var(--text-muted)] hover:text-[var(--text)] no-underline">{l}</a>
          ))}
        </div>
      )}
    </nav>
  )
}
