import { useEffect, useState } from 'react';
import QRCode from 'qrcode';
import { api, type Budget } from '../lib/api';

export function Logo() {
  return (
    <a href="#/" className="logo" aria-label="3DDD home">
      <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
        <ellipse cx="9" cy="10" rx="6.5" ry="4" transform="rotate(-25 9 10)" fill="var(--accent)" opacity="0.95" />
        <ellipse cx="15.5" cy="14" rx="5" ry="3" transform="rotate(35 15.5 14)" fill="#7dd3fc" opacity="0.8" />
        <ellipse cx="11" cy="17" rx="3.4" ry="2" transform="rotate(-10 11 17)" fill="#f472b6" opacity="0.75" />
      </svg>
      <span>3DDD</span>
    </a>
  );
}

export function BudgetBadge() {
  const [b, setB] = useState<Budget | null>(null);
  useEffect(() => {
    let alive = true;
    let id: ReturnType<typeof setInterval> | undefined;
    // Spend is an operator concern: hidden on the hosted site.
    api.health().then((h) => {
      if (h.public || !alive) return;
      const tick = () => api.budget().then((x) => alive && setB(x), () => {});
      void tick();
      id = setInterval(tick, 30000);
    }, () => {});
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);
  if (!b) return null;
  const frac = Math.min(1, b.spent / b.cap);
  return (
    <div className="budget" title={`Modal metered: $${(b.metered ?? 0).toFixed(2)} · local ledger: $${b.ledger.toFixed(2)}`}>
      <span className="eyebrow">GPU budget</span>
      <span className="mono">
        ${b.spent.toFixed(2)} <span className="faint">/ ${b.cap.toFixed(0)}</span>
      </span>
      <span className="budget-bar">
        <span style={{ width: `${Math.max(2, frac * 100)}%` }} />
      </span>
    </div>
  );
}

export function TopBar({ children }: { children?: React.ReactNode }) {
  return (
    <header className="topbar">
      <Logo />
      <div className="topbar-mid">{children}</div>
      <BudgetBadge />
    </header>
  );
}

export function QrCode({ text, size = 148 }: { text: string; size?: number }) {
  const [svg, setSvg] = useState('');
  useEffect(() => {
    QRCode.toString(text, { type: 'svg', margin: 1, color: { dark: '#0b0c0f', light: '#ffffff' } }).then(setSvg, () => {});
  }, [text]);
  return (
    <div
      className="qr"
      style={{ width: size, height: size }}
      role="img"
      aria-label={`QR code for ${text}`}
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}

export function Stat({ label, value, sub }: { label: string; value: React.ReactNode; sub?: React.ReactNode }) {
  return (
    <div className="stat">
      <div className="eyebrow">{label}</div>
      <div className="stat-value mono">{value}</div>
      {sub != null && <div className="stat-sub faint">{sub}</div>}
    </div>
  );
}
