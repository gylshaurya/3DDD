interface Props {
  values: number[];
  width?: number;
  height?: number;
  color?: string;
  /** Fixed lower bound for the y-axis (e.g. 0 for counts); otherwise fit to data. */
  floor?: number;
}

/** Minimal area sparkline; downsamples to at most `width` points. */
export function Sparkline({ values, width = 220, height = 44, color = 'var(--accent)', floor }: Props) {
  if (values.length < 2) return <svg width={width} height={height} aria-hidden="true" />;
  const step = Math.max(1, Math.ceil(values.length / width));
  const v = values.filter((_, i) => i % step === 0 || i === values.length - 1);
  const lo = floor ?? Math.min(...v);
  const hi = Math.max(...v);
  const span = hi - lo || 1;
  const pts = v.map((y, i) => [(i / (v.length - 1)) * width, height - 3 - ((y - lo) / span) * (height - 6)]);
  const line = pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join('');
  const id = `g${color.replace(/[^a-z0-9]/gi, '')}`;
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden="true" className="spark">
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.28" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={`${line}L${width},${height}L0,${height}Z`} fill={`url(#${id})`} />
      <path d={line} fill="none" stroke={color} strokeWidth="1.6" strokeLinejoin="round" />
    </svg>
  );
}
