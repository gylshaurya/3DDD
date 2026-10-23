import { useCallback, useRef, useState } from 'react';
import { SplatViewer } from './components/SplatViewer';
import type { Viewer, ViewerStats } from './viewer';
import './App.css';

const SAMPLES = [
  { label: 'Train', url: '/samples/train.splat' },
  { label: 'Bonsai (SH3)', url: '/samples/bonsai-7k.ply' },
  { label: 'Plush', url: '/samples/plush.splat' },
  { label: 'Sneaker', url: '/samples/nike.splat' },
];

export default function App() {
  const [src, setSrc] = useState<string>(SAMPLES[0].url);
  const [stats, setStats] = useState<ViewerStats | null>(null);
  const [orbit, setOrbit] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const viewerRef = useRef<Viewer | null>(null);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (!file || !viewerRef.current) return;
    setError(null);
    setSrc('');
    viewerRef.current.load(file).catch((err: Error) => setError(err.message));
  }, []);

  return (
    <div
      className={`app ${dragging ? 'dragging' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <SplatViewer
        className="stage"
        src={src || null}
        autoOrbit={orbit}
        onReady={(v) => { viewerRef.current = v; }}
        onStats={setStats}
        onError={(e) => setError(e.message)}
      />
      <header className="bar">
        <strong className="brand">Splatcast</strong>
        <nav>
          {SAMPLES.map((s) => (
            <button key={s.url} className={s.url === src ? 'on' : ''} onClick={() => { setError(null); setSrc(s.url); }}>
              {s.label}
            </button>
          ))}
          <button className={orbit ? 'on' : ''} onClick={() => setOrbit((o) => !o)}>Auto-orbit</button>
        </nav>
      </header>
      {stats && (
        <div className="hud">
          <span>{stats.fps.toFixed(0)} fps</span>
          <span>{stats.splatCount.toLocaleString()} splats</span>
          <span>sort {stats.sortMs >= 0 ? `${stats.sortMs.toFixed(1)} ms` : 'n/a'} · {stats.gpuSort ? 'GPU radix' : 'CPU'}</span>
          {stats.loadProgress < 1 && <span>loading {(stats.loadProgress * 100).toFixed(0)}%</span>}
        </div>
      )}
      {error && <div className="error" role="alert">{error}</div>}
      <p className="hint">Drag to orbit · right-drag / shift to pan · scroll to zoom · F fly · R reset · drop a .ply or .splat</p>
    </div>
  );
}
