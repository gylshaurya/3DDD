import { useEffect, useRef, useState, type CSSProperties } from 'react';
import { createViewer, showWebGPUError, type UpAxis, type Viewer, type ViewerStats } from '../viewer';

export interface SplatViewerProps {
  /** Scene URL to load (re-loads when it changes). */
  src?: string | null;
  up?: UpAxis;
  autoOrbit?: boolean;
  background?: [number, number, number];
  className?: string;
  style?: CSSProperties;
  /** Called once the GPU viewer exists. Use it for live snapshot swaps etc. */
  onReady?: (viewer: Viewer) => void;
  onStats?: (stats: ViewerStats) => void;
  onError?: (err: Error) => void;
  onLoaded?: () => void;
}

/** React wrapper around the WebGPU splat viewer. Owns the canvas and the GPU device lifetime. */
export function SplatViewer({
  src, up, autoOrbit = false, background, className, style, onReady, onStats, onError, onLoaded,
}: SplatViewerProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [viewer, setViewer] = useState<Viewer | null>(null);
  const cbs = useRef({ onReady, onStats, onError, onLoaded });
  cbs.current = { onReady, onStats, onError, onLoaded };

  useEffect(() => {
    let cancelled = false;
    let v: Viewer | null = null;
    createViewer(canvasRef.current!, { up, background, onError: (e) => cbs.current.onError?.(e) })
      .then((created) => {
        if (cancelled) {
          created.dispose();
          return;
        }
        v = created;
        v.onStats((s) => cbs.current.onStats?.(s));
        setViewer(v);
        cbs.current.onReady?.(v);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (wrapRef.current) showWebGPUError(wrapRef.current, err);
        cbs.current.onError?.(err instanceof Error ? err : new Error(String(err)));
      });
    return () => {
      cancelled = true;
      v?.dispose();
      setViewer(null);
    };
    // The GPU device is created once per mount. Option changes are applied below.
  }, []);

  useEffect(() => {
    viewer?.setAutoOrbit(autoOrbit);
  }, [viewer, autoOrbit]);

  useEffect(() => {
    if (viewer && up) viewer.setUpAxis(up);
  }, [viewer, up]);

  useEffect(() => {
    if (!viewer || !src) return;
    viewer.load(src).then(
      () => cbs.current.onLoaded?.(),
      (err: Error) => {
        if (err.message !== 'Load superseded') cbs.current.onError?.(err);
      },
    );
  }, [viewer, src]);

  return (
    // Positioning comes from className/style (the error overlay makes the box relative if needed).
    <div ref={wrapRef} className={className} style={{ overflow: 'hidden', ...style }}>
      <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block', outline: 'none' }} />
    </div>
  );
}
