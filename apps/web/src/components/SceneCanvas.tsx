import { useEffect, useRef, type CSSProperties } from 'react';
import type { SceneMeta } from '../lib/api';
import type { Viewer, ViewerStats } from '../viewer';
import { SplatViewer } from './SplatViewer';

/** Applies a scene's gravity + home view to a viewer, then loads without re-framing. */
export function frameScene(v: Viewer, meta: SceneMeta | null | undefined, zoom = 1): void {
  if (!meta) return;
  v.setUpAxis(meta.up);
  const eye = meta.home_eye.map((e, i) => meta.target[i] + (e - meta.target[i]) * zoom) as [number, number, number];
  v.lookAt(eye, meta.target);
}

interface Props {
  src: string | null;
  meta?: SceneMeta | null;
  autoOrbit?: boolean;
  className?: string;
  style?: CSSProperties;
  onReady?: (v: Viewer) => void;
  onStats?: (s: ViewerStats) => void;
  onLoaded?: () => void;
  onError?: (e: Error) => void;
}

/** A splat viewer that opens a captured scene from its "home" capture viewpoint. */
export function SceneCanvas({ src, meta, autoOrbit, className, style, onReady, onStats, onLoaded, onError }: Props) {
  const viewer = useRef<Viewer | null>(null);
  const loaded = useRef<string | null>(null);
  const cbs = useRef({ onLoaded, onError });
  cbs.current = { onLoaded, onError };

  const load = () => {
    const v = viewer.current;
    if (!v || !src || loaded.current === src) return;
    loaded.current = src;
    if (meta) frameScene(v, meta);
    v.load(src, { keepCamera: !!meta }).then(
      () => cbs.current.onLoaded?.(),
      (e: Error) => e.message !== 'Load superseded' && cbs.current.onError?.(e),
    );
  };

  useEffect(load);

  return (
    <SplatViewer
      className={className}
      style={style}
      autoOrbit={autoOrbit}
      onStats={onStats}
      onError={onError}
      onReady={(v) => {
        viewer.current = v;
        onReady?.(v);
        load();
      }}
    />
  );
}
