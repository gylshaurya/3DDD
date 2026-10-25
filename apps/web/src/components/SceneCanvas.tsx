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

/** Download a file with progress (0..1); resolves to its bytes. */
export async function fetchWithProgress(
  url: string,
  onProgress: (f: number) => void,
  signal?: AbortSignal,
): Promise<ArrayBuffer> {
  const res = await fetch(url, { signal });
  if (!res.ok || !res.body) throw new Error(`Download failed (${res.status})`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (total) onProgress(got / total);
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  onProgress(1);
  return out.buffer;
}

/**
 * Show the compact .splat right away, then fetch the full .ply (view-dependent colour) in the
 * background and crossfade to it without moving the camera. Big scenes open in seconds on slow
 * connections and still end up at full quality.
 */
export async function loadProgressive(
  v: Viewer,
  splatUrl: string,
  plyUrl: string | null,
  onFull?: (f: number) => void,
  signal?: AbortSignal,
): Promise<void> {
  await v.load(splatUrl, { keepCamera: true });
  if (!plyUrl || signal?.aborted) return;
  const buf = await fetchWithProgress(plyUrl, (f) => onFull?.(f), signal);
  if (signal?.aborted) return;
  await v.setSplatsFromBuffer(buf, 'ply', { keepCamera: true });
}

interface Props {
  src: string | null;
  /** Optional full-quality version swapped in once downloaded. */
  fullSrc?: string | null;
  meta?: SceneMeta | null;
  autoOrbit?: boolean;
  className?: string;
  style?: CSSProperties;
  onReady?: (v: Viewer) => void;
  onStats?: (s: ViewerStats) => void;
  onLoaded?: () => void;
  onFullProgress?: (f: number) => void;
  onError?: (e: Error) => void;
}

/** A splat viewer that opens a captured scene from its "home" capture viewpoint. */
export function SceneCanvas({
  src, fullSrc, meta, autoOrbit, className, style, onReady, onStats, onLoaded, onFullProgress, onError,
}: Props) {
  const viewer = useRef<Viewer | null>(null);
  const loaded = useRef<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const cbs = useRef({ onLoaded, onError, onFullProgress });
  cbs.current = { onLoaded, onError, onFullProgress };

  const load = () => {
    const v = viewer.current;
    if (!v || !src || loaded.current === src) return;
    loaded.current = src;
    abort.current?.abort();
    abort.current = new AbortController();
    const { signal } = abort.current;
    if (meta) frameScene(v, meta);
    v.load(src, { keepCamera: !!meta }).then(
      () => {
        cbs.current.onLoaded?.();
        if (!fullSrc) return;
        fetchWithProgress(fullSrc, (f) => cbs.current.onFullProgress?.(f), signal)
          .then((buf) => (signal.aborted ? undefined : v.setSplatsFromBuffer(buf, 'ply', { keepCamera: true })))
          .catch((e: Error) => e.name !== 'AbortError' && cbs.current.onError?.(e));
      },
      (e: Error) => e.message !== 'Load superseded' && cbs.current.onError?.(e),
    );
  };

  useEffect(load);
  useEffect(() => () => abort.current?.abort(), []);

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
