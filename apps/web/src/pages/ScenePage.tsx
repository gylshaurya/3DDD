import { useEffect, useRef, useState } from 'react';
import { Stat, TopBar } from '../components/Chrome';
import { frameScene, SceneCanvas } from '../components/SceneCanvas';
import { api, formatSeconds, ownerToken, type SceneSummary } from '../lib/api';
import type { Viewer, ViewerStats } from '../viewer';

export function ScenePage({ id }: { id: string }) {
  const [scene, setScene] = useState<SceneSummary | null>(null);
  const [missing, setMissing] = useState(false);
  const [stats, setStats] = useState<ViewerStats | null>(null);
  const [orbit, setOrbit] = useState(false);
  const [fly, setFly] = useState(false);
  const [loading, setLoading] = useState(0);
  const [recording, setRecording] = useState<number | null>(null);
  const [info, setInfo] = useState(true);
  const [splatScale, setSplatScale] = useState(1);
  const [full, setFull] = useState<number | null>(null);
  const [isPublic, setIsPublic] = useState(false);
  const viewer = useRef<Viewer | null>(null);
  const wrap = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.health().then((h) => setIsPublic(h.public), () => {});
    api.scenes([id]).then((all) => {
      const s = all.find((x) => x.id === id);
      if (s) setScene(s);
      else setMissing(true);
    }, () => setMissing(true));
  }, [id]);

  useEffect(() => viewer.current?.setAutoOrbit(orbit), [orbit]);
  useEffect(() => viewer.current?.setFlyMode(fly), [fly]);
  useEffect(() => viewer.current?.setSplatScale(splatScale), [splatScale]);
  useEffect(() => setLoading(stats ? stats.loadProgress : 0), [stats]);

  const screenshot = () => {
    const v = viewer.current;
    if (!v) return;
    const a = document.createElement('a');
    a.href = v.captureFrame();
    a.download = `${scene?.name ?? 'scene'}.png`;
    a.click();
  };

  /** Record a 360 degree orbit from the canvas. */
  const recordTurntable = async () => {
    const v = viewer.current;
    const canvas = wrap.current?.querySelector('canvas');
    if (!v || !canvas || recording !== null) return;
    const mime = ['video/mp4;codecs=avc1', 'video/webm;codecs=vp9', 'video/webm'].find((m) => MediaRecorder.isTypeSupported(m));
    if (!mime) return;
    frameScene(v, scene?.scene);
    setFly(false);
    setOrbit(true);
    const rec = new MediaRecorder(canvas.captureStream(60), { mimeType: mime, videoBitsPerSecond: 16_000_000 });
    const chunks: Blob[] = [];
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    const seconds = 12;
    const started = performance.now();
    const tick = setInterval(() => setRecording(Math.min(1, (performance.now() - started) / (seconds * 1000))), 200);
    rec.start(250);
    setRecording(0);
    await new Promise((r) => setTimeout(r, seconds * 1000));
    rec.stop();
    await new Promise((r) => (rec.onstop = r));
    clearInterval(tick);
    setRecording(null);
    setOrbit(false);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob(chunks, { type: mime }));
    a.download = `${scene?.name ?? 'scene'}-turntable.${mime.startsWith('video/mp4') ? 'mp4' : 'webm'}`;
    a.click();
  };

  if (missing)
    return (
      <div className="center-page">
        <TopBar />
        <div className="empty panel">
          <div>That scene does not exist (any more).</div>
          <a className="btn small" href="#/">
            Back home
          </a>
        </div>
      </div>
    );

  const m = scene?.metrics;
  return (
    <div className="scene-page" ref={wrap}>
      <SceneCanvas
        className="stage-canvas"
        src={scene ? scene.splat : null}
        fullSrc={scene?.ply ?? null}
        onFullProgress={setFull}
        meta={scene?.scene}
        onReady={(v) => {
          viewer.current = v;
        }}
        onStats={setStats}
      />
      <div className="overlay-top">
        <TopBar>{scene && <div className="job-title"><span className="job-name">{scene.name}</span></div>}</TopBar>
      </div>
      {loading > 0 && loading < 1 && (
        <div className="load-bar">
          <span style={{ width: `${loading * 100}%` }} />
        </div>
      )}

      {info && m && (
        <aside className="side right panel scene-info">
          <button className="btn ghost small close" onClick={() => setInfo(false)} aria-label="Hide details">
            <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
              <path d="M3 3l6 6M9 3l-6 6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
            </svg>
          </button>
          <div className="metrics">
            <div className="eyebrow">How this was made</div>
            {m.sfm && (
              <Stat label="Camera solve" value={`${m.sfm.registered}/${m.sfm.num_frames}`} sub={`cameras · ${m.sfm.mean_reproj_error.toFixed(2)} px reprojection error`} />
            )}
            {m.train?.gaussians != null && <Stat label="Gaussians" value={m.train.gaussians.toLocaleString()} sub={`${m.train.steps?.toLocaleString()} training steps`} />}
            {m.train?.test_psnr != null && <Stat label="Held-out PSNR" value={`${m.train.test_psnr.toFixed(2)} dB`} sub="on views the model never trained on" />}
            {m.total_seconds != null && <Stat label="Video → 3D" value={formatSeconds(m.total_seconds)} sub={scene?.ply_mb ? `${scene.ply_mb} MB · SH degree 3` : undefined} />}
          </div>
        </aside>
      )}

      <div className="caption">
        <div className="caption-tools">
          <button className={`btn small ${orbit ? 'on' : ''}`} onClick={() => setOrbit((o) => !o)}>
            Auto-orbit
          </button>
          <button className={`btn small ${fly ? 'on' : ''}`} onClick={() => setFly((f) => !f)} title="WASD + mouse to fly">
            Fly mode
          </button>
          <button className="btn small" onClick={() => viewer.current && frameScene(viewer.current, scene?.scene)}>
            Reset view
          </button>
          <label className="scale-slider" title="Shrink every Gaussian to see what the scene is made of">
            <span className="eyebrow">Gaussian size</span>
            <input
              type="range"
              min={0.02}
              max={1}
              step={0.01}
              value={splatScale}
              onChange={(e) => setSplatScale(+e.target.value)}
              aria-label="Gaussian size"
            />
            <span className="mono">{Math.round(splatScale * 100)}%</span>
          </label>
          <button className="btn small" onClick={screenshot}>
            Screenshot
          </button>
          <button className="btn small" onClick={() => void recordTurntable()} disabled={recording !== null}>
            {recording !== null ? `Recording ${Math.round(recording * 100)}%` : 'Record turntable'}
          </button>
          {scene && (
            <a className="btn small" href={scene.ply} download={`${scene.name}.ply`}>
              Download .ply
            </a>
          )}
          {scene && (!isPublic || (!scene.featured && ownerToken(scene.id))) && (
            <button
              className="btn small ghost"
              onClick={() => {
                if (confirm(`Delete “${scene.name}” permanently?`)) void api.remove(scene.id).then(() => location.assign('/#/'));
              }}
            >
              Delete
            </button>
          )}
          {!info && (
            <button className="btn small" onClick={() => setInfo(true)}>
              Details
            </button>
          )}
        </div>
        {full !== null && full < 1 && (
          <div className="mono faint hud">Loading full-quality colour… {Math.round(full * 100)}%</div>
        )}
        {stats && (
          <div className="mono faint hud">
            {stats.fps.toFixed(0)} fps · {stats.splatCount.toLocaleString()} splats · {stats.visibleCount.toLocaleString()} visible · GPU radix sort{' '}
            {stats.sortMs >= 0 ? `${stats.sortMs.toFixed(1)} ms` : '-'}
            {fly ? ' · WASD to move, drag to look' : ' · drag to orbit, right-drag to pan, scroll to zoom'}
          </div>
        )}
      </div>
    </div>
  );
}
