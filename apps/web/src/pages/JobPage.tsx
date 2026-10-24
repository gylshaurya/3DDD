import { useEffect, useRef, useState } from 'react';
import { Stat, TopBar } from '../components/Chrome';
import { frameScene } from '../components/SceneCanvas';
import { Sparkline } from '../components/Sparkline';
import { SplatViewer } from '../components/SplatViewer';
import { api, formatSeconds, STAGES, type Stage } from '../lib/api';
import { useJobReplay, useJobStream, type JobStreamState } from '../lib/useJobStream';
import type { Viewer, ViewerStats } from '../viewer';

type Showing = 'none' | 'sfm' | 'snapshot' | 'final';

export function JobPage({ id }: { id: string }) {
  const s = useJobStream(id);
  return <JobView id={id} s={s} />;
}

/** Plays back a finished run exactly as it streamed live (demo insurance against flaky Wi-Fi). */
export function ReplayPage({ id }: { id: string }) {
  const [run, setRun] = useState(0);
  const s = useJobReplay(id, 70, run);
  return <JobView key={run} id={id} s={s} replay onRestart={() => setRun((r) => r + 1)} />;
}

function JobView({ id, s, replay = false, onRestart }: { id: string; s: JobStreamState; replay?: boolean; onRestart?: () => void }) {
  const viewer = useRef<Viewer | null>(null);
  const [ready, setReady] = useState(false);
  const [showing, setShowing] = useState<Showing>('none');
  const [orbit, setOrbit] = useState(true);
  const [stats, setStats] = useState<ViewerStats | null>(null);
  const shownStep = useRef(0);
  const showingRef = useRef<Showing>('none');
  showingRef.current = showing;
  const inflight = useRef(false);
  const pending = useRef<string | null>(null);

  const job = s.job;
  const meta = s.sfm?.scene ?? job?.scene ?? null;

  // 1) Camera solve finished: show the sparse cloud + camera frusta from a wide vantage point.
  useEffect(() => {
    const v = viewer.current;
    if (!ready || !v || !s.live || !s.sfm || showing !== 'none' || s.snapshot || s.final) return;
    frameScene(v, s.sfm.scene, 2.4);
    void v.load(s.sfm.url, { keepCamera: true });
    setShowing('sfm');
  }, [ready, s.live, s.sfm, s.snapshot, s.final, showing]);

  // 2) Live training snapshots: always fetch the newest, never queue a backlog.
  useEffect(() => {
    const v = viewer.current;
    if (!ready || !v || !s.live || !s.snapshot || s.final) return;
    const url = s.snapshot.url;
    const step = s.snapshot.step;
    if (step <= shownStep.current) return;
    if (inflight.current) {
      pending.current = url;
      return;
    }
    const pull = async (u: string, st: number) => {
      inflight.current = true;
      try {
        const buf = await fetch(u).then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error('gone'))));
        const first = shownStep.current === 0;
        shownStep.current = st;
        await v.setSplatsFromBuffer(buf, 'splat', { keepCamera: true });
        if (first && meta) {
          // Glide from the wide solve view into the capture's home viewpoint
          // (or jump straight there if the page was opened mid-training).
          if (showingRef.current === 'sfm') v.lookAt(meta.home_eye, meta.target, false);
          else frameScene(v, meta);
        }
        setShowing('snapshot');
      } catch {
        /* snapshot rotated away on the server; the next one will arrive shortly */
      } finally {
        inflight.current = false;
        const next = pending.current;
        pending.current = null;
        if (next && next !== u) {
          const m = /snap_(\d+)/.exec(next);
          if (m && +m[1] > shownStep.current) void pull(next, +m[1]);
        }
      }
    };
    void pull(url, step);
  }, [ready, s.live, s.snapshot, s.final, meta]);

  // 3) Done: swap in the full-quality scene (view-dependent SH colour) without moving the camera.
  useEffect(() => {
    const v = viewer.current;
    if (!ready || !v || !s.live || !s.final || showing === 'final') return;
    if (showing === 'none') frameScene(v, meta);
    void v.load(s.final.ply, { keepCamera: true });
    setShowing('final');
  }, [ready, s.live, s.final, showing, meta]);

  useEffect(() => {
    viewer.current?.setAutoOrbit(orbit);
  }, [orbit, ready]);

  const running = job?.status === 'running' || job?.status === 'queued';

  return (
    <div className="job-page">
      <SplatViewer
        className="stage-canvas"
        onReady={(v) => {
          viewer.current = v;
          v.setAutoOrbit(orbit);
          setReady(true);
        }}
        onStats={setStats}
      />
      {showing === 'none' && s.thumb && (
        <div className="thumb-backdrop" style={{ backgroundImage: `url(${s.thumb})` }}>
          <div className="scanline" />
        </div>
      )}
      {showing === 'none' && s.frames && <ContactSheet frames={s.frames} solving={s.job?.stage === 'sfm'} />}
      <div className="overlay-top">
        <TopBar>
          {job && (
            <div className="job-title">
              <span className="job-name">{job.name}</span>
              <StatusPill status={job.status} />
              {replay && <span className="pill">replay</span>}
            </div>
          )}
        </TopBar>
      </div>

      <aside className="side left panel">
        <Timeline s={s} replay={replay} />
        <Console s={s} />
        <div className="side-actions">
          {running && !replay && (
            <button className="btn small" onClick={() => void api.cancel(id)}>
              Cancel run
            </button>
          )}
          {replay && (
            <button className="btn small" onClick={onRestart}>
              Restart replay
            </button>
          )}
          {!replay && job?.status === 'done' && (
            <a className="btn small" href={`#/replay/${id}`}>
              Replay training
            </a>
          )}
          {job?.status === 'done' && (
            <>
              <a className="btn small primary" href={`#/scene/${id}`}>
                Open scene
              </a>
              {s.final && (
                <a className="btn small" href={s.final.ply} download={`${job.name}.ply`}>
                  Download .ply
                </a>
              )}
            </>
          )}
        </div>
      </aside>

      <div className="right-col">
        <aside className="side right panel">
          <Metrics s={s} />
        </aside>
        {s.compares.length > 0 && <Compares s={s} />}
      </div>

      <div className="caption">
        <Caption s={s} showing={showing} />
        <div className="caption-tools">
          <button className={`btn small ${orbit ? 'on' : ''}`} onClick={() => setOrbit((o) => !o)}>
            Auto-orbit
          </button>
          {stats && (
            <span className="mono faint hud">
              {stats.fps.toFixed(0)} fps · {stats.splatCount.toLocaleString()} splats · sort{' '}
              {stats.sortMs >= 0 ? `${stats.sortMs.toFixed(1)} ms` : '—'}
            </span>
          )}
        </div>
      </div>

      {s.connectionLost && <div className="toast">Reconnecting to server…</div>}
    </div>
  );
}

function ContactSheet({ frames, solving }: { frames: { count: number; urls: string[] }; solving: boolean }) {
  return (
    <div className={`contact ${solving ? 'solving' : ''}`} aria-hidden="true">
      <div className="contact-grid">
        {frames.urls.map((u, i) => (
          <div key={u} className="contact-cell" style={{ animationDelay: `${i * 45}ms` }}>
            <img src={u} alt="" />
            <span className="contact-idx mono">{String(Math.round((i * frames.count) / frames.urls.length)).padStart(3, '0')}</span>
          </div>
        ))}
      </div>
      <div className="contact-caption eyebrow">
        {frames.count} sharpest frames selected{solving ? ' · matching features across views' : ''}
      </div>
    </div>
  );
}

function StatusPill({ status }: { status: string }) {
  const cls = status === 'done' ? 'ok' : status === 'failed' ? 'bad' : status === 'cancelled' ? '' : 'run';
  return (
    <span className={`pill ${cls}`}>
      {(status === 'running' || status === 'queued') && <span className="dot pulse" />}
      {status}
    </span>
  );
}

function Timeline({ s, replay }: { s: JobStreamState; replay: boolean }) {
  const job = s.job;
  const order = STAGES.map((x) => x.key);
  const cur = job?.stage ? order.indexOf(job.stage) : job?.status === 'done' ? order.length : -1;
  return (
    <ol className="timeline">
      {STAGES.map((st, i) => {
        const state = job?.status === 'done' || i < cur ? 'done' : i === cur ? (job?.status === 'running' ? 'active' : 'stopped') : 'todo';
        const secs = job?.timings[st.key]?.seconds;
        const p = s.stageProgress[st.key] ?? 0;
        return (
          <li key={st.key} className={`tl ${state}`}>
            <span className="tl-mark">{state === 'done' ? '✓' : i + 1}</span>
            <div className="tl-body">
              <div className="tl-head">
                <span className="tl-label">{st.label}</span>
                <span className="mono faint">
                  {secs != null ? formatSeconds(secs) : state === 'active' && !replay ? <Elapsed since={job?.timings[st.key]?.start} /> : ''}
                </span>
              </div>
              <div className="tl-detail faint">{state === 'active' ? (s.stageMsg[st.key] ?? st.detail) : st.detail}</div>
              {state === 'active' && (
                <div className="bar thin">
                  <span style={{ width: `${Math.max(3, p * 100)}%` }} />
                </div>
              )}
              {st.key === 'sfm' && s.sfm && (
                <div className="tl-facts mono">
                  {s.sfm.stats.registered}/{s.sfm.stats.num_frames} cameras · {(s.sfm.stats.points3d / 1000).toFixed(1)}k pts ·{' '}
                  {s.sfm.stats.mean_reproj_error.toFixed(2)} px
                </div>
              )}
            </div>
          </li>
        );
      })}
      {s.job?.error && <li className="tl-error">{s.job.error}</li>}
    </ol>
  );
}

function Elapsed({ since }: { since?: number }) {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now() / 1000), 500);
    return () => clearInterval(id);
  }, []);
  return <>{since ? formatSeconds(now - since) : ''}</>;
}

function Console({ s }: { s: JobStreamState }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    ref.current?.scrollTo({ top: ref.current.scrollHeight });
  }, [s.logs.length]);
  return (
    <div className="console mono" ref={ref} aria-live="polite">
      {s.logs.slice(-60).map((l, i) => (
        <div key={i} className={l.error ? 'err' : ''}>
          <span className="faint">{new Date(l.t * 1000).toLocaleTimeString([], { hour12: false })}</span> {l.msg}
        </div>
      ))}
    </div>
  );
}

function Metrics({ s }: { s: JobStreamState }) {
  const p = s.progress;
  const t = s.job?.metrics.train;
  const psnr = s.curve.map((c) => c.psnr);
  const gs = s.curve.map((c) => c.gaussians);
  const isLocal = s.job?.backend === 'local';
  return (
    <div className="metrics">
      <div className="eyebrow">{isLocal ? 'Brush · this Mac' : `Modal · ${s.job?.gpu ?? ''}`}</div>
      <Stat
        label="Step"
        value={p ? `${p.step.toLocaleString()}` : '—'}
        sub={p ? `of ${p.steps.toLocaleString()}${p.it_s ? ` · ${p.it_s} it/s` : ''}` : 'waiting for GPU'}
      />
      {!isLocal && (
        <>
          <Stat
            label="Training PSNR"
            value={p?.psnr != null ? `${p.psnr.toFixed(2)} dB` : '—'}
            sub={t?.test_psnr != null ? `held-out views: ${t.test_psnr.toFixed(2)} dB` : 'how closely renders match the photos'}
          />
          <Sparkline values={psnr} />
          <Stat
            label="Gaussians"
            value={p?.gaussians != null ? p.gaussians.toLocaleString() : '—'}
            sub="MCMC densification, capped"
          />
          <Sparkline values={gs} color="#7dd3fc" floor={0} />
        </>
      )}
      <Stat label="Live snapshots" value={s.snapshotCount || '—'} sub={s.snapshot ? `${s.snapshot.splats.toLocaleString()} splats streamed` : undefined} />
      {s.job?.metrics.total_seconds != null && (
        <Stat label="Video → 3D" value={formatSeconds(s.job.metrics.total_seconds)} sub={s.job.metrics.ply_mb ? `${s.job.metrics.ply_mb} MB scene` : undefined} />
      )}
    </div>
  );
}

function Caption({ s, showing }: { s: JobStreamState; showing: Showing }) {
  const stage: Stage | null | undefined = s.job?.stage;
  let text = 'Connecting…';
  if (s.job?.status === 'failed') text = 'Run failed. See the log for details.';
  else if (s.job?.status === 'cancelled') text = 'Run cancelled.';
  else if (showing === 'final') text = 'Final scene, full spherical-harmonic colour. Drag to orbit, scroll to zoom.';
  else if (showing === 'snapshot' && s.snapshot)
    text = `Live from the GPU: step ${s.snapshot.step.toLocaleString()}, ${s.snapshot.splats.toLocaleString()} Gaussians`;
  else if (showing === 'sfm' && s.sfm)
    text = `Camera solve: ${s.sfm.stats.registered} phone positions recovered (gold), ${s.sfm.stats.points3d.toLocaleString()} 3D points`;
  else if (stage === 'frames') text = 'Reading the video and picking the sharpest frames…';
  else if (stage === 'sfm') text = 'Solving where the phone was for every frame…';
  else if (s.job?.status === 'queued') text = 'Queued behind another run…';
  return <div className="caption-text">{text}</div>;
}

function Compares({ s }: { s: JobStreamState }) {
  const [open, setOpen] = useState(true);
  return (
    <div className={`compares panel ${open ? 'open' : ''}`}>
      <button className="compares-head" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <span>
          <span className="eyebrow">Unseen views</span> photo <span className="faint">|</span> render
        </span>
        <span className="faint">{open ? 'Hide' : 'Show'}</span>
      </button>
      {open && (
        <div className="compares-row">
          {s.compares.map((c) => (
            <a key={c.url} href={c.url} target="_blank" rel="noreferrer">
              <img src={c.url} alt="Real photo next to the rendered view" loading="lazy" />
            </a>
          ))}
        </div>
      )}
    </div>
  );
}
