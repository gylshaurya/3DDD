import { useEffect, useState } from 'react';
import { TopBar } from '../components/Chrome';
import { NewCapture } from '../components/NewCapture';
import { SceneCanvas } from '../components/SceneCanvas';
import { api, formatSeconds, myJobs, type Health, type Job, type SceneSummary } from '../lib/api';

export function Home() {
  const [scenes, setScenes] = useState<SceneSummary[]>([]);
  const [mine, setMine] = useState<SceneSummary[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [heroReady, setHeroReady] = useState(false);

  useEffect(() => {
    let id: ReturnType<typeof setInterval> | undefined;
    api.health().then((h) => {
      setHealth(h);
      // On the hosted site the gallery shows featured scenes, and your uploads come from this browser.
      const ids = h.public ? myJobs().map((j) => j.id) : undefined;
      api.scenes().then(setScenes, () => {});
      if (ids?.length) api.scenes(ids).then((s) => setMine(s.filter((x) => !x.featured)), () => {});
      const tick = () => (ids && !ids.length ? Promise.resolve() : api.jobs(ids).then(setJobs, () => {}));
      void tick();
      id = setInterval(tick, 4000);
    }, () => {});
    return () => clearInterval(id);
  }, []);
  const localAvailable = !!health?.local_trainer;

  // Show the scene with the best held-out PSNR.
  const hero = scenes.reduce<SceneSummary | undefined>(
    (best, s) => ((s.metrics.train?.test_psnr ?? 0) > (best?.metrics.train?.test_psnr ?? 0) ? s : best),
    scenes[0],
  );
  const active = jobs.filter((j) => j.status === 'running' || j.status === 'queued');

  return (
    <div className="home">
      <section className="hero">
        {hero && !heroReady && <div className="hero-poster" style={{ backgroundImage: `url(${hero.thumb})` }} />}
        {hero && (
          <SceneCanvas
            className={`hero-canvas ${heroReady ? 'ready' : ''}`}
            src={hero.splat}
            meta={hero.scene}
            autoOrbit
            onLoaded={() => setHeroReady(true)}
          />
        )}
        <div className="hero-shade" />
        <TopBar />
        <div className="hero-content">
          <div className="hero-copy">
            <div className="eyebrow">Video → structure-from-motion → 3D Gaussian Splatting → WebGPU</div>
            <h1>
              Film anything.
              <br />
              <span className="accent">Walk around inside it</span> minutes later.
            </h1>
            <p className="dim lede">
              3DDD turns a phone video into a photoreal, explorable 3D scene. It solves every camera frame, trains
              hundreds of thousands of Gaussians on a cloud GPU while you watch, and renders the result here in a
             WebGPU renderer.
            </p>
            {active.length > 0 && (
              <div className="active-jobs">
                {active.map((j) => (
                  <a key={j.id} href={`#/job/${j.id}`} className="active-job panel">
                    <span className="pill run">
                      <span className="dot pulse" /> {j.status === 'queued' ? 'Queued' : 'Live'}
                    </span>
                    <span>{j.name}</span>
                    <span className="faint mono">{j.message}</span>
                    <span className="go">Watch →</span>
                  </a>
                ))}
              </div>
            )}
          </div>
          <NewCapture localAvailable={localAvailable} health={health} />
        </div>
        {hero && (
          <a className="hero-caption" href={`#/scene/${hero.id}`}>
            <span className="faint">Now orbiting</span> {hero.name} <span className="faint">· open →</span>
          </a>
        )}
      </section>

      <section className="gallery-section">
        <div className="section-head">
          <h2>Scenes</h2>
          <span className="faint">{scenes.length} captured</span>
        </div>
        {scenes.length === 0 ? (
          <div className="empty panel">
            <div>No scenes yet.</div>
            <div className="faint">Upload a video above, or run the sample, and your first scene will appear here.</div>
          </div>
        ) : (
          <div className="gallery">
            {scenes.map((s) => (
              <a key={s.id} className="card" href={`#/scene/${s.id}`}>
                <div className="card-img" style={{ backgroundImage: `url(${s.thumb})` }} />
                <div className="card-body">
                  <div className="card-title">{s.name}</div>
                  <div className="card-meta mono faint">
                    {s.metrics.train?.gaussians ? `${(s.metrics.train.gaussians / 1000).toFixed(0)}k splats` : ''}
                    {s.metrics.train?.test_psnr ? ` · ${s.metrics.train.test_psnr.toFixed(1)} dB` : ''}
                    {s.metrics.total_seconds ? ` · ${formatSeconds(s.metrics.total_seconds)}` : ''}
                  </div>
                </div>
              </a>
            ))}
          </div>
        )}
        {mine.length > 0 && (
          <>
            <div className="section-head mine-head">
              <h2>Your captures</h2>
              <span className="faint">only visible to you, via their links</span>
            </div>
            <div className="gallery">
              {mine.map((s) => (
                <a key={s.id} className="card" href={`#/scene/${s.id}`}>
                  <div className="card-img" style={{ backgroundImage: `url(${s.thumb})` }} />
                  <div className="card-body">
                    <div className="card-title">{s.name}</div>
                    <div className="card-meta mono faint">
                      {s.metrics.train?.test_psnr ? `${s.metrics.train.test_psnr.toFixed(1)} dB` : ''}
                      {s.metrics.total_seconds ? ` · ${formatSeconds(s.metrics.total_seconds)}` : ''}
                    </div>
                  </div>
                </a>
              ))}
            </div>
          </>
        )}
        <RecentJobs jobs={jobs.filter((j) => j.status === 'failed' || j.status === 'cancelled')} />
      </section>
    </div>
  );
}

function RecentJobs({ jobs }: { jobs: Job[] }) {
  if (!jobs.length) return null;
  return (
    <details className="recent">
      <summary className="faint">
        {jobs.length} unfinished run{jobs.length > 1 ? 's' : ''}
      </summary>
      {jobs.map((j) => (
        <a key={j.id} href={`#/job/${j.id}`} className="recent-row">
          <span className={`pill ${j.status === 'failed' ? 'bad' : ''}`}>{j.status}</span>
          <span>{j.name}</span>
          <span className="faint">{j.error ?? j.message}</span>
        </a>
      ))}
    </details>
  );
}
