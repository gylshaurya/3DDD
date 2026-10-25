import { useEffect, useRef, useState } from 'react';
import { api, createJob, GPUS, type Backend, type Health, type Preset } from '../lib/api';
import { navigate } from '../lib/router';
import { QrCode } from './Chrome';

// Rough timings measured on the counter scene (150 frames) on an M4.
const SFM_S = 90;
const TRAIN_S: Record<Backend, Record<Preset, Record<string, number>>> = {
  modal: {
    preview: { L4: 110, L40S: 80, H100: 70 },
    fast: { L4: 450, L40S: 185, H100: 150 },
    quality: { L4: 1000, L40S: 420, H100: 330 },
  },
  local: { preview: { mac: 210 }, fast: { mac: 420 }, quality: { mac: 1100 } },
};
const RATE: Record<string, number> = { L4: 0.8, L40S: 1.95, H100: 3.95 };

const PRESETS: { id: Preset; label: string }[] = [
  { id: 'preview', label: 'Preview' },
  { id: 'fast', label: 'Fast' },
  { id: 'quality', label: 'Quality' },
];

export function NewCapture({ localAvailable, health }: { localAvailable: boolean; health: Health | null }) {
  const [backend, setBackend] = useState<Backend>('modal');
  const [preset, setPreset] = useState<Preset>('fast');
  const [gpu, setGpu] = useState('L40S');
  const [drag, setDrag] = useState(false);
  const [upload, setUpload] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [samples, setSamples] = useState<string[]>([]);
  const [lan, setLan] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.samples().then((s) => setSamples(s.map((x) => x.name)), () => {});
  }, []);
  useEffect(() => {
    if (!health) return;
    // Hosted: the public site. Local: this laptop's Wi-Fi address.
    if (health.public) setLan(`${location.origin}/capture`);
    else api.lan().then((l) => setLan(l.url), () => {});
  }, [health]);
  const limits = health?.limits ?? null;
  const presets = limits ? PRESETS.filter((p) => limits.presets.includes(p.id)) : PRESETS;
  const gpus = limits ? GPUS.filter((g) => limits.gpus.includes(g.id)) : GPUS;

  const trainS = TRAIN_S[backend][preset][backend === 'modal' ? gpu : 'mac'];
  const cost = backend === 'modal' ? ((RATE[gpu] + 0.45) * (trainS + 40)) / 3600 : 0;
  const opts = { backend, preset, gpu };

  const start = async (source: { file: File } | { sample: string }) => {
    setError(null);
    setUpload(0);
    try {
      const job = await createJob(source, opts, setUpload);
      navigate(`#/job/${job.id}`);
    } catch (e) {
      setError((e as Error).message);
      setUpload(null);
    }
  };

  const onFiles = (files: FileList | null) => {
    const f = files?.[0];
    if (!f) return;
    if (!f.type.startsWith('video/') && !/\.(mp4|mov|m4v|webm|mkv)$/i.test(f.name)) {
      setError('That does not look like a video file.');
      return;
    }
    void start({ file: f });
  };

  return (
    <div className="capture-card panel">
      <div
        className={`drop ${drag ? 'drag' : ''}`}
        role="button"
        tabIndex={0}
        onClick={() => input.current?.click()}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          onFiles(e.dataTransfer.files);
        }}
      >
        <input ref={input} type="file" accept="video/*" hidden onChange={(e) => onFiles(e.target.files)} />
        {upload === null ? (
          <>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
              <path d="M12 16V4m0 0-4.5 4.5M12 4l4.5 4.5M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />
            </svg>
            <div className="drop-title">Drop a video, or click to choose</div>
            <div className="drop-sub faint">Walk slowly in a circle around something · 20-60 s · MP4 / MOV</div>
          </>
        ) : (
          <>
            <div className="drop-title">{upload < 1 ? 'Uploading…' : 'Starting pipeline…'}</div>
            <div className="bar">
              <span style={{ width: `${Math.round(upload * 100)}%` }} />
            </div>
            <div className="drop-sub mono faint">{Math.round(upload * 100)}%</div>
          </>
        )}
      </div>

      <div className="opts">
        {!health?.public && (
        <div className="opt-row">
          <span className="opt-label">Train on</span>
          <div className="seg">
            <button aria-pressed={backend === 'modal'} onClick={() => setBackend('modal')}>
              Cloud GPU
            </button>
            <button
              aria-pressed={backend === 'local'}
              onClick={() => setBackend('local')}
              disabled={!localAvailable}
              title={localAvailable ? 'Train offline on this computer\'s GPU with Brush' : 'Brush is not installed: run uv run python -m threeddd.setup_tools'}
            >
              This computer
            </button>
          </div>
        </div>
        )}
        <div className="opt-row">
          <span className="opt-label">Quality</span>
          <div className="seg">
            {presets.map((p) => (
              <button key={p.id} aria-pressed={preset === p.id} onClick={() => setPreset(p.id)}>
                {p.label}
              </button>
            ))}
          </div>
        </div>
        {backend === 'modal' && (
          <div className="opt-row">
            <span className="opt-label">GPU</span>
            <div className="seg">
              {gpus.map((g) => (
                <button key={g.id} aria-pressed={gpu === g.id} onClick={() => setGpu(g.id)} title={g.note}>
                  {g.label}
                </button>
              ))}
            </div>
          </div>
        )}
        <div className="estimate mono">
          ≈ {Math.round((SFM_S + trainS) / 60)} min end-to-end
          {backend === 'modal' ? ` · ≈ $${cost.toFixed(2)} GPU` : ' · free, offline'}
        </div>
      </div>

      {limits && (
        <div className="limits faint">
          Free hosted demo: videos up to {limits.max_video_seconds} s / {limits.max_upload_mb} MB,{' '}
          {limits.jobs_per_ip_per_hour} captures per hour.
        </div>
      )}
      {error && <div className="error-line">{error}</div>}

      <div className="capture-foot">
        {lan && (
          <div className="phone">
            <QrCode text={lan} size={84} />
            <div>
              <div className="phone-title">Film it on your phone</div>
              <div className="faint phone-sub">
                {health?.public
                  ? 'Scan to record and upload straight from your phone camera.'
                  : 'Scan on the same Wi-Fi to record and upload straight from the camera.'}
              </div>
            </div>
          </div>
        )}
        {samples.length > 0 && (
          <div className="samples">
            <span className="faint">No video handy? Try a sample:</span>
            {samples.map((s) => (
              <button
                key={s}
                className="btn small"
                disabled={upload !== null}
                onClick={() => void start({ sample: s })}
                aria-label={`Run sample ${s}`}
              >
                {s}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
