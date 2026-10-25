import { useEffect, useRef, useState } from 'react';
import { Logo } from '../components/Chrome';
import { api, createJob, type Preset } from '../lib/api';

/** Phone page opened from the QR code: record, upload, then watch it train. */
export function Capture() {
  const input = useRef<HTMLInputElement>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [preset, setPreset] = useState<Preset>('fast');
  const [presets, setPresets] = useState<Preset[]>(['preview', 'fast', 'quality']);
  useEffect(() => {
    api.health().then((h) => h.limits && setPresets(h.limits.presets), () => {});
  }, []);

  const onFile = async (file: File | undefined) => {
    if (!file) return;
    setError(null);
    setProgress(0);
    try {
      const job = await createJob({ file }, { backend: 'modal', preset, gpu: 'L40S', name: 'Phone capture' }, setProgress);
      location.assign(`/#/job/${job.id}`);
    } catch (e) {
      setError((e as Error).message);
      setProgress(null);
    }
  };

  return (
    <div className="capture-page">
      <Logo />
      <h1>Capture a scene</h1>
      <ol className="tips">
        <li>Pick one subject: a person, a plant, a shoe, a statue.</li>
        <li>Walk slowly all the way around it, keeping it in the centre of the frame.</li>
        <li>Keep it steady and well lit. 20-60 seconds is plenty.</li>
      </ol>
      <div className="seg">
        {presets.map((p) => (
          <button key={p} aria-pressed={preset === p} onClick={() => setPreset(p)}>
            {p[0].toUpperCase() + p.slice(1)}
          </button>
        ))}
      </div>
      <input ref={input} type="file" accept="video/*" capture="environment" hidden onChange={(e) => void onFile(e.target.files?.[0])} />
      {progress === null ? (
        <button className="btn primary big" onClick={() => input.current?.click()}>
          Record or choose video
        </button>
      ) : (
        <div className="uploading">
          <div>{progress < 1 ? `Uploading ${Math.round(progress * 100)}%` : 'Starting…'}</div>
          <div className="bar">
            <span style={{ width: `${Math.round(progress * 100)}%` }} />
          </div>
        </div>
      )}
      {error && <div className="error-line">{error}</div>}
    </div>
  );
}
