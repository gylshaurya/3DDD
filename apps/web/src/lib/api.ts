// Typed client for the Splatcast server (server/src/splatcast/api.py).

export type Vec3 = [number, number, number];
export type JobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled';
export type Stage = 'frames' | 'sfm' | 'train' | 'export';
export type Backend = 'modal' | 'local';
export type Preset = 'preview' | 'fast' | 'quality';

export interface SceneMeta {
  up: Vec3;
  target: Vec3;
  home_eye: Vec3;
  orbit_radius: number;
  num_cameras: number;
}

export interface SfmStats {
  num_frames: number;
  registered: number;
  points3d: number;
  mean_reproj_error: number;
  mapper: string;
  seconds: number;
}

export interface TrainMetrics {
  steps?: number;
  gaussians?: number;
  train_psnr?: number;
  test_psnr?: number;
  train_seconds?: number;
}

export interface Job {
  id: string;
  name: string;
  backend: Backend;
  preset: Preset;
  gpu: string;
  created: number;
  status: JobStatus;
  stage: Stage | null;
  stage_progress: number;
  message: string;
  error: string | null;
  metrics: { sfm?: SfmStats; train?: TrainMetrics; total_seconds?: number; ply_mb?: number };
  scene: SceneMeta | null;
  timings: Partial<Record<Stage, { start: number; seconds?: number }>>;
}

export interface TrainProgress {
  step: number;
  steps: number;
  loss?: number;
  psnr?: number;
  gaussians?: number;
  it_s?: number | null;
  elapsed?: number;
}

export type JobEvent = { t: number } & (
  | { type: 'job'; job: Job }
  | { type: 'stage'; stage: Stage }
  | { type: 'stage_progress'; stage: Stage; progress: number; msg: string }
  | { type: 'log'; msg: string; level?: 'error' }
  | { type: 'thumb'; url: string }
  | { type: 'frames'; count: number; urls: string[] }
  | { type: 'sfm'; url: string; scene: SceneMeta; stats: SfmStats }
  | ({ type: 'train_progress' } & TrainProgress)
  | { type: 'snapshot'; step: number; url: string; splats: number }
  | { type: 'compare'; url: string; held_out: boolean }
  | { type: 'metrics'; metrics: Job['metrics'] }
  | { type: 'done'; ply: string; splat: string; metrics: Job['metrics'] }
);

export interface SceneSummary {
  id: string;
  name: string;
  metrics: Job['metrics'];
  scene: SceneMeta;
  ply: string;
  splat: string;
  thumb: string;
  ply_mb: number | null;
}

export interface Budget {
  cap: number;
  metered: number | null;
  ledger: number;
  spent: number;
  remaining: number;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = ((await res.json()) as { detail?: string }).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => fetch('/api/health').then((r) => json<{ ok: boolean; local_trainer: boolean }>(r)),
  budget: () => fetch('/api/budget').then((r) => json<Budget>(r)),
  lan: () => fetch('/api/lan').then((r) => json<{ ip: string; url: string }>(r)),
  samples: () => fetch('/api/samples').then((r) => json<{ name: string; size_mb: number }[]>(r)),
  jobs: () => fetch('/api/jobs').then((r) => json<Job[]>(r)),
  job: (id: string) => fetch(`/api/jobs/${id}`).then((r) => json<Job>(r)),
  scenes: () => fetch('/api/scenes').then((r) => json<SceneSummary[]>(r)),
  cancel: (id: string) => fetch(`/api/jobs/${id}/cancel`, { method: 'POST' }).then((r) => json<unknown>(r)),
  remove: (id: string) => fetch(`/api/jobs/${id}`, { method: 'DELETE' }).then((r) => json<unknown>(r)),
};

export interface JobOptions {
  backend: Backend;
  preset: Preset;
  gpu: string;
  name?: string;
}

/** Upload with progress (fetch has no upload progress events). */
export function createJob(
  source: { file: File } | { sample: string },
  opts: JobOptions,
  onProgress?: (fraction: number) => void,
): Promise<Job> {
  const form = new FormData();
  if ('file' in source) form.append('video', source.file);
  else form.append('sample', source.sample);
  form.append('backend', opts.backend);
  form.append('preset', opts.preset);
  form.append('gpu', opts.gpu);
  if (opts.name) form.append('name', opts.name);
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/jobs');
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
    xhr.onload = () => {
      try {
        const body = JSON.parse(xhr.responseText) as Job & { detail?: string };
        if (xhr.status >= 200 && xhr.status < 300) resolve(body);
        else reject(new Error(body.detail ?? `Upload failed (${xhr.status})`));
      } catch {
        reject(new Error(`Upload failed (${xhr.status})`));
      }
    };
    xhr.onerror = () => reject(new Error('Network error during upload'));
    xhr.send(form);
  });
}

export function formatSeconds(s: number | undefined | null): string {
  if (s == null || !Number.isFinite(s)) return '—';
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s - m * 60).toString().padStart(2, '0')}s`;
}

export const STAGES: { key: Stage; label: string; detail: string }[] = [
  { key: 'frames', label: 'Frames', detail: 'Pick the sharpest frames from the video' },
  { key: 'sfm', label: 'Camera solve', detail: 'SIFT → matching → global structure-from-motion' },
  { key: 'train', label: 'Gaussian training', detail: 'Optimise millions of 3D Gaussians on the GPU' },
  { key: 'export', label: 'Export', detail: 'Pack the scene for the web' },
];

export const GPUS = [
  { id: 'L4', label: 'L4', note: '$0.80/h · thrifty' },
  { id: 'L40S', label: 'L40S', note: '$1.95/h · fast' },
  { id: 'H100', label: 'H100', note: '$3.95/h · fastest' },
];
