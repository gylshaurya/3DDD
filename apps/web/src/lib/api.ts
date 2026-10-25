// Typed client for the 3DDD server (server/src/threeddd/api.py).

/** API origin. Empty = same origin (local ./run.sh); set VITE_API_BASE for the hosted site. */
export const API_BASE = ((import.meta.env.VITE_API_BASE as string | undefined) ?? '').replace(/\/$/, '');

/** Resolve a server path ("/api/...", "/files/...") against the API origin. */
export const u = (path: string): string => (path.startsWith('/') ? API_BASE + path : path);

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
  featured?: boolean;
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

export interface Health {
  ok: boolean;
  local_trainer: boolean;
  /** Hosted deployment: anyone can upload, with limits. */
  public: boolean;
  limits: { max_upload_mb: number; max_video_seconds: number; presets: Preset[]; gpus: string[]; jobs_per_ip_per_hour: number } | null;
}

// ---- captures this browser started (hosted mode keeps strangers' uploads private) ----------
const MINE_KEY = 'threeddd.myJobs';
interface MyJob {
  id: string;
  token: string;
  created: number;
}

export function myJobs(): MyJob[] {
  try {
    return JSON.parse(localStorage.getItem(MINE_KEY) ?? '[]') as MyJob[];
  } catch {
    return [];
  }
}

function rememberJob(id: string, token: string): void {
  try {
    localStorage.setItem(MINE_KEY, JSON.stringify([{ id, token, created: Date.now() }, ...myJobs()].slice(0, 30)));
  } catch {
    /* private mode: the job still works, it just isn't listed under "your captures" */
  }
}

export function forgetJob(id: string): void {
  try {
    localStorage.setItem(MINE_KEY, JSON.stringify(myJobs().filter((j) => j.id !== id)));
  } catch {
    /* ignore */
  }
}

export const ownerToken = (id: string): string | undefined => myJobs().find((j) => j.id === id)?.token;
const ownerHeaders = (id: string): HeadersInit => {
  const t = ownerToken(id);
  return t ? { 'x-owner-token': t } : {};
};

/** Absolutise server-relative file URLs so the hosted site can load them from the API origin. */
function withUrls<T extends SceneSummary>(s: T): T {
  return { ...s, ply: u(s.ply), splat: u(s.splat), thumb: u(s.thumb) };
}

export const api = {
  health: () => fetch(u('/api/health')).then((r) => json<Health>(r)),
  budget: () => fetch(u('/api/budget')).then((r) => json<Budget>(r)),
  lan: () => fetch(u('/api/lan')).then((r) => json<{ ip: string; url: string }>(r)),
  samples: () => fetch(u('/api/samples')).then((r) => json<{ name: string; size_mb: number }[]>(r)),
  jobs: (ids?: string[]) =>
    fetch(u(ids ? `/api/jobs?ids=${encodeURIComponent(ids.join(','))}` : '/api/jobs')).then((r) => json<Job[]>(r)),
  job: (id: string) => fetch(u(`/api/jobs/${id}`)).then((r) => json<Job>(r)),
  scenes: (ids?: string[]) =>
    fetch(u(ids ? `/api/scenes?ids=${encodeURIComponent(ids.join(','))}` : '/api/scenes'))
      .then((r) => json<SceneSummary[]>(r))
      .then((list) => list.map(withUrls)),
  cancel: (id: string) =>
    fetch(u(`/api/jobs/${id}/cancel`), { method: 'POST', headers: ownerHeaders(id) }).then((r) => json<unknown>(r)),
  remove: (id: string) =>
    fetch(u(`/api/jobs/${id}`), { method: 'DELETE', headers: ownerHeaders(id) })
      .then((r) => json<unknown>(r))
      .then((r) => {
        forgetJob(id);
        return r;
      }),
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
    xhr.open('POST', u('/api/jobs'));
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
    xhr.onload = () => {
      try {
        const body = JSON.parse(xhr.responseText) as Job & { detail?: string; owner_token?: string };
        if (xhr.status >= 200 && xhr.status < 300) {
          if (body.owner_token) rememberJob(body.id, body.owner_token);
          resolve(body);
        }
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
